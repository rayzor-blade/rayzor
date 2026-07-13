package nue.sampling;

import nue.arch.LlamaModel;
import nue.tokenizer.BPETokenizer;
import rayzor.ds.Tensor;

/**
 * Prompt-lookup speculative decode for direct-run experiments.
 *
 * This does not use a draft model. Instead it finds an earlier matching
 * n-gram in the already-tokenized prompt/output, drafts the continuation,
 * verifies the draft with one batched target-model forward, and rewinds the
 * KV cache to the accepted prefix on mismatch.
 */
class SpeculativeGenerationLoop {
    public var model:LlamaModel;
    public var tokenizer:BPETokenizer;
    public var sampler:LocalTempSampler;
    public var eosId:Int;
    public var maxNewTokens:Int;
    public var maxDraft:Int;
    public var ngram:Int;

    var batches:Int;
    var drafted:Int;
    var accepted:Int;
    var rejected:Int;
    var fallback:Int;
    var profile:Bool;
    var specDisabled:Bool;
    var minAcceptPct:Int;

    public function new(
        model:LlamaModel,
        tokenizer:BPETokenizer,
        sampler:LocalTempSampler,
        eosId:Int,
        maxNewTokens:Int
    ) {
        this.model = model;
        this.tokenizer = tokenizer;
        this.sampler = sampler;
        this.eosId = eosId;
        this.maxNewTokens = maxNewTokens;
        this.maxDraft = envInt("RAYZOR_SPEC_MAX", 4);
        this.ngram = envInt("RAYZOR_SPEC_NGRAM", 4);
        if (this.maxDraft < 2) this.maxDraft = 2;
        if (this.ngram < 1) this.ngram = 1;
        this.minAcceptPct = envInt("RAYZOR_SPEC_MIN_ACCEPT", 70);
        if (this.minAcceptPct < 0) this.minAcceptPct = 0;
        if (this.minAcceptPct > 100) this.minAcceptPct = 100;
        this.profile = envOn("RAYZOR_SPEC_PROFILE");
        this.batches = 0;
        this.drafted = 0;
        this.accepted = 0;
        this.rejected = 0;
        this.fallback = 0;
        this.specDisabled = false;
    }

    public function generate(prompt:String, onToken:Int->String->Bool):String {
        if (profile) {
            trace("[spec] prompt_lookup max=" + maxDraft + " ngram=" + ngram);
        }

        model.resetCache();

        var ids:Array<Int> = tokenizer.encode(prompt);
        if (ids.length == 0) return prompt;

        var logits:Tensor = useFastLastLogits()
            ? model.forwardLastLogits(ids)
            : model.forwardIds(ids);
        var lr0:Tensor = lastRow(logits);
        var nextId:Int = sampler.sample(lr0);
        if (lr0 != logits) lr0.free();

        var parts:Array<String> = [];
        var carryHolder = [haxe.io.Bytes.alloc(0)];
        var step = 0;

        while (true) {
            if (nextId == eosId) break;
            if (maxNewTokens > 0 && step >= maxNewTokens) break;

            var remaining = maxDraft;
            if (maxNewTokens > 0) {
                remaining = maxNewTokens - step;
                if (remaining > maxDraft) remaining = maxDraft;
            }

            if (!specDisabled && remaining > 1) {
                var draftIds:Array<Int> = draftPromptLookup(ids, nextId, remaining, ngram);
                if (draftIds.length > 1) {
                    batches++;
                    drafted += draftIds.length - 1;

                    var baseLen:Int = model.cacheLen();
                    logits.free();

                    var verifyLogits:Tensor = model.forwardIds(draftIds);
                    var emitted = 0;
                    var stopNow = false;
                    var resumedAfterReject = false;
                    var i = 0;

                    if (!emitToken(draftIds[0], parts, carryHolder, onToken)) stopNow = true;
                    ids.push(draftIds[0]);
                    step++;
                    emitted++;
                    if (maxNewTokens > 0 && step >= maxNewTokens) stopNow = true;

                    i = 1;
                    while (!stopNow && i < draftIds.length) {
                        var row:Tensor = rowAt(verifyLogits, i - 1);
                        var target:Int = sampler.sample(row);
                        if (row != verifyLogits) row.free();

                        if (target == draftIds[i]) {
                            accepted++;
                            if (!emitToken(target, parts, carryHolder, onToken)) stopNow = true;
                            ids.push(target);
                            step++;
                            emitted++;
                            if (maxNewTokens > 0 && step >= maxNewTokens) stopNow = true;
                            i++;
                        } else {
                            rejected++;
                            if (shouldDisableSpec()) specDisabled = true;
                            model.rewindCache(baseLen + emitted);
                            var terminalAfterReject = false;

                            if (target == eosId) {
                                terminalAfterReject = true;
                            } else {
                                if (!emitToken(target, parts, carryHolder, onToken)) {
                                    terminalAfterReject = true;
                                }
                                ids.push(target);
                                step++;

                                if (maxNewTokens > 0 && step >= maxNewTokens) {
                                    terminalAfterReject = true;
                                }

                                if (!terminalAfterReject) {
                                    verifyLogits.free();
                                    var replacementIds:Array<Int> = [target];
                                    logits = model.forwardLastLogits(replacementIds);
                                    var lrR:Tensor = lastRow(logits);
                                    nextId = sampler.sample(lrR);
                                    if (lrR != logits) lrR.free();
                                    resumedAfterReject = true;
                                } else {
                                    stopNow = true;
                                }
                            }
                            break;
                        }
                    }

                    if (resumedAfterReject) continue;

                    if (stopNow) {
                        model.rewindCache(baseLen + emitted);
                        logits = verifyLogits;
                        break;
                    }

                    if (i >= draftIds.length) {
                        var lrS:Tensor = rowAt(verifyLogits, draftIds.length - 1);
                        nextId = sampler.sample(lrS);
                        if (lrS != verifyLogits) lrS.free();
                        logits = verifyLogits;
                        continue;
                    }

                    logits = verifyLogits;
                    break;
                }
            }

            fallback++;
            ids.push(nextId);
            if (!emitToken(nextId, parts, carryHolder, onToken)) break;
            step++;
            if (maxNewTokens > 0 && step >= maxNewTokens) break;

            logits.free();
            var nextIds:Array<Int> = [nextId];
            logits = model.forwardLastLogits(nextIds);
            var lrN:Tensor = lastRow(logits);
            nextId = sampler.sample(lrN);
            if (lrN != logits) lrN.free();
        }

        if (profile) {
            var rate = (drafted > 0) ? (accepted * 100.0 / drafted) : 0.0;
            trace("[profile-spec] batches=" + batches
                + " drafted=" + drafted
                + " accepted=" + accepted
                + " rejected=" + rejected
                + " fallback=" + fallback
                + " disabled=" + (specDisabled ? 1 : 0)
                + " accept_pct=" + rate);
        }

        logits.free();
        var tail = (carryHolder[0].length > 0) ? carryHolder[0].toString() : "";
        return prompt + parts.join("") + tail;
    }

    function emitToken(
        id:Int,
        parts:Array<String>,
        carryHolder:Array<haxe.io.Bytes>,
        onToken:Int->String->Bool
    ):Bool {
        var delta:String = tokenizer.decodeStreamStep(carryHolder, id);
        parts.push(delta);
        if (onToken != null) return onToken(id, delta);
        return true;
    }

    static function lastRow(logits:Tensor):Tensor {
        var shape:Array<Int> = logits.shape();
        if (shape.length <= 1) return logits;
        var lastIdx = shape[0] - 1;
        var vocab = shape[shape.length - 1];
        var sliced:Tensor = logits.slice(0, lastIdx, lastIdx + 1);
        var reshaped:Tensor = sliced.reshape([vocab]);
        sliced.free();
        return reshaped;
    }

    static function useFastLastLogits():Bool {
        var v = Sys.getEnvOr("NUE_PREFILL_LAST_LOGITS", "RAYZOR_PREFILL_LAST_LOGITS");
        return v != null && v != "0" && v != "" && v.toLowerCase() != "false";
    }

    static function rowAt(logits:Tensor, row:Int):Tensor {
        var shape:Array<Int> = logits.shape();
        if (shape.length <= 1) return logits;
        var vocab = shape[shape.length - 1];
        var sliced:Tensor = logits.slice(0, row, row + 1);
        var reshaped:Tensor = sliced.reshape([vocab]);
        sliced.free();
        return reshaped;
    }

    static function envOn(name:String):Bool {
        var v = Sys.getEnv(name);
        return v != null && v != "0" && v != "" && v.toLowerCase() != "false";
    }

    static function envInt(name:String, fallbackValue:Int):Int {
        var v = Sys.getEnv(name);
        if (v == null || v == "") return fallbackValue;
        var parsed = Std.parseInt(v);
        if (parsed == null) return fallbackValue;
        return parsed;
    }

    function shouldDisableSpec():Bool {
        if (drafted <= 0) return false;
        // The verifier batch uses seqQ>1 and therefore cannot take the
        // single-token Q8 flash path. If prompt lookup misses immediately,
        // keep the rest of the run on the known-fast decode route.
        if (accepted == 0) return true;
        if (batches < 3) return false;
        return accepted * 100 < drafted * minAcceptPct;
    }

    static function draftPromptLookup(
        ids:Array<Int>, firstId:Int, maxDraft:Int, ngram:Int
    ):Array<Int> {
        var out:Array<Int> = [firstId];
        if (maxDraft <= 1 || ids.length == 0) return out;

        if (ngram <= 1) {
            var best1 = -1;
            var p1 = 0;
            while (p1 + 1 < ids.length) {
                if (ids[p1] == firstId) best1 = p1;
                p1++;
            }
            if (best1 >= 0) {
                var q1 = best1 + 1;
                while (out.length < maxDraft && q1 < ids.length) {
                    out.push(ids[q1]);
                    q1++;
                }
            }
            return out;
        }

        var prefixLen = ngram - 1;
        if (ids.length < prefixLen) return out;

        var best = -1;
        var p = 0;
        while (p + ngram < ids.length) {
            var ok = true;
            var j = 0;
            while (j < prefixLen && ok) {
                if (ids[p + j] != ids[ids.length - prefixLen + j]) ok = false;
                j++;
            }
            if (ok && ids[p + prefixLen] == firstId) best = p;
            p++;
        }

        if (best >= 0) {
            var q = best + ngram;
            while (out.length < maxDraft && q < ids.length) {
                out.push(ids[q]);
                q++;
            }
        }
        return out;
    }
}
