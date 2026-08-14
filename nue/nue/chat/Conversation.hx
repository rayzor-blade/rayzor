package nue.chat;

import nue.CausalLanguageModel;
import nue.tokenizer.BPETokenizer;
import nue.sampling.GenerationLoop;
#if nue_spec_decode
import nue.sampling.SpeculativeGenerationLoop;
#end
import nue.sampling.LocalTempSampler;
import nue.model.ModelMetadata;
import nue.model.Architecture.Architectures;

/**
 * High-level multi-turn chat API over a causal language model.
 *
 * Owns the pieces a chat app otherwise hand-rolls: an arch-aware prompt
 * template (Llama-3 headers vs ChatML, auto-detected), per-arch stop
 * tokens, sampling config, and turn history. Each turn reports token usage
 * and a finish reason via `ChatResponse`.
 *
 * Typical use:
 * ```haxe
 *   var chat = Conversation.fromLoaded(model, tok, meta);
 *   chat.system("You are a helpful assistant.");
 *   var res = chat.ask("What is a Voronoi diagram?", onToken);
 *   trace(res.text + " [" + res.completionTokens + " tokens, " + res.finishReason + "]");
 * ```
 *
 * Turns are stateless: each `ask` renders the full history through the
 * template and prefills it, so a reply never depends on leftover cache
 * state from a prior turn.
 */
class Conversation {
    public var model:CausalLanguageModel;
    public var tokenizer:BPETokenizer;
    public var template:ChatTemplate;
    public var messages:Array<Message>;
    /** Sampling + length controls applied to each `ask`. */
    public var config:GenerationConfig;

    var stopIdList:Array<Int>;

    public function new(
        model:CausalLanguageModel, tokenizer:BPETokenizer, template:ChatTemplate
    ) {
        this.model = model;
        this.tokenizer = tokenizer;
        this.template = template;
        this.messages = [];
        this.config = new GenerationConfig();
        this.stopIdList = template.stopIds(tokenizer);
    }

    /**
     * Build a Conversation from a loader result, auto-selecting the chat
     * template from the architecture + the tokenizer's registered specials.
     */
    public static function fromLoaded(
        model:CausalLanguageModel, tokenizer:BPETokenizer, meta:ModelMetadata
    ):Conversation {
        var arch = Architectures.fromString((meta != null) ? meta.architecture : null);
        return new Conversation(model, tokenizer, ChatTemplate.forModel(arch, tokenizer));
    }

    // ---- History ----------------------------------------------------------

    public function system(text:String):Conversation {
        messages.push(new Message("system", text));
        return this;
    }

    public function addUser(text:String):Conversation {
        messages.push(new Message("user", text));
        return this;
    }

    public function addAssistant(text:String):Conversation {
        messages.push(new Message("assistant", text));
        return this;
    }

    /** Clear all turns (keeps model/tokenizer/template/config). */
    public function reset():Void {
        messages = [];
    }

    // ---- Introspection ----------------------------------------------------

    /** The stop-token ids that end an assistant turn for this template. */
    public function stopIds():Array<Int> {
        return stopIdList;
    }

    /**
     * The exact prompt `ask(userText, …)` will send to the model, without
     * mutating history. Useful for logging or a tokenizer diff harness.
     */
    public function nextPrompt(userText:String):String {
        var preview:Array<Message> = [];
        for (m in messages) preview.push(m);
        preview.push(new Message("user", userText));
        return template.render(preview, true);
    }

    // ---- Generation -------------------------------------------------------

    /**
     * Add a user turn, generate the assistant reply, record it in history,
     * and return a `ChatResponse` (reply text + token usage + finish reason).
     * `onToken(id, delta)` streams each token's decoded text; return `false`
     * to abort. Generation stops at a template stop token, `maxNewTokens`, or
     * the KV-cache context bound — whichever comes first.
     */
    public function ask(text:String, onToken:Int->String->Bool):ChatResponse {
        messages.push(new Message("user", text));
        var prompt = template.render(messages, true);
        var promptTokens = tokenizer.encode(prompt).length;
        var sampler = buildSampler();

        // Count completion tokens as they stream, then forward to the caller.
        var completion = [0];
        var counting = function(id:Int, delta:String):Bool {
            completion[0] = completion[0] + 1;
            return (onToken != null) ? onToken(id, delta) : true;
        };

        var eos = (stopIdList.length > 0) ? stopIdList[0] : -1;
        var full:String;
        var finish:String;
        if (config.useSpeculative) {
            #if nue_spec_decode
            var loop = new SpeculativeGenerationLoop(model, tokenizer, sampler, eos, config.maxNewTokens);
            loop.stopIds = stopIdList;
            full = loop.generate(prompt, counting);
            finish = loop.finishReason;
            #else
            var loop = new GenerationLoop(model, tokenizer, sampler, eos, config.maxNewTokens);
            loop.stopIds = stopIdList;
            full = loop.generate(prompt, counting);
            finish = loop.finishReason;
            #end
        } else {
            var loop = new GenerationLoop(model, tokenizer, sampler, eos, config.maxNewTokens);
            loop.stopIds = stopIdList;
            full = loop.generate(prompt, counting);
            finish = loop.finishReason;
        }

        // generate() returns prompt + tail; the assistant reply is the tail.
        var reply = (full.length >= prompt.length) ? full.substr(prompt.length) : full;
        messages.push(new Message("assistant", reply));
        return new ChatResponse(
            reply, promptTokens, completion[0], ChatResponse.reasonFrom(finish)
        );
    }

    /** Non-streaming convenience: generate silently and return the response. */
    public function askSilent(text:String):ChatResponse {
        return ask(text, function(_id:Int, _delta:String) return true);
    }

    // ---- Internals --------------------------------------------------------

    function buildSampler():LocalTempSampler {
        // temp ~0 → deterministic-greedy (small top-K, tiny temperature); the
        // rep-penalty still applies before the pick so even greedy breaks
        // loops without adding randomness.
        var k = (config.temperature > 0.0) ? config.topK : 8;
        var t = (config.temperature > 0.0) ? config.temperature : 1e-4;
        return new LocalTempSampler(t, config.repetitionPenalty, k, config.seed);
    }
}
