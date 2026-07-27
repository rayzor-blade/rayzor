package nue.chat;

import nue.tokenizer.BPETokenizer;
import nue.model.Architecture;

/**
 * Arch-aware chat prompt formatter. Turns a message list into the exact
 * string an instruct model was fine-tuned on, and reports the stop-token
 * ids that terminate an assistant turn.
 *
 * Construct via `ChatTemplate.forModel(arch, tok)`: it probes the
 * tokenizer's registered specials (the reliable signal) and falls back to
 * the architecture name. This is what lets a single example serve Llama-3
 * (header format) and Qwen2 (ChatML) with no per-model branching in user
 * code — the previous llama-chat example hardcoded the Llama-3 markup and
 * silently fed Qwen a raw, untemplated prompt.
 */
class ChatTemplate {
    public static inline var LLAMA3 = "llama3"; // <|start_header_id|>role<|end_header_id|>\n\n … <|eot_id|>
    public static inline var CHATML = "chatml"; // <|im_start|>role\n … <|im_end|>\n   (Qwen2, Yi, …)
    public static inline var MISTRAL = "mistral"; // <s>[INST] … [/INST] …</s>  (Mistral, Llama-2)
    public static inline var RAW = "raw";       // no wrapping — base (non-instruct) models

    public var kind:String;

    public function new(kind:String) {
        this.kind = kind;
    }

    /**
     * Pick a template from the model's registered specials, falling back
     * to the architecture family. Registered specials win: a GGUF that
     * carries the Llama-3 header ids is a Llama-3 chat model regardless of
     * what the `architecture` string claims.
     */
    public static function forModel(arch:Architecture, tok:BPETokenizer):ChatTemplate {
        // Registered specials win — a GGUF carrying the Llama-3 header ids is
        // a Llama-3 chat model regardless of the declared architecture.
        if (tok.specialId("<|start_header_id|>") >= 0) return new ChatTemplate(LLAMA3);
        if (tok.specialId("<|im_start|>") >= 0) return new ChatTemplate(CHATML);
        // No modern chat specials, but a sentencepiece-era `</s>`: this is the
        // [INST] family (Mistral, Llama-2). It must be checked BEFORE the arch
        // fallback, because a Mistral GGUF declares `general.architecture` as
        // "llama" — `fromString` can never return `Mistral` for one — and the
        // Llama-3 branch would then emit <|start_header_id|> markup as literal
        // text the tokenizer shreds into subwords the model never saw.
        if (tok.specialId("</s>") >= 0) return new ChatTemplate(MISTRAL);
        // Fall back to the architecture family.
        switch (arch) {
            case Qwen2: return new ChatTemplate(CHATML);
            case Mistral: return new ChatTemplate(MISTRAL);
            case Llama: return new ChatTemplate(LLAMA3);
            default: return new ChatTemplate(RAW);
        }
    }

    /**
     * Render `messages` into a single prompt string. When
     * `addGenerationPrompt` is true, append the opener for a fresh
     * assistant turn so the model continues in the assistant role.
     */
    public function render(messages:Array<Message>, addGenerationPrompt:Bool):String {
        if (kind == CHATML) return renderChatML(messages, addGenerationPrompt);
        if (kind == MISTRAL) return renderMistral(messages, addGenerationPrompt);
        if (kind == RAW) return renderRaw(messages, addGenerationPrompt);
        return renderLlama3(messages, addGenerationPrompt);
    }

    function renderLlama3(messages:Array<Message>, addGen:Bool):String {
        var parts:Array<String> = ["<|begin_of_text|>"];
        for (m in messages) {
            parts.push("<|start_header_id|>");
            parts.push(m.role);
            parts.push("<|end_header_id|>\n\n");
            parts.push(m.content);
            parts.push("<|eot_id|>");
        }
        if (addGen) parts.push("<|start_header_id|>assistant<|end_header_id|>\n\n");
        return parts.join("");
    }

    function renderChatML(messages:Array<Message>, addGen:Bool):String {
        var parts:Array<String> = [];
        for (m in messages) {
            parts.push("<|im_start|>");
            parts.push(m.role);
            parts.push("\n");
            parts.push(m.content);
            parts.push("<|im_end|>\n");
        }
        if (addGen) parts.push("<|im_start|>assistant\n");
        return parts.join("");
    }

    function renderMistral(messages:Array<Message>, _addGen:Bool):String {
        // Mistral / Llama-2 instruction format. Neither family has a system
        // ROLE: the system text is folded into the first instruction, which is
        // what both were fine-tuned on. The trailing `[/INST]` IS the assistant
        // opener, so `addGen` needs no separate branch — a rendered user turn
        // always invites a completion.
        var sys = "";
        for (m in messages) if (m.role == "system") sys = m.content;
        var parts:Array<String> = ["<s>"];
        for (m in messages) {
            if (m.role == "system") continue;
            if (m.role == "assistant") {
                parts.push(m.content);
                parts.push("</s>");
                continue;
            }
            parts.push("[INST] ");
            if (sys != "") {
                parts.push(sys);
                parts.push("\n\n");
                sys = ""; // first instruction only
            }
            parts.push(m.content);
            parts.push(" [/INST]");
        }
        return parts.join("");
    }

    function renderRaw(messages:Array<Message>, _addGen:Bool):String {
        // Base models have no turn structure — concatenate content in order.
        // A generation-prompt opener would be meaningless here, so ignore it.
        var parts:Array<String> = [];
        for (m in messages) {
            parts.push(m.content);
            parts.push("\n");
        }
        return parts.join("");
    }

    /**
     * Stop-token ids that end an assistant turn, most-specific first,
     * filtered to those the tokenizer actually registers (-1 dropped).
     */
    public function stopIds(tok:BPETokenizer):Array<Int> {
        var names:Array<String> =
            (kind == CHATML) ? ["<|im_end|>", "<|endoftext|>"]
            : (kind == MISTRAL) ? ["</s>", "<|end_of_text|>"]
            : (kind == RAW) ? ["<|end_of_text|>", "<|endoftext|>"]
            : ["<|eot_id|>", "<|end_of_text|>"];
        var ids:Array<Int> = [];
        for (n in names) {
            var id = tok.specialId(n);
            if (id >= 0) ids.push(id);
        }
        return ids;
    }
}
