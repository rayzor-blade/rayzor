package nue.chat;

/**
 * Why an assistant turn ended.
 *
 * - Stop:        a stop token was emitted (the turn ended naturally)
 * - Length:      the `maxNewTokens` cap was reached
 * - ContextFull: the KV cache reached the context window
 * - Aborted:     a streaming callback returned false
 * - Unknown:     reason could not be determined
 */
enum FinishReason {
    Stop;
    Length;
    ContextFull;
    Aborted;
    Unknown;
}

/**
 * The result of one assistant turn: the generated text plus token usage and
 * the reason generation stopped. Mirrors the shape of a modern chat
 * completion so callers can branch on `finishReason` (e.g. warn on `Length`,
 * offer "continue" on `ContextFull`) and account for token usage.
 */
class ChatResponse {
    /** The assistant's reply text (the generated tail, no prompt markup). */
    public var text:String;
    /** Tokens in the rendered prompt that seeded this turn. */
    public var promptTokens:Int;
    /** Tokens the model generated in this turn. */
    public var completionTokens:Int;
    /** Why generation stopped. */
    public var finishReason:FinishReason;

    public function new(
        text:String, promptTokens:Int, completionTokens:Int, finishReason:FinishReason
    ) {
        this.text = text;
        this.promptTokens = promptTokens;
        this.completionTokens = completionTokens;
        this.finishReason = finishReason;
    }

    public function totalTokens():Int {
        return promptTokens + completionTokens;
    }

    /** Lowercase display name of the finish reason. */
    public function finishReasonName():String {
        var n = "unknown";
        switch (finishReason) {
            case Stop: n = "stop";
            case Length: n = "length";
            case ContextFull: n = "context";
            case Aborted: n = "aborted";
            case Unknown: n = "unknown";
        }
        return n;
    }

    /** Map a loop's string finish reason to the typed enum. */
    public static function reasonFrom(s:String):FinishReason {
        if (s == "stop") return Stop;
        if (s == "length") return Length;
        if (s == "context") return ContextFull;
        if (s == "aborted") return Aborted;
        return Unknown;
    }
}
