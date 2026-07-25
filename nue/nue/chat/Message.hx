package nue.chat;

/**
 * One turn in a conversation.
 *
 * `role` is an open string ("system" | "user" | "assistant", or any custom
 * role a template understands) so callers aren't limited to the three
 * standard roles.
 */
class Message {
    public var role:String;
    public var content:String;

    public function new(role:String, content:String) {
        this.role = role;
        this.content = content;
    }
}
