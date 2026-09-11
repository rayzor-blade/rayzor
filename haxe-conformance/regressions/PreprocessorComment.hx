// A trailing comment on a #if line is not part of the condition. The stdlib
// and the Haxe test corpus both write `#if !hl // too many arguments in HL`;
// feeding the comment to the evaluator made the condition false and silently
// compiled the wrong branch -- which, for a test whose whole body sits in the
// active branch, left an empty method the backend then trap-stubbed.
class PreprocessorComment {
    static function main() {
        var taken = 0;
        #if !hl
        taken++;
        #else
        throw "plain condition";
        #end

        #if !hl // trailing line comment
        taken++;
        #else
        throw "condition with a trailing line comment";
        #end

        #if !hl /* trailing block comment */
        taken++;
        #else
        throw "condition with a trailing block comment";
        #end

        #if hl // inverted, with a comment
        throw "negative case took the wrong branch";
        #elseif !hl // and on the elseif too
        taken++;
        #else
        throw "elseif with a trailing comment";
        #end

        if (taken != 4) throw "expected 4 branches taken, got " + taken;
        trace("CONFORMANCE_OK");
    }
}
