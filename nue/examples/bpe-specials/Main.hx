import nue.tokenizer.BPETokenizer;
import nue.tokenizer.MergeRule;
import nue.tokenizer.Vocab;
import haxe.ds.StringMap;

class Main {
    static function main() {
        var s = "Paris";
        var bytes = haxe.io.Bytes.ofString(s);
        trace("s.len=" + s.length);
        trace("s.cca0=" + s.charCodeAt(0));
        trace("bytes.len=" + bytes.length);
        trace("bytes0=" + bytes.get(0));
        var spaceAlias = String.fromCharCode(0x120);
        trace("alias.len=" + spaceAlias.length);
        trace("alias.cca0=" + spaceAlias.charCodeAt(0));
        trace("alias.cca1=" + spaceAlias.charCodeAt(1));
        var sm = new StringMap<Int>();
        sm.set("P", 80);
        sm.set(spaceAlias, 32);
        trace("map.P.exists=" + sm.exists("P"));
        trace("map.P=" + sm.get("P"));
        trace("map.alias.exists=" + sm.exists(spaceAlias));
        trace("map.alias=" + sm.get(spaceAlias));
        var table:Array<String> = [for (_ in 0...256) ""];
        var hexCount = 0;
        for (b in 0x21...0x7F) hexCount++;
        var decCount = 0;
        for (b in 33...127) decCount++;
        trace("hexCount=" + hexCount);
        trace("decCount=" + decCount);
        trace("hexLo=" + 0x21);
        trace("hexHi=" + 0x7F);
        var t2:Array<String> = [for (_ in 0...256) ""];
        trace("t2.length=" + t2.length);
        t2[80] = String.fromCharCode(80);
        trace("t2.80.direct.len=" + t2[80].length);
        var ch80 = String.fromCharCode(80);
        t2[81] = ch80;
        trace("t2.81.temp.len=" + t2[81].length);
        var t3:Array<String> = [for (_ in 0...256) ""];
        trace("t3.length=" + t3.length);
        for (b in 33...127) {
            var ch = String.fromCharCode(b);
            t3[b] = ch;
        }
        trace("t3.80.tempLoop.len=" + t3[80].length);
        trace("t3.80.tempLoop.cca0=" + t3[80].charCodeAt(0));
        var t4:Array<String> = [];
        var next4 = 0x100;
        for (b in 0...256) {
            if ((b >= 0x21 && b < 0x7F) || (b >= 0xA1 && b < 0xAD) || (b >= 0xAE && b < 0x100)) {
                t4.push(String.fromCharCode(b));
            } else {
                t4.push(String.fromCharCode(next4));
                next4++;
            }
        }
        trace("t4.80.push.len=" + t4[80].length);
        trace("t4.length=" + t4.length);
        trace("t4.80.push.cca0=" + t4[80].charCodeAt(0));
        trace("t4.32.push.len=" + t4[32].length);
        trace("t4.32.push.cca0=" + t4[32].charCodeAt(0));
        trace("t4.32.push.cca1=" + t4[32].charCodeAt(1));
        for (b in 0x21...0x7F) table[b] = String.fromCharCode(b);
        trace("table.length=" + table.length);
        for (b in 0xA1...0xAD) table[b] = String.fromCharCode(b);
        for (b in 0xAE...0x100) table[b] = String.fromCharCode(b);
        var next = 0x100;
        for (b in 0...256) {
            if (table[b] == "") {
                table[b] = String.fromCharCode(next);
                next++;
            }
        }
        trace("table80.len=" + table[80].length);
        trace("table80.cca0=" + table[80].charCodeAt(0));
        trace("table80.eqP=" + (table[80] == "P"));
        trace("table32.len=" + table[32].length);
        trace("table32.cca0=" + table[32].charCodeAt(0));
        trace("table32.cca1=" + table[32].charCodeAt(1));
        trace("table32.eqAlias=" + (table[32] == spaceAlias));

        var vocab = new Vocab();
        vocab.add("P");
        vocab.add("a");
        vocab.add("r");
        vocab.add("s");
        vocab.add("h");
        vocab.add("i");
        var specialId = vocab.add("<|x|>");

        var merges:Array<MergeRule> = [];
        var tok = new BPETokenizer(vocab, merges, true);
        tok.addSpecial("<|x|>", specialId);

        trace("sid=" + tok.specialId("<|x|>"));

        var ids = tok.encode("<|x|>");
        trace("len=" + ids.length);
        trace("id0=" + ids[0]);
        var raw = tok.encode("Paris");
        trace("raw.len=" + raw.length);
        for (i in 0...raw.length) trace("raw." + i + "=" + raw[i]);
    }
}
