import sys.io.File;

class Main {
    static function main() {
        // Write test file with binary data
        var output = File.write("/tmp/rayzor_test_readall.bin");
        output.writeByte(65);  // 'A'
        output.writeByte(66);  // 'B'
        output.writeByte(67);  // 'C'
        output.close();

        // Test readAll
        var input = File.read("/tmp/rayzor_test_readall.bin");
        var bytes = input.readAll();
        input.close();

        trace(bytes.length);  // 3
        trace(bytes.get(0));  // 65
        trace(bytes.get(1));  // 66
        trace(bytes.get(2));  // 67

        // Test readLine
        var out2 = File.write("/tmp/rayzor_test_readline.txt");
        // Write "Hello\nWorld\n"
        out2.writeByte(72);   // H
        out2.writeByte(101);  // e
        out2.writeByte(108);  // l
        out2.writeByte(108);  // l
        out2.writeByte(111);  // o
        out2.writeByte(10);   // \n
        out2.writeByte(87);   // W
        out2.writeByte(111);  // o
        out2.writeByte(114);  // r
        out2.writeByte(108);  // l
        out2.writeByte(100);  // d
        out2.writeByte(10);   // \n
        out2.close();

        var in2 = File.read("/tmp/rayzor_test_readline.txt");
        var line1 = in2.readLine();
        trace(line1);  // Hello
        var line2 = in2.readLine();
        trace(line2);  // World
        in2.close();

        trace("done");
    }
}
