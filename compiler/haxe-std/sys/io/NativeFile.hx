package sys.io;

import sys.io.NativeFileInput;
import sys.io.NativeFileOutput;

@:noDoc
extern class NativeFile {
    public static function read(path:String, binary:Bool):NativeFileInput;
    public static function write(path:String, binary:Bool):NativeFileOutput;
    public static function append(path:String, binary:Bool):NativeFileOutput;
    public static function update(path:String, binary:Bool):NativeFileOutput;
}
