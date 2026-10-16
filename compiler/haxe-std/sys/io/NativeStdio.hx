package sys.io;

import sys.io.NativeFileInput;
import sys.io.NativeFileOutput;

@:noDoc
extern class NativeStdio {
    public static function stdin():NativeFileInput;
    public static function stdout():NativeFileOutput;
    public static function stderr():NativeFileOutput;
}
