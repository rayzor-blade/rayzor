package sys.io;

import sys.io.NativeFileInput;
import sys.io.NativeFileOutput;

@:noDoc
extern class NativeProcess {
    public function new(cmd:String, args:Array<String>, detached:Bool):Void;
    public function takeStdin():NativeFileOutput;
    public function takeStdout():NativeFileInput;
    public function takeStderr():NativeFileInput;
    public function getPid():Int;
    public function hasExited():Bool;
    public function waitForExit():Int;
    public function kill():Void;
    public function close():Void;
}
