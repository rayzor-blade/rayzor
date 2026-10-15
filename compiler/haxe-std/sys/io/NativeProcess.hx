package sys.io;

import sys.io.FileInput;
import sys.io.FileOutput;

@:noDoc
extern class NativeProcess {
    public function new(cmd:String, args:Array<String>, detached:Bool):Void;
    public function takeStdin():FileOutput;
    public function takeStdout():FileInput;
    public function takeStderr():FileInput;
    public function getPid():Int;
    public function hasExited():Bool;
    public function waitForExit():Int;
    public function kill():Void;
    public function close():Void;
}
