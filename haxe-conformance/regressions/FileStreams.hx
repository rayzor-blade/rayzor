import haxe.io.Input;
import haxe.io.Output;
import sys.io.File;
import sys.io.FileSeek;

class FileStreams {
 static function check(value:Bool, label:String):Void { if (!value) throw label; }
 static function exercise(path:String):Void {
  var file = File.write(path, false);
  var output:Output = file;
  output.writeString("λ\nabc!");
  output.flush();
  check(file.tell() == 7, "output position");
  output.close();
  output.close();
  var source = File.read(path);
  var input:Input = source;
  check(input.readLine() == "λ", "Unicode line");
  check(source.tell() == 3, "input position");
  check(input.readUntil(33) == "abc", "read until delimiter");
  source.seek(3, SeekBegin);
  check(input.readString(3) == "abc", "seek then read string");
  source.seek(0, SeekBegin);
  check(input.readAll().toString() == "λ\nabc!", "seek then read all");
  check(source.eof(), "end of file");
  input.close();
  input.close();
  var append = File.append(path, false);
  var appended:Output = append;
  appended.writeString("z");
  appended.close();
  var update = File.update(path);
  update.seek(3, SeekBegin);
  var updated:Output = update;
  updated.writeByte(65);
  updated.close();
  check(File.getContent(path) == "λ\nAbc!z", "append and non-truncating update");
  var empty = File.write(path);
  empty.close();
  var reader:Input = File.read(path, false);
  var text = reader.readAll().toString();
  check(text != null && text == "", "empty file string");
  reader.close();
 }
 static function main():Void {
  var directory = Sys.getEnv("TMPDIR");
  if (directory == null) directory = Sys.getEnv("TEMP");
  if (directory == null) directory = ".";
  var path = directory + "/rayzor-stream-" + Std.random(1000000000);
  try { exercise(path); }
  catch (error:Dynamic) {
   if (sys.FileSystem.exists(path)) sys.FileSystem.deleteFile(path);
   throw error;
  }
  sys.FileSystem.deleteFile(path);
  trace("CONFORMANCE_OK");
 }
}
