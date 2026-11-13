package draw3d;

#if my_gpu
typedef Frame = core.Frame;
#else
typedef Frame = Dynamic;
#end

interface Pass {
	function prepare(frame:Frame):Void;
}
