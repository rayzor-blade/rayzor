package toy.draw;
typedef TextStyle = {
	/** In layout units; 14 by default. **/
	?size:Float,
	/** A family by name; the system's interface face when unset. **/
	?family:String,
	/** 0 the system's, 1 monospace, 2 serif, 3 sans-serif: the face when `family` is unset or missing. **/
	?generic:Int,
	/** 100 to 900; 400 by default. **/
	?weight:Int,
	?italic:Bool,
	/** Added after each character, in layout units. **/
	?letterSpacing:Float,
	/** Between lines, a multiple of the font size, as CSS's `line-height`; the face's own line by default. **/
	?lineHeight:Float,
	/** Which of its ends or its middle the point is; its start by default. **/
	?align:TextAlignment,
	/** Which of its lines the point is on: the baseline by default. **/
	?baseline:TextBaseline
}

enum abstract TextAlignment(Int) {
	var Start;
	var Middle;
	var End;
}

enum abstract TextBaseline(Int) {
	/** The baseline its letters stand on. **/
	var Alphabetic;

	/** The top of its tallest letters. **/
	var Top;

	/** Halfway between its top and its bottom. **/
	var Middle;

	/** The bottom of its deepest descenders. **/
	var Bottom;
}

class GlyphOutlines { public static var n = 0; }
