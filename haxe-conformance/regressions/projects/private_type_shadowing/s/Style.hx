package s;

enum abstract Align(Int) to Int {
	var Auto = -1;
	var Start = 0;
	var End = 1;
}

enum Prop {
	AlignItems;
	AlignSelf;
}
