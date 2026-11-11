/**
	The build macro runs for each implementer; the word in this comment
	must not hide the metadata below.
**/
#if !macro
@:autoBuild(ChainMac.build())
#end
interface ChainBase {}

interface ChainMid extends ChainBase {}
