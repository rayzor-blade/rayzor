enum ForeignMetaish {
    U8(v:Int);
    I8(v:Int);
    U16(v:Int);
    I16(v:Int);
    U32(v:Int);
    I32(v:Int);
    U64(v:Int);
    I64(v:Int);
    F32(v:Float);
    F64(v:Float);
    Bool(v:Bool);
    Str(s:String);
    Arr(elemType:Int, items:Array<ForeignMetaish>);
}
