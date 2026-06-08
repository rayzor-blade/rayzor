class ForeignModule {
    public static function make(v:Int):ForeignModuleValue {
        return U32(v);
    }
}

enum ForeignModuleValue {
    U8(v:Int);
    U32(v:Int);
    Str(v:String);
}
