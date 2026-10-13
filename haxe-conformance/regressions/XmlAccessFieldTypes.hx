class XmlAccessFieldTypes {
    static function main() {
        var node = Xml.parse('<root get="a:b" set="value"/>').firstElement();
        var access = new haxe.xml.Access(node);
        if (!access.has.get || !access.has.set || access.has.missing) throw "Bool resolver";
        if (access.att.get != "a:b" || access.att.set != "value") throw "String resolver";
        var parts = access.att.get.split(":");
        var joined = "";
        for (part in parts) joined += part;
        if (joined != "ab") throw "resolver result method";
        trace("CONFORMANCE_OK");
    }
}
