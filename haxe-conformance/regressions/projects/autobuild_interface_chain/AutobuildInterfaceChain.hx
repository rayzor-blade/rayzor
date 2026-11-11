import Chain;

class ViaQualified implements Chain.ChainMid {}
class ViaImport implements ChainMid {}
class ViaBase implements Chain.ChainBase {}

class AutobuildInterfaceChain {
	static function main() {
		var got = [ViaQualified.TAG, ViaImport.TAG, ViaBase.TAG].join(",");
		var want = "ViaQualified,ViaImport,ViaBase";
		if (got != want)
			throw got + " != " + want;
		Sys.println("CONFORMANCE_OK");
	}
}
