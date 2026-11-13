// A bare inline constant inside a class is that class's own, not a
// same-named constant of another class compiled before it.
import a.Mesh;
import b.MeshData;

class InlineConstantOwnClass {
	static function main() {
		var got = [Mesh.stride(), MeshData.stride(), MeshData.bytes(3), Mesh.STRIDE, MeshData.STRIDE].join(",");
		if (got != "7,48,144,7,48")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
