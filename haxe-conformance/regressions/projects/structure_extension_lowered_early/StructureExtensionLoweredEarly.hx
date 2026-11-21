// A `> Base` extension typed before Base's fields were known is the merged
// structure: a literal of it converts each field and readers find them.
class StructureExtensionLoweredEarly {
	static function main() {
		Check.run();
	}
}
