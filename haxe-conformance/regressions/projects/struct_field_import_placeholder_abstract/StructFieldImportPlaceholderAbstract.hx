// A structure field typed by an abstract that its module imported before the
// abstract's module was read still converts through the abstract's `@:from`.
class StructFieldImportPlaceholderAbstract {
	static function main() {
		Ext2.main();
	}
}
