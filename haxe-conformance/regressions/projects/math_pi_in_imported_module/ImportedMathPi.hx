import geometry.Circle;

class ImportedMathPi {
    static function main() {
        if (Math.abs(new Circle(2).area() - 12.566370614359172) > 1e-12) throw "Math.PI in an imported module";
        trace("CONFORMANCE_OK");
    }
}
