package utest.ui.common;

interface IReport<T:IReport<T>> {
    public var displaySuccessResults:SuccessResultsDisplayMode;
    public var displayHeader:HeaderDisplayMode;
    public function setHandler(handler:T->Void):Void;
}
