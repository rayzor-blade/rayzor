package utest.ui;

import utest.ui.common.HeaderDisplayMode;
import utest.ui.common.IReport;

/** Minimal stand-in for utest's Report: holds the display modes a case sets. */
class Report {
    public static function create(runner:Runner, ?displaySuccessResults:SuccessResultsDisplayMode,
            ?headerDisplayMode:HeaderDisplayMode):IReport<Dynamic> {
        var report = new PlainReport();
        if (displaySuccessResults != null)
            report.displaySuccessResults = displaySuccessResults;
        if (headerDisplayMode != null)
            report.displayHeader = headerDisplayMode;
        return report;
    }
}

private class PlainReport implements IReport<PlainReport> {
    public var displaySuccessResults:SuccessResultsDisplayMode = ShowSuccessResultsWithNoErrors;
    public var displayHeader:HeaderDisplayMode = ShowHeaderWithResults;

    public function new() {}

    public function setHandler(handler:PlainReport->Void):Void {}
}
