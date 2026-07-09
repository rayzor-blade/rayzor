import rayzor.concurrent.CpuTopology;

class Main {
    static function main() {
        var cpus = CpuTopology.cpuCount();
        var perf = CpuTopology.perfCoreCount();
        var relax = CpuTopology.poolRelaxDefault();
        var spins = CpuTopology.poolSpinDefault();
        if (cpus < 1) throw "cpu count < 1";
        if (perf < 1) throw "perf core count < 1";
        if (relax != 0 && relax != 1) throw "relax default must be 0 or 1";
        if (spins < 1) throw "spin default < 1";
        Sys.println("PASS cpu-topology cpus=" + cpus
            + " perf=" + perf
            + " relax=" + relax
            + " spins=" + spins);
    }
}
