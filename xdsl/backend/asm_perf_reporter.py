import subprocess
from abc import ABC, abstractmethod
from tempfile import NamedTemporaryFile

from xdsl.backend.assembly_printer import AssemblyPrinter
from xdsl.dialects.builtin import ModuleOp


class AssemblyPerformanceReporter(ABC):
    arch: str
    module: ModuleOp
    src_path: str

    def __init__(self, arch: str, module: ModuleOp):
        self.arch = arch
        self.module = module

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def cmd(self) -> list[str]:
        pass

    def produce_report(self) -> str:
        cmd = self.cmd()
        with NamedTemporaryFile(mode="w+", suffix=".s", delete=False) as report_file:
            _ = subprocess.run(
                cmd, stdout=report_file, stderr=subprocess.DEVNULL, check=True
            )
            report_file.seek(0)
            return report_file.read()

    @abstractmethod
    def process_report(self, report: str) -> float | None:
        pass

    def estimate_cost(self) -> float | None:
        with NamedTemporaryFile(mode="w+", delete=False, suffix=".s") as tmp_file:
            self.src_path = tmp_file.name
            printer = AssemblyPrinter(stream=tmp_file)
            printer.print_module(self.module)

        try:
            report = self.produce_report()
            return self.process_report(report)
        finally:
            pass
            # if os.path.exists(self.src_path):
            #     os.remove(self.src_path)


class MCAReporter(AssemblyPerformanceReporter):
    def name(self) -> str:
        return "llvm-mca"

    def cmd(self) -> list[str]:
        return ["llvm-mca", f"-mcpu={self.arch}", self.src_path]

    def process_report(self, report: str) -> float | None:
        cycles = None
        iterations = None
        assert report is not None

        for l in report.splitlines():
            if l.startswith("Iterations"):
                iterations = float(l.split(":")[1])
            elif l.startswith("Total Cycles"):
                cycles = float(l.split(":")[1])

        if iterations is not None and cycles is not None:
            return cycles / iterations

        return None
