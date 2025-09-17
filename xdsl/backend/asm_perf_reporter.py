import os
import subprocess
from abc import ABC, abstractmethod
from shutil import which
from tempfile import NamedTemporaryFile

from xdsl.backend.assembly_printer import AssemblyPrintable, AssemblyPrinter
from xdsl.ir import Block


class AssemblyPerformanceReporter(ABC):
    arch: str
    src_path: str

    def __init__(self, arch: str):
        self.arch = arch

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

    def is_installed(self) -> bool:
        return which(self.name()) is not None

    def estimate_throughput(self, block: Block) -> float | None:
        with NamedTemporaryFile(mode="w+", delete=False, suffix=".s") as tmp_file:
            self.src_path = tmp_file.name
            printer = AssemblyPrinter(stream=tmp_file)
            for op in block.walk():
                assert isinstance(op, AssemblyPrintable), (
                    f"Block operation {op} should be an assembly instruction"
                )
                op.print_assembly(printer)

        try:
            report = self.produce_report()
            return self.process_report(report)
        finally:
            if os.path.exists(self.src_path):
                os.remove(self.src_path)


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
