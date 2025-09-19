"""
This module defines an abstract base class `BlockThroughputCostModel` for estimating the
throughput of a given `Block`.

It includes a concrete implementation, `MCABlockThroughputCostModel`, that uses
the `llvm-mca` tool to perform this estimation.
"""

import os
import subprocess
from abc import ABC, abstractmethod
from shutil import which
from tempfile import NamedTemporaryFile

from xdsl.backend.assembly_printer import AssemblyPrintable, AssemblyPrinter
from xdsl.ir import Block


class BlockThroughputCostModel(ABC):
    """
    An abstract base class for a throughput cost model.

    This class provides an interface for estimating the throughput of a block
    of assembly-like operations for a specific microarchitecture.
    """

    @abstractmethod
    def estimate_throughput(self, block: Block) -> float | None:
        """
        Estimates the throughput of a block of assembly-like operations.

        Throughput is defined as the number of cycles per iteration of a block.
        The throughput value gives an indication of the performance of the block.

        Args:
            block: The block of operations to analyze.

        Returns:
            The estimated throughput as a float, or None if the estimation fails.
        """
        pass


class ExternalBlockThroughputCostModel(BlockThroughputCostModel):
    """
    An abstract base class for throughput cost models that use external command-line tools.

    This class extends `BlockThroughputCostModel` and provides common functionality
    for models that rely on an external command-line tool to produce a report
    from a source file. Subclasses must implement methods to define the command
    to run (`cmd`) and name (`tool_name`), and to parse the tool's output (`process_report`).
    """

    arch: str
    src_path: str

    def __init__(self, arch: str):
        self.arch = arch

    @abstractmethod
    def tool_name(self) -> str:
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
        return which(self.tool_name()) is not None

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


class MCABlockThroughputCostModel(ExternalBlockThroughputCostModel):
    """
    A throughput cost model that uses the `llvm-mca` tool.
    """

    def tool_name(self) -> str:
        return "llvm-mca"

    def cmd(self) -> list[str]:
        return [self.tool_name(), f"-mcpu={self.arch}", self.src_path]

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
