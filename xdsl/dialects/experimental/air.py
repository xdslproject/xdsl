"""
Port of the AMD Xilinx AIR dialect for programming the AIEs on the AMD Xilinx Versal FPGA architecture.
This is a higher-level dialect than the AIE dialect. It is used to program Versal cards over PCIe.
AIE is a hardened systolic array present in the Versal devices. The dialect describes netlists of AIE
components and it can be lowered to the processor's assembly using the vendor's compiler. A description
of the original dialect can be found here https://xilinx.github.io/mlir-air/AIRDialect.html
"""

from xdsl.ir import Dialect

AIR = Dialect("air", [], [])
