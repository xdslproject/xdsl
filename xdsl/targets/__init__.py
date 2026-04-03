from collections.abc import Callable

from xdsl.utils.target import Target


def get_all_targets() -> dict[str, Callable[[], type[Target]]]:
    """Return the list of all available targets."""

    def get_arm_asm():
        from xdsl.dialects.arm import ARMAsmTarget

        return ARMAsmTarget

    def get_csl():
        from xdsl.backend.csl.print_csl import CSLTarget

        return CSLTarget

    def get_llvm():
        from xdsl.backend.llvm.convert import LLVMTarget

        return LLVMTarget

    def get_mlir():
        from xdsl.targets.mlir import MLIRTarget

        return MLIRTarget

    def get_mps():
        from xdsl.backend.mps.print_mps import MPSTarget

        return MPSTarget

    def get_riscemu():
        from xdsl.targets.riscemu import RISCVEmulatorTarget

        return RISCVEmulatorTarget

    def get_riscv_asm():
        from xdsl.dialects.riscv.abstract_ops import RISCVAsmTarget

        return RISCVAsmTarget

    def get_wat():
        from xdsl.dialects.wasm.wat import WatTarget

        return WatTarget

    def get_wgsl():
        from xdsl.backend.wgsl.wgsl_printer import WGSLTarget

        return WGSLTarget

    def get_x86_asm():
        from xdsl.dialects.x86.ops import X86AsmTarget

        return X86AsmTarget

    return {
        "arm-asm": get_arm_asm,
        "csl": get_csl,
        "llvm": get_llvm,
        "mlir": get_mlir,
        "mps": get_mps,
        "riscemu": get_riscemu,
        "riscv-asm": get_riscv_asm,
        "wat": get_wat,
        "wgsl": get_wgsl,
        "x86-asm": get_x86_asm,
    }
