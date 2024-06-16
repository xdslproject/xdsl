from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from xdsl.ir import Attribute, Dialect, Operation


def get_all_dialects() -> dict[str, Callable[[], Dialect]]:
    """Returns all available dialects."""

    def get_accfg():
        from xdsl.dialects.accfg import ACCFG

        return ACCFG

    def get_affine():
        from xdsl.dialects.affine import Affine

        return Affine

    def get_aie():
        from xdsl.dialects.experimental.aie import AIE

        return AIE

    def get_air():
        from xdsl.dialects.experimental.air import AIR

        return AIR

    def get_arith():
        from xdsl.dialects.arith import Arith

        return Arith

    def get_bufferization():
        from xdsl.dialects.bufferization import Bufferization

        return Bufferization

    def get_builtin():
        from xdsl.dialects.builtin import Builtin

        return Builtin

    def get_cf():
        from xdsl.dialects.cf import Cf

        return Cf

    def get_cmath():
        from xdsl.dialects.cmath import CMath

        return CMath

    def get_comb():
        from xdsl.dialects.comb import Comb

        return Comb

    def get_csl():
        from xdsl.dialects.csl import CSL

        return CSL

    def get_dmp():
        from xdsl.dialects.experimental.dmp import DMP

        return DMP

    def get_fir():
        from xdsl.dialects.experimental.fir import FIR

        return FIR

    def get_fsm():
        from xdsl.dialects.fsm import FSM

        return FSM

    def get_func():
        from xdsl.dialects.func import Func

        return Func

    def get_gpu():
        from xdsl.dialects.gpu import GPU

        return GPU

    def get_hlfir():
        from xdsl.dialects.experimental.hlfir import HLFIR

        return HLFIR

    def get_hls():
        from xdsl.dialects.experimental.hls import HLS

        return HLS

    def get_hw():
        from xdsl.dialects.hw import HW

        return HW

    def get_linalg():
        from xdsl.dialects.linalg import Linalg

        return Linalg

    def get_irdl():
        from xdsl.dialects.irdl.irdl import IRDL

        return IRDL

    def get_llvm():
        from xdsl.dialects.llvm import LLVM

        return LLVM

    def get_ltl():
        from xdsl.dialects.ltl import LTL

        return LTL

    def get_math():
        from xdsl.dialects.experimental.math import Math

        return Math

    def get_memref():
        from xdsl.dialects.memref import MemRef

        return MemRef

    def get_memref_stream():
        from xdsl.dialects.memref_stream import MemrefStream

        return MemrefStream

    def get_ml_program():
        from xdsl.dialects.ml_program import MLProgram

        return MLProgram

    def get_mpi():
        from xdsl.dialects.mpi import MPI

        return MPI

    def get_omp():
        from xdsl.dialects.omp import OMP

        return OMP

    def get_onnx():
        from xdsl.dialects.onnx import ONNX

        return ONNX

    def get_pdl():
        from xdsl.dialects.pdl import PDL

        return PDL

    def get_printf():
        from xdsl.dialects.printf import Printf

        return Printf

    def get_riscv_debug():
        from xdsl.dialects.riscv_debug import RISCV_Debug

        return RISCV_Debug

    def get_riscv():
        from xdsl.dialects.riscv import RISCV

        return RISCV

    def get_riscv_func():
        from xdsl.dialects.riscv_func import RISCV_Func

        return RISCV_Func

    def get_riscv_scf():
        from xdsl.dialects.riscv_scf import RISCV_Scf

        return RISCV_Scf

    def get_riscv_cf():
        from xdsl.dialects.riscv_cf import RISCV_Cf

        return RISCV_Cf

    def get_riscv_snitch():
        from xdsl.dialects.riscv_snitch import RISCV_Snitch

        return RISCV_Snitch

    def get_scf():
        from xdsl.dialects.scf import Scf

        return Scf

    def get_seq():
        from xdsl.dialects.seq import Seq

        return Seq

    def get_snitch():
        from xdsl.dialects.snitch import Snitch

        return Snitch

    def get_snitch_runtime():
        from xdsl.dialects.snitch_runtime import SnitchRuntime

        return SnitchRuntime

    def get_snitch_stream():
        from xdsl.dialects.snitch_stream import SnitchStream

        return SnitchStream

    def get_stencil():
        from xdsl.dialects.stencil import Stencil

        return Stencil

    def get_stream():
        from xdsl.dialects.stream import Stream

        return Stream

    def get_symref():
        from xdsl.frontend.symref import Symref

        return Symref

    def get_tensor():
        from xdsl.dialects.tensor import Tensor

        return Tensor

    def get_test():
        from xdsl.dialects.test import Test

        return Test

    def get_vector():
        from xdsl.dialects.vector import Vector

        return Vector

    def get_x86():
        from xdsl.dialects.x86 import X86

        return X86

    return {
        "accfg": get_accfg,
        "affine": get_affine,
        "aie": get_aie,
        "air": get_air,
        "arith": get_arith,
        "bufferization": get_bufferization,
        "builtin": get_builtin,
        "cf": get_cf,
        "cmath": get_cmath,
        "comb": get_comb,
        "csl": get_csl,
        "dmp": get_dmp,
        "fir": get_fir,
        "fsm": get_fsm,
        "func": get_func,
        "gpu": get_gpu,
        "hlfir": get_hlfir,
        "hls": get_hls,
        "hw": get_hw,
        "linalg": get_linalg,
        "irdl": get_irdl,
        "llvm": get_llvm,
        "ltl": get_ltl,
        "math": get_math,
        "memref": get_memref,
        "memref_stream": get_memref_stream,
        "ml_program": get_ml_program,
        "mpi": get_mpi,
        "omp": get_omp,
        "onnx": get_onnx,
        "pdl": get_pdl,
        "printf": get_printf,
        "riscv": get_riscv,
        "riscv_debug": get_riscv_debug,
        "riscv_func": get_riscv_func,
        "riscv_scf": get_riscv_scf,
        "riscv_cf": get_riscv_cf,
        "riscv_snitch": get_riscv_snitch,
        "scf": get_scf,
        "seq": get_seq,
        "snitch": get_snitch,
        "snrt": get_snitch_runtime,
        "snitch_stream": get_snitch_stream,
        "stencil": get_stencil,
        "stream": get_stream,
        "symref": get_symref,
        "tensor": get_tensor,
        "test": get_test,
        "vector": get_vector,
        "x86": get_x86,
    }


@dataclass
class MLContext:
    """Contains structures for operations/attributes registration."""

    allow_unregistered: bool = field(default=False)

    _loaded_dialects: dict[str, Dialect] = field(default_factory=dict)
    _loaded_ops: dict[str, type[Operation]] = field(default_factory=dict)
    _loaded_attrs: dict[str, type[Attribute]] = field(default_factory=dict)
    _registered_dialects: dict[str, Callable[[], Dialect]] = field(default_factory=dict)
    """
    A dictionary of all registered dialects that are not yet loaded. This is used to
    only load the respective Python files when the dialect is actually used.
    """

    def register_all_dialects(self):
        """
        Register all dialects that can be used.

        Add other/additional dialects by overloading this function.
        """
        for dialect_name, dialect_factory in get_all_dialects().items():
            self.register_dialect(dialect_name, dialect_factory)

    def clone(self) -> "MLContext":
        return MLContext(
            self.allow_unregistered,
            self._loaded_dialects.copy(),
            self._loaded_ops.copy(),
            self._loaded_attrs.copy(),
            self._registered_dialects.copy(),
        )

    @property
    def loaded_ops(self) -> Iterable[type[Operation]]:
        """
        Returns all the loaded operations. Not valid across mutations of this object.
        """
        return self._loaded_ops.values()

    @property
    def loaded_attrs(self) -> Iterable[type[Attribute]]:
        """
        Returns all the loaded attributes. Not valid across mutations of this object.
        """
        return self._loaded_attrs.values()

    @property
    def loaded_dialects(self) -> Iterable[Dialect]:
        """
        Returns all the loaded attributes. Not valid across mutations of this object.
        """
        return self._loaded_dialects.values()

    @property
    def registered_dialect_names(self) -> Iterable[str]:
        """
        Returns the names of all registered dialects. Not valid across mutations of this object.
        """
        return self._registered_dialects.keys()

    def register_dialect(
        self, name: str, dialect_factory: Callable[[], Dialect]
    ) -> None:
        """
        Register a dialect without loading it. The dialect is only loaded in the context
        when an operation or attribute of that dialect is parsed, or when explicitely
        requested with `load_registered_dialect`.
        """
        if name in self._registered_dialects:
            raise ValueError(f"'{name}' dialect is already registered")
        self._registered_dialects[name] = dialect_factory

    def load_registered_dialect(self, name: str) -> None:
        """Load a dialect that is already registered in the context."""
        if name not in self._registered_dialects:
            raise ValueError(f"'{name}' dialect is not registered")
        dialect = self._registered_dialects[name]()
        self._loaded_dialects[dialect.name] = dialect

        for op in dialect.operations:
            self.load_op(op)

        for attr in dialect.attributes:
            self.load_attr(attr)

    def load_dialect(self, dialect: Dialect):
        """
        Load a dialect. Operation and Attribute names should be unique.
        If the dialect is already registered in the context, use
        `load_registered_dialect` instead.
        """
        if dialect.name in self._registered_dialects:
            raise ValueError(
                f"'{dialect.name}' dialect is already registered, use 'load_registered_dialect' instead"
            )
        self.register_dialect(dialect.name, lambda: dialect)
        self.load_registered_dialect(dialect.name)

    def load_op(self, op: type[Operation]) -> None:
        """Load an operation definition. Operation names should be unique."""
        if op.name in self._loaded_ops:
            raise Exception(f"Operation {op.name} has already been loaded")
        self._loaded_ops[op.name] = op

    def load_attr(self, attr: type[Attribute]) -> None:
        """Load an attribute definition. Attribute names should be unique."""
        if attr.name in self._loaded_attrs:
            raise Exception(f"Attribute {attr.name} has already been loaded")
        self._loaded_attrs[attr.name] = attr

    def get_optional_op(self, name: str) -> type[Operation] | None:
        """
        Get an operation class from its name if it exists.
        If the operation is not registered, return None unless unregistered operations
        are allowed in the context, in which case return an UnregisteredOp.
        """
        # If the operation is already loaded, returns it.
        if name in self._loaded_ops:
            return self._loaded_ops[name]

        # Otherwise, check if the operation dialect is registered.
        if "." in name:
            dialect_name, _ = name.split(".", 1)
            if (
                dialect_name in self._registered_dialects
                and dialect_name not in self._loaded_dialects
            ):
                self.load_registered_dialect(dialect_name)
                return self.get_optional_op(name)

        # If the dialect is unregistered, but the context allows unregistered
        # operations, return an UnregisteredOp.
        if self.allow_unregistered:
            from xdsl.dialects.builtin import UnregisteredOp

            op_type = UnregisteredOp.with_name(name)
            self._loaded_ops[name] = op_type
            return op_type
        return None

    def get_op(self, name: str) -> type[Operation]:
        """
        Get an operation class from its name.
        If the operation is not registered, raise an exception unless unregistered
        operations are allowed in the context, in which case return an UnregisteredOp.
        """
        if op_type := self.get_optional_op(name):
            return op_type
        raise Exception(f"Operation {name} is not registered")

    def get_optional_attr(
        self,
        name: str,
        create_unregistered_as_type: bool = False,
    ) -> type[Attribute] | None:
        """
        Get an attribute class from its name if it exists.
        If the attribute is not registered, return None unless unregistered attributes
        are allowed in the context, in which case return an UnregisteredAttr.
        Since UnregisteredAttr may be a type (for MLIR compatibility), an
        additional flag is required to create an UnregisterAttr that is
        also a type.
        """
        # If the attribute is already loaded, returns it.
        if name in self._loaded_attrs:
            return self._loaded_attrs[name]

        # Otherwise, check if the attribute dialect is registered.
        dialect_name, _ = name.split(".", 1)
        if (
            dialect_name in self._registered_dialects
            and dialect_name not in self._loaded_dialects
        ):
            self.load_registered_dialect(dialect_name)
            return self.get_optional_attr(name)

        # If the dialect is unregistered, but the context allows unregistered
        # attributes, return an UnregisteredOp.
        if self.allow_unregistered:
            from xdsl.dialects.builtin import UnregisteredAttr

            attr_type = UnregisteredAttr.with_name_and_type(
                name, create_unregistered_as_type
            )
            self._loaded_attrs[name] = attr_type
            return attr_type

        return None

    def get_attr(
        self,
        name: str,
        create_unregistered_as_type: bool = False,
    ) -> type[Attribute]:
        """
        Get an attribute class from its name.
        If the attribute is not registered, raise an exception unless unregistered
        attributes are allowed in the context, in which case return an UnregisteredAttr.
        Since UnregisteredAttr may be a type (for MLIR compatibility), an
        additional flag is required to create an UnregisterAttr that is
        also a type.
        """
        if attr_type := self.get_optional_attr(name, create_unregistered_as_type):
            return attr_type
        raise Exception(f"Attribute {name} is not registered")

    def get_dialect(self, name: str) -> Dialect:
        if (dialect := self.get_optional_dialect(name)) is None:
            raise Exception(f"Dialect {name} is not registered")
        return dialect

    def get_optional_dialect(self, name: str) -> Dialect | None:
        if name in self._loaded_dialects:
            return self._loaded_dialects[name]
        return None
