from io import StringIO


from xdsl.builder import Builder

from xdsl.dialects.builtin import (
    ArrayAttr,
    DenseArrayBase,
    DenseIntOrFPElementsAttr,
    FunctionType,
    IndexType,
    ModuleOp,
    StringAttr,
    f64,
    i32,
)
from xdsl.dialects import riscv, riscv_func, llvm
from xdsl.ir import Attribute

from xdsl.printer import Printer
from xdsl.riscv_asm_writer import riscv_code
from xdsl.transforms.lower_riscv_func import LowerRISCVFunc
from xdsl.transforms.riscv_register_allocation import RISCVRegisterAllocation

from ..rewrites.lower_toy import LowerToy
from ..rewrites.optimise_toy import OptimiseToy
from ..rewrites.optimise_vector import OptimiseVector
from ..rewrites.lower_vector import LowerVector
from ..rewrites.lower_llvm import LowerLLVM

from ..compiler import (
    compile,
    parse_toy,
    context,
)
from ..emulator.emulator_iop import run_riscv
from ..emulator.toy_accelerator import ToyAccelerator

from ..dialects import toy, vector


ctx = context()

toy_program = """
def main() {
  # Define a variable `a` with shape <2, 3>, initialized with the literal value.
  # The shape is inferred from the supplied literal.
  var a = [[1, 2, 3], [4, 5, 6]];

  # b is identical to a, the literal tensor is implicitly reshaped: defining new
  # variables is the way to reshape tensors (element count must match).
  var b<3, 2> = [1, 2, 3, 4, 5, 6];

  # There is a built-in print instruction to display the contents of the tensor
  print(b);

  # Reshapes are implicit on assignment
  var c<2, 3> = b;

  # There are + and * operators for pointwise addition and multiplication
  var d = a + c;

  print(d);
}
"""


def desc(op: ModuleOp) -> str:
    stream = StringIO()
    Printer(stream=stream).print(op)
    return stream.getvalue()


@ModuleOp
@Builder.implicit_region
def toy_0():
    main_type = FunctionType.from_lists([], [])

    @Builder.implicit_region
    def main() -> None:
        a = toy.ConstantOp.from_list([1, 2, 3, 4, 5, 6], [2, 3]).res
        b_0 = toy.ConstantOp.from_list([1, 2, 3, 4, 5, 6], [6]).res
        b = toy.ReshapeOp(b_0, [3, 2]).res
        toy.PrintOp(b)
        c = toy.ReshapeOp(b, [2, 3]).res
        d = toy.AddOp(a, c).res
        toy.PrintOp(d)
        toy.ReturnOp()

    toy.FuncOp("main", main_type, main)


@ModuleOp
@Builder.implicit_region
def toy_1():
    main_type = FunctionType.from_lists([], [])

    @Builder.implicit_region
    def main() -> None:
        a = toy.ConstantOp.from_list([1, 2, 3, 4, 5, 6], [2, 3]).res
        b = toy.ConstantOp.from_list([1, 2, 3, 4, 5, 6], [3, 2]).res
        toy.PrintOp(b)
        c = toy.ConstantOp.from_list([1, 2, 3, 4, 5, 6], [2, 3]).res
        d = toy.AddOp(a, c).res
        toy.PrintOp(d)
        toy.ReturnOp()

    toy.FuncOp("main", main_type, main)


@ModuleOp
@Builder.implicit_region
def vir_0():
    main_type = FunctionType.from_lists([], [])

    def vector_i32(elements: list[int]) -> DenseIntOrFPElementsAttr:
        return DenseIntOrFPElementsAttr.vector_from_list(elements, i32)

    def vector_f64(elements: list[float]) -> DenseIntOrFPElementsAttr:
        return DenseIntOrFPElementsAttr.vector_from_list(elements, f64)

    def tensor_type(shape: list[int]) -> toy.TensorTypeF64:
        return toy.TensorTypeF64.from_type_and_list(f64, shape)

    @Builder.implicit_region
    def main() -> None:
        a_shape = vector.VectorConstantOp(vector_i32([2, 3]), "tensor_shape").res
        a_data = vector.VectorConstantOp(
            vector_f64([1, 2, 3, 4, 5, 6]), "tensor_data"
        ).res
        a = vector.TensorMakeOp(a_shape, a_data, tensor_type([2, 3])).tensor
        b_shape = vector.VectorConstantOp(vector_i32([3, 2]), "tensor_shape").res
        b_data = vector.VectorConstantOp(
            vector_f64([1, 2, 3, 4, 5, 6]), "tensor_data"
        ).res
        b = vector.TensorMakeOp(b_shape, b_data, tensor_type([3, 2])).tensor
        llvm.CallIntrinsicOp("tensor.print", (b,), ())

        c_shape = vector.VectorConstantOp(vector_i32([2, 3]), "tensor_shape").res
        c_data = vector.VectorConstantOp(
            vector_f64([1, 2, 3, 4, 5, 6]), "tensor_data"
        ).res
        c = vector.TensorMakeOp(c_shape, c_data, tensor_type([2, 3])).tensor

        d_shape = vector.TensorShapeOp(a).data
        lhs = vector.TensorDataOp(a).data
        rhs = vector.TensorDataOp(c).data
        d_data = vector.VectorAddOp(lhs, rhs).res
        d = vector.TensorMakeOp(d_shape, d_data, tensor_type([2, 3])).tensor

        llvm.CallIntrinsicOp("tensor.print", (d,), ())
        toy.ReturnOp()

    toy.FuncOp("main", main_type, main)


@ModuleOp
@Builder.implicit_region
def vir_1():
    main_type = FunctionType.from_lists([], [])

    def vector_i32(elements: list[int]) -> DenseIntOrFPElementsAttr:
        return DenseIntOrFPElementsAttr.vector_from_list(elements, i32)

    def vector_f64(elements: list[float]) -> DenseIntOrFPElementsAttr:
        return DenseIntOrFPElementsAttr.vector_from_list(elements, f64)

    def tensor_type(shape: list[int]) -> toy.TensorTypeF64:
        return toy.TensorTypeF64.from_type_and_list(f64, shape)

    @Builder.implicit_region
    def main() -> None:
        a_shape = vector.VectorConstantOp(vector_i32([2, 3]), "tensor_shape").res
        a_data = vector.VectorConstantOp(
            vector_f64([1, 2, 3, 4, 5, 6]), "tensor_data"
        ).res

        b_shape = vector.VectorConstantOp(vector_i32([3, 2]), "tensor_shape").res
        b_data = vector.VectorConstantOp(
            vector_f64([1, 2, 3, 4, 5, 6]), "tensor_data"
        ).res
        b = vector.TensorMakeOp(b_shape, b_data, tensor_type([3, 2])).tensor
        llvm.CallIntrinsicOp("tensor.print", (b,), ())

        c_data = vector.VectorConstantOp(
            vector_f64([1, 2, 3, 4, 5, 6]), "tensor_data"
        ).res

        d_data = vector.VectorAddOp(a_data, c_data).res
        d = vector.TensorMakeOp(a_shape, d_data, tensor_type([2, 3])).tensor

        llvm.CallIntrinsicOp("tensor.print", (d,), ())
        toy.ReturnOp()

    toy.FuncOp("main", main_type, main)


@ModuleOp
@Builder.implicit_region
def llvm_0():
    main_type = llvm.LLVMFunctionType.from_lists([], llvm.VoidType())

    def vector_i32(
        elements: list[int],
    ) -> tuple[DenseIntOrFPElementsAttr, Attribute]:
        vec = DenseIntOrFPElementsAttr.vector_from_list(elements, i32)
        return vec, vec.type

    def vector_f64(elements: list[float]) -> tuple[DenseIntOrFPElementsAttr, Attribute]:
        vec = DenseIntOrFPElementsAttr.vector_from_list(elements, f64)
        return vec, vec.type

    tensor_type = llvm.LLVMStructType([StringAttr("toy_tensor"), ArrayAttr([i32, i32])])

    @Builder.implicit_region
    def main() -> None:
        a_shape = llvm.ConstantOp(*vector_i32([2, 3])).result
        a_data = llvm.ConstantOp(*vector_f64([1, 2, 3, 4, 5, 6])).result

        b_shape = llvm.ConstantOp(*vector_i32([3, 2])).result
        b_data = llvm.ConstantOp(*vector_f64([1, 2, 3, 4, 5, 6])).result

        b0 = llvm.UndefOp(tensor_type).res
        b1 = llvm.InsertValueOp(
            DenseArrayBase.create_dense_int_or_index(IndexType(), [0]),
            b0,
            b_shape,
        ).res
        b = llvm.InsertValueOp(
            DenseArrayBase.create_dense_int_or_index(IndexType(), [1]),
            b1,
            b_data,
        ).res

        llvm.CallIntrinsicOp("tensor.print", (b,), ())

        c_data = llvm.ConstantOp(*vector_f64([1, 2, 3, 4, 5, 6])).result

        d_data = llvm.FAddOp(a_data, c_data).res

        d0 = llvm.UndefOp(tensor_type).res
        d1 = llvm.InsertValueOp(
            DenseArrayBase.create_dense_int_or_index(IndexType(), [0]),
            d0,
            a_shape,
        ).res
        d = llvm.InsertValueOp(
            DenseArrayBase.create_dense_int_or_index(IndexType(), [1]),
            d1,
            d_data,
        ).res

        llvm.CallIntrinsicOp("tensor.print", (d,), ())

        llvm.ReturnOp()

    llvm.FuncOp("main", main_type, main)


@ModuleOp
@Builder.implicit_region
def risc_0():
    @Builder.implicit_region
    def bss_region():
        riscv.LabelOp("heap")
        riscv.DirectiveOp(".space", "1024")

    riscv.DirectiveOp(".bss", None, bss_region)

    @Builder.implicit_region
    def data_region():
        riscv.LabelOp("main.0")
        riscv.DirectiveOp(".word", "0x2, 0x2, 0x3")
        riscv.LabelOp("main.1")
        riscv.DirectiveOp(".word", "0x6, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6")
        riscv.LabelOp("main.2")
        riscv.DirectiveOp(".word", "0x2, 0x3, 0x2")
        riscv.LabelOp("main.3")
        riscv.DirectiveOp(".word", "0x6, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6")
        riscv.LabelOp("main.4")
        riscv.DirectiveOp(".word", "0x6, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6")
        pass

    riscv.DirectiveOp(".data", None, data_region)

    @Builder.implicit_region
    def text_region():
        @Builder.implicit_region
        def main() -> None:
            heap = riscv.LiOp("heap")
            riscv.AddiOp(
                heap,
                1020,
                rd=riscv.Registers.SP,
                comment="stack grows from the top of the heap",
            )
            ts0 = riscv.LiOp("main.0").rd
            td0 = riscv.LiOp("main.1").rd
            ts1 = riscv.LiOp("main.2").rd
            td1 = riscv.LiOp("main.3").rd

            sp0 = riscv.GetRegisterOp(riscv.Registers.SP).res
            riscv.CommentOp('Reserve 8 bytes on stack for element type "toy_tensor"')
            t0 = riscv.AddiOp(sp0, -8, rd=riscv.Registers.SP).rd
            riscv.SwOp(ts1, t0, 0, comment='Set "toy_tensor" @ 0')
            riscv.SwOp(td1, t0, 4, comment='Set "toy_tensor" @ 1')

            riscv.CustomEmulatorInstructionOp("tensor.print", (t0,), ())
            td2 = riscv.LiOp("main.4").rd
            result_vector = riscv.CustomEmulatorInstructionOp(
                "vector.copy",
                (td0,),
                (riscv.RegisterType(riscv.Register()),),
            ).results[0]

            riscv.CustomEmulatorInstructionOp(
                "vector.add",
                (result_vector, td2),
                (),
            )

            sp1 = riscv.GetRegisterOp(riscv.Registers.SP).res
            riscv.CommentOp('Reserve 8 bytes on stack for element type "toy_tensor"')
            t1 = riscv.AddiOp(sp1, -8, rd=riscv.Registers.SP).rd
            riscv.SwOp(ts0, t1, 0, comment='Set "toy_tensor" @ 0')
            riscv.SwOp(result_vector, t1, 4, comment='Set "toy_tensor" @ 1')

            riscv.CustomEmulatorInstructionOp("tensor.print", (t1,), ())
            riscv_func.ReturnOp()

        riscv_func.FuncOp("main", main)

    riscv.DirectiveOp(".text", None, text_region)


riscv_asm = """.bss
heap:
.space 1024
.data
main.0:
.word 0x2, 0x2, 0x3
main.1:
.word 0x6, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6
main.2:
.word 0x2, 0x3, 0x2
main.3:
.word 0x6, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6
main.4:
.word 0x6, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6
.text
main:
    li j0, heap
    addi sp, j0, 1020                            # stack grows from the top of the heap
    li j1, main.0
    li j2, main.1
    li j3, main.2
    li j4, main.3
    # Reserve 8 bytes on stack for element type "toy_tensor"
    addi sp, sp, -8
    sw j3, sp, 0                                 # Set "toy_tensor" @ 0
    sw j4, sp, 4                                 # Set "toy_tensor" @ 1
    tensor.print sp
    li j5, main.4
    vector.copy j6, j2
    vector.add j6, j5
    # Reserve 8 bytes on stack for element type "toy_tensor"
    addi sp, sp, -8
    sw j1, sp, 0                                 # Set "toy_tensor" @ 0
    sw j6, sp, 4                                 # Set "toy_tensor" @ 1
    tensor.print sp
    li a7, 93
    ecall
"""


def test_compile():
    code = compile(toy_program)
    assert code == riscv_asm


def test_parse_toy():
    assert desc(toy_0) == desc(parse_toy(toy_program))
    assert toy_0.is_structurally_equivalent(parse_toy(toy_program))


def test_optimise_toy():
    copy = toy_0.clone()
    OptimiseToy().apply(ctx, copy)
    assert desc(toy_1) == desc(copy)
    assert toy_1.is_structurally_equivalent(copy)


def test_lower_from_toy():
    copy = toy_1.clone()
    LowerToy().apply(ctx, copy)
    assert desc(vir_0) == desc(copy)
    assert vir_0.is_structurally_equivalent(copy)


def test_optimise_vir():
    copy = vir_0.clone()
    OptimiseVector().apply(ctx, copy)
    assert desc(vir_1) == desc(copy)
    assert vir_1.is_structurally_equivalent(copy)


def test_lower_to_llvm():
    copy = vir_1.clone()

    LowerVector().apply(ctx, copy)

    assert desc(llvm_0) == desc(copy)
    assert llvm_0.is_structurally_equivalent(copy)


def test_lower_to_riscv_func():
    copy = llvm_0.clone()

    LowerLLVM().apply(ctx, copy)

    assert desc(risc_0) == desc(copy)
    assert risc_0.is_structurally_equivalent(copy)


def test_lower_riscv_func():
    copy = risc_0.clone()

    LowerRISCVFunc().apply(ctx, copy)
    RISCVRegisterAllocation().apply(ctx, copy)

    assert riscv_asm == riscv_code(copy)


def test_execution():
    stream = StringIO()
    ToyAccelerator.stream = stream
    run_riscv(riscv_asm, extensions=[ToyAccelerator], unlimited_regs=True, verbosity=1)
    assert "[[2, 4, 6], [8, 10, 12]]" in stream.getvalue()
