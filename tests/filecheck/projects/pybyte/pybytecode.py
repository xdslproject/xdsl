# RUN: python %s | filecheck %s

import dis
from collections.abc import Callable
from types import FunctionType
from typing import Any

from bytecode import Bytecode, Instr

from xdsl.builder import Builder
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.pybc import (
    BinaryOpOp,
    FunctionOp,
    LoadConstOp,
    LoadFastOp,
    PybcOp,
    # ResumeOp,
    ReturnValueOp,
    StoreFastOp,
)
from xdsl.ir import Block, Operation, Region, SSAValue
from xdsl.irdl import IRDLOperation
from xdsl.rewriter import InsertPoint

_BINARY_OP_MAP: dict[str, PybcOp] = {
    "+": PybcOp.OP_ADD,
    "-": PybcOp.OP_SUB,
    "*": PybcOp.OP_MUL,
    "/": PybcOp.OP_DIV,
}


def expect_name_hint(ssa: SSAValue | Block) -> str:
    assert ssa.name_hint
    return ssa.name_hint


def expect_first_op(b: Block) -> Operation:
    assert b.first_op
    return b.first_op


def expect_first_block(b: Region) -> Block:
    assert b.first_block
    return b.first_block


class PybcGen:
    def __init__(self) -> None:
        self.stack: list[IRDLOperation] = []
        self.str_ssa_mapping: dict[str, SSAValue] = dict()

    def inst_to_pybc_op(self, inst: dis.Instruction) -> IRDLOperation | None:
        match inst.opname:
            case "RESUME":
                return None

            case "LOAD_FAST":
                assert inst.arg is not None

                self.stack.append(
                    LoadFastOp(inst.arg, self.str_ssa_mapping[inst.argval])
                )
                return self.stack[-1]

            case "LOAD_CONST":
                self.stack.append(LoadConstOp(inst.argval))
                return self.stack[-1]

            case "STORE_FAST":
                assert inst.arg is not None
                self.stack.pop()
                return StoreFastOp(inst.arg, self.str_ssa_mapping[inst.argval])

            case "BINARY_OP":
                symbol = inst.argrepr
                if symbol not in _BINARY_OP_MAP:
                    raise NotImplementedError(f"Unknown binary operator: {symbol!r}")

                lhs = self.stack.pop()
                rhs = self.stack.pop()
                self.stack.append(BinaryOpOp(_BINARY_OP_MAP[symbol], lhs, rhs))

                return self.stack[-1]

            case "RETURN_VALUE":
                return ReturnValueOp(self.stack.pop())

            case other:
                raise NotImplementedError(f"Instruction not supported: {other}")

    def pybcop_to_inst(self, op: Operation) -> Instr | None:
        match op:
            case LoadFastOp():
                return Instr("LOAD_FAST", expect_name_hint(op.var_name))

            case StoreFastOp():
                return Instr("STORE_FAST", expect_name_hint(op.var_name))

            case LoadConstOp():
                return Instr("LOAD_CONST", op.const.value.data)

            case BinaryOpOp():
                return Instr("BINARY_OP", op.op.data.value)

            case ReturnValueOp():
                return Instr("RETURN_VALUE")

            case other:
                raise NotImplementedError(f"Instruction {other} not supported")

    def gen_pybytecode(self, func: FunctionType) -> ModuleOp:
        name = func.__name__
        code = func.__code__

        arg_count = code.co_argcount + code.co_kwonlyargcount

        sym_list = code.co_varnames
        arg_list = sym_list[:arg_count]
        var_list = sym_list[arg_count:]

        funcop = FunctionOp(
            name,
            arg_list,
            var_list,
            ops=[],
        )
        builder = Builder(
            insertion_point=InsertPoint.at_start(expect_first_block(funcop.body))
        )
        assert funcop.body.first_block is not None
        for i, arg in enumerate(funcop.body.first_block.args):
            self.str_ssa_mapping[arg_list[i]] = arg

        for inst in dis.get_instructions(func):
            op = self.inst_to_pybc_op(inst)
            if op:
                builder.insert(op)

        self.stack = []
        self.str_ssa_mapping = dict()

        return ModuleOp([funcop])

    def gen_bytecode(self, module: ModuleOp) -> FunctionType | None:
        block = module.body.first_block
        if not block:
            return None

        func_op = block.first_op
        assert func_op
        assert isinstance(func_op, FunctionOp)

        arg_names = [expect_name_hint(arg) for arg in func_op.get_args()]

        insts: list[Instr] = [Instr("RESUME", 0)]
        for op in func_op.walk():
            if isinstance(op, FunctionOp):
                continue

            inst = self.pybcop_to_inst(op)
            if inst:
                insts.append(inst)

        bc = Bytecode(insts)
        bc.argnames = arg_names
        bc.argcount = len(arg_names)
        bc.name = func_op.sym_name.data.removeprefix('"').removesuffix('"')

        code = bc.to_code()
        new_func = FunctionType(code, globals(), name=bc.name)
        return new_func


class PybcJIT:
    @classmethod
    def disas(cls, func: FunctionType) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            print("Bytecode: ")
            dis.dis(func)
            generator = PybcGen()
            module = generator.gen_pybytecode(func)

            print("Module: ")
            print(module)

            rebuild = generator.gen_bytecode(module)
            assert rebuild is not None
            print("Regenerated bytecode: ")
            dis.dis(rebuild)

            result = func(*args, **kwargs)
            rebuilt_result = rebuild(*args, **kwargs)

            print(f"Result:\nExpect: {result}\nGot: {rebuilt_result}")
            return result

        return wrapper


@PybcJIT.disas
def func(a: float, b: float) -> float:
    return a + b


func(2, 3)
# CHECK: Bytecode:
# CHECK-NEXT: 197           0 RESUME                   0
# CHECK: 199           2 LOAD_FAST                0 (a)
# CHECK-NEXT:               4 LOAD_FAST                1 (b)
# CHECK-NEXT:               6 BINARY_OP                0 (+)
# CHECK-NEXT:              10 RETURN_VALUE
# CHECK-NEXT: Module:
# CHECK-NEXT: builtin.module {
# CHECK-NEXT:   pybc.function @func() {
# CHECK-NEXT:   ^bb0(%a: #pybc.object"'Unknown'", %b: #pybc.object"'Unknown'"):
# CHECK-NEXT:     %0 = "pybc.load_fast"(%a) <{var_index = 0 : i32}> : (#pybc.object"'Unknown'") -> #pybc.object"'Unknown'"
# CHECK-NEXT:     %1 = "pybc.load_fast"(%b) <{var_index = 1 : i32}> : (#pybc.object"'Unknown'") -> #pybc.object"'Unknown'"
# CHECK-NEXT:     %2 = "pybc.binary_op"(%1, %0) <{op = #pybc.op<op_add>}> : (#pybc.object"'Unknown'", #pybc.object"'Unknown'") -> #pybc.object"'Unknown'"
# CHECK-NEXT:     "pybc.return_value"(%2) : (#pybc.object"'Unknown'") -> ()
# CHECK-NEXT:   }
# CHECK-NEXT: }
# CHECK-NEXT: Regenerated bytecode:
# CHECK-NEXT:   1           0 RESUME                   0
# CHECK-NEXT:               2 LOAD_FAST                0 (a)
# CHECK-NEXT:               4 LOAD_FAST                1 (b)
# CHECK-NEXT:               6 BINARY_OP                0 (+)
# CHECK-NEXT:              10 RETURN_VALUE
# CHECK-NEXT: Result:
# CHECK-NEXT: Expect: 5
# CHECK-NEXT: Got: 5
