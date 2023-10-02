"""
This file contains the implementations of the custom accelerator instructions for riscemu.

These are temporary, until xDSL supports all the functionality necessary to lower from
builtin+printf to riscv.
"""

from typing import IO, ClassVar

from riscemu.instructions.instruction_set import InstructionSet
from riscemu.types.instruction import Instruction
from riscemu.types.int32 import Int32

from xdsl.interpreters.riscv import RawPtr
from xdsl.interpreters.shaped_array import ShapedArray


# Define a RISC-V ISA extension by subclassing InstructionSet
class ToyAccelerator(InstructionSet):
    # each method beginning with instruction_ will be available to the Emulator

    stream: ClassVar[IO[str] | None] = None

    # add typed helpers

    def ptr_read(self, ptr: int, /, offset: int = 0) -> int:
        byte_array = self.mmu.read(ptr + offset * 4, 4)
        return int.from_bytes(byte_array, byteorder="little")

    def ptr_write(self, ptr: int, /, value: int, offset: int = 0):
        byte_array = bytearray(value.to_bytes(4, byteorder="little"))
        self.mmu.write(ptr + offset * 4, 4, byte_array)

    def buffer_read(self, ptr: int, len: int) -> RawPtr:
        return RawPtr(self.mmu.read(ptr, len))

    def buffer_write(self, ptr: int, buffer: RawPtr):
        self.mmu.write(ptr, len(buffer.memory), buffer.memory)

    def buffer_copy(self, /, source: int, destination: int, count: int):
        self.mmu.write(destination, count * 4, self.mmu.read(source, count * 4))

    # Custom instructions

    def instruction_tensor_print1d(self, ins: Instruction):
        """
        This instruction prints a formatted tensor
        [1, 2, 3, 4, 5, 6]
        """

        b_ptr, b_els = (self.regs.get(ins.get_reg(i)).value for i in range(2))

        data = self.buffer_read(b_ptr, b_els)

        shaped_array = ShapedArray(data.float64.get_list(b_els), [b_els])

        print(f"{shaped_array}", file=type(self).stream)

    def instruction_tensor_print2d(self, ins: Instruction):
        """
        This instruction prints a formatted tensor
        [[1, 2, 3], [4, 5, 6]]
        """

        b_ptr, b_rows, b_cols = (self.regs.get(ins.get_reg(i)).value for i in range(3))

        data = self.buffer_read(b_ptr, b_rows * b_cols * 8)

        shaped_array = ShapedArray(
            data.float64.get_list(b_rows * b_cols), [b_rows, b_cols]
        )

        print(f"{shaped_array}", file=type(self).stream)

    def instruction_buffer_alloc(self, ins: Instruction):
        """
        Custom instruction to allocate a buffer of n words in the dedicated space.
        """

        destination_ptr_reg = ins.get_reg(0)
        count_reg = ins.get_reg(1)

        count = self.regs.get(count_reg).value

        # Magic value of the start of the address space
        # The .bss instruction is the first one inserted in the code, and
        # 'heap' is the first label, so this will point to the start of the
        # address space.
        heap_ptr = 0x100

        # The first word of the heap contains the size in words of the used space
        heap_count = self.ptr_read(heap_ptr)

        # The second word of the heap is the start of the allocated space
        heap_start = heap_ptr + 4

        # The first element past the end of the allocated space
        result_ptr = heap_start + heap_count * 8

        # Update the allocates space counter
        new_heap_count = heap_count + count
        self.ptr_write(heap_ptr, value=new_heap_count)

        self.regs.set(destination_ptr_reg, Int32(result_ptr))

    # should be in riscemu

    def instruction_fmul_d(self, ins: Instruction):
        """
        Multiplies two double values.
        """

        result_reg = ins.get_reg(0)
        lhs_reg = ins.get_reg(1)
        rhs_reg = ins.get_reg(2)

        lhs = self.regs.get_f(lhs_reg)
        rhs = self.regs.get_f(rhs_reg)

        self.regs.set_f(result_reg, lhs * rhs)

    def instruction_fadd_d(self, ins: Instruction):
        """
        Adds two double values.
        """

        result_reg = ins.get_reg(0)
        lhs_reg = ins.get_reg(1)
        rhs_reg = ins.get_reg(2)

        lhs = self.regs.get_f(lhs_reg)
        rhs = self.regs.get_f(rhs_reg)

        self.regs.set_f(result_reg, lhs + rhs)

    def instruction_fld(self, ins: Instruction):
        """
        Loads a double value into a float register.
        """

        result_reg = ins.get_reg(0)
        value_ptr_reg = ins.get_reg(1)
        offset = ins.get_imm(2)

        value_ptr = self.regs.get(value_ptr_reg).value

        buffer = self.buffer_read(value_ptr + offset, 8)

        value = buffer.float64.get_list(1)[0]

        self.regs.set_f(result_reg, value)

    def instruction_fsd(self, ins: Instruction):
        """
        Stores a double value from a float register.
        """

        value_reg = ins.get_reg(0)
        destination_ptr_reg = ins.get_reg(1)
        offset = ins.get_imm(2)

        value = self.regs.get_f(value_reg).value
        destination_ptr = self.regs.get(destination_ptr_reg).value

        buffer = RawPtr.new_float64([value])

        self.buffer_write(destination_ptr + offset, buffer)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, ToyAccelerator):
            return False
        return self.stream is __value.stream

    def __hash__(self) -> int:
        return hash(id(self.stream))
