"""
This file contains the implementations of the custom accelerator instructions for riscemu.

These are temporary, until xDSL supports all the functionality necessary to lower from
builtin+printf to riscv.
"""

from typing import IO, ClassVar

from riscemu.instructions.instruction_set import InstructionSet
from riscemu.types.instruction import Instruction
from riscemu.types.int32 import Int32

from xdsl.interpreters.shaped_array import ShapedArray


# Define a RISC-V ISA extension by subclassing InstructionSet
class ToyAccelerator(InstructionSet):
    # each method beginning with instruction_ will be available to the Emulator

    stream: ClassVar[IO[str] | None] = None

    # add typed helpers

    def set_reg(self, reg: str, value: int):
        self.regs.set(reg, Int32(value))

    def get_reg(self, reg: str) -> int:
        return self.regs.get(reg).value

    def ptr_read(self, ptr: int, /, offset: int = 0) -> int:
        byte_array = self.mmu.read(ptr + offset * 4, 4)
        return int.from_bytes(byte_array, byteorder="little")

    def ptr_write(self, ptr: int, /, value: int, offset: int = 0):
        byte_array = bytearray(value.to_bytes(4, byteorder="little"))
        self.mmu.write(ptr + offset * 4, 4, byte_array)

    def buffer_read(self, ptr: int, len: int, /, offset: int = 0) -> list[int]:
        return [self.ptr_read(ptr, offset) for offset in range(offset, offset + len)]

    def buffer_write(self, ptr: int, /, data: list[int], offset: int = 0):
        for i, value in enumerate(data):
            self.ptr_write(ptr, value=value, offset=offset + i)

    def buffer_copy(self, /, source: int, destination: int, count: int):
        self.mmu.write(destination, count * 4, self.mmu.read(source, count * 4))

    # Custom instructions

    def instruction_tensor_print1d(self, ins: Instruction):
        """
        This instruction prints a formatted tensor
        [1, 2, 3, 4, 5, 6]
        """

        b_ptr, b_els = (self.get_reg(ins.get_reg(i)) for i in range(2))

        data = self.buffer_read(b_ptr, b_els)

        shaped_array = ShapedArray([float(value) for value in data], [b_els])

        print(f"{shaped_array}", file=type(self).stream)

    def instruction_tensor_print2d(self, ins: Instruction):
        """
        This instruction prints a formatted tensor
        [[1, 2, 3], [4, 5, 6]]
        """

        b_ptr, b_rows, b_cols = (self.get_reg(ins.get_reg(i)) for i in range(3))

        size = b_rows * b_cols

        data = self.buffer_read(b_ptr, size)

        shaped_array = ShapedArray([float(value) for value in data], [b_rows, b_cols])

        print(f"{shaped_array}", file=type(self).stream)

    def instruction_tensor_transpose2d(self, ins: Instruction):
        """
        This instruction transposes a 2d tensor, swapping its row and column indices:
        [[1, 2, 3], [4, 5, 6]] -> [[1, 4], [2, 5], [3, 6]]
        """

        dest_ptr, source_ptr, b_rows, b_cols = (
            self.get_reg(ins.get_reg(i)) for i in range(4)
        )

        size = b_rows * b_cols

        data = self.buffer_read(source_ptr, size)

        shaped_array = ShapedArray(data, [b_rows, b_cols])
        result_shaped_array = shaped_array.transposed(0, 1)

        self.buffer_write(dest_ptr, data=result_shaped_array.data)

    def instruction_buffer_copy(self, ins: Instruction):
        """
        Copies the elements of one buffer to another.
        """
        c_reg, d_reg, s_reg = [ins.get_reg(i) for i in range(3)]

        count = self.get_reg(c_reg)

        s_ptr = self.get_reg(s_reg)
        d_ptr = self.get_reg(d_reg)

        s_data = self.buffer_read(s_ptr, count)

        self.buffer_write(d_ptr, data=s_data)

    def instruction_buffer_add(self, ins: Instruction):
        """
        Pointwise addition of two buffers.
        """
        c_reg, d_reg, s_reg = [ins.get_reg(i) for i in range(3)]

        count = self.get_reg(c_reg)

        s_ptr = self.get_reg(s_reg)
        d_ptr = self.get_reg(d_reg)

        s_data = self.buffer_read(s_ptr, count)
        d_data = self.buffer_read(d_ptr, count)

        self.buffer_write(
            d_ptr, data=[l_el + r_el for l_el, r_el in zip(s_data, d_data)]
        )

    def instruction_buffer_mul(self, ins: Instruction):
        """
        Pointwise multiplication of two buffers.
        """
        c_reg, d_reg, s_reg = [ins.get_reg(i) for i in range(3)]

        count = self.get_reg(c_reg)

        s_ptr = self.get_reg(s_reg)
        d_ptr = self.get_reg(d_reg)

        s_data = self.buffer_read(s_ptr, count)
        d_data = self.buffer_read(d_ptr, count)

        self.buffer_write(
            d_ptr, data=[l_el * r_el for l_el, r_el in zip(s_data, d_data)]
        )

    def instruction_buffer_alloc(self, ins: Instruction):
        """
        Custom instruction to allocate a buffer of n words in the dedicated space.
        """

        destination_ptr_reg = ins.get_reg(0)
        count_reg = ins.get_reg(1)

        count = self.get_reg(count_reg)

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
        result_ptr = heap_start + heap_count * 4

        # Update the allocates space counter
        new_heap_count = heap_count + count
        self.ptr_write(heap_ptr, value=new_heap_count)

        self.set_reg(destination_ptr_reg, result_ptr)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, ToyAccelerator):
            return False
        return self.stream is __value.stream

    def __hash__(self) -> int:
        return hash(id(self.stream))
