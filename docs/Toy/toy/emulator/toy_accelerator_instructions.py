"""
This file contains the implementations of the custom accelerator instructions for riscemu.

These are temporary, until xDSL supports all the functionality necessary to lower from
builtin+printf to riscv.
"""

from typing import IO, ClassVar

from riscemu.core import Float64, Instruction, Int32
from riscemu.instructions.instruction_set import InstructionSet

from xdsl.interpreters.ptr import RawPtr


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

    # Missing fcvt.d.w instruction

    def instruction_fcvt_d_w(self, ins: Instruction):
        """
        Converts a 32-bit unsigned integer, in integer register rs1 into a double-precision floating-point number in floating-point register rd.

        f[rd] = f64_{u32}(x[rs1])

        https://msyksphinz-self.github.io/riscv-isadoc/html/rvfd.html#fcvt-d-wu
        """

        rd = ins.get_reg(0)
        rs1 = ins.get_reg(1)

        rs1_val = self.regs.get(rs1)
        self.regs.set_f(rd, Float64(float(rs1_val.value)))

    # Custom instructions

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

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, ToyAccelerator):
            return False
        return self.stream is __value.stream

    def __hash__(self) -> int:
        return hash(id(self.stream))
