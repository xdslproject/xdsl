from collections.abc import Sequence
from typing import cast

from xdsl.backend.register_queue import RegisterQueue
from xdsl.backend.register_type import RegisterType
from xdsl.ir import Attribute, SSAValue
from xdsl.rewriter import Rewriter
from xdsl.utils.exceptions import DiagnosticException


class ValueAllocator:
    """
    Base class for register allocators.
    """

    available_registers: RegisterQueue
    register_base_class: type[RegisterType]
    new_value_by_old_value: dict[SSAValue, SSAValue]

    def __init__(
        self,
        available_registers: RegisterQueue,
        register_base_class: type[RegisterType],
    ) -> None:
        self.available_registers = available_registers
        self.register_base_class = register_base_class
        self.new_value_by_old_value = {}

    def new_type_for_value(self, reg: SSAValue) -> RegisterType | None:
        if isinstance(reg.type, self.register_base_class) and not reg.type.is_allocated:
            return self.available_registers.pop(type(reg.type))

    def _replace_value_with_new_type(
        self, val: SSAValue, new_type: Attribute
    ) -> SSAValue:
        new_val = Rewriter.replace_value_with_new_type(val, new_type)
        self.new_value_by_old_value[val] = new_val
        return new_val

    def allocate_value(self, val: SSAValue) -> SSAValue | None:
        """
        Allocate a register if not already allocated.
        """
        if val in self.new_value_by_old_value:
            return
        new_type = self.new_type_for_value(val)
        if new_type is not None:
            return self._replace_value_with_new_type(val, new_type)

    def allocate_values_same_reg(self, vals: Sequence[SSAValue]) -> bool:
        """
        Allocates the values passed in to the same register.
        If some of the values are already allocated, they must be allocated to the same
        register, and unallocated values are then allocated to this register.
        If the values passed in are already allocated to differing registers, a
        `DiagnosticException` is raised.
        """
        reg_types = set(val.type for val in vals)
        assert all(
            isinstance(reg_type, self.register_base_class) for reg_type in reg_types
        )
        reg_types = cast(set[RegisterType], reg_types)

        match len(reg_types):
            case 0:
                # No inputs, nothing to do
                return False
            case 1:
                # Single input, may already be allocated
                (reg_type,) = reg_types
                if reg_type.is_allocated:
                    return False
                else:
                    reg_type = self.available_registers.pop(type(reg_type))
            case 2:
                # Two inputs, either one is allocated or two
                reg_type_0, reg_type_1 = reg_types
                if reg_type_0.is_allocated:
                    if reg_type_1.is_allocated:
                        reg_names = [f"{reg_type}" for reg_type in reg_types]
                        reg_names.sort()
                        raise DiagnosticException(
                            f"Cannot allocate registers to the same register {reg_names}"
                        )
                    else:
                        reg_type = reg_type_0
                else:
                    reg_type = reg_type_1
            case _:
                # More than one input is allocated, meaning we can't allocate them to be
                # the same, error.
                reg_names = [f"{reg_type}" for reg_type in reg_types]
                reg_names.sort()
                raise DiagnosticException(
                    f"Cannot allocate registers to the same register {reg_names}"
                )

        did_allocate = False

        for val in vals:
            if val.type != reg_type:
                self._replace_value_with_new_type(val, reg_type)
                did_allocate = True

        return did_allocate

    def free_value(self, val: SSAValue) -> None:
        if isinstance(val.type, self.register_base_class) and val.type.is_allocated:
            self.available_registers.push(val.type)
