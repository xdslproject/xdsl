"""
List dialect to represent (recursively) linked lists.
"""

from __future__ import annotations

from typing import ClassVar

from xdsl.dialects import test
from xdsl.dialects.builtin import i32
from xdsl.ir import Dialect, ParametrizedAttribute, SSAValue, TypeAttribute
from xdsl.irdl import (
    AnyAttr,
    IRDLOperation,
    ParameterDef,
    VarConstraint,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.traits import Pure


@irdl_attr_definition
class List(ParametrizedAttribute, TypeAttribute):
    """
    The `list.list` type represents a linked list of elements of type T.
    """

    name = "list.list"

    element_type: ParameterDef[TypeAttribute]

    def __init__(self, element_type: TypeAttribute):
        super().__init__((element_type,))


@irdl_op_definition
class ConsOp(IRDLOperation):
    """
    The `list.cons` operation constructs a new list by combining two operands.
    Both operands can be either an element of type T or a list of type List[T].
    """

    name = "list.cons"

    car = operand_def(AnyAttr())
    cdr = opt_operand_def(AnyAttr())
    result = result_def(AnyAttr())

    traits = traits_def(Pure())

    assembly_format = (
        "`(` operands `)` attr-dict `:` `(` type(operands) `)` `->` type($result)"
    )

    def __init__(
        self,
        car: SSAValue,
        cdr: SSAValue | None = None,
        result_type: List | None = None,
    ):
        # Infer the element type from the car operand
        car_type = car.type
        if isinstance(car_type, List):
            element_type = car_type.element_type
        else:
            element_type = car_type

        # If cdr is provided, verify type consistency
        if cdr is not None:
            cdr_type = cdr.type
            if isinstance(cdr_type, List):
                assert element_type == cdr_type.element_type, "Element types must match"
            else:
                assert element_type == cdr_type, "Element types must match"

        assert isinstance(element_type, TypeAttribute), (
            "Expected element type to be a TypeAttribute"
        )

        # Create the result list type
        if result_type is not None:
            # Verify consistency with inferred type
            assert result_type.element_type == element_type, (
                "Explicit result type must match inferred element type"
            )
            final_result_type = result_type
        else:
            final_result_type = List(element_type)

        super().__init__(operands=(car, cdr), result_types=(final_result_type,))

    def verify_(self) -> None:
        """
        Manually verify type constraints since we use AnyAttr for flexible typing.
        Ensures all operands and result have compatible element types.
        """
        car_type = self.car.type

        # Extract element type from car operand
        if isinstance(car_type, List):
            element_type = car_type.element_type
        else:
            element_type = car_type

        # Check cdr operand if present
        if self.cdr is not None:
            cdr_type = self.cdr.type
            if isinstance(cdr_type, List):
                if element_type != cdr_type.element_type:
                    raise Exception(
                        f"Element type mismatch: car has {element_type}, cdr has List[{cdr_type.element_type}]"
                    )
            else:
                if element_type != cdr_type:
                    raise Exception(
                        f"Element type mismatch: car has {element_type}, cdr has {cdr_type}"
                    )

        # Check result type
        result_type = self.result.type
        if not isinstance(result_type, List):
            raise Exception(f"Result must be a List type, got {result_type}")

        if result_type.element_type != element_type:
            raise Exception(
                f"Result element type mismatch: expected List[{element_type}], got {result_type}"
            )


@irdl_op_definition
class ListOp(IRDLOperation):
    """
    The `list.op` takes a number of list elements and creates a list containing those elements.
    """

    name = "list.list"

    T: ClassVar = VarConstraint("T", AnyAttr())

    values = var_operand_def(T)
    result = result_def(List)

    traits = traits_def(Pure())

    assembly_format = (
        "`(`$values`)` `:` `(` type($values) `)` attr-dict `:` type($result)"
    )

    def __init__(self, *values: SSAValue):
        """
        Initialize the ListOp with a variable number of SSA values.
        """
        T = values[0].type
        assert isinstance(T, TypeAttribute), (
            "Expected SSAValue type to be a type attribute"
        )
        super().__init__(operands=(values,), result_types=(List(T),))


LinkedList = Dialect(
    "list",
    [
        ConsOp,
        ListOp,
    ],
    [
        List,
    ],
)

if __name__ == "__main__":
    # Demonstrate the flexible ConsOp implementation
    print("=== List Dialect Demo ===")

    # Create test values
    a = test.TestOp(result_types=(i32,)).results[0]
    b = test.TestOp(result_types=(i32,)).results[0]
    from xdsl.dialects.builtin import f32

    c = test.TestOp(result_types=(f32,)).results[0]

    # Create lists
    lo = ListOp(a, b, a)
    print(f"List creation: {lo}")

    # Demonstrate all ConsOp variants
    l = lo.result
    print(f"Element + List: {ConsOp(car=a, cdr=l)}")
    print(f"List + Element: {ConsOp(car=l, cdr=a)}")
    print(f"List + List: {ConsOp(car=l, cdr=ListOp(b).result)}")
    print(f"Element only: {ConsOp(car=a)}")

    # Show type safety in action
    print("\n=== Type Safety Demo ===")
    lo_f32 = ListOp(c)

    # These should fail
    invalid_ops = [
        ("i32 element + f32 list", lambda: ConsOp(car=a, cdr=lo_f32.result)),
        ("f32 element + i32 list", lambda: ConsOp(car=c, cdr=l)),
    ]

    for desc, create_op in invalid_ops:
        try:
            create_op().verify()
            print(f"✗ {desc}: Should have been rejected!")
        except Exception:
            print(f"✓ {desc}: Correctly rejected")

    print("✓ Type constraints working correctly")
