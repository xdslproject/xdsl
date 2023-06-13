"""
Test the usage of builtin traits.
"""

import pytest
from xdsl.dialects import arith, builtin

from xdsl.dialects.builtin import ModuleOp
from xdsl.irdl import (
    IRDLOperation,
    OptSuccessor,
    irdl_op_definition,
    opt_successor_def,
    region_def,
)
from xdsl.ir import Region, Block
from xdsl.traits import (
    HasParent,
    IsTerminator,
    NoTerminator,
    IsolatedFromAbove,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.dialects.test import TestOp


@irdl_op_definition
class ParentOp(IRDLOperation):
    name = "test.parent"

    region: Region = region_def()

    traits = frozenset([NoTerminator()])


@irdl_op_definition
class Parent2Op(IRDLOperation):
    name = "test.parent2"

    region: Region = region_def()

    traits = frozenset([NoTerminator()])


@irdl_op_definition
class HasParentOp(IRDLOperation):
    """
    An operation expecting a 'test.parent_op' parent.
    """

    name = "test.has_parent"

    traits = frozenset([HasParent(ParentOp)])


@irdl_op_definition
class HasMultipleParentOp(IRDLOperation):
    """
    An operation expecting one of multiple parent types.
    """

    name = "test.has_multiple_parent"

    traits = frozenset([HasParent((ParentOp, Parent2Op))])


def test_has_parent_no_parent():
    """
    Test that an operation with an HasParentOp trait
    fails with no parents.
    """
    has_parent_op = HasParentOp()
    with pytest.raises(
        VerifyException, match="'test.has_parent' expects parent op 'test.parent'"
    ):
        has_parent_op.verify()

    has_multiple_parent_op = HasMultipleParentOp()
    message = (
        "'test.has_multiple_parent' expects parent op to "
        "be one of 'test.parent', 'test.parent2'"
    )
    with pytest.raises(VerifyException, match=message):
        has_multiple_parent_op.verify()


def test_has_parent_wrong_parent():
    """
    Test that an operation with an HasParentOp trait
    fails with a wrong parent.
    """
    module = ModuleOp([HasParentOp()])
    with pytest.raises(
        VerifyException, match="'test.has_parent' expects parent op 'test.parent'"
    ):
        module.verify()

    module = ModuleOp([HasMultipleParentOp()])
    message = (
        "'test.has_multiple_parent' expects parent op to "
        "be one of 'test.parent', 'test.parent2'"
    )
    with pytest.raises(VerifyException, match=message):
        module.verify()


def test_has_parent_verify():
    """
    Test that an operation with an HasParentOp trait
    expects a parent operation of the right type.
    """
    op = ParentOp(regions=[[HasParentOp()]])
    op.verify()

    op = ParentOp(regions=[[HasMultipleParentOp()]])
    op.verify()

    op = Parent2Op(regions=[[HasMultipleParentOp()]])
    op.verify()


@irdl_op_definition
class HasNoTerminatorOp(IRDLOperation):
    """
    An operation that can opt out from having a terminator.
    This requires the operation to have a single block.
    """

    name = "test.has_no_terminator"

    region: Region = region_def()

    traits = frozenset([NoTerminator()])


def test_has_no_terminator_empty_block_with_single_block_region_requires_no_terminator():
    """
    Tests that an empty block belonging to a single-block region with parent
    operation requires no terminator operation if it has the NoTerminator trait.
    """
    block0 = Block([])
    region0 = Region([block0])
    _ = HasNoTerminatorOp.create(regions=[region0])

    block0.verify()


@irdl_op_definition
class IsTerminatorOp(IRDLOperation):
    """
    An operation that provides the IsTerminator trait.
    """

    name = "test.is_terminator"

    successor: OptSuccessor = opt_successor_def()

    traits = frozenset([IsTerminator()])


def test_is_terminator_without_successors_verify():
    """
    Test that an operation with an IsTerminator trait may not have successor
    blocks.
    """
    block0 = Block([])
    block1 = Block([IsTerminatorOp.create()])
    region0 = Region([block0, block1])
    op0 = TestOp.create(regions=[region0])

    op0.verify()


def test_is_terminator_fails_if_not_last_operation_parent_block():
    """
    Test that an operation with an IsTerminator trait fails if it is not the
    last operation in its parent block.
    """
    block0 = Block([IsTerminatorOp.create(), TestOp.create()])
    region0 = Region([block0])
    op0 = TestOp.create(regions=[region0])

    with pytest.raises(
        VerifyException, match="must be the last operation in the parent block"
    ):
        op0.verify()


@irdl_op_definition
class IsolatedFromAboveOp(IRDLOperation):
    """
    An isolated from above operation.
    """

    name = "test.isolated_from_above"

    region: Region = region_def()

    traits = frozenset([IsolatedFromAbove(), NoTerminator()])


def test_isolated_from_above():
    # Empty Isolated is fine
    op = IsolatedFromAboveOp(regions=[Region()])
    op.verify()

    block = Block(arg_types=[builtin.i32])
    block.add_op(arith.Addi(block.args[0], block.args[0]))

    # Test a simple, properly Isolated
    op = IsolatedFromAboveOp(regions=[Region([block])])
    op.verify()

    # Check a simple isolation violation
    out_cst = arith.Constant.from_int_and_width(0, builtin.i32)
    out_block = Block(
        [
            out_cst,
            IsolatedFromAboveOp(
                regions=[Region(Block([arith.Addi(out_cst, out_cst)]))]
            ),
        ]
    )
    message = "Operation using value defined out of its IsolatedFromAbove parent!"
    with pytest.raises(VerifyException, match=message):
        out_block.verify()

    # Check a nested isolation violation
    out_cst = arith.Constant.from_int_and_width(0, builtin.i32)
    out_block = Block(
        [
            # This one is fine
            out_isolated := IsolatedFromAboveOp(
                regions=[
                    Region(
                        Block(
                            [
                                out_cst,
                                # This one is not!
                                in_isolated := IsolatedFromAboveOp(
                                    regions=[
                                        Region(Block([arith.Addi(out_cst, out_cst)]))
                                    ]
                                ),
                            ],
                        )
                    )
                ]
            ),
        ]
    )
    # Check that the IR as a whole is wrong
    message = "Operation using value defined out of its IsolatedFromAbove parent!"
    with pytest.raises(VerifyException, match=message):
        out_block.verify()
    # Check that the outer one in itself is fine
    out_isolated.verify(verify_nested_ops=False)
    # Check that the inner one is indeed failing to verify.
    with pytest.raises(VerifyException, match=message):
        in_isolated.verify()
