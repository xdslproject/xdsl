"""
Test the usage of builtin traits.
"""

import pytest

from xdsl.dialects.builtin import ModuleOp
from xdsl.irdl import IRDLOperation, OptSuccessor, irdl_op_definition
from xdsl.ir import Region, Block
from xdsl.traits import HasParent, IsTerminator
from xdsl.utils.exceptions import VerifyException
from xdsl.dialects.test import TestOp


@irdl_op_definition
class ParentOp(IRDLOperation):
    name = "test.parent"

    region: Region


@irdl_op_definition
class Parent2Op(IRDLOperation):
    name = "test.parent2"

    region: Region


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
class IsTerminatorOp(IRDLOperation):
    """
    An operation that provides the IsTerminator trait.
    """

    name = "test.is_terminator"

    successor: OptSuccessor

    traits = frozenset([IsTerminator()])


def test_is_terminator_verify():
    """
    Test that an operation with an IsTerminator trait expects successor blocks.
    """
    block0 = Block([])
    block1 = Block([IsTerminatorOp.create(successors=[block0])])
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
