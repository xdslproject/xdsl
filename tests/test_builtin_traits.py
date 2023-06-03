"""
Test the usage of builtin traits.
"""

import pytest

from xdsl.dialects.builtin import ModuleOp
from xdsl.irdl import IRDLOperation, irdl_op_definition, Region
from xdsl.ir import Block
from xdsl.traits import HasParent, NoTerminator
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
    with pytest.raises(VerifyException) as exc_info:
        has_parent_op.verify()
    assert str(exc_info.value) == "'test.has_parent' expects parent op 'test.parent'"

    has_multiple_parent_op = HasMultipleParentOp()
    with pytest.raises(VerifyException) as exc_info:
        has_multiple_parent_op.verify()
    assert str(exc_info.value) == (
        "'test.has_multiple_parent' expects parent op to "
        "be one of 'test.parent', 'test.parent2'"
    )


def test_has_parent_wrong_parent():
    """
    Test that an operation with an HasParentOp trait
    fails with a wrong parent.
    """
    module = ModuleOp([HasParentOp()])
    with pytest.raises(VerifyException) as exc_info:
        module.verify()
    assert str(exc_info.value) == "'test.has_parent' expects parent op 'test.parent'"

    module = ModuleOp([HasMultipleParentOp()])
    with pytest.raises(VerifyException) as exc_info:
        module.verify()
    assert str(exc_info.value) == (
        "'test.has_multiple_parent' expects parent op to "
        "be one of 'test.parent', 'test.parent2'"
    )


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

    region: Region

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
