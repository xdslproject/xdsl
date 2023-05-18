"""
Test the usage of builtin traits.
"""

import pytest

from xdsl.dialects.builtin import ModuleOp
from xdsl.irdl import IRDLOperation, irdl_op_definition
from xdsl.ir import Region
from xdsl.traits import HasParent
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class ParentOp(IRDLOperation):
    name = "test.parent_op"

    region: Region


@irdl_op_definition
class HasParentOp(IRDLOperation):
    """
    An operation expecting a 'test.parent_op' parent.
    """

    name = "test.has_parent_op"

    traits = frozenset([HasParent(ParentOp)])


def test_has_parent_no_parent():
    """
    Test that an operation with an HasParentOp trait
    fails with no parents.
    """
    has_parent_op = HasParentOp()
    with pytest.raises(VerifyException) as exc_info:
        has_parent_op.verify()
    assert "has no parent" in str(exc_info.value)


def test_has_parent_wrong_parent():
    """
    Test that an operation with an HasParentOp trait
    fails with a wrong parent.
    """
    module = ModuleOp([HasParentOp()])
    with pytest.raises(VerifyException) as exc_info:
        module.verify()
    assert "has a parent of type 'builtin.module'" in str(exc_info.value)


def test_has_parent_pass():
    """
    Test that an operation with an HasParentOp trait
    expects a parent operation of the right type.
    """
    parent_op = HasParentOp()
    has_parent_op = ParentOp(regions=[[parent_op]])
    has_parent_op.verify()

    module = ModuleOp([has_parent_op])
    module.verify()
