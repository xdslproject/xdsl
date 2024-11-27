"""
Test the usage of builtin traits.
"""

import pytest

from xdsl.dialects import arith, builtin
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.test import TestOp
from xdsl.ir import Block, Region
from xdsl.irdl import (
    AttrSizedRegionSegments,
    IRDLOperation,
    irdl_op_definition,
    lazy_traits_def,
    opt_region_def,
    opt_successor_def,
    region_def,
    traits_def,
)
from xdsl.traits import (
    HasParent,
    IsolatedFromAbove,
    IsTerminator,
    NoTerminator,
    SingleBlockImplicitTerminator,
    ensure_terminator,
)
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class ParentOp(IRDLOperation):
    name = "test.parent"

    region = region_def()

    traits = traits_def(NoTerminator())


@irdl_op_definition
class Parent2Op(IRDLOperation):
    name = "test.parent2"

    region = region_def()

    traits = traits_def(NoTerminator())


@irdl_op_definition
class HasParentOp(IRDLOperation):
    """
    An operation expecting a 'test.parent_op' parent.
    """

    name = "test.has_parent"

    traits = traits_def(HasParent(ParentOp))


@irdl_op_definition
class HasMultipleParentOp(IRDLOperation):
    """
    An operation expecting one of multiple parent types.
    """

    name = "test.has_multiple_parent"

    traits = traits_def(HasParent(ParentOp, Parent2Op))


def test_has_parent_no_parent():
    """
    A detached op with a HasParent trait should be verifyable when detached
    """

    single_parent = HasParentOp()

    single_parent.verify()

    multiple_parent = HasMultipleParentOp()

    multiple_parent.verify()


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

    region = region_def()

    traits = traits_def(NoTerminator())


def test_has_no_terminator_empty_block_with_single_block_region_requires_no_terminator():
    """
    Tests that an empty block belonging to a single-block region with parent
    operation requires no terminator operation if it has the NoTerminator trait.
    """
    block0 = Block([])
    region0 = Region([block0])
    _ = HasNoTerminatorOp.create(regions=[region0])

    block0.verify()


def test_has_no_terminator_empty_block_with_multi_block_region_fails():
    """
    Tests that an empty block belonging to a multi-block region with parent
    operation fails if it has the NoTerminator trait.
    """
    block0 = Block([])
    block1 = Block([])
    region0 = Region([block0, block1])
    op0 = HasNoTerminatorOp.create(regions=[region0])

    with pytest.raises(VerifyException, match="does not contain single-block regions"):
        op0.verify()


@irdl_op_definition
class IsTerminatorOp(IRDLOperation):
    """
    An operation that provides the IsTerminator trait.
    """

    name = "test.is_terminator"

    successor = opt_successor_def()

    traits = traits_def(IsTerminator())


def test_is_terminator_without_successors_multi_block_parent_region_verify():
    """
    Test that an operation with an IsTerminator trait may not have successor
    blocks in a multi-block parent region.
    """

    block0 = Block([])
    # term op is the single op in its block
    block1 = Block([IsTerminatorOp.create()])
    region0 = Region([block0, block1])
    op0 = TestOp.create(regions=[region0])

    op0.verify()

    block2 = Block([])
    # term op with other ops in its block
    block3 = Block([TestOp.create(), IsTerminatorOp.create()])
    region1 = Region([block2, block3])
    op1 = TestOp.create(regions=[region1])

    op1.verify()


def test_is_terminator_without_successors_single_block_parent_region_verify():
    """
    Test that an operation with an IsTerminator trait may not have successor
    blocks in a single-block parent region.
    """
    # term op is the single op in its block
    block0 = Block([IsTerminatorOp.create()])
    region0 = Region([block0])
    op0 = TestOp.create(regions=[region0])

    op0.verify()

    # term op with other ops in its block
    block1 = Block([TestOp.create(), IsTerminatorOp.create()])
    region1 = Region([block1])
    op1 = TestOp.create(regions=[region1])

    op1.verify()


def test_is_terminator_fails_if_not_last_op_parent_block_in_single_block_region():
    """
    Test that an operation with an IsTerminator trait fails if it is not the
    last operation in its parent block in a single-block region.
    """
    block0 = Block([IsTerminatorOp.create(), TestOp.create()])
    region0 = Region([block0])
    op0 = TestOp.create(regions=[region0])

    with pytest.raises(
        VerifyException, match="must be the last operation in its parent block"
    ):
        op0.verify()


def test_is_terminator_fails_if_not_last_op_parent_block_in_multi_block_region():
    """
    Test that an operation without an IsTerminator trait verifies if it is not
    the last operation in its parent block in a multi-block region.
    """
    block0 = Block([IsTerminatorOp.create(), TestOp.create()])
    block1 = Block([])
    region0 = Region([block0, block1])
    op0 = TestOp.create(regions=[region0])

    with pytest.raises(
        VerifyException, match="must be the last operation in its parent block"
    ):
        op0.verify()


def test_no_terminator_op_with_is_terminator_op():
    """
    Test that an operation with a NoTerminator trait verifies if it contains a
    terminator operation (i.e., has the IsTerminator trait).
    """
    block0 = Block([IsTerminatorOp.create()])
    region0 = Region([block0])
    op0 = HasNoTerminatorOp.create(regions=[region0])

    op0.verify()


@irdl_op_definition
class IsSingleBlockImplicitTerminatorOp(IRDLOperation):
    """
    An operation that implements terminator to be used with an operation that uses the
    SingleBlockImplicitTerminator trait.
    """

    name = "test.is_single_block_implicit_terminator"

    traits = lazy_traits_def(
        lambda: (IsTerminator(), HasParent(HasSingleBlockImplicitTerminatorOp))
    )


@irdl_op_definition
class HasSingleBlockImplicitTerminatorOp(IRDLOperation):
    """
    An operation that expects a single-block region and an implicit terminator trait for
    that block.
    """

    name = "test.has_single_block_implicit_terminator"

    irdl_options = [AttrSizedRegionSegments()]

    region = region_def()
    opt_region = opt_region_def()

    traits = traits_def(
        SingleBlockImplicitTerminator(IsSingleBlockImplicitTerminatorOp)
    )

    def __post_init__(self):
        for trait in self.get_traits_of_type(SingleBlockImplicitTerminator):
            ensure_terminator(self, trait)


@irdl_op_definition
class HasSingleBlockImplicitTerminatorWrongCreationOp(IRDLOperation):
    """
    An operation that expects a single-block region and an implicit terminator trait for
    that block, but ensure_terminator() has not been called during construction.
    This exercises the SingleBlockImplicitTerminator.verify() checks.
    """

    name = "test.has_single_block_implicit_terminator_wrong_creation"

    irdl_options = [AttrSizedRegionSegments()]

    region = region_def()
    opt_region = opt_region_def()

    traits = traits_def(
        SingleBlockImplicitTerminator(IsSingleBlockImplicitTerminatorOp)
    )


@irdl_op_definition
class HasSingleBlockImplicitTerminatorWrongCreation2Op(IRDLOperation):
    """
    An operation that expects a single-block region and an implicit terminator trait for
    that block, but ensure_terminator() has not been called during construction.
    This exercises the SingleBlockImplicitTerminator.verify() checks that expects at
    least a terminator.
    This is achieved by adding the trait NoTerminator, but it should catch cases where
    these traits are used in a conflicting manner.
    """

    name = "test.has_single_block_implicit_terminator_wrong_creation2"

    irdl_options = [AttrSizedRegionSegments()]

    region = region_def()
    opt_region = opt_region_def()

    traits = traits_def(
        NoTerminator(),
        SingleBlockImplicitTerminator(IsSingleBlockImplicitTerminatorOp),
    )


def test_single_block_implicit_terminator_verify():
    # test empty single-region op
    op0 = HasSingleBlockImplicitTerminatorOp(regions=[Region(), []])
    op0.verify()
    assert len(op0.region.block.ops) == 1

    # test empty multi-region op
    op1 = HasSingleBlockImplicitTerminatorOp(regions=[Region(), Region()])
    op1.verify()
    assert len(op1.region.block.ops) == 1
    assert op1.opt_region is not None
    assert len(op1.opt_region.block.ops) == 1

    # test non-empty multi-region op
    op2 = HasSingleBlockImplicitTerminatorOp(regions=[Region(Block()), Region()])
    op2.verify()
    assert len(op2.region.block.ops) == 1
    assert op2.opt_region is not None
    assert len(op2.opt_region.block.ops) == 1

    # test non-empty multi-region op with non-terminator operation
    op3 = HasSingleBlockImplicitTerminatorOp(
        regions=[Region(Block([TestOp.create()])), Region()]
    )
    op3.verify()
    assert len(op3.region.block.ops) == 2
    assert op3.opt_region is not None
    assert len(op3.opt_region.block.ops) == 1

    # test non-empty multi-region op with correct terminator already there
    op4 = HasSingleBlockImplicitTerminatorOp(
        regions=[Region(Block([IsSingleBlockImplicitTerminatorOp.create()])), Region()]
    )
    op4.verify()
    assert len(op4.region.block.ops) == 1
    assert op4.opt_region is not None
    assert len(op4.opt_region.block.ops) == 1


def test_single_block_implicit_terminator_with_correct_construction_fail():
    """
    Tests SingleBlockImplicitTerminator when ensure_terminator has been called during
    operation creation
    """

    # test multi-block region op
    with pytest.raises(VerifyException, match="does not contain single-block regions"):
        HasSingleBlockImplicitTerminatorOp(
            regions=[Region([Block(), Block()]), Region()]
        )

    # test single-block region op with wrong terminator
    with pytest.raises(
        VerifyException, match="terminates with operation test.is_terminator"
    ):
        HasSingleBlockImplicitTerminatorOp(
            regions=[Region(Block([IsTerminatorOp.create()])), Region()]
        )


def test_single_block_implicit_terminator_with_wrong_construction_fail():
    """
    Tests SingleBlockImplicitTerminator when ensure_terminator has not been called during
    operation creation
    """

    op0 = HasSingleBlockImplicitTerminatorWrongCreationOp(
        regions=[Region([Block(), Block()]), Region()]
    )
    # test multi-block region op
    with pytest.raises(VerifyException, match="does not contain single-block regions"):
        op0.verify()

    op1 = HasSingleBlockImplicitTerminatorWrongCreationOp(
        regions=[Region(Block([IsTerminatorOp.create()])), Region()]
    )
    # test single-block region op with wrong terminator
    with pytest.raises(
        VerifyException, match="terminates with operation test.is_terminator"
    ):
        op1.verify()

    op2 = HasSingleBlockImplicitTerminatorWrongCreation2Op(
        regions=[Region(Block()), Region()]
    )
    # test single-block region op with wrong terminator
    with pytest.raises(
        VerifyException,
        match="contains empty block instead of at least terminating with",
    ):
        op2.verify()


@irdl_op_definition
class IsolatedFromAboveOp(IRDLOperation):
    """
    An isolated from above operation.
    """

    name = "test.isolated_from_above"

    region = region_def()

    traits = traits_def(IsolatedFromAbove(), NoTerminator())


def test_isolated_from_above():
    # Empty Isolated is fine
    op = IsolatedFromAboveOp(regions=[Region()])
    op.verify()

    block = Block(arg_types=[builtin.i32])
    block.add_op(arith.AddiOp(block.args[0], block.args[0]))

    # Test a simple, properly Isolated
    op = IsolatedFromAboveOp(regions=[Region([block])])
    op.verify()

    # Check a simple isolation violation
    out_cst = arith.ConstantOp.from_int_and_width(0, builtin.i32)
    out_block = Block(
        [
            out_cst,
            IsolatedFromAboveOp(
                regions=[Region(Block([arith.AddiOp(out_cst, out_cst)]))]
            ),
        ]
    )
    message = r"Operation using value defined out of its IsolatedFromAbove parent: AddiOp\(%\d+ = arith.addi %\d+, %\d+ : i32\)"
    with pytest.raises(VerifyException, match=message):
        out_block.verify()

    # Check a nested isolation violation
    out_cst = arith.ConstantOp.from_int_and_width(0, builtin.i32)
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
                                        Region(Block([arith.AddiOp(out_cst, out_cst)]))
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
    message = r"Operation using value defined out of its IsolatedFromAbove parent: AddiOp\(%\d+ = arith.addi %\d+, %\d+ : i32\)"
    with pytest.raises(VerifyException, match=message):
        out_block.verify()
    # Check that the outer one in itself is fine
    out_isolated.verify(verify_nested_ops=False)
    # Check that the inner one is indeed failing to verify.
    with pytest.raises(VerifyException, match=message):
        in_isolated.verify()
