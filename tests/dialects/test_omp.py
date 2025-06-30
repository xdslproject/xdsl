import pytest

from xdsl.dialects import builtin, omp
from xdsl.ir import Block, Operation, Region
from xdsl.irdl import IRDLOperation, irdl_op_definition, region_def, traits_def
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class DummyLoopWrapper(IRDLOperation):
    name = "dummy.loop_wrapper"
    body = region_def()

    traits = traits_def(omp.LoopWrapper())


def dummy_wrapper_module(cls: type, *ops_list: list[Operation]):
    return builtin.ModuleOp([cls(regions=[Region(Block(ops)) for ops in ops_list])])


loop_op = omp.LoopNestOp(
    operands=[[], [], []], regions=[Region(Block([omp.TerminatorOp()]))]
)


def test_loop_wrapper_correct():
    single_loop_nest = dummy_wrapper_module(DummyLoopWrapper, [loop_op.clone()])
    single_loop_nest.verify()

    wrapped_loop_nest = dummy_wrapper_module(
        DummyLoopWrapper, [DummyLoopWrapper(regions=[Region(Block([loop_op.clone()]))])]
    )
    wrapped_loop_nest.verify()


def test_loop_wrapper_no_region():
    @irdl_op_definition
    class DummyLoopWrapperNoRegion(IRDLOperation):
        name = "dummy.loop_wrapper"

        traits = traits_def(omp.LoopWrapper())

    with pytest.raises(
        VerifyException, match="is not a LoopWrapper: has 0 region, expected 1"
    ):
        mod = dummy_wrapper_module(DummyLoopWrapperNoRegion)
        mod.verify()


def test_loop_wrapper_many_regions():
    @irdl_op_definition
    class DummyLoopWrapperManyRegions(IRDLOperation):
        name = "dummy.loop_wrapper"

        traits = traits_def(omp.LoopWrapper())
        b1 = region_def()
        b2 = region_def()

    with pytest.raises(
        VerifyException, match="is not a LoopWrapper: has 2 region, expected 1"
    ):
        mod = dummy_wrapper_module(
            DummyLoopWrapperManyRegions, [loop_op.clone()], [loop_op.clone()]
        )
        mod.verify()


def test_loop_wrapper_many_ops():
    with pytest.raises(
        VerifyException, match="is not a LoopWrapper: has 2 ops, expected 1"
    ):
        mod = dummy_wrapper_module(DummyLoopWrapper, [loop_op.clone(), loop_op.clone()])
        mod.verify()


def test_loop_wrapper_wrong_op():
    with pytest.raises(
        VerifyException,
        match="is not a LoopWrapper: should have a single operation which is either another LoopWrapper or omp.loop_nest",
    ):
        mod = dummy_wrapper_module(DummyLoopWrapper, [omp.TerminatorOp()])
        mod.verify()
