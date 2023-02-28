from xdsl.dialects.arith import Constant
from xdsl.dialects.builtin import SymbolRefAttr, i32
from xdsl.dialects.gpu import AllReduceOperationAttr, BarrierOp, BlockDimOp, BlockIdOp, GlobalIdOp, GridDimOp, LaneIdOp, ModuleEndOp, ModuleOp, DimensionAttr, NumSubgroupsOp, SetDefaultDeviceOp, SubgroupIdOp, SubgroupSizeOp, ThreadIdOp
from xdsl.ir import Operation


def test_dimension():
    dim = DimensionAttr.from_dimension("x")

    assert dim.data == "x"


def test_all_reduce_operation():
    op = AllReduceOperationAttr.from_op("add")

    assert op.data == "add"


def test_barrier():
    barrier = BarrierOp.get()

    assert isinstance(barrier, BarrierOp)


def test_block_dim():
    dim = DimensionAttr.from_dimension("x")

    block_dim = BlockDimOp.get(dim)

    assert isinstance(block_dim, BlockDimOp)
    assert block_dim.dimension is dim


def test_block_id():
    dim = DimensionAttr.from_dimension("x")

    block_id = BlockIdOp.get(dim)

    assert isinstance(block_id, BlockIdOp)
    assert block_id.dimension is dim


def test_gpu_module():
    name = SymbolRefAttr.from_str("gpu")

    ops: list[Operation] = [ModuleEndOp.get()]

    gpu_module = ModuleOp.get(name, ops)

    assert isinstance(gpu_module, ModuleOp)
    assert gpu_module.body.ops == ops
    assert gpu_module.sym_name is name


def test_gpu_module_end():
    module_end = ModuleEndOp.get()

    assert isinstance(module_end, ModuleEndOp)


def test_global_id():
    dim = DimensionAttr.from_dimension("x")

    global_id = GlobalIdOp.get(dim)

    assert isinstance(global_id, GlobalIdOp)
    assert global_id.dimension is dim


def test_grid_dim():
    dim = DimensionAttr.from_dimension("x")

    grid_dim = GridDimOp.get(dim)

    assert isinstance(grid_dim, GridDimOp)
    assert grid_dim.dimension is dim


def test_lane_id():
    lane_id = LaneIdOp.get()

    assert isinstance(lane_id, LaneIdOp)


def test_num_subgroups():
    num_subgroups = NumSubgroupsOp.get()

    assert isinstance(num_subgroups, NumSubgroupsOp)


def test_set_default_device():
    devIndex = Constant.from_int_and_width(0, i32)

    set_default_device = SetDefaultDeviceOp.get(devIndex)

    assert isinstance(set_default_device, SetDefaultDeviceOp)
    assert set_default_device.devIndex is devIndex.result


def test_subgroup_id():
    subgroup_id = SubgroupIdOp.get()

    assert isinstance(subgroup_id, SubgroupIdOp)


def test_subgroup_size():
    subgroup_size = SubgroupSizeOp.get()

    assert isinstance(subgroup_size, SubgroupSizeOp)


def test_thread_id():
    dim = DimensionAttr.from_dimension("x")

    thread_id = ThreadIdOp.get(dim)

    assert isinstance(thread_id, ThreadIdOp)
    assert thread_id.dimension is dim
