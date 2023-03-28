from xdsl.dialects import builtin, arith, memref
from xdsl.dialects.gpu import (
    AllocOp, AllReduceOp, AllReduceOperationAttr, AsyncTokenType, BarrierOp,
    BlockDimOp, BlockIdOp, DeallocOp, GlobalIdOp, GridDimOp, HostRegisterOp,
    LaneIdOp, LaunchOp, ModuleEndOp, ModuleOp, DimensionAttr, NumSubgroupsOp,
    SetDefaultDeviceOp, SubgroupIdOp, SubgroupSizeOp, TerminatorOp, ThreadIdOp,
    YieldOp)
from xdsl.ir import Block, Operation, Region, SSAValue


def test_dimension():
    dim = DimensionAttr.from_dimension("x")

    assert dim.data == "x"


def test_alloc():
    typ = memref.MemRefType.from_element_type_and_shape(
        builtin.Float32Type(), [10, 10, 10])
    alloc = AllocOp.get(typ, is_async=True)

    assert isinstance(alloc, AllocOp)
    assert alloc.result.typ is typ
    assert len(alloc.asyncDependencies) == 0
    assert len(alloc.dynamicSizes) == 0
    assert alloc.asyncToken is not None
    assert isinstance(alloc.asyncToken.typ, AsyncTokenType)
    assert alloc.hostShared is None

    dyntyp = memref.MemRefType.from_element_type_and_shape(
        builtin.Float32Type(), [-1, -1, -1])
    ten = arith.Constant.from_int_and_width(10, builtin.IndexType())
    dynamic_sizes = [ten, ten, ten]
    token = alloc.asyncToken

    full_alloc = AllocOp.get(return_type=dyntyp,
                             dynamic_sizes=dynamic_sizes,
                             host_shared=True,
                             async_dependencies=[token])

    assert isinstance(full_alloc, AllocOp)
    assert full_alloc.result.typ is dyntyp
    assert len(full_alloc.asyncDependencies) == 1
    assert full_alloc.asyncDependencies[0] is token
    assert len(full_alloc.dynamicSizes) == 3
    assert full_alloc.dynamicSizes == tuple(d.result for d in dynamic_sizes)
    assert full_alloc.asyncToken is None
    assert "hostShared" in full_alloc.attributes.keys()
    assert isinstance(full_alloc.hostShared, builtin.UnitAttr)


def test_all_reduce_operation():
    op = AllReduceOperationAttr.from_op("add")

    assert op.data == "add"


def test_all_reduce():
    op = AllReduceOperationAttr.from_op("add")

    init = arith.Constant.from_int_and_width(0, builtin.IndexType())

    all_reduce = AllReduceOp.from_op(op, init)

    assert isinstance(all_reduce, AllReduceOp)
    assert all_reduce.op is op
    assert all_reduce.operand is init.result
    assert all_reduce.uniform is None
    assert all_reduce.result.typ is all_reduce.operand.typ

    body_block = Block.from_arg_types(
        [builtin.IndexType(), builtin.IndexType()])

    ops: list[Operation] = [
        sum := Operation.clone(arith.Addi.get(*body_block.args)),
        YieldOp.get([sum])
    ]

    body = Region.from_operation_list(ops)

    all_reduce_body = AllReduceOp.from_body(body, init)

    assert isinstance(all_reduce_body, AllReduceOp)
    assert all_reduce_body.op is None
    assert all_reduce_body.operand is init.result
    assert all_reduce_body.uniform is None
    assert all_reduce_body.result.typ is all_reduce_body.operand.typ


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


def test_dealloc():

    typ = memref.MemRefType.from_element_type_and_shape(
        builtin.Float32Type(), [10, 10, 10])
    alloc = AllocOp.get(typ, is_async=True)

    assert alloc.asyncToken is not None  # For pyright

    dealloc = DeallocOp.get(buffer=alloc.result,
                            async_dependencies=[alloc.asyncToken],
                            is_async=True)

    assert dealloc.asyncToken is not None
    assert isinstance(dealloc.asyncToken.typ, AsyncTokenType)
    assert dealloc.buffer is alloc.result
    assert dealloc.asyncDependencies == tuple([alloc.asyncToken])

    alloc2 = AllocOp.get(typ, is_async=True)
    sync_dealloc = DeallocOp.get(buffer=alloc2.result)

    assert sync_dealloc.asyncToken is None
    assert sync_dealloc.buffer is alloc2.result
    assert len(sync_dealloc.asyncDependencies) == 0


def test_gpu_module():
    name = builtin.SymbolRefAttr("gpu")

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


def test_host_register():
    memref_type = memref.MemRefType.from_element_type_and_shape(
        builtin.i32, [10, 10])
    ref = memref.Alloca.get(memref_type, 0)

    unranked = memref.Cast.build(
        operands=[ref],
        result_types=[memref.UnrankedMemrefType.from_type(builtin.i32)])

    register = HostRegisterOp.from_memref(unranked)

    assert isinstance(register, HostRegisterOp)
    assert register.value is unranked.results[0]


def test_lane_id():
    lane_id = LaneIdOp.get()

    assert isinstance(lane_id, LaneIdOp)


def test_launch():
    body = Region.get([])
    ten = arith.Constant.from_int_and_width(10, builtin.IndexType())
    gridSize: list[Operation | SSAValue] = [ten, ten, ten]
    blockSize: list[Operation | SSAValue] = [ten, ten, ten]
    launch = LaunchOp.get(body, gridSize, blockSize)

    assert isinstance(launch, LaunchOp)
    assert launch.body is body
    assert launch.gridSizeX is ten.result
    assert launch.gridSizeY is ten.result
    assert launch.gridSizeZ is ten.result
    assert launch.blockSizeX is ten.result
    assert launch.blockSizeY is ten.result
    assert launch.blockSizeZ is ten.result
    assert launch.asyncToken is None
    assert launch.asyncDependencies == tuple()
    assert launch.dynamicSharedMemorySize is None

    asyncDependencies = []

    body2 = Region()

    nd_launch = LaunchOp.get(body2, gridSize, blockSize, True,
                             asyncDependencies, ten)

    assert isinstance(launch, LaunchOp)
    assert nd_launch.body is body2
    assert nd_launch.gridSizeX is ten.result
    assert nd_launch.gridSizeY is ten.result
    assert nd_launch.gridSizeZ is ten.result
    assert nd_launch.blockSizeX is ten.result
    assert nd_launch.blockSizeY is ten.result
    assert nd_launch.blockSizeZ is ten.result
    assert nd_launch.asyncToken is not None
    assert nd_launch.asyncToken.typ == AsyncTokenType()
    assert nd_launch.asyncDependencies == tuple()
    assert nd_launch.dynamicSharedMemorySize is ten.result


def test_num_subgroups():
    num_subgroups = NumSubgroupsOp.get()

    assert isinstance(num_subgroups, NumSubgroupsOp)


def test_set_default_device():
    devIndex = arith.Constant.from_int_and_width(0, builtin.i32)

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


def test_terminator():
    terminator = TerminatorOp.get()

    assert isinstance(terminator, TerminatorOp)


def test_yield():

    operands: list[SSAValue | Operation] = [
        o for o in [
            arith.Constant.from_int_and_width(42, builtin.i32),
            arith.Constant.from_int_and_width(19, builtin.IndexType()),
            arith.Constant.from_int_and_width(84, builtin.i64),
        ]
    ]
    yield_op = YieldOp.get(operands)

    assert isinstance(yield_op, YieldOp)
    assert yield_op.operands == tuple([SSAValue.get(o) for o in operands])
