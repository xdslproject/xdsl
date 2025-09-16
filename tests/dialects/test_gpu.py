from xdsl.builder import Builder
from xdsl.dialects import arith, builtin, memref
from xdsl.dialects.gpu import (
    AllocOp,
    AllReduceOp,
    AllReduceOpAttr,
    AllReduceOpEnum,
    AsyncTokenType,
    BarrierOp,
    BlockDimOp,
    BlockIdOp,
    DeallocOp,
    DimensionAttr,
    DimensionEnum,
    FuncOp,
    GlobalIdOp,
    GridDimOp,
    HostRegisterOp,
    HostUnregisterOp,
    LaneIdOp,
    LaunchFuncOp,
    LaunchOp,
    MemcpyOp,
    ModuleOp,
    NumSubgroupsOp,
    ReturnOp,
    SetDefaultDeviceOp,
    SubgroupIdOp,
    SubgroupSizeOp,
    TerminatorOp,
    ThreadIdOp,
    WaitOp,
    YieldOp,
)
from xdsl.ir import Block, Operation, Region, SSAValue


def test_dimension():
    dim = DimensionAttr(DimensionEnum.X)

    assert dim.data == DimensionEnum.X


def test_alloc():
    memref_type = memref.MemRefType(builtin.f32, [10, 10, 10])
    alloc = AllocOp(memref_type, is_async=True)

    assert isinstance(alloc, AllocOp)
    assert alloc.result.type is memref_type
    assert len(alloc.asyncDependencies) == 0
    assert len(alloc.dynamicSizes) == 0
    assert alloc.asyncToken is not None
    assert isinstance(alloc.asyncToken.type, AsyncTokenType)
    assert alloc.hostShared is None

    dyn_type = memref.MemRefType(builtin.f32, [-1, -1, -1])
    ten = arith.ConstantOp.from_int_and_width(10, builtin.IndexType())
    dynamic_sizes = [ten, ten, ten]
    token = alloc.asyncToken

    full_alloc = AllocOp(
        return_type=dyn_type,
        dynamic_sizes=dynamic_sizes,
        host_shared=True,
        async_dependencies=[token],
    )

    assert isinstance(full_alloc, AllocOp)
    assert full_alloc.result.type is dyn_type
    assert len(full_alloc.asyncDependencies) == 1
    assert full_alloc.asyncDependencies[0] is token
    assert len(full_alloc.dynamicSizes) == 3
    assert full_alloc.dynamicSizes == tuple(d.result for d in dynamic_sizes)
    assert full_alloc.asyncToken is None
    assert "hostShared" in full_alloc.attributes.keys()
    assert isinstance(full_alloc.hostShared, builtin.UnitAttr)


def test_all_reduce_operation():
    op = AllReduceOpAttr(AllReduceOpEnum.Add)

    assert op.data == AllReduceOpEnum.Add


def test_all_reduce():
    op = AllReduceOpAttr(AllReduceOpEnum.Add)

    init = arith.ConstantOp.from_int_and_width(0, builtin.IndexType())

    all_reduce = AllReduceOp.from_op(op, init)

    assert isinstance(all_reduce, AllReduceOp)
    assert all_reduce.op is op
    assert all_reduce.operand is init.result
    assert all_reduce.uniform is None
    assert all_reduce.result.type is all_reduce.operand.type

    body_block = Block(arg_types=[builtin.IndexType(), builtin.IndexType()])

    @Builder.implicit_region
    def body():
        sum = Operation.clone(arith.AddiOp(body_block.args[0], body_block.args[1]))
        YieldOp([sum])

    all_reduce_body = AllReduceOp.from_body(body, init)

    assert isinstance(all_reduce_body, AllReduceOp)
    assert all_reduce_body.op is None
    assert all_reduce_body.operand is init.result
    assert all_reduce_body.uniform is None
    assert all_reduce_body.result.type is all_reduce_body.operand.type


def test_barrier():
    barrier = BarrierOp()

    assert isinstance(barrier, BarrierOp)


def test_block_dim():
    dim = DimensionAttr(DimensionEnum.X)

    block_dim = BlockDimOp(dim)

    assert isinstance(block_dim, BlockDimOp)
    assert block_dim.dimension is dim


def test_block_id():
    dim = DimensionAttr(DimensionEnum.X)

    block_id = BlockIdOp(dim)

    assert isinstance(block_id, BlockIdOp)
    assert block_id.dimension is dim


def test_dealloc():
    memref_type = memref.MemRefType(builtin.f32, [10, 10, 10])
    alloc = AllocOp(memref_type, is_async=True)

    assert alloc.asyncToken is not None  # For pyright

    dealloc = DeallocOp(
        buffer=alloc.result, async_dependencies=[alloc.asyncToken], is_async=True
    )

    assert dealloc.asyncToken is not None
    assert isinstance(dealloc.asyncToken.type, AsyncTokenType)
    assert dealloc.buffer is alloc.result
    assert dealloc.asyncDependencies == tuple([alloc.asyncToken])

    alloc2 = AllocOp(memref_type, is_async=True)
    sync_dealloc = DeallocOp(buffer=alloc2.result)

    assert sync_dealloc.asyncToken is None
    assert sync_dealloc.buffer is alloc2.result
    assert len(sync_dealloc.asyncDependencies) == 0


def test_gpu_module():
    name = builtin.SymbolRefAttr("gpu")

    gpu_module = ModuleOp(name, Region(Block()))

    assert isinstance(gpu_module, ModuleOp)
    assert list(gpu_module.body.ops) == []
    assert gpu_module.sym_name is name


def test_global_id():
    dim = DimensionAttr(DimensionEnum.X)

    global_id = GlobalIdOp(dim)

    assert isinstance(global_id, GlobalIdOp)
    assert global_id.dimension is dim


def test_grid_dim():
    dim = DimensionAttr(DimensionEnum.X)

    grid_dim = GridDimOp(dim)

    assert isinstance(grid_dim, GridDimOp)
    assert grid_dim.dimension is dim


def test_host_register():
    memref_type = memref.MemRefType(builtin.i32, [-1])
    unranked = memref.AllocaOp.get(memref_type, 0)

    register = HostRegisterOp(unranked)

    assert isinstance(register, HostRegisterOp)
    assert register.value is unranked.results[0]


def test_host_unregister():
    memref_type = memref.MemRefType(builtin.i32, [-1])
    unranked = memref.AllocaOp.get(memref_type, 0)

    unregister = HostUnregisterOp(unranked)

    assert isinstance(unregister, HostUnregisterOp)
    assert unregister.value is unranked.results[0]


def test_lane_id():
    lane_id = LaneIdOp()

    assert isinstance(lane_id, LaneIdOp)


def test_func():
    kernel = "mygpufunc"
    inputs = [builtin.IndexType()]
    known_block_size = (32, 8, 4)
    known_grid_size = (4, 16, 32)

    body = Region(Block([ReturnOp([])]))

    func = FuncOp(kernel, (inputs, []), body, True, known_block_size, known_grid_size)

    assert isinstance(func, FuncOp)
    assert func.kernel == builtin.UnitAttr()
    assert func.sym_name == builtin.StringAttr(kernel)
    assert func.known_block_size is not None
    assert func.known_block_size.get_values() == known_block_size
    assert func.known_grid_size is not None
    assert func.known_grid_size.get_values() == known_grid_size

    assert func.function_type == builtin.FunctionType.from_lists(inputs, [])
    assert func.body is body


def test_launch():
    body = Region([])
    ten = arith.ConstantOp.from_int_and_width(10, builtin.IndexType())
    gridSize: list[Operation | SSAValue] = [ten, ten, ten]
    blockSize: list[Operation | SSAValue] = [ten, ten, ten]
    launch = LaunchOp(body, gridSize, blockSize)

    assert isinstance(launch, LaunchOp)
    assert launch.body is body
    assert launch.gridSizeX is ten.result
    assert launch.gridSizeY is ten.result
    assert launch.gridSizeZ is ten.result
    assert launch.blockSizeX is ten.result
    assert launch.blockSizeY is ten.result
    assert launch.blockSizeZ is ten.result
    assert launch.clusterSizeX is None
    assert launch.clusterSizeY is None
    assert launch.clusterSizeZ is None
    assert launch.asyncToken is None
    assert launch.asyncDependencies == tuple()
    assert launch.dynamicSharedMemorySize is None

    body2 = Region()

    nd_launch = LaunchOp(body2, gridSize, blockSize, [], True, [], ten)

    assert isinstance(launch, LaunchOp)
    assert nd_launch.body is body2
    assert nd_launch.gridSizeX is ten.result
    assert nd_launch.gridSizeY is ten.result
    assert nd_launch.gridSizeZ is ten.result
    assert nd_launch.blockSizeX is ten.result
    assert nd_launch.blockSizeY is ten.result
    assert nd_launch.blockSizeZ is ten.result
    assert nd_launch.clusterSizeX is None
    assert nd_launch.clusterSizeY is None
    assert nd_launch.clusterSizeZ is None
    assert nd_launch.asyncToken is not None
    assert nd_launch.asyncToken.type == AsyncTokenType()
    assert nd_launch.asyncDependencies == tuple()
    assert nd_launch.dynamicSharedMemorySize is ten.result


def test_launchfunc():
    kernel = builtin.SymbolRefAttr("root", ["gpu", "kernel"])
    args = [arith.ConstantOp.from_int_and_width(10, builtin.IndexType())]
    ten = arith.ConstantOp.from_int_and_width(10, builtin.IndexType())
    gridSize: list[Operation | SSAValue] = [ten, ten, ten]
    blockSize: list[Operation | SSAValue] = [ten, ten, ten]
    launch = LaunchFuncOp(kernel, gridSize, blockSize)

    assert isinstance(launch, LaunchFuncOp)
    assert launch.kernel is kernel
    assert launch.gridSizeX is ten.result
    assert launch.gridSizeY is ten.result
    assert launch.gridSizeZ is ten.result
    assert launch.blockSizeX is ten.result
    assert launch.blockSizeY is ten.result
    assert launch.blockSizeZ is ten.result
    assert launch.asyncToken is None
    assert launch.asyncDependencies == ()
    assert launch.dynamicSharedMemorySize is None
    assert launch.kernelOperands == ()

    kernel = builtin.SymbolRefAttr("root", ["gpu", "kernel"])

    launch = LaunchFuncOp(
        kernel,
        gridSize,
        blockSize,
        None,
        args,
        True,
        [],
        ten,
    )

    assert isinstance(launch, LaunchFuncOp)
    assert launch.kernel is kernel
    assert launch.gridSizeX is ten.result
    assert launch.gridSizeY is ten.result
    assert launch.gridSizeZ is ten.result
    assert launch.blockSizeX is ten.result
    assert launch.blockSizeY is ten.result
    assert launch.blockSizeZ is ten.result
    assert launch.asyncToken is not None
    assert launch.asyncToken.type == AsyncTokenType()
    assert launch.asyncDependencies == tuple()
    assert launch.dynamicSharedMemorySize is ten.result
    assert tuple(a.owner for a in launch.kernelOperands) == tuple(args)


def test_memcpy():
    memref_type = memref.MemRefType(builtin.f32, [10, 10, 10])
    host_alloc = memref.AllocOp.get(builtin.f32, 0, [10, 10, 10])
    alloc = AllocOp(memref_type, is_async=True)

    assert alloc.asyncToken is not None  # for Pyright

    memcpy = MemcpyOp(host_alloc, alloc.result, [alloc.asyncToken])

    assert isinstance(memcpy, MemcpyOp)
    assert memcpy.src is host_alloc.memref
    assert memcpy.dst is alloc.result
    assert memcpy.asyncDependencies == tuple([alloc.asyncToken])
    assert memcpy.asyncToken is None

    memcpy2 = MemcpyOp(
        alloc.result, host_alloc.memref, [alloc.asyncToken], is_async=True
    )

    assert isinstance(memcpy2, MemcpyOp)
    assert memcpy2.src is alloc.result
    assert memcpy2.dst is host_alloc.memref
    assert memcpy2.asyncDependencies == tuple([alloc.asyncToken])
    assert memcpy2.asyncToken is not None
    assert isinstance(memcpy2.asyncToken.type, AsyncTokenType)


def test_num_subgroups():
    num_subgroups = NumSubgroupsOp()

    assert isinstance(num_subgroups, NumSubgroupsOp)


def test_set_default_device():
    devIndex = arith.ConstantOp.from_int_and_width(0, builtin.i32)

    set_default_device = SetDefaultDeviceOp(devIndex)

    assert isinstance(set_default_device, SetDefaultDeviceOp)
    assert set_default_device.devIndex is devIndex.result


def test_subgroup_id():
    subgroup_id = SubgroupIdOp()

    assert isinstance(subgroup_id, SubgroupIdOp)


def test_subgroup_size():
    subgroup_size = SubgroupSizeOp()

    assert isinstance(subgroup_size, SubgroupSizeOp)


def test_thread_id():
    dim = DimensionAttr(DimensionEnum.X)

    thread_id = ThreadIdOp(dim)

    assert isinstance(thread_id, ThreadIdOp)
    assert thread_id.dimension is dim


def test_terminator():
    terminator = TerminatorOp()

    assert isinstance(terminator, TerminatorOp)


def test_wait():
    waitOp = WaitOp()

    assert isinstance(waitOp, WaitOp)
    assert waitOp.asyncToken is not None
    assert isinstance(waitOp.asyncToken.type, AsyncTokenType)

    waitOp1 = WaitOp()

    waitOpWithDep = WaitOp([waitOp, waitOp1])
    assert waitOpWithDep.asyncToken is not None
    assert waitOpWithDep.asyncDependencies[0] is waitOp.asyncToken
    assert waitOpWithDep.asyncDependencies[1] is waitOp1.asyncToken


def test_yield():
    operands: list[SSAValue | Operation] = [
        o
        for o in [
            arith.ConstantOp.from_int_and_width(42, builtin.i32),
            arith.ConstantOp.from_int_and_width(19, builtin.IndexType()),
            arith.ConstantOp.from_int_and_width(84, builtin.i64),
        ]
    ]
    yield_op = YieldOp(operands)

    assert isinstance(yield_op, YieldOp)
    assert list(yield_op.operands) == [SSAValue.get(o) for o in operands]
