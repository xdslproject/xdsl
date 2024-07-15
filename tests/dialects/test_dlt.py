from xdsl.dialects import arith, builtin
from xdsl.dialects.experimental import dlt
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriteWalker
from xdsl.transforms.dead_code_elimination import RemoveUnusedOperations
from xdsl.transforms.experimental.dlt import lower_dlt_to_


def test_typetype_has_selectable_type():
    t1 = dlt.TypeType([
        dlt.ElementAttr([("a","1"), ("b","1")], [], builtin.f32),
        dlt.ElementAttr([("a", "1"), ("b","2")], [], builtin.f32),
        dlt.ElementAttr([("a", "1"), ("b","3")], [], builtin.f32)
    ])
    t2 =  dlt.TypeType([
        dlt.ElementAttr([("b","1")], [], builtin.f32),
        dlt.ElementAttr([("b","2")], [], builtin.f32),
    ])
    selections = t1.has_selectable_type(t2)
    assert selections == {(dlt.SetAttr([dlt.MemberAttr("a", "1")]), dlt.SetAttr([]))}


def test_make_ConstantLayout():
    layout = dlt.ConstantLayoutAttr(builtin.IntegerAttr(25, builtin.i32))
    layout = dlt.DenseLayoutAttr(layout, dlt.DimensionAttr("i", 10))
    assert layout.contents_type == dlt.TypeType([dlt.ElementAttr([],[("i",10)], builtin.i32)])

    ptr = dlt.PtrType(layout.contents_type, layout, base=True)
    alloc_op = dlt.AllocOp(ptr, {})
    three = arith.Constant(builtin.IntegerAttr(3, builtin.IndexType()))
    select_op = dlt.SelectOp(alloc_op.res, [], [dlt.DimensionAttr("i", 10)], [three])
    get_op = dlt.GetOp(select_op.res, builtin.i32)
    scope = dlt.LayoutScopeOp([], [alloc_op, three, select_op, get_op])
    module = builtin.ModuleOp([scope])
    print(module)

    module.verify()

    dlt_to_llvm_applier = PatternRewriteWalker(GreedyRewritePatternApplier(
        [
         # lower_dlt_to_.DLTSelectRewriter(),
         # lower_dlt_to_.DLTGetRewriter(),
         # lower_dlt_to_.DLTSetRewriter(),
         lower_dlt_to_.DLTAllocRewriter(),
         lower_dlt_to_.DLTDeallocRewriter(),
         # lower_dlt_to_.DLTIterateRewriter(),
         # lower_dlt_to_.DLTCopyRewriter(),
         # lower_dlt_to_.DLTExtractExtentRewriter(),
         ]),
        walk_regions_first=False)
    dlt_to_llvm_applier.rewrite_module(module)

    print(module)


def test_make_ArithDropLayout():
    layout = dlt.PrimitiveLayoutAttr(builtin.f32)
    layout = dlt.ArithDropLayoutAttr(layout, dlt.DimensionAttr("i", 10))
    assert layout.contents_type == dlt.TypeType([dlt.ElementAttr([],[("i",10)], builtin.f32)])

    ptr = dlt.PtrType(layout.contents_type, layout)
    alloc_op = dlt.AllocOp(ptr, {})
    three = arith.Constant(builtin.IntegerAttr(3, builtin.IndexType()))
    select_op = dlt.SelectOp(alloc_op.res, [], [dlt.DimensionAttr("i", 10)], [three])
    get_op = dlt.GetOp(select_op.res, builtin.f32)
    set_op = dlt.SetOp(select_op.res, get_op.res)
    module = builtin.ModuleOp([alloc_op, three, select_op, get_op, set_op])
    print(module)
    dlt_to_llvm_applier = PatternRewriteWalker(GreedyRewritePatternApplier(
        [RemoveUnusedOperations(),
         lower_dlt_to_.DLTSelectRewriter(),
         lower_dlt_to_.DLTGetRewriter(),
         lower_dlt_to_.DLTSetRewriter(),
         lower_dlt_to_.DLTAllocRewriter(),
         lower_dlt_to_.DLTDeallocRewriter(),
         lower_dlt_to_.DLTIterateRewriter(),
         lower_dlt_to_.DLTCopyRewriter(),
         lower_dlt_to_.DLTExtractExtentRewriter(),
         ]),
        walk_regions_first=False)
    dlt_to_llvm_applier.rewrite_module(module)

    print(module)

def test_make_UnpackedCOOLayout():

    direct = dlt.PrimitiveLayoutAttr(dlt.IndexRangeType())
    direct = dlt.DenseLayoutAttr(direct, i := dlt.DimensionAttr("i", dlt.StaticExtentAttr(10)))
    sparse = dlt.PrimitiveLayoutAttr(builtin.i32)
    sparse = dlt.DenseLayoutAttr(sparse, k := dlt.DimensionAttr("k", dlt.StaticExtentAttr(3)))
    sparse = dlt.UnpackedCOOLayoutAttr(sparse, [j := dlt.DimensionAttr("j", dlt.StaticExtentAttr(1000))])
    layout = dlt.IndexingLayoutAttr(direct, sparse)
    assert layout.contents_type == dlt.TypeType([
        dlt.ElementAttr([],[("i",10), ("k",3), ("j",1000)], builtin.i32),
    ])

    init_layout = dlt.DenseLayoutAttr(dlt.ConstantLayoutAttr(builtin.IntegerAttr(9999, builtin.i32)), j)
    init_layout = dlt.DenseLayoutAttr(init_layout, k)
    init_layout = dlt.DenseLayoutAttr(init_layout, i)
    init_values_op = dlt.AllocOp(dlt.PtrType(init_layout.contents_type, init_layout, base=True), {})

    ptr = dlt.PtrType(layout.contents_type, layout, base=True)
    alloc_op = dlt.AllocOp(ptr, {}, [init_values_op.res])

    i_get = arith.Constant(builtin.IntegerAttr(5, builtin.IndexType()))
    j_get = arith.Constant(builtin.IntegerAttr(11, builtin.IndexType()))
    k_get = arith.Constant(builtin.IntegerAttr(1, builtin.IndexType()))
    sel_op = dlt.SelectOp(alloc_op.res, [], [i, j, k], [i_get, j_get, k_get])
    get_op = dlt.GetOp(sel_op.res, builtin.i32)
    module = builtin.ModuleOp([dlt.LayoutScopeOp([],[init_values_op, alloc_op, i_get, j_get, k_get, sel_op, get_op])])

    dlt_to_llvm_applier = PatternRewriteWalker(GreedyRewritePatternApplier(
        [RemoveUnusedOperations(),
         lower_dlt_to_.DLTSelectRewriter(),
         lower_dlt_to_.DLTGetRewriter(),
         lower_dlt_to_.DLTSetRewriter(),
         lower_dlt_to_.DLTAllocRewriter(),
         lower_dlt_to_.DLTDeallocRewriter(),
         lower_dlt_to_.DLTIterateRewriter(),
         lower_dlt_to_.DLTCopyRewriter(),
         lower_dlt_to_.DLTExtractExtentRewriter(),
         ]),
        walk_regions_first=False)
    dlt_to_llvm_applier.rewrite_module(module)

    print(module)
    module.verify()



