from xdsl.dialects.memref import *
from xdsl.dialects.std import *
from xdsl.printer import Printer


def get_example_memref_program(ctx: MLContext, std: Std,
                               memref: MemRef) -> Operation:
    def test() -> List[Operation]:
        # yapf: disable
        return [

                Global.get("g", IndexType(), IntegerAttr.from_index_int_value(0),
                    "public"),

                global_ref := GetGlobal.get("g", MemRefType.from_params(IndexType())),

                index_0 := std.constant(0, IndexType()),

                ref := Alloca.get(IndexType(), 0),
                val := std.constant(42, IndexType()),
                Store.get(val, ref, [index_0]),
                val2 := Load.get(ref, [index_0]),

                arr := Alloc.get(IndexType(), 0, [10, 2]),
                Store.get(val, arr, [val, val2]),

                Dealloc.get(ref),
                Dealloc.get(arr)

        ]
    # yapf: enable

    f = FuncOp.from_callable("test", [], [], test)
    return f


def test_memref():
    ctx = MLContext()
    std = Std(ctx)
    memref = MemRef(ctx)

    f = get_example_memref_program(ctx, std, memref)

    f.verify()
    printer = Printer()
    printer.print_op(f)
    print()

    print("Done")


if __name__ == "__main__":
    test_memref()
