from xdsl.dialects.memref import *
from xdsl.dialects.std import *
from xdsl.printer import Printer
from xdsl.util import block, func


def get_example_memref_program(ctx: MLContext, std: Std,
                               memref: MemRef) -> Operation:
    def test() -> List[Operation]:
        # yapf: disable
        return [

                memref.global_("g", IndexType(), IntegerAttr.from_index_int_value(0),
                    "public"),

                global_ref := memref.get_global("g", MemRefType.from_params(IndexType())),

                index_0 := std.constant(0, IndexType()),

                ref := memref.alloca(0, IndexType()),
                val := std.constant(42, IndexType()),
                memref.store(val, ref, [index_0]),
                val2 := memref.load(ref, [index_0]),

                arr := memref.alloc(0, IndexType(), [10,2]),
                memref.store(val, arr, [val, val2]),

                memref.dealloc(ref),
                memref.dealloc(arr)

        ]
    # yapf: enable

    f = func("test", [], [], test)
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
