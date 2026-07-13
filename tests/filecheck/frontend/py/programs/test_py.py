# RUN: uv run ../../../../../xdsl/frontend/pydialect/main.py -T %s | filecheck %s


def foo(x: float, y: float) -> float:
    return 0.5 * x + foo(1, 2)


# CHECK:      builtin.module {
# CHECK-NEXT:   "py.func"() <{sym_name = "foo", function_type = (!py.type<float>, !py.type<float>) -> !py.type<float>}> ({
# CHECK-NEXT:   ^bb0(%x: !py.type<float>, %y: !py.type<float>):
# CHECK-NEXT:     %0 = "py.constant"() <{value = #py.const<0.5>}> : () -> !py.type<float>
# CHECK-NEXT:     %1 = "py.call"(%0, %x) <{callee = "__mul__"}> : (!py.type<float>, !py.type<float>) -> !py.type<float>
# CHECK-NEXT:     %2 = "py.constant"() <{value = #py.const<1>}> : () -> !py.type<int>
# CHECK-NEXT:     %3 = "py.constant"() <{value = #py.const<2>}> : () -> !py.type<int>
# CHECK-NEXT:     %4 = "py.call"(%2, %3) <{callee = "foo"}> : (!py.type<int>, !py.type<int>) -> !py.type<Unknown>
# CHECK-NEXT:     %5 = "py.call"(%1, %4) <{callee = "__add__"}> : (!py.type<float>, !py.type<Unknown>) -> !py.type<Unknown>
# CHECK-NEXT:     "py.return"(%5) : (!py.type<Unknown>) -> ()
# CHECK-NEXT:   }) : () -> ()
# CHECK-NEXT: }
