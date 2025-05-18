// RUN: XDSL_ROUNDTRIP

"builtin.module"() ({

    // CHECK: "emitc.include"() <{include = "test.h"}> : () -> ()
    "emitc.include"() <{include = "test.h"}> : () -> ()

    "func.func"() <{function_type = (i32, !emitc.opaque<"int32_t">) -> (), sym_name = "f"}> ({
    ^bb0(%arg0: i32, %arg1: !emitc.opaque<"int32_t">):
        %0 = "emitc.call_opaque"() <{callee = "blah"}> : () -> i64
        "emitc.call_opaque"(%0) <{args = [0 : index, dense<[0, 1]> : tensor<2xi32>, 0 : index], callee = "foo"}> : (i64) -> ()
        "func.return"() : () -> ()
    }) : () -> ()

    "emitc.declare_func"() <{sym_name = @func}> : () -> ()

    "emitc.func"() <{function_type = (i32) -> (), sym_name = "func"}> ({
    ^bb0(%arg0: i32):
    "emitc.call_opaque"(%arg0) <{callee = "foo"}> : (i32) -> ()
    "emitc.return"() : () -> ()
    }) : () -> ()

    "emitc.func"() <{function_type = () -> i32, specifiers = ["static", "inline"], sym_name = "return_i32"}> ({
        %0 = "emitc.call_opaque"() <{callee = "foo"}> : () -> i32
        "emitc.return"(%0) : (i32) -> ()
    }) : () -> ()

    "func.func"() <{function_type = (!emitc.ptr<f32>, i32, !emitc.opaque<"unsigned int">) -> (), sym_name = "add_pointer"}> ({
    ^bb0(%arg0: !emitc.ptr<f32>, %arg1: i32, %arg2: !emitc.opaque<"unsigned int">):
        %0 = "emitc.add"(%arg0, %arg1) : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
        %1 = "emitc.add"(%arg0, %arg2) : (!emitc.ptr<f32>, !emitc.opaque<"unsigned int">) -> !emitc.ptr<f32>
        "func.return"() : () -> ()
    }) : () -> ()

    "func.func"() <{function_type = (!emitc.array<2x3xf32>, !emitc.ptr<i32>, !emitc.opaque<"std::map<char, int>">, index, i32, !emitc.opaque<"char">) -> (), sym_name = "test_subscript"}> ({
    ^bb0(%arg0: !emitc.array<2x3xf32>, %arg1: !emitc.ptr<i32>, %arg2: !emitc.opaque<"std::map<char, int>">, %arg3: index, %arg4: i32, %arg5: !emitc.opaque<"char">):
        %0 = "emitc.subscript"(%arg0, %arg3, %arg4) : (!emitc.array<2x3xf32>, index, i32) -> !emitc.lvalue<f32>
        %1 = "emitc.subscript"(%arg1, %arg3) : (!emitc.ptr<i32>, index) -> !emitc.lvalue<i32>
        %2 = "emitc.subscript"(%arg2, %arg5) : (!emitc.opaque<"std::map<char, int>">, !emitc.opaque<"char">) -> !emitc.lvalue<!emitc.opaque<"int">>
        "func.return"() : () -> ()
    }) : () -> ()
}) : () -> ()
