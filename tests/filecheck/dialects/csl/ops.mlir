// RUN: XDSL_ROUNDTRIP

csl.func @initialize() {

    %lb, %ub = "test.op"() : () -> (i16, i16)

    %thing = "csl.import_module"() <{module = "<thing>"}> : () -> !csl.imported_module

    "csl.member_call"(%thing, %lb, %ub) <{field = "some_func", operandSegmentSizes = array<i32: 1, 2>}> : (!csl.imported_module, i16, i16) -> ()

    %res = "csl.member_call"(%thing, %lb, %ub) <{field = "some_func", operandSegmentSizes = array<i32: 1, 2>}> : (!csl.imported_module, i16, i16) -> (i32)

    %11 = "csl.member_access"(%thing) <{field = "some_field"}> : (!csl.imported_module) -> !csl.comptime_struct

    %single_const = "test.op"() : () -> !csl.ptr<i32, #csl<ptr_kind single>, #csl<ptr_const const>>
    %single_var = "test.op"() : () -> !csl.ptr<i32, #csl<ptr_kind single>, #csl<ptr_const var>>
    %many_const = "test.op"() : () -> !csl.ptr<i32, #csl<ptr_kind many>, #csl<ptr_const const>>
    %many_var = "test.op"() : () -> !csl.ptr<i32, #csl<ptr_kind many>, #csl<ptr_const var>>

    %col = "test.op"() : () -> !csl.color

  csl.return
}


// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   csl.func @initialize() {
// CHECK-NEXT:     %lb, %ub = "test.op"() : () -> (i16, i16)
// CHECK-NEXT:     %thing = "csl.import_module"() <{"module" = "<thing>"}> : () -> !csl.imported_module
// CHECK-NEXT:     "csl.member_call"(%thing, %lb, %ub) <{"field" = "some_func", "operandSegmentSizes" = array<i32: 1, 2>}> : (!csl.imported_module, i16, i16) -> ()
// CHECK-NEXT:     %res = "csl.member_call"(%thing, %lb, %ub) <{"field" = "some_func", "operandSegmentSizes" = array<i32: 1, 2>}> : (!csl.imported_module, i16, i16) -> i32
// CHECK-NEXT:     %0 = "csl.member_access"(%thing) <{"field" = "some_field"}> : (!csl.imported_module) -> !csl.comptime_struct
// CHECK-NEXT:     %single_const = "test.op"() : () -> !csl.ptr<i32, #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:     %single_var = "test.op"() : () -> !csl.ptr<i32, #csl<ptr_kind single>, #csl<ptr_const var>>
// CHECK-NEXT:     %many_const = "test.op"() : () -> !csl.ptr<i32, #csl<ptr_kind many>, #csl<ptr_const const>>
// CHECK-NEXT:     %many_var = "test.op"() : () -> !csl.ptr<i32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:     %col = "test.op"() : () -> !csl.color
// CHECK-NEXT:     csl.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
