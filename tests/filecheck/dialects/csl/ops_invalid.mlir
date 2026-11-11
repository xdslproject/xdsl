// RUN: xdsl-opt %s --verify-diagnostics --split-input-file | filecheck %s

%ptr = "test.op"() : () -> !csl.ptr<i32, #csl<ptr_kind single>, #csl<ptr_const var>>
%0 = csl.intptrcast %ptr : !csl.ptr<i32, #csl<ptr_kind single>, #csl<ptr_const var>> to !csl.ptr<i16, #csl<ptr_kind single>, #csl<ptr_const var>>

// CHECK: Expected exactly one of input and result to be a pointer, got !csl.ptr<i32, #csl<ptr_kind single>, #csl<ptr_const var>> and !csl.ptr<i16, #csl<ptr_kind single>, #csl<ptr_const var>>

// -----

%int = "test.op"() : () -> i32
%0 = csl.intptrcast %int : i32 to index

// CHECK: Expected exactly one of input and result to be a pointer, got i32 and index

// -----

%f = "test.op"() : () -> f32
%0 = csl.intptrcast %f : f32 to !csl.ptr<i32, #csl<ptr_kind single>, #csl<ptr_const var>>

// CHECK: Unexpected attribute f32
