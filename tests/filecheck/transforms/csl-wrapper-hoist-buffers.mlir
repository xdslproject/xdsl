// RUN: xdsl-opt %s -p "csl-wrapper-hoist-buffers" | filecheck %s

builtin.module {
  "csl_wrapper.module"() <{"width" = 1022 : i16, "height" = 510 : i16, "params" = [], "program_name" = "gauss_seidel_func", "target" = "wse2"}> ({
  ^bb0(%0 : i16, %1 : i16, %2 : i16, %3 : i16):
    "csl_wrapper.yield"() <{"fields" = []}> : () -> ()
  }, {
  ^bb1(%4 : i16, %5 : i16):
    %arg0 = memref.alloc() : memref<512xf32>
    %6 = "csl.addressof"(%arg0) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
    "csl.export"(%6) <{"var_name" = "arg0", "type" = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
    "csl.export"() <{"var_name" = @gauss_seidel_func, "type" = () -> ()}> : () -> ()
    csl.func @gauss_seidel_func() {
      %7 = memref.alloc() {"alignment" = 64 : i64} : memref<510xf32>
      "test.op"(%7) : (memref<510xf32>) -> ()
      csl.return
    }
    "csl_wrapper.yield"() <{"fields" = []}> : () -> ()
  }) : () -> ()
}

// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   "csl_wrapper.module"() <{width = 1022 : i16, height = 510 : i16, params = [], program_name = "gauss_seidel_func", target = "wse2"}> ({
// CHECK-NEXT:   ^bb0(%0 : i16, %1 : i16, %2 : i16, %3 : i16):
// CHECK-NEXT:     "csl_wrapper.yield"() <{fields = []}> : () -> ()
// CHECK-NEXT:   }, {
// CHECK-NEXT:   ^bb1(%4 : i16, %5 : i16):
// CHECK-NEXT:     %6 = memref.alloc() {alignment = 64 : i64} : memref<510xf32>
// CHECK-NEXT:     %arg0 = memref.alloc() : memref<512xf32>
// CHECK-NEXT:     %arg0_ptr = "csl.addressof"(%arg0) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:     "csl.export"(%arg0_ptr) <{var_name = "arg0", type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:     "csl.export"() <{var_name = @gauss_seidel_func, type = () -> ()}> : () -> ()
// CHECK-NEXT:     csl.func @gauss_seidel_func() {
// CHECK-NEXT:       "test.op"(%6) : (memref<510xf32>) -> ()
// CHECK-NEXT:       csl.return
// CHECK-NEXT:     }
// CHECK-NEXT:     "csl_wrapper.yield"() <{fields = []}> : () -> ()
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }
