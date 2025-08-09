// RUN: xdsl-opt %s -p "csl-stencil-set-global-coeffs" --split-input-file | filecheck %s


"csl_wrapper.module"() <{width = 1022 : i16, height = 510 : i16, params = [#csl_wrapper.param<"pattern" default=2 : i16>], program_name = "same_coeffs_but_reversed", target = "wse2"}> ({
^bb0(%0 : i16, %1 : i16, %2 : i16, %3 : i16, %4 : i16):
  "csl_wrapper.yield"() <{fields = []}> : () -> ()
}, {
^bb1(%5 : i16, %6 : i16, %7 : i16):
  %8 = "csl_wrapper.import"() <{module = "<memcpy/memcpy>", fields = []}> : () -> !csl.imported_module
  %9 = "csl_wrapper.import"() <{module = "stencil_comms.csl", fields = []}> : () -> !csl.imported_module
  %arg0 = memref.alloc() : memref<512xf32>
  %arg1 = memref.alloc() : memref<512xf32>
  %10 = "csl.addressof"(%arg0) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
  %11 = "csl.addressof"(%arg1) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
  "csl.export"(%10) <{var_name = "arg0", type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
  "csl.export"(%11) <{var_name = "arg1", type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
  "csl.export"() <{var_name = @gauss_seidel_func, type = () -> ()}> : () -> ()
  %12 = memref.alloc() {alignment = 64 : i64} : memref<510xf32>
  csl.func @same_coeffs_but_reversed() {
    %this_comes_before_inserted_api_call = "test.op"() : () -> i1
    csl_stencil.apply(%arg0 : memref<512xf32>, %12 : memref<510xf32>) outs (%arg1 : memref<512xf32>) <{bounds = #stencil.bounds<[0, 0], [1, 1]>, num_chunks = 2 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>, swaps = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], topo = #dmp.topo<1022x510>, coeffs = [#csl_stencil.coeff<#stencil.index<[1, 0]>, 0.234567806 : f32>, #csl_stencil.coeff<#stencil.index<[0, -1]>, 3.141500e-01 : f32>]}> ({
    ^bb2(%arg5 : memref<4x255xf32>, %arg6 : index, %arg7 : memref<510xf32>):
      csl_stencil.yield %arg7 : memref<510xf32>
    }, {
    ^bb3(%arg5_1 : memref<512xf32>, %arg6_1 : memref<510xf32>):
      %13 = arith.constant 1.666600e-01 : f32
      csl_stencil.yield %arg6_1 : memref<510xf32>
    }) to <[0, 0], [1, 1]>
    csl_stencil.apply(%arg1 : memref<512xf32>, %12 : memref<510xf32>) outs (%arg0 : memref<512xf32>) <{bounds = #stencil.bounds<[0, 0], [1, 1]>, num_chunks = 2 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>, swaps = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], topo = #dmp.topo<1022x510>, coeffs = [#csl_stencil.coeff<#stencil.index<[0, -1]>, 3.141500e-01 : f32>, #csl_stencil.coeff<#stencil.index<[1, 0]>, 0.234567806 : f32>]}> ({
    ^bb2(%arg5 : memref<4x255xf32>, %arg6 : index, %arg7 : memref<510xf32>):
      csl_stencil.yield %arg7 : memref<510xf32>
    }, {
    ^bb3(%arg5_1 : memref<512xf32>, %arg6_1 : memref<510xf32>):
      %13 = arith.constant 0.123456702 : f32
      csl_stencil.yield %arg6_1 : memref<510xf32>
    }) to <[0, 0], [1, 1]>
    "csl.member_call"(%8) <{field = "unblock_cmd_stream"}> : (!csl.imported_module) -> ()
    csl.return
  }
  "csl_wrapper.yield"() <{fields = []}> : () -> ()
}) : () -> ()

// CHECK:       "csl_wrapper.module"() <{width = 1022 : i16, height = 510 : i16, params = [#csl_wrapper.param<"pattern" default=2 : i16>], program_name = "same_coeffs_but_reversed", target = "wse2"}> ({
// CHECK-NEXT:  ^bb0(%0 : i16, %1 : i16, %2 : i16, %3 : i16, %4 : i16):
// CHECK-NEXT:    "csl_wrapper.yield"() <{fields = []}> : () -> ()
// CHECK-NEXT:  }, {
// CHECK-NEXT:  ^bb1(%5 : i16, %6 : i16, %7 : i16):
// CHECK-NEXT:    %8 = "csl_wrapper.import"() <{module = "<memcpy/memcpy>", fields = []}> : () -> !csl.imported_module
// CHECK-NEXT:    %9 = "csl_wrapper.import"() <{module = "stencil_comms.csl", fields = []}> : () -> !csl.imported_module
// CHECK-NEXT:    %arg0 = memref.alloc() : memref<512xf32>
// CHECK-NEXT:    %arg1 = memref.alloc() : memref<512xf32>
// CHECK-NEXT:    %10 = "csl.addressof"(%arg0) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:    %11 = "csl.addressof"(%arg1) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:    "csl.export"(%10) <{var_name = "arg0", type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:    "csl.export"(%11) <{var_name = "arg1", type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:    "csl.export"() <{var_name = @gauss_seidel_func, type = () -> ()}> : () -> ()
// CHECK-NEXT:    %12 = memref.alloc() {alignment = 64 : i64} : memref<510xf32>
// CHECK-NEXT:    csl.func @same_coeffs_but_reversed() {
// CHECK-NEXT:      %this_comes_before_inserted_api_call = "test.op"() : () -> i1
// CHECK-NEXT:      %north = arith.constant dense<[0.000000e+00, 3.141500e-01]> : memref<2xf32>
// CHECK-NEXT:      %south = arith.constant dense<[0.000000e+00, 1.000000e+00]> : memref<2xf32>
// CHECK-NEXT:      %east = arith.constant dense<[0.000000e+00, 1.000000e+00]> : memref<2xf32>
// CHECK-NEXT:      %west = arith.constant dense<[0.000000e+00, 0.234567806]> : memref<2xf32>
// CHECK-NEXT:      %13 = "csl.addressof"(%east) : (memref<2xf32>) -> !csl.ptr<memref<2xf32>, #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:      %14 = "csl.addressof"(%west) : (memref<2xf32>) -> !csl.ptr<memref<2xf32>, #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:      %15 = "csl.addressof"(%south) : (memref<2xf32>) -> !csl.ptr<memref<2xf32>, #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:      %16 = "csl.addressof"(%north) : (memref<2xf32>) -> !csl.ptr<memref<2xf32>, #csl<ptr_kind single>, #csl<ptr_const const>>
// CHECK-NEXT:      "csl.member_call"(%9, %13, %14, %15, %16) <{field = "setCoeffs"}> : (!csl.imported_module, !csl.ptr<memref<2xf32>, #csl<ptr_kind single>, #csl<ptr_const const>>, !csl.ptr<memref<2xf32>, #csl<ptr_kind single>, #csl<ptr_const const>>, !csl.ptr<memref<2xf32>, #csl<ptr_kind single>, #csl<ptr_const const>>, !csl.ptr<memref<2xf32>, #csl<ptr_kind single>, #csl<ptr_const const>>) -> ()
// CHECK-NEXT:      csl_stencil.apply(%arg0 : memref<512xf32>, %12 : memref<510xf32>) outs (%arg1 : memref<512xf32>) <{bounds = #stencil.bounds<[0, 0], [1, 1]>, num_chunks = 2 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>, swaps = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], topo = #dmp.topo<1022x510>}> ({
// CHECK-NEXT:      ^bb2(%arg5 : memref<4x255xf32>, %arg6 : index, %arg7 : memref<510xf32>):
// CHECK-NEXT:        csl_stencil.yield %arg7 : memref<510xf32>
// CHECK-NEXT:      }, {
// CHECK-NEXT:      ^bb3(%arg5_1 : memref<512xf32>, %arg6_1 : memref<510xf32>):
// CHECK-NEXT:        %17 = arith.constant 1.666600e-01 : f32
// CHECK-NEXT:        csl_stencil.yield %arg6_1 : memref<510xf32>
// CHECK-NEXT:      }) to <[0, 0], [1, 1]>
// CHECK-NEXT:      csl_stencil.apply(%arg1 : memref<512xf32>, %12 : memref<510xf32>) outs (%arg0 : memref<512xf32>) <{bounds = #stencil.bounds<[0, 0], [1, 1]>, num_chunks = 2 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>, swaps = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], topo = #dmp.topo<1022x510>}> ({
// CHECK-NEXT:      ^bb2(%arg5 : memref<4x255xf32>, %arg6 : index, %arg7 : memref<510xf32>):
// CHECK-NEXT:        csl_stencil.yield %arg7 : memref<510xf32>
// CHECK-NEXT:      }, {
// CHECK-NEXT:      ^bb3(%arg5_1 : memref<512xf32>, %arg6_1 : memref<510xf32>):
// CHECK-NEXT:        %17 = arith.constant 0.123456702 : f32
// CHECK-NEXT:        csl_stencil.yield %arg6_1 : memref<510xf32>
// CHECK-NEXT:      }) to <[0, 0], [1, 1]>
// CHECK-NEXT:      "csl.member_call"(%8) <{field = "unblock_cmd_stream"}> : (!csl.imported_module) -> ()
// CHECK-NEXT:      csl.return
// CHECK-NEXT:    }
// CHECK-NEXT:    "csl_wrapper.yield"() <{fields = []}> : () -> ()
// CHECK-NEXT:  }) : () -> ()

// -----

"csl_wrapper.module"() <{width = 1022 : i16, height = 510 : i16, params = [#csl_wrapper.param<"pattern" default=2 : i16>], program_name = "no_global_coeffs", target = "wse2"}> ({
^bb0(%0 : i16, %1 : i16, %2 : i16, %3 : i16, %4 : i16):
  "csl_wrapper.yield"() <{fields = []}> : () -> ()
}, {
^bb1(%5 : i16, %6 : i16, %7 : i16):
  %8 = "csl_wrapper.import"() <{module = "<memcpy/memcpy>", fields = []}> : () -> !csl.imported_module
  %9 = "csl_wrapper.import"() <{module = "stencil_comms.csl", fields = []}> : () -> !csl.imported_module
  %arg0 = memref.alloc() : memref<512xf32>
  %arg1 = memref.alloc() : memref<512xf32>
  %10 = "csl.addressof"(%arg0) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
  %11 = "csl.addressof"(%arg1) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
  "csl.export"(%10) <{var_name = "arg0", type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
  "csl.export"(%11) <{var_name = "arg1", type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
  "csl.export"() <{var_name = @gauss_seidel_func, type = () -> ()}> : () -> ()
  %12 = memref.alloc() {alignment = 64 : i64} : memref<510xf32>
  csl.func @no_global_coeffs() {
    %this_comes_before_inserted_api_call = "test.op"() : () -> i1
    csl_stencil.apply(%arg0 : memref<512xf32>, %12 : memref<510xf32>) outs (%arg1 : memref<512xf32>) <{bounds = #stencil.bounds<[0, 0], [1, 1]>, num_chunks = 2 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>, swaps = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], topo = #dmp.topo<1022x510>, coeffs = [#csl_stencil.coeff<#stencil.index<[1, 0]>, 123.4567890 : f32>, #csl_stencil.coeff<#stencil.index<[0, -1]>, 3.141500e-01 : f32>]}> ({
    ^bb2(%arg5 : memref<4x255xf32>, %arg6 : index, %arg7 : memref<510xf32>):
      csl_stencil.yield %arg7 : memref<510xf32>
    }, {
    ^bb3(%arg5_1 : memref<512xf32>, %arg6_1 : memref<510xf32>):
      %13 = arith.constant 1.666600e-01 : f32
      csl_stencil.yield %arg6_1 : memref<510xf32>
    }) to <[0, 0], [1, 1]>
    csl_stencil.apply(%arg1 : memref<512xf32>, %12 : memref<510xf32>) outs (%arg0 : memref<512xf32>) <{bounds = #stencil.bounds<[0, 0], [1, 1]>, num_chunks = 2 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>, swaps = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], topo = #dmp.topo<1022x510>, coeffs = [#csl_stencil.coeff<#stencil.index<[0, -1]>, 3.141500e-01 : f32>, #csl_stencil.coeff<#stencil.index<[1, 0]>, 0.234567806 : f32>]}> ({
    ^bb2(%arg5 : memref<4x255xf32>, %arg6 : index, %arg7 : memref<510xf32>):
      csl_stencil.yield %arg7 : memref<510xf32>
    }, {
    ^bb3(%arg5_1 : memref<512xf32>, %arg6_1 : memref<510xf32>):
      %13 = arith.constant 0.123456702 : f32
      csl_stencil.yield %arg6_1 : memref<510xf32>
    }) to <[0, 0], [1, 1]>
    "csl.member_call"(%8) <{field = "unblock_cmd_stream"}> : (!csl.imported_module) -> ()
    csl.return
  }
  "csl_wrapper.yield"() <{fields = []}> : () -> ()
}) : () -> ()

// CHECK:       "csl_wrapper.module"() <{width = 1022 : i16, height = 510 : i16, params = [#csl_wrapper.param<"pattern" default=2 : i16>], program_name = "no_global_coeffs", target = "wse2"}> ({
// CHECK-NEXT:  ^bb0(%0 : i16, %1 : i16, %2 : i16, %3 : i16, %4 : i16):
// CHECK-NEXT:    "csl_wrapper.yield"() <{fields = []}> : () -> ()
// CHECK-NEXT:  }, {
// CHECK-NEXT:  ^bb1(%5 : i16, %6 : i16, %7 : i16):
// CHECK-NEXT:    %8 = "csl_wrapper.import"() <{module = "<memcpy/memcpy>", fields = []}> : () -> !csl.imported_module
// CHECK-NEXT:    %9 = "csl_wrapper.import"() <{module = "stencil_comms.csl", fields = []}> : () -> !csl.imported_module
// CHECK-NEXT:    %arg0 = memref.alloc() : memref<512xf32>
// CHECK-NEXT:    %arg1 = memref.alloc() : memref<512xf32>
// CHECK-NEXT:    %10 = "csl.addressof"(%arg0) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:    %11 = "csl.addressof"(%arg1) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:    "csl.export"(%10) <{var_name = "arg0", type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:    "csl.export"(%11) <{var_name = "arg1", type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:    "csl.export"() <{var_name = @gauss_seidel_func, type = () -> ()}> : () -> ()
// CHECK-NEXT:    %12 = memref.alloc() {alignment = 64 : i64} : memref<510xf32>
// CHECK-NEXT:    csl.func @no_global_coeffs() {
// CHECK-NEXT:      %this_comes_before_inserted_api_call = "test.op"() : () -> i1
// CHECK-NEXT:      csl_stencil.apply(%arg0 : memref<512xf32>, %12 : memref<510xf32>) outs (%arg1 : memref<512xf32>) <{bounds = #stencil.bounds<[0, 0], [1, 1]>, num_chunks = 2 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>, swaps = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], topo = #dmp.topo<1022x510>, coeffs = [#csl_stencil.coeff<#stencil.index<[1, 0]>, 123.456787 : f32>, #csl_stencil.coeff<#stencil.index<[0, -1]>, 3.141500e-01 : f32>]}> ({
// CHECK-NEXT:      ^bb2(%arg5 : memref<4x255xf32>, %arg6 : index, %arg7 : memref<510xf32>):
// CHECK-NEXT:        csl_stencil.yield %arg7 : memref<510xf32>
// CHECK-NEXT:      }, {
// CHECK-NEXT:      ^bb3(%arg5_1 : memref<512xf32>, %arg6_1 : memref<510xf32>):
// CHECK-NEXT:        %13 = arith.constant 1.666600e-01 : f32
// CHECK-NEXT:        csl_stencil.yield %arg6_1 : memref<510xf32>
// CHECK-NEXT:      }) to <[0, 0], [1, 1]>
// CHECK-NEXT:      csl_stencil.apply(%arg1 : memref<512xf32>, %12 : memref<510xf32>) outs (%arg0 : memref<512xf32>) <{bounds = #stencil.bounds<[0, 0], [1, 1]>, num_chunks = 2 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>, swaps = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], topo = #dmp.topo<1022x510>, coeffs = [#csl_stencil.coeff<#stencil.index<[0, -1]>, 3.141500e-01 : f32>, #csl_stencil.coeff<#stencil.index<[1, 0]>, 0.234567806 : f32>]}> ({
// CHECK-NEXT:      ^bb2(%arg5 : memref<4x255xf32>, %arg6 : index, %arg7 : memref<510xf32>):
// CHECK-NEXT:        csl_stencil.yield %arg7 : memref<510xf32>
// CHECK-NEXT:      }, {
// CHECK-NEXT:      ^bb3(%arg5_1 : memref<512xf32>, %arg6_1 : memref<510xf32>):
// CHECK-NEXT:        %13 = arith.constant 0.123456702 : f32
// CHECK-NEXT:        csl_stencil.yield %arg6_1 : memref<510xf32>
// CHECK-NEXT:      }) to <[0, 0], [1, 1]>
// CHECK-NEXT:      "csl.member_call"(%8) <{field = "unblock_cmd_stream"}> : (!csl.imported_module) -> ()
// CHECK-NEXT:      csl.return
// CHECK-NEXT:    }
// CHECK-NEXT:    "csl_wrapper.yield"() <{fields = []}> : () -> ()
// CHECK-NEXT:  }) : () -> ()

// -----

"csl_wrapper.module"() <{width = 1022 : i16, height = 510 : i16, params = [#csl_wrapper.param<"pattern" default=2 : i16>], program_name = "no_coeffs", target = "wse2"}> ({
^bb0(%0 : i16, %1 : i16, %2 : i16, %3 : i16, %4 : i16):
  "csl_wrapper.yield"() <{fields = []}> : () -> ()
}, {
^bb1(%5 : i16, %6 : i16, %7 : i16):
  %8 = "csl_wrapper.import"() <{module = "<memcpy/memcpy>", fields = []}> : () -> !csl.imported_module
  %9 = "csl_wrapper.import"() <{module = "stencil_comms.csl", fields = []}> : () -> !csl.imported_module
  %arg0 = memref.alloc() : memref<512xf32>
  %arg1 = memref.alloc() : memref<512xf32>
  %10 = "csl.addressof"(%arg0) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
  %11 = "csl.addressof"(%arg1) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
  "csl.export"(%10) <{var_name = "arg0", type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
  "csl.export"(%11) <{var_name = "arg1", type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
  "csl.export"() <{var_name = @gauss_seidel_func, type = () -> ()}> : () -> ()
  %12 = memref.alloc() {alignment = 64 : i64} : memref<510xf32>
  csl.func @no_coeffs() {
    %this_comes_before_inserted_api_call = "test.op"() : () -> i1
    csl_stencil.apply(%arg0 : memref<512xf32>, %12 : memref<510xf32>) outs (%arg1 : memref<512xf32>) <{bounds = #stencil.bounds<[0, 0], [1, 1]>, num_chunks = 2 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>, swaps = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], topo = #dmp.topo<1022x510>}> ({
    ^bb2(%arg5 : memref<4x255xf32>, %arg6 : index, %arg7 : memref<510xf32>):
      csl_stencil.yield %arg7 : memref<510xf32>
    }, {
    ^bb3(%arg5_1 : memref<512xf32>, %arg6_1 : memref<510xf32>):
      %13 = arith.constant 1.666600e-01 : f32
      csl_stencil.yield %arg6_1 : memref<510xf32>
    }) to <[0, 0], [1, 1]>
    "csl.member_call"(%8) <{field = "unblock_cmd_stream"}> : (!csl.imported_module) -> ()
    csl.return
  }
  "csl_wrapper.yield"() <{fields = []}> : () -> ()
}) : () -> ()

// CHECK:      "csl_wrapper.module"() <{width = 1022 : i16, height = 510 : i16, params = [#csl_wrapper.param<"pattern" default=2 : i16>], program_name = "no_coeffs", target = "wse2"}> ({
// CHECK-NEXT:  ^bb0(%0 : i16, %1 : i16, %2 : i16, %3 : i16, %4 : i16):
// CHECK-NEXT:    "csl_wrapper.yield"() <{fields = []}> : () -> ()
// CHECK-NEXT:  }, {
// CHECK-NEXT:  ^bb1(%5 : i16, %6 : i16, %7 : i16):
// CHECK-NEXT:    %8 = "csl_wrapper.import"() <{module = "<memcpy/memcpy>", fields = []}> : () -> !csl.imported_module
// CHECK-NEXT:    %9 = "csl_wrapper.import"() <{module = "stencil_comms.csl", fields = []}> : () -> !csl.imported_module
// CHECK-NEXT:    %arg0 = memref.alloc() : memref<512xf32>
// CHECK-NEXT:    %arg1 = memref.alloc() : memref<512xf32>
// CHECK-NEXT:    %10 = "csl.addressof"(%arg0) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:    %11 = "csl.addressof"(%arg1) : (memref<512xf32>) -> !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>
// CHECK-NEXT:    "csl.export"(%10) <{var_name = "arg0", type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:    "csl.export"(%11) <{var_name = "arg1", type = !csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>}> : (!csl.ptr<f32, #csl<ptr_kind many>, #csl<ptr_const var>>) -> ()
// CHECK-NEXT:    "csl.export"() <{var_name = @gauss_seidel_func, type = () -> ()}> : () -> ()
// CHECK-NEXT:    %12 = memref.alloc() {alignment = 64 : i64} : memref<510xf32>
// CHECK-NEXT:    csl.func @no_coeffs() {
// CHECK-NEXT:      %this_comes_before_inserted_api_call = "test.op"() : () -> i1
// CHECK-NEXT:      csl_stencil.apply(%arg0 : memref<512xf32>, %12 : memref<510xf32>) outs (%arg1 : memref<512xf32>) <{bounds = #stencil.bounds<[0, 0], [1, 1]>, num_chunks = 2 : i64, operandSegmentSizes = array<i32: 1, 1, 0, 0, 1>, swaps = [#csl_stencil.exchange<to [1, 0]>, #csl_stencil.exchange<to [-1, 0]>, #csl_stencil.exchange<to [0, 1]>, #csl_stencil.exchange<to [0, -1]>], topo = #dmp.topo<1022x510>}> ({
// CHECK-NEXT:      ^bb2(%arg5 : memref<4x255xf32>, %arg6 : index, %arg7 : memref<510xf32>):
// CHECK-NEXT:        csl_stencil.yield %arg7 : memref<510xf32>
// CHECK-NEXT:      }, {
// CHECK-NEXT:      ^bb3(%arg5_1 : memref<512xf32>, %arg6_1 : memref<510xf32>):
// CHECK-NEXT:        %13 = arith.constant 1.666600e-01 : f32
// CHECK-NEXT:        csl_stencil.yield %arg6_1 : memref<510xf32>
// CHECK-NEXT:      }) to <[0, 0], [1, 1]>
// CHECK-NEXT:      "csl.member_call"(%8) <{field = "unblock_cmd_stream"}> : (!csl.imported_module) -> ()
// CHECK-NEXT:      csl.return
// CHECK-NEXT:    }
// CHECK-NEXT:    "csl_wrapper.yield"() <{fields = []}> : () -> ()
// CHECK-NEXT:  }) : () -> ()
