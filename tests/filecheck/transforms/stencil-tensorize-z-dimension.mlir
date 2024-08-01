// RUN: xdsl-opt %s -p "stencil-tensorize-z-dimension" | filecheck %s

builtin.module {
// CHECK:    builtin.module {

  func.func @gauss_seidel(%a : memref<1024x512x512xf32>, %b : memref<1024x512x512xf32>) {
    %0 = "stencil.external_load"(%a) : (memref<1024x512x512xf32>) -> !stencil.field<[-1,1023]x[-1,511]x[-1,511]xf32>
    %1 = "stencil.load"(%0) : (!stencil.field<[-1,1023]x[-1,511]x[-1,511]xf32>) -> !stencil.temp<[-1,1023]x[-1,511]x[-1,511]xf32>
    %2 = "stencil.external_load"(%b) : (memref<1024x512x512xf32>) -> !stencil.field<[-1,1023]x[-1,511]x[-1,511]xf32>
    %3 = "stencil.apply"(%1) ({
    ^0(%4 : !stencil.temp<[-1,1023]x[-1,511]x[-1,511]xf32>):
      %5 = arith.constant 1.666600e-01 : f32
      %6 = "stencil.access"(%4) {"offset" = #stencil.index<[1, 0, 0]>} : (!stencil.temp<[-1,1023]x[-1,511]x[-1,511]xf32>) -> f32
      %7 = "stencil.access"(%4) {"offset" = #stencil.index<[-1, 0, 0]>} : (!stencil.temp<[-1,1023]x[-1,511]x[-1,511]xf32>) -> f32
      %8 = "stencil.access"(%4) {"offset" = #stencil.index<[0, 0, 1]>} : (!stencil.temp<[-1,1023]x[-1,511]x[-1,511]xf32>) -> f32
      %9 = "stencil.access"(%4) {"offset" = #stencil.index<[0, 0, -1]>} : (!stencil.temp<[-1,1023]x[-1,511]x[-1,511]xf32>) -> f32
      %10 = "stencil.access"(%4) {"offset" = #stencil.index<[0, 1, 0]>} : (!stencil.temp<[-1,1023]x[-1,511]x[-1,511]xf32>) -> f32
      %11 = "stencil.access"(%4) {"offset" = #stencil.index<[0, -1, 0]>} : (!stencil.temp<[-1,1023]x[-1,511]x[-1,511]xf32>) -> f32
      %12 = arith.addf %11, %10 : f32
      %13 = arith.addf %12, %9 : f32
      %14 = arith.addf %13, %8 : f32
      %15 = arith.addf %14, %7 : f32
      %16 = arith.addf %15, %6 : f32
      %17 = arith.mulf %16, %5 : f32
      "stencil.return"(%17) : (f32) -> ()
    }) : (!stencil.temp<[-1,1023]x[-1,511]x[-1,511]xf32>) -> !stencil.temp<[0,1022]x[0,510]x[0,510]xf32>
    "stencil.store"(%3, %2) {"bounds" = #stencil.bounds<[0, 0, 0], [1022, 510, 510]>} : (!stencil.temp<[0,1022]x[0,510]x[0,510]xf32>, !stencil.field<[-1,1023]x[-1,511]x[-1,511]xf32>) -> ()
    func.return
  }

// CHECK:           func.func @gauss_seidel(%a : memref<1024x512x512xf32>, %b : memref<1024x512x512xf32>) {
// CHECK-NEXT:        %0 = stencil.external_load %a : memref<1024x512x512xf32> -> !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:        %1 = stencil.load %0 : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>> -> !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:        %2 = stencil.external_load %b : memref<1024x512x512xf32> -> !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:        %3 = stencil.apply(%4 = %1 : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>) -> (!stencil.temp<[0,1022]x[0,510]xtensor<510xf32>>) {
// CHECK-NEXT:          %5 = arith.constant 1.666600e-01 : f32
// CHECK-NEXT:          %6 = stencil.access %4[1, 0] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:          %7 = "tensor.extract_slice"(%6) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:          %8 = stencil.access %4[-1, 0] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:          %9 = "tensor.extract_slice"(%8) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:          %10 = stencil.access %4[0, 0] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:          %11 = "tensor.extract_slice"(%10) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:          %12 = stencil.access %4[0, 0] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:          %13 = "tensor.extract_slice"(%12) <{"static_offsets" = array<i64: -1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:          %14 = stencil.access %4[0, 1] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:          %15 = "tensor.extract_slice"(%14) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:          %16 = stencil.access %4[0, -1] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:          %17 = "tensor.extract_slice"(%16) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:          %18 = arith.addf %17, %15 : tensor<510xf32>
// CHECK-NEXT:          %19 = arith.addf %18, %13 : tensor<510xf32>
// CHECK-NEXT:          %20 = arith.addf %19, %11 : tensor<510xf32>
// CHECK-NEXT:          %21 = arith.addf %20, %9 : tensor<510xf32>
// CHECK-NEXT:          %22 = arith.addf %21, %7 : tensor<510xf32>
// CHECK-NEXT:          %23 = tensor.empty() : tensor<510xf32>
// CHECK-NEXT:          %24 = linalg.fill ins(%5 : f32) outs(%23 : tensor<510xf32>) -> tensor<510xf32>
// CHECK-NEXT:          %25 = arith.mulf %22, %24 : tensor<510xf32>
// CHECK-NEXT:          stencil.return %25 : tensor<510xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        stencil.store %3 to %2 (<[0, 0], [1022, 510]>) : !stencil.temp<[0,1022]x[0,510]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:        func.return
// CHECK-NEXT:      }

  func.func @gauss_seidel_func(%a : !stencil.field<[-1,1023]x[-1,511]x[-1,511]xf32>, %b : !stencil.field<[-1,1023]x[-1,511]x[-1,511]xf32>) {
    %0 = "stencil.load"(%a) : (!stencil.field<[-1,1023]x[-1,511]x[-1,511]xf32>) -> !stencil.temp<[-1,1023]x[-1,511]x[-1,511]xf32>
    %1 = "stencil.apply"(%0) ({
    ^0(%2 : !stencil.temp<[-1,1023]x[-1,511]x[-1,511]xf32>):
      %3 = arith.constant 1.666600e-01 : f32
      %4 = "stencil.access"(%2) {"offset" = #stencil.index<[1, 0, 0]>} : (!stencil.temp<[-1,1023]x[-1,511]x[-1,511]xf32>) -> f32
      %5 = "stencil.access"(%2) {"offset" = #stencil.index<[-1, 0, 0]>} : (!stencil.temp<[-1,1023]x[-1,511]x[-1,511]xf32>) -> f32
      %6 = "stencil.access"(%2) {"offset" = #stencil.index<[0, 0, 1]>} : (!stencil.temp<[-1,1023]x[-1,511]x[-1,511]xf32>) -> f32
      %7 = "stencil.access"(%2) {"offset" = #stencil.index<[0, 0, -1]>} : (!stencil.temp<[-1,1023]x[-1,511]x[-1,511]xf32>) -> f32
      %8 = "stencil.access"(%2) {"offset" = #stencil.index<[0, 1, 0]>} : (!stencil.temp<[-1,1023]x[-1,511]x[-1,511]xf32>) -> f32
      %9 = "stencil.access"(%2) {"offset" = #stencil.index<[0, -1, 0]>} : (!stencil.temp<[-1,1023]x[-1,511]x[-1,511]xf32>) -> f32
      %10 = arith.addf %9, %8 : f32
      %11 = arith.addf %10, %7 : f32
      %12 = arith.addf %11, %6 : f32
      %13 = arith.addf %12, %5 : f32
      %14 = arith.addf %13, %4 : f32
      %15 = arith.mulf %14, %3 : f32
      "stencil.return"(%15) : (f32) -> ()
    }) : (!stencil.temp<[-1,1023]x[-1,511]x[-1,511]xf32>) -> !stencil.temp<[0,1022]x[0,510]x[0,510]xf32>
    "stencil.store"(%1, %b) {"bounds" = #stencil.bounds<[0, 0, 0], [1022, 510, 510]>} : (!stencil.temp<[0,1022]x[0,510]x[0,510]xf32>, !stencil.field<[-1,1023]x[-1,511]x[-1,511]xf32>) -> ()
    func.return
  }

// CHECK:          func.func @gauss_seidel_func(%a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) {
// CHECK-NEXT:       %0 = stencil.load %a : !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>> -> !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:       %1 = stencil.apply(%2 = %0 : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>) -> (!stencil.temp<[0,1022]x[0,510]xtensor<510xf32>>) {
// CHECK-NEXT:         %3 = arith.constant 1.666600e-01 : f32
// CHECK-NEXT:         %4 = stencil.access %2[1, 0] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:         %5 = "tensor.extract_slice"(%4) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:         %6 = stencil.access %2[-1, 0] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:         %7 = "tensor.extract_slice"(%6) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:         %8 = stencil.access %2[0, 0] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:         %9 = "tensor.extract_slice"(%8) <{"static_offsets" = array<i64: 1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:         %10 = stencil.access %2[0, 0] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:         %11 = "tensor.extract_slice"(%10) <{"static_offsets" = array<i64: -1>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:         %12 = stencil.access %2[0, 1] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:         %13 = "tensor.extract_slice"(%12) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:         %14 = stencil.access %2[0, -1] : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:         %15 = "tensor.extract_slice"(%14) <{"static_offsets" = array<i64: 0>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64: 1>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<512xf32>) -> tensor<510xf32>
// CHECK-NEXT:         %16 = arith.addf %15, %13 : tensor<510xf32>
// CHECK-NEXT:         %17 = arith.addf %16, %11 : tensor<510xf32>
// CHECK-NEXT:         %18 = arith.addf %17, %9 : tensor<510xf32>
// CHECK-NEXT:         %19 = arith.addf %18, %7 : tensor<510xf32>
// CHECK-NEXT:         %20 = arith.addf %19, %5 : tensor<510xf32>
// CHECK-NEXT:         %21 = tensor.empty() : tensor<510xf32>
// CHECK-NEXT:         %22 = linalg.fill ins(%3 : f32) outs(%21 : tensor<510xf32>) -> tensor<510xf32>
// CHECK-NEXT:         %23 = arith.mulf %20, %22 : tensor<510xf32>
// CHECK-NEXT:         stencil.return %23 : tensor<510xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:       stencil.store %1 to %b (<[0, 0], [1022, 510]>) : !stencil.temp<[0,1022]x[0,510]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:       func.return
// CHECK-NEXT:     }

}
// CHECK-NEXT:    }