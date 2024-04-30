// RUN: xdsl-opt %s -p "stencil-tensorize-z-dimension" | filecheck %s

builtin.module {
  func.func @gauss_seidel(%a : memref<1024x512x512xf32>, %b : memref<1024x512x512xf32>) {
    %0 = "stencil.external_load"(%a) : (memref<1024x512x512xf32>) -> !stencil.field<[-1,1023]x[-1,511]x[-1,511]xf32>
    %1 = "stencil.load"(%0) : (!stencil.field<[-1,1023]x[-1,511]x[-1,511]xf32>) -> !stencil.temp<[-1,1023]x[-1,511]x[-1,511]xf32>
    %2 = "stencil.external_load"(%b) : (memref<1024x512x512xf32>) -> !stencil.field<[-1,1023]x[-1,511]x[-1,511]xf32>
    %3 = "stencil.apply"(%1) ({
    ^0(%4 : !stencil.temp<[-1,1023]x[-1,511]x[-1,511]xf32>):
      %5 = arith.constant 1.666600e-01 : f32
      %6 = "stencil.access"(%4) {"offset" = #stencil.index<1, 0, 0>} : (!stencil.temp<[-1,1023]x[-1,511]x[-1,511]xf32>) -> f32
      %7 = "stencil.access"(%4) {"offset" = #stencil.index<-1, 0, 0>} : (!stencil.temp<[-1,1023]x[-1,511]x[-1,511]xf32>) -> f32
      %8 = "stencil.access"(%4) {"offset" = #stencil.index<0, 0, 1>} : (!stencil.temp<[-1,1023]x[-1,511]x[-1,511]xf32>) -> f32
      %9 = "stencil.access"(%4) {"offset" = #stencil.index<0, 0, -1>} : (!stencil.temp<[-1,1023]x[-1,511]x[-1,511]xf32>) -> f32
      %10 = "stencil.access"(%4) {"offset" = #stencil.index<0, 1, 0>} : (!stencil.temp<[-1,1023]x[-1,511]x[-1,511]xf32>) -> f32
      %11 = "stencil.access"(%4) {"offset" = #stencil.index<0, -1, 0>} : (!stencil.temp<[-1,1023]x[-1,511]x[-1,511]xf32>) -> f32
      %12 = arith.addf %11, %10 : f32
      %13 = arith.addf %12, %9 : f32
      %14 = arith.addf %13, %8 : f32
      %15 = arith.addf %14, %7 : f32
      %16 = arith.addf %15, %6 : f32
      %17 = arith.mulf %16, %5 : f32
      "stencil.return"(%17) : (f32) -> ()
    }) : (!stencil.temp<[-1,1023]x[-1,511]x[-1,511]xf32>) -> !stencil.temp<[0,1022]x[0,510]x[0,510]xf32>
    "stencil.store"(%3, %2) {"lb" = #stencil.index<0, 0, 0>, "ub" = #stencil.index<1022, 510, 510>} : (!stencil.temp<[0,1022]x[0,510]x[0,510]xf32>, !stencil.field<[-1,1023]x[-1,511]x[-1,511]xf32>) -> ()
    func.return
  }
}

// CHECK-NEXT:    builtin.module {
// CHECK-NEXT:      func.func @gauss_seidel(%a : memref<1024x512xtensor<512xf32>>, %b : memref<1024x512xtensor<512xf32>>) {
// CHECK-NEXT:        %0 = "stencil.external_load"(%a) : (memref<1024x512xtensor<512xf32>>) -> !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:        %1 = "stencil.load"(%0) : (!stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) -> !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:        %2 = "stencil.external_load"(%b) : (memref<1024x512xtensor<512xf32>>) -> !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>
// CHECK-NEXT:        %3 = "stencil.apply"(%1) ({
// CHECK-NEXT:        ^0(%4 : !stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>):
// CHECK-NEXT:          %5 = arith.constant 1.666600e-01 : f32
// CHECK-NEXT:          %6 = "stencil.access"(%4) {"offset" = #stencil.index<1, 0>} : (!stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>) -> tensor<510xf32>
// CHECK-NEXT:          %7 = "stencil.access"(%4) {"offset" = #stencil.index<-1, 0>} : (!stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>) -> tensor<510xf32>
// CHECK-NEXT:          %8 = "stencil.access"(%4) {"offset" = #stencil.index<0, 0>} : (!stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>) -> tensor<510xf32>
// CHECK-NEXT:          %9 = "tensor.extract_slice"(%8) <{"static_offsets" = array<i64>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<510xf32>) -> tensor<510xf32>
// CHECK-NEXT:          %10 = "stencil.access"(%4) {"offset" = #stencil.index<0, 0>} : (!stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>) -> tensor<510xf32>
// CHECK-NEXT:          %11 = "tensor.extract_slice"(%10) <{"static_offsets" = array<i64>, "static_sizes" = array<i64: 510>, "static_strides" = array<i64>, "operandSegmentSizes" = array<i32: 1, 0, 0, 0>}> : (tensor<510xf32>) -> tensor<510xf32>
// CHECK-NEXT:          %12 = "stencil.access"(%4) {"offset" = #stencil.index<0, 1>} : (!stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>) -> tensor<510xf32>
// CHECK-NEXT:          %13 = "stencil.access"(%4) {"offset" = #stencil.index<0, -1>} : (!stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>) -> tensor<510xf32>
// CHECK-NEXT:          %14 = arith.addf %13, %12 : tensor<510xf32>
// CHECK-NEXT:          %15 = arith.addf %14, %11 : tensor<510xf32>
// CHECK-NEXT:          %16 = arith.addf %15, %9 : tensor<510xf32>
// CHECK-NEXT:          %17 = arith.addf %16, %7 : tensor<510xf32>
// CHECK-NEXT:          %18 = arith.addf %17, %6 : tensor<510xf32>
// CHECK-NEXT:          %19 = tensor.empty() : tensor<510xf32>
// CHECK-NEXT:          %20 = linalg.fill ins(%5 : f32) outs(%19 : tensor<510xf32>) -> tensor<510xf32>
// CHECK-NEXT:          %21 = arith.mulf %18, %20 : tensor<510xf32>
// CHECK-NEXT:          "stencil.return"(%21) : (tensor<510xf32>) -> ()
// CHECK-NEXT:        }) : (!stencil.temp<[-1,1023]x[-1,511]xtensor<512xf32>>) -> !stencil.temp<[0,1022]x[0,510]xtensor<510xf32>>
// CHECK-NEXT:        "stencil.store"(%3, %2) {"lb" = #stencil.index<0, 0, 0>, "ub" = #stencil.index<1022, 510, 510>} : (!stencil.temp<[0,1022]x[0,510]xtensor<510xf32>>, !stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>>) -> ()
// CHECK-NEXT:        func.return
// CHECK-NEXT:      }
// CHECK-NEXT:    }
