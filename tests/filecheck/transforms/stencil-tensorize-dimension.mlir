// RUN: xdsl-opt %s -p "stencil-tensorize-dimension{dimension=2}" | filecheck %s

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
    }) : (!stencil.temp<[-1,1023]x[-1,511]x[-1,511]xf32>) -> !stencil.temp<[0,1021]x[0,509]x[0,509]xf32>
    "stencil.store"(%3, %2) {"lb" = #stencil.index<0, 0, 0>, "ub" = #stencil.index<1022, 510, 510>} : (!stencil.temp<[0,1021]x[0,509]x[0,509]xf32>, !stencil.field<[-1,1023]x[-1,511]x[-1,511]xf32>) -> ()
    func.return
  }
}

// CHECK-NEXT:    builtin.module {
// CHECK-NEXT:      func.func @gauss_seidel(%a : memref<1024x512xtensor<512,f32>>, %b : memref<1024x512xtensor<512,f32>>) {
// CHECK-NEXT:        %0 = "stencil.external_load"(%a) : (memref<1024x512xtensor<512,f32>>) -> !stencil.field<[-1,1023]x[-1,511]xtensor<[-1,511],f32>>
// CHECK-NEXT:        %1 = "stencil.load"(%0) : (!stencil.field<[-1,1023]x[-1,511]xtensor<[-1,511],f32>>) -> !stencil.temp<[-1,1023]x[-1,511]xtensor<[-1,511],f32>>
// CHECK-NEXT:        %2 = "stencil.external_load"(%b) : (memref<1024x512xtensor<512,f32>>) -> !stencil.field<[-1,1023]x[-1,511]xtensor<[-1,511],f32>>
// CHECK-NEXT:        %3 = "stencil.apply"(%1) ({
// CHECK-NEXT:        ^0(%4 : !stencil.temp<[-1,1023]x[-1,511]xtensor<[-1,511],f32>>):
// CHECK-NEXT:          %5 = arith.constant 1.666600e-01 : f32
// CHECK-NEXT:          %6 = "stencil.access"(%4) {"offset" = #stencil.index<1, 0>} : (!stencil.temp<[-1,1023]x[-1,511]xtensor<[-1,511],f32>>) -> tensor<[-1,511],f32>
// CHECK-NEXT:          %7 = "stencil.access"(%4) {"offset" = #stencil.index<-1, 0>} : (!stencil.temp<[-1,1023]x[-1,511]xtensor<[-1,511],f32>>) -> tensor<[-1,511],f32>
// CHECK-NEXT:          %8 = "stencil.access"(%4) {"offset" = #stencil.index<0, 1>} : (!stencil.temp<[-1,1023]x[-1,511]xtensor<[-1,511],f32>>) -> tensor<[-1,511],f32>
// CHECK-NEXT:          %9 = "stencil.access"(%4) {"offset" = #stencil.index<0, -1>} : (!stencil.temp<[-1,1023]x[-1,511]xtensor<[-1,511],f32>>) -> tensor<[-1,511],f32>
// CHECK-NEXT:
// CHECK-NEXT:          %10 = "stencil.access"(%4) {"offset" = #stencil.index<0, 0>} : (!stencil.temp<[-1,1023]x[-1,511]xtensor<[-1,511],f32>>) -> tensor<[-1,511],f32>
// CHECK-NEXT:          %11 = tensor.extract_slice %10[-1][511][1] : tensor<[-1,511],f32> -> tensor<[-1,511],f32>
// CHECK-NEXT:          %12 = tensor.extract_slice %10[ 1][511][1] : tensor<[-1,511],f32> -> tensor<[-1,511],f32>
// CHECK-NEXT:
// CHECK-NEXT:          %13 = linalg.add %7, %6 : tensor<[-1,511],f32>
// CHECK-NEXT:          %14 = linalg.add %12, %8 : tensor<[-1,511],f32>
// CHECK-NEXT:          %15 = linalg.add %13, %9 : tensor<[-1,511],f32>
// CHECK-NEXT:          %16 = linalg.add %15, %11: tensor<[-1,511],f32>
// CHECK-NEXT:          %17 = linalg.add %16, %12: tensor<[-1,511],f32>
// CHECK-NEXT:
// CHECK-NEXT:          // not sure that linalg.mulf supports tensor*scalar, might have to use different op
// CHECK-NEXT:          %18 = linalg.mulf %17, %5 : tensor<[-1,511],f32>
// CHECK-NEXT:
// CHECK-NEXT:          "stencil.return"(%18) : (tensor<[-1,511],f32>) -> ()
// CHECK-NEXT:        }) : (!stencil.temp<[-1,1023]x[-1,511]xtensor<[-1,511],f32>>) -> !stencil.temp<[0,1021]x[0,509]xtensor<[0,509],f32>>
// CHECK-NEXT:        "stencil.store"(%3, %2) {"lb" = #stencil.index<0, 0, 0>, "ub" = #stencil.index<1022, 510, 510>} : (!stencil.temp<[0,1021]x[0,509]xtensor<[0,509],f32>>, !stencil.field<[-1,1023]x[-1,511]xtensor<[-1,511],f32>>) -> ()
// CHECK-NEXT:        func.return
// CHECK-NEXT:      }
// CHECK-NEXT:    }
