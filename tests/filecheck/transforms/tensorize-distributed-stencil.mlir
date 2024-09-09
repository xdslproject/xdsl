

// func.func @gauss_seidel_func(%a : !stencil.field<[-1,1023]x[-1,511]x[-1,511]xf32>, %b : !stencil.field<[-1,1023]x[-1,511]x[-1,511]xf32>) {
//   %0 = stencil.load %a : !stencil.field<[-1,1023]x[-1,511]x[-1,511]xf32> -> !stencil.temp<[-1,2]x[-1,2]x[-1,511]xf32>
//   %1 = "dmp.swap"(%0) {"strategy" = #dmp.grid_slice_2d<#dmp.topo<1022x510>, false>, "swaps" = [#dmp.exchange<at [1, 0, 0] size [1, 1, 510] source offset [-1, 0, 0] to [1, 0, 0]>, #dmp.exchange<at [-1, 0, 0] size [1, 1, 510] source offset [1, 0, 0] to [-1, 0, 0]>, #dmp.exchange<at [0, 1, 0] size [1, 1, 510] source offset [0, -1, 0] to [0, 1, 0]>, #dmp.exchange<at [0, -1, 0] size [1, 1, 510] source offset [0, 1, 0] to [0, -1, 0]>]} : (!stencil.temp<[-1,2]x[-1,2]x[-1,511]xf32>) -> !stencil.temp<[-1,2]x[-1,2]x[-1,511]xf32>
//   %2 = stencil.apply(%3 = %1 : !stencil.temp<[-1,2]x[-1,2]x[-1,511]xf32>) -> (!stencil.temp<[0,1]x[0,1]x[0,510]xf32>) {
//     %4 = arith.constant 1.234500e-01 : f32
//     %5 = stencil.access %3[1, 0, 0] : !stencil.temp<[-1,2]x[-1,2]x[-1,511]xf32>
//     %6 = stencil.access %3[-1, 0, 0] : !stencil.temp<[-1,2]x[-1,2]x[-1,511]xf32>
//     %7 = stencil.access %3[0, 0, 1] : !stencil.temp<[-1,2]x[-1,2]x[-1,511]xf32>
//     %8 = stencil.access %3[0, 0, -1] : !stencil.temp<[-1,2]x[-1,2]x[-1,511]xf32>
//     %9 = stencil.access %3[0, 1, 0] : !stencil.temp<[-1,2]x[-1,2]x[-1,511]xf32>
//     %10 = stencil.access %3[0, -1, 0] : !stencil.temp<[-1,2]x[-1,2]x[-1,511]xf32>
//     %11 = arith.addf %10, %9 : f32
//     %12 = arith.addf %11, %8 : f32
//     %13 = arith.addf %12, %7 : f32
//     %14 = arith.addf %13, %6 : f32
//     %15 = arith.addf %14, %5 : f32
//     %16 = arith.mulf %15, %4 : f32
//     stencil.return %16 : f32
//   }
//   stencil.store %2 to %b(<[0, 0, 0], [1, 1, 510]>) : !stencil.temp<[0,1]x[0,1]x[0,510]xf32> to !stencil.field<[-1,1023]x[-1,511]x[-1,511]xf32>
//   func.return
// }

//CHECK:      func.func @gauss_seidel_func(%a : !stencil.field<[-1,1023]x[-1,511]x[-1,511]xtensor<510xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]x[-1,511]xtensor<510xf32>>) {
//CHECK-NEXT:   %0 = stencil.load %a : !stencil.field<[-1,1023]x[-1,511]x[-1,511]xtensor<510xf32>> -> !stencil.temp<[-1,2]x[-1,2]x[-1,511]xtensor<510xf32>>
//CHECK-NEXT:   %1 = "dmp.swap"(%0) {"strategy" = #dmp.grid_slice_2d<#dmp.topo<1022x510>, false>, "swaps" = [#dmp.exchange<at [1, 0, 0] size [1, 1, 510] source offset [-1, 0, 0] to [1, 0, 0]>, #dmp.exchange<at [-1, 0, 0] size [1, 1, 510] source offset [1, 0, 0] to [-1, 0, 0]>, #dmp.exchange<at [0, 1, 0] size [1, 1, 510] source offset [0, -1, 0] to [0, 1, 0]>, #dmp.exchange<at [0, -1, 0] size [1, 1, 510] source offset [0, 1, 0] to [0, -1, 0]>]} : (!stencil.temp<[-1,2]x[-1,2]x[-1,511]xtensor<510xf32>>) -> !stencil.temp<[-1,2]x[-1,2]x[-1,511]xtensor<510xf32>>
//CHECK-NEXT:   %2 = stencil.apply(%3 = %1 : !stencil.temp<[-1,2]x[-1,2]x[-1,511]xtensor<510xf32>>) -> (!stencil.temp<[0,1]x[0,1]x[0,510]xtensor<510xf32>>) {
//CHECK-NEXT:     %4 = arith.constant dense<1.234500e-01> : tensor<510xf32>
//CHECK-NEXT:     %5 = arith.constant 1.234500e-01 : f32
//CHECK-NEXT:     %6 = stencil.access %3[1, 0, 0] : !stencil.temp<[-1,2]x[-1,2]x[-1,511]xtensor<510xf32>>
//CHECK-NEXT:     %7 = stencil.access %3[-1, 0, 0] : !stencil.temp<[-1,2]x[-1,2]x[-1,511]xtensor<510xf32>>
//CHECK-NEXT:     %8 = stencil.access %3[0, 0, 1] : !stencil.temp<[-1,2]x[-1,2]x[-1,511]xtensor<510xf32>>
//CHECK-NEXT:     %9 = stencil.access %3[0, 0, -1] : !stencil.temp<[-1,2]x[-1,2]x[-1,511]xtensor<510xf32>>
//CHECK-NEXT:     %10 = stencil.access %3[0, 1, 0] : !stencil.temp<[-1,2]x[-1,2]x[-1,511]xtensor<510xf32>>
//CHECK-NEXT:     %11 = stencil.access %3[0, -1, 0] : !stencil.temp<[-1,2]x[-1,2]x[-1,511]xtensor<510xf32>>
//CHECK-NEXT:     %12 = arith.addf %11, %10 : tensor<510xf32>
//CHECK-NEXT:     %13 = arith.addf %12, %9 : tensor<510xf32>
//CHECK-NEXT:     %14 = arith.addf %13, %8 : tensor<510xf32>
//CHECK-NEXT:     %15 = arith.addf %14, %7 : tensor<510xf32>
//CHECK-NEXT:     %16 = arith.addf %15, %6 : tensor<510xf32>
//CHECK-NEXT:     %17 = arith.mulf %16, %4 : tensor<510xf32>
//CHECK-NEXT:     stencil.return %17 : tensor<510xf32>
//CHECK-NEXT:   }
//CHECK-NEXT:   stencil.store %2 to %b(<[0, 0, 0], [1, 1, 510]>) : !stencil.temp<[0,1]x[0,1]x[0,510]xtensor<510xf32>> to !stencil.field<[-1,1023]x[-1,511]x[-1,511]xtensor<510xf32>>
//CHECK-NEXT:   func.return
//CHECK-NEXT: }

func.func @bufferized(%a : !stencil.field<[-1,1023]x[-1,511]x[-1,511]xf32>, %b : !stencil.field<[-1,1023]x[-1,511]x[-1,511]xf32>) {
  "dmp.swap"(%a) {"strategy" = #dmp.grid_slice_2d<#dmp.topo<1022x510>, false>, "swaps" = [
  #dmp.exchange<at [1, 0, 0] size [1, 1, 510] source offset [-1, 0, 0] to [1, 0, 0]>,
  #dmp.exchange<at [-1, 0, 0] size [1, 1, 510] source offset [1, 0, 0] to [-1, 0, 0]>,
  #dmp.exchange<at [0, 1, 0] size [1, 1, 510] source offset [0, -1, 0] to [0, 1, 0]>,
  #dmp.exchange<at [0, -1, 0] size [1, 1, 510] source offset [0, 1, 0] to [0, -1, 0]>]} : (!stencil.field<[-1,1023]x[-1,511]x[-1,511]xf32>) -> ()
  stencil.apply(%0 = %a : !stencil.field<[-1,1023]x[-1,511]x[-1,511]xf32>) outs (%b : !stencil.field<[-1,1023]x[-1,511]x[-1,511]xf32>) {
    %1 = arith.constant 1.234500e-01 : f32
    %2 = stencil.access %0[1, 0, 0] : !stencil.field<[-1,1023]x[-1,511]x[-1,511]xf32>
    %3 = stencil.access %0[-1, 0, 0] : !stencil.field<[-1,1023]x[-1,511]x[-1,511]xf32>
    %4 = stencil.access %0[0, 0, 1] : !stencil.field<[-1,1023]x[-1,511]x[-1,511]xf32>
    %5 = stencil.access %0[0, 0, -1] : !stencil.field<[-1,1023]x[-1,511]x[-1,511]xf32>
    %6 = stencil.access %0[0, 1, 0] : !stencil.field<[-1,1023]x[-1,511]x[-1,511]xf32>
    %7 = stencil.access %0[0, -1, 0] : !stencil.field<[-1,1023]x[-1,511]x[-1,511]xf32>
    %8 = arith.addf %7, %6 : f32
    %9 = arith.addf %8, %5 : f32
    %10 = arith.addf %9, %4 : f32
    %11 = arith.addf %10, %3 : f32
    %12 = arith.addf %11, %2 : f32
    %13 = arith.mulf %12, %1 : f32
    stencil.return %13 : f32
  } to <[0, 0, 0], [1, 1, 510]>
  func.return
}

//CHECK:      func.func @bufferized(%a : !stencil.field<[-1,1023]x[-1,511]x[-1,511]xtensor<510xf32>>, %b : !stencil.field<[-1,1023]x[-1,511]x[-1,511]xtensor<510xf32>>) {
//CHECK-NEXT:   "dmp.swap"(%a) {"strategy" = #dmp.grid_slice_2d<#dmp.topo<1022x510>, false>, "swaps" = [#dmp.exchange<at [1, 0, 0] size [1, 1, 510] source offset [-1, 0, 0] to [1, 0, 0]>, #dmp.exchange<at [-1, 0, 0] size [1, 1, 510] source offset [1, 0, 0] to [-1, 0, 0]>, #dmp.exchange<at [0, 1, 0] size [1, 1, 510] source offset [0, -1, 0] to [0, 1, 0]>, #dmp.exchange<at [0, -1, 0] size [1, 1, 510] source offset [0, 1, 0] to [0, -1, 0]>]} : (!stencil.field<[-1,1023]x[-1,511]x[-1,511]xtensor<510xf32>>) -> ()
//CHECK-NEXT:   stencil.apply(%0 = %a : !stencil.field<[-1,1023]x[-1,511]x[-1,511]xtensor<510xf32>>) outs (%b : !stencil.field<[-1,1023]x[-1,511]x[-1,511]xtensor<510xf32>>) {
//CHECK-NEXT:     %1 = arith.constant dense<1.234500e-01> : tensor<510xf32>
//CHECK-NEXT:     %2 = arith.constant 1.234500e-01 : f32
//CHECK-NEXT:     %3 = stencil.access %0[1, 0, 0] : !stencil.field<[-1,1023]x[-1,511]x[-1,511]xtensor<510xf32>>
//CHECK-NEXT:     %4 = stencil.access %0[-1, 0, 0] : !stencil.field<[-1,1023]x[-1,511]x[-1,511]xtensor<510xf32>>
//CHECK-NEXT:     %5 = stencil.access %0[0, 0, 1] : !stencil.field<[-1,1023]x[-1,511]x[-1,511]xtensor<510xf32>>
//CHECK-NEXT:     %6 = stencil.access %0[0, 0, -1] : !stencil.field<[-1,1023]x[-1,511]x[-1,511]xtensor<510xf32>>
//CHECK-NEXT:     %7 = stencil.access %0[0, 1, 0] : !stencil.field<[-1,1023]x[-1,511]x[-1,511]xtensor<510xf32>>
//CHECK-NEXT:     %8 = stencil.access %0[0, -1, 0] : !stencil.field<[-1,1023]x[-1,511]x[-1,511]xtensor<510xf32>>
//CHECK-NEXT:     %9 = arith.addf %8, %7 : tensor<510xf32>
//CHECK-NEXT:     %10 = arith.addf %9, %6 : tensor<510xf32>
//CHECK-NEXT:     %11 = arith.addf %10, %5 : tensor<510xf32>
//CHECK-NEXT:     %12 = arith.addf %11, %4 : tensor<510xf32>
//CHECK-NEXT:     %13 = arith.addf %12, %3 : tensor<510xf32>
//CHECK-NEXT:     %14 = arith.mulf %13, %1 : tensor<510xf32>
//CHECK-NEXT:     stencil.return %14 : tensor<510xf32>
//CHECK-NEXT:   } to <[0, 0, 0], [1, 1, 510]>
//CHECK-NEXT:   func.return
//CHECK-NEXT: }
