// RUN: xdsl-opt %s -p "csl-stencil-tensorize" | filecheck %s

func.func @gs(%u_vec0 : !stencil.field<[-4,604]x[-4,604]x[-4,604]xf32>, %u_vec1 : !stencil.field<[-4,604]x[-4,604]x[-4,604]xf32>) {
  %0 = tensor.empty() : tensor<1xf32>
  csl_stencil.apply(%u_vec0 : !stencil.field<[-4,604]x[-4,604]x[-4,604]xf32>, %0 : tensor<1xf32>) outs (%u_vec1 : !stencil.field<[-4,604]x[-4,604]x[-4,604]xf32>) <{"swaps" = [#dmp.exchange<at [1, 0, 0] size [2, 1, 600] source offset [-2, 0, 0] to [1, 0, 0]>, #dmp.exchange<at [-2, 0, 0] size [2, 1, 600] source offset [2, 0, 0] to [-1, 0, 0]>, #dmp.exchange<at [0, 1, 0] size [1, 2, 600] source offset [0, -2, 0] to [0, 1, 0]>, #dmp.exchange<at [0, -2, 0] size [1, 2, 600] source offset [0, 2, 0] to [0, -1, 0]>], "topo" = #dmp.topo<600x600>, "num_chunks" = 2 : i64, "bounds" = #stencil.bounds<[0, 0, 0], [1, 1, 600]>, "operandSegmentSizes" = array<i32: 1, 1, 0, 1>}> ({
  ^0(%1 : !stencil.field<[-4,604]x[-4,604]x[-4,604]xf32>, %2 : index, %3 : f32):
    %4 = csl_stencil.access %1[-2, 0, 0] : !stencil.field<[-4,604]x[-4,604]x[-4,604]xf32>
    %5 = csl_stencil.access %1[2, 0, 0] : !stencil.field<[-4,604]x[-4,604]x[-4,604]xf32>
    %6 = csl_stencil.access %1[0, -2, 0] : !stencil.field<[-4,604]x[-4,604]x[-4,604]xf32>
    %7 = csl_stencil.access %1[0, 2, 0] : !stencil.field<[-4,604]x[-4,604]x[-4,604]xf32>
    %8 = arith.addf %4, %5 : f32
    %9 = arith.addf %8, %6 : f32
    %10 = arith.addf %9, %7 : f32
    csl_stencil.yield %10 : f32
  }, {
  ^1(%11 : !stencil.field<[-4,604]x[-4,604]x[-4,604]xf32>, %12 : f32):
    %13 = csl_stencil.access %11[0, 0, -2] : !stencil.field<[-4,604]x[-4,604]x[-4,604]xf32>
    %14 = csl_stencil.access %11[0, 0, 2] : !stencil.field<[-4,604]x[-4,604]x[-4,604]xf32>
    %15 = arith.addf %12, %13 : f32
    %16 = arith.addf %15, %14 : f32
    csl_stencil.yield %16 : f32
  }) to <[0, 0, 0], [1, 1, 600]>
  func.return
}
