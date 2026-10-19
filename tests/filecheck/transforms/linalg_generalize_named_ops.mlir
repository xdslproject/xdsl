// RUN: xdsl-opt -p linalg-generalize-named-ops %s | filecheck %s

//CHECK:   %lhs, %rhs, %out = "test.op"() : () -> (memref<4x5xf32>, memref<4x5xf32>, memref<4x5xf32>)
//CHECK-NEXT:   %a, %b, %c = "test.op"() : () -> (memref<2x3xf32>, memref<3x4xf32>, memref<2x4xf32>)
//CHECK-NEXT:   %cond, %t, %f, %res = "test.op"() : () -> (memref<4x5xi1>, memref<4x5xf32>, memref<4x5xf32>, memref<4x5xf32>)
//CHECK-NEXT:   %n1, %n2, %n3 = "test.op"() : () -> (memref<1x4x16x16xf32>, memref<6x4x3x3xf32>, memref<1x6x6x4xf32>)
//CHECK-NEXT:   %h1, %h2, %h3 = "test.op"() : () -> (memref<1x16x16x4xf32>, memref<3x3x4x6xf32>, memref<1x6x4x6xf32>)
//CHECK-NEXT:   %w1, %w2, %w3 = "test.op"() : () -> (memref<1x16x16x4xf32>, memref<6x3x3x4xf32>, memref<1x6x4x6xf32>)
//CHECK-NEXT:   %g1, %g2, %g3 = "test.op"() : () -> (memref<1x2x4x16x16xf32>, memref<6x2x4x3x3xf32>, memref<1x2x6x6x4xf32>)
//CHECK-NEXT:   %q1, %q2, %q3 = "test.op"() : () -> (memref<1x2x4x16x16xf32>, memref<2x6x4x3x3xf32>, memref<1x2x6x6x4xf32>)
//CHECK-NEXT:   %z1, %z2, %z3 = "test.op"() : () -> (memref<1x16x16x2x4xf32>, memref<2x6x3x3x4xf32>, memref<1x6x4x2x6xf32>)
//CHECK-NEXT:   %p1, %p2, %p3 = "test.op"() : () -> (memref<1x4x16x16xf32>, memref<3x3xf32>, memref<1x4x6x4xf32>)
//CHECK-NEXT:   %v, %filled = "test.op"() : () -> (f32, memref<4x5xf32>)
%lhs, %rhs, %out = "test.op"() : () -> (memref<4x5xf32>, memref<4x5xf32>, memref<4x5xf32>)
%a, %b, %c = "test.op"() : () -> (memref<2x3xf32>, memref<3x4xf32>, memref<2x4xf32>)
%cond, %t, %f, %res = "test.op"() : () -> (memref<4x5xi1>, memref<4x5xf32>, memref<4x5xf32>, memref<4x5xf32>)
%n1, %n2, %n3 = "test.op"() : () -> (memref<1x4x16x16xf32>, memref<6x4x3x3xf32>, memref<1x6x6x4xf32>)
%h1, %h2, %h3 = "test.op"() : () -> (memref<1x16x16x4xf32>, memref<3x3x4x6xf32>, memref<1x6x4x6xf32>)
%w1, %w2, %w3 = "test.op"() : () -> (memref<1x16x16x4xf32>, memref<6x3x3x4xf32>, memref<1x6x4x6xf32>)
%g1, %g2, %g3 = "test.op"() : () -> (memref<1x2x4x16x16xf32>, memref<6x2x4x3x3xf32>, memref<1x2x6x6x4xf32>)
%q1, %q2, %q3 = "test.op"() : () -> (memref<1x2x4x16x16xf32>, memref<2x6x4x3x3xf32>, memref<1x2x6x6x4xf32>)
%z1, %z2, %z3 = "test.op"() : () -> (memref<1x16x16x2x4xf32>, memref<2x6x3x3x4xf32>, memref<1x6x4x2x6xf32>)
%p1, %p2, %p3 = "test.op"() : () -> (memref<1x4x16x16xf32>, memref<3x3xf32>, memref<1x4x6x4xf32>)
%v, %filled = "test.op"() : () -> (f32, memref<4x5xf32>)

//CHECK-NEXT:   linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%lhs, %rhs : memref<4x5xf32>, memref<4x5xf32>) outs(%out : memref<4x5xf32>) {
//CHECK-NEXT:   ^bb0(%0: f32, %1: f32, %2: f32):
//CHECK-NEXT:     %3 = arith.addf %0, %1 : f32
//CHECK-NEXT:     linalg.yield %3 : f32
//CHECK-NEXT:   }
linalg.add ins(%lhs, %rhs : memref<4x5xf32>, memref<4x5xf32>) outs(%out : memref<4x5xf32>)

//CHECK-NEXT:   linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%a, %b : memref<2x3xf32>, memref<3x4xf32>) outs(%c : memref<2x4xf32>) {
//CHECK-NEXT:   ^bb0(%4: f32, %5: f32, %6: f32):
//CHECK-NEXT:     %7 = arith.mulf %4, %5 : f32
//CHECK-NEXT:     %8 = arith.addf %7, %6 : f32
//CHECK-NEXT:     linalg.yield %8 : f32
//CHECK-NEXT:   }
linalg.matmul ins(%a, %b : memref<2x3xf32>, memref<3x4xf32>) outs(%c : memref<2x4xf32>)

//CHECK-NEXT:   linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%cond, %t, %f : memref<4x5xi1>, memref<4x5xf32>, memref<4x5xf32>) outs(%res : memref<4x5xf32>) {
//CHECK-NEXT:   ^bb0(%9: i1, %10: f32, %11: f32, %12: f32):
//CHECK-NEXT:     %13 = arith.select %9, %10, %11 : f32
//CHECK-NEXT:     linalg.yield %13 : f32
//CHECK-NEXT:   }
linalg.select ins(%cond, %t, %f : memref<4x5xi1>, memref<4x5xf32>, memref<4x5xf32>) outs(%res : memref<4x5xf32>)

//CHECK-NEXT:   linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, ((d2 * 2) + (d5 * 2)), ((d3 * 3) + (d6 * 3)))>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%n1, %n2 : memref<1x4x16x16xf32>, memref<6x4x3x3xf32>) outs(%n3 : memref<1x6x6x4xf32>) {
//CHECK-NEXT:   ^bb0(%14: f32, %15: f32, %16: f32):
//CHECK-NEXT:     %17 = arith.mulf %14, %15 : f32
//CHECK-NEXT:     %18 = arith.addf %17, %16 : f32
//CHECK-NEXT:     linalg.yield %18 : f32
//CHECK-NEXT:   }
linalg.conv_2d_nchw_fchw {dilations = dense<[2, 3]> : tensor<2xi64>, strides = dense<[2, 3]> : tensor<2xi64>} ins(%n1, %n2 : memref<1x4x16x16xf32>, memref<6x4x3x3xf32>) outs(%n3 : memref<1x6x6x4xf32>)

//CHECK-NEXT:   linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, ((d1 * 2) + (d4 * 2)), ((d2 * 3) + (d5 * 3)), d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%h1, %h2 : memref<1x16x16x4xf32>, memref<3x3x4x6xf32>) outs(%h3 : memref<1x6x4x6xf32>) {
//CHECK-NEXT:   ^bb0(%19: f32, %20: f32, %21: f32):
//CHECK-NEXT:     %22 = arith.mulf %19, %20 : f32
//CHECK-NEXT:     %23 = arith.addf %22, %21 : f32
//CHECK-NEXT:     linalg.yield %23 : f32
//CHECK-NEXT:   }
linalg.conv_2d_nhwc_hwcf {dilations = dense<[2, 3]> : tensor<2xi64>, strides = dense<[2, 3]> : tensor<2xi64>} ins(%h1, %h2 : memref<1x16x16x4xf32>, memref<3x3x4x6xf32>) outs(%h3 : memref<1x6x4x6xf32>)

//CHECK-NEXT:   linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, ((d1 * 2) + (d4 * 2)), ((d2 * 3) + (d5 * 3)), d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%w1, %w2 : memref<1x16x16x4xf32>, memref<6x3x3x4xf32>) outs(%w3 : memref<1x6x4x6xf32>) {
//CHECK-NEXT:   ^bb0(%24: f32, %25: f32, %26: f32):
//CHECK-NEXT:     %27 = arith.mulf %24, %25 : f32
//CHECK-NEXT:     %28 = arith.addf %27, %26 : f32
//CHECK-NEXT:     linalg.yield %28 : f32
//CHECK-NEXT:   }
linalg.conv_2d_nhwc_fhwc {dilations = dense<[2, 3]> : tensor<2xi64>, strides = dense<[2, 3]> : tensor<2xi64>} ins(%w1, %w2 : memref<1x16x16x4xf32>, memref<6x3x3x4xf32>) outs(%w3 : memref<1x6x4x6xf32>)

//CHECK-NEXT:   linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d5, ((d3 * 2) + (d6 * 2)), ((d4 * 3) + (d7 * 3)))>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d2, d1, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%g1, %g2 : memref<1x2x4x16x16xf32>, memref<6x2x4x3x3xf32>) outs(%g3 : memref<1x2x6x6x4xf32>) {
//CHECK-NEXT:   ^bb0(%29: f32, %30: f32, %31: f32):
//CHECK-NEXT:     %32 = arith.mulf %29, %30 : f32
//CHECK-NEXT:     %33 = arith.addf %32, %31 : f32
//CHECK-NEXT:     linalg.yield %33 : f32
//CHECK-NEXT:   }
linalg.conv_2d_ngchw_fgchw {dilations = dense<[2, 3]> : tensor<2xi64>, strides = dense<[2, 3]> : tensor<2xi64>} ins(%g1, %g2 : memref<1x2x4x16x16xf32>, memref<6x2x4x3x3xf32>) outs(%g3 : memref<1x2x6x6x4xf32>)

//CHECK-NEXT:   linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d5, ((d3 * 2) + (d6 * 2)), ((d4 * 3) + (d7 * 3)))>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, d2, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%q1, %q2 : memref<1x2x4x16x16xf32>, memref<2x6x4x3x3xf32>) outs(%q3 : memref<1x2x6x6x4xf32>) {
//CHECK-NEXT:   ^bb0(%34: f32, %35: f32, %36: f32):
//CHECK-NEXT:     %37 = arith.mulf %34, %35 : f32
//CHECK-NEXT:     %38 = arith.addf %37, %36 : f32
//CHECK-NEXT:     linalg.yield %38 : f32
//CHECK-NEXT:   }
linalg.conv_2d_ngchw_gfchw {dilations = dense<[2, 3]> : tensor<2xi64>, strides = dense<[2, 3]> : tensor<2xi64>} ins(%q1, %q2 : memref<1x2x4x16x16xf32>, memref<2x6x4x3x3xf32>) outs(%q3 : memref<1x2x6x6x4xf32>)

//CHECK-NEXT:   linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, ((d1 * 2) + (d5 * 2)), ((d2 * 3) + (d6 * 3)), d3, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%z1, %z2 : memref<1x16x16x2x4xf32>, memref<2x6x3x3x4xf32>) outs(%z3 : memref<1x6x4x2x6xf32>) {
//CHECK-NEXT:   ^bb0(%39: f32, %40: f32, %41: f32):
//CHECK-NEXT:     %42 = arith.mulf %39, %40 : f32
//CHECK-NEXT:     %43 = arith.addf %42, %41 : f32
//CHECK-NEXT:     linalg.yield %43 : f32
//CHECK-NEXT:   }
linalg.conv_2d_nhwgc_gfhwc {dilations = dense<[2, 3]> : tensor<2xi64>, strides = dense<[2, 3]> : tensor<2xi64>} ins(%z1, %z2 : memref<1x16x16x2x4xf32>, memref<2x6x3x3x4xf32>) outs(%z3 : memref<1x6x4x2x6xf32>)

//CHECK-NEXT:   linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, ((d2 * 2) + (d4 * 2)), ((d3 * 3) + (d5 * 3)))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%p1, %p2 : memref<1x4x16x16xf32>, memref<3x3xf32>) outs(%p3 : memref<1x4x6x4xf32>) {
//CHECK-NEXT:   ^bb0(%44: f32, %45: f32, %46: f32):
//CHECK-NEXT:     %47 = arith.maximumf %44, %45 : f32
//CHECK-NEXT:     linalg.yield %47 : f32
//CHECK-NEXT:   }
linalg.pooling_nchw_max {dilations = dense<[2, 3]> : tensor<2xi64>, strides = dense<[2, 3]> : tensor<2xi64>} ins(%p1, %p2 : memref<1x4x16x16xf32>, memref<3x3xf32>) outs(%p3 : memref<1x4x6x4xf32>)

//CHECK-NEXT:   linalg.generic {indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%v : f32) outs(%filled : memref<4x5xf32>) {
//CHECK-NEXT:   ^bb0(%48: f32, %49: f32):
//CHECK-NEXT:     linalg.yield %48 : f32
//CHECK-NEXT:   }
linalg.fill ins(%v : f32) outs(%filled : memref<4x5xf32>)
