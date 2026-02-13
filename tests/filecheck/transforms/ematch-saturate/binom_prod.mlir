// RUN: xdsl-opt -p 'ematch-saturate{max_iterations=4 pdl_file="%p/binom_prod_pdl_interp.mlir"}' %s

func.func @product_of_binomials(%0 : f32) -> f32 {
    %res = equivalence.graph %0 : (f32) -> f32 {
    ^bb0(%a: f32):
        %2 = arith.constant 3.000000e+00 : f32
        %4 = arith.addf %a, %2 : f32
        %6 = arith.constant 1.000000e+00 : f32
        %8 = arith.addf %a, %6 : f32
        %10 = arith.mulf %4, %8 : f32
        equivalence.yield %10 : f32 // (a + 3) * (a + 1) 
    }
    func.return %res : f32
}


// CHECK:      func.func @product_of_binomials(%0 : f32) -> f32 {
// CHECK-NEXT:   %res = equivalence.graph %0 : (f32) -> f32 {
// CHECK-NEXT:   ^bb0(%a : f32):
// CHECK-NEXT:     %1 = arith.constant 3.000000e+00 : f32
// CHECK-NEXT:     %2 = arith.addf %1, %3 : f32
// CHECK-NEXT:     %4 = arith.addf %3, %1 : f32
// CHECK-NEXT:     %5 = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:     %6 = arith.addf %7, %3 : f32
// CHECK-NEXT:     %8 = arith.addf %3, %7 : f32
// CHECK-NEXT:     %9 = arith.addf %10, %3 : f32
// CHECK-NEXT:     %11 = arith.addf %3, %10 : f32
// CHECK-NEXT:     %12 = arith.mulf %3, %13 : f32
// CHECK-NEXT:     %14 = equivalence.class %15, %12, %9, %11 : f32
// CHECK-NEXT:     %16 = arith.mulf %1, %3 : f32
// CHECK-NEXT:     %17 = arith.mulf %1, %7 : f32
// CHECK-NEXT:     %18 = arith.addf %19, %20 : f32
// CHECK-NEXT:     %21 = arith.addf %20, %19 : f32
// CHECK-NEXT:     %22 = arith.mulf %1, %13 : f32
// CHECK-NEXT:     %23 = equivalence.class %24, %22, %18, %21 : f32
// CHECK-NEXT:     %24 = arith.mulf %13, %1 : f32
// CHECK-NEXT:     %25 = arith.addf %14, %23 : f32
// CHECK-NEXT:     %26 = arith.addf %23, %14 : f32
// CHECK-NEXT:     %27 = arith.mulf %7, %28 : f32
// CHECK-NEXT:     %29 = arith.mulf %28, %7 : f32
// CHECK-NEXT:     %30 = arith.mulf %7, %13 : f32
// CHECK-NEXT:     %13 = equivalence.class %31, %30, %8, %6 : f32
// CHECK-NEXT:     %31 = arith.mulf %13, %7 : f32
// CHECK-NEXT:     %32 = arith.mulf %13, %33 : f32
// CHECK-NEXT:     %15 = arith.mulf %13, %3 : f32
// CHECK-NEXT:     %34 = arith.mulf %13, %20 : f32
// CHECK-NEXT:     %35 = arith.addf %14, %34 : f32
// CHECK-NEXT:     %36 = arith.addf %34, %14 : f32
// CHECK-NEXT:     %28 = equivalence.class %37, %32, %38, %25, %26, %39, %29, %40, %41, %27, %35, %36, %42, %43, %44, %45, %46, %47 : f32
// CHECK-NEXT:     %48 = arith.mulf %7, %49 : f32
// CHECK-NEXT:     %50 = arith.mulf %49, %7 : f32
// CHECK-NEXT:     %3 = equivalence.class %51, %52, %a : f32
// CHECK-NEXT:     %51 = arith.mulf %3, %7 : f32
// CHECK-NEXT:     %53 = arith.mulf %3, %33 : f32
// CHECK-NEXT:     %54 = arith.mulf %3, %20 : f32
// CHECK-NEXT:     %55 = arith.addf %10, %54 : f32
// CHECK-NEXT:     %56 = arith.addf %54, %10 : f32
// CHECK-NEXT:     %10 = arith.mulf %3, %3 : f32
// CHECK-NEXT:     %19 = equivalence.class %57, %16 : f32
// CHECK-NEXT:     %57 = arith.mulf %3, %1 : f32
// CHECK-NEXT:     %58 = arith.addf %10, %19 : f32
// CHECK-NEXT:     %59 = arith.addf %19, %10 : f32
// CHECK-NEXT:     %49 = equivalence.class %60, %53, %50, %58, %59, %48, %55, %56 : f32
// CHECK-NEXT:     %60 = arith.mulf %33, %3 : f32
// CHECK-NEXT:     %7 = equivalence.class %61, %5 : f32
// CHECK-NEXT:     %61 = arith.mulf %7, %7 : f32
// CHECK-NEXT:     %62 = arith.mulf %7, %20 : f32
// CHECK-NEXT:     %63 = arith.addf %3, %62 : f32
// CHECK-NEXT:     %64 = arith.addf %62, %3 : f32
// CHECK-NEXT:     %52 = arith.mulf %7, %3 : f32
// CHECK-NEXT:     %20 = equivalence.class %65, %17 : f32
// CHECK-NEXT:     %65 = arith.mulf %7, %1 : f32
// CHECK-NEXT:     %66 = arith.addf %3, %20 : f32
// CHECK-NEXT:     %67 = arith.addf %20, %3 : f32
// CHECK-NEXT:     %68 = arith.mulf %7, %33 : f32
// CHECK-NEXT:     %33 = equivalence.class %69, %68, %4, %2, %66, %67, %63, %64 : f32
// CHECK-NEXT:     %69 = arith.mulf %33, %7 : f32
// CHECK-NEXT:     %70 = arith.addf %33, %10 : f32
// CHECK-NEXT:     %42 = arith.addf %70, %19 : f32
// CHECK-NEXT:     %71 = arith.addf %33, %19 : f32
// CHECK-NEXT:     %43 = arith.addf %71, %10 : f32
// CHECK-NEXT:     %39 = arith.addf %33, %49 : f32
// CHECK-NEXT:     %72 = arith.addf %3, %49 : f32
// CHECK-NEXT:     %73 = equivalence.class %74, %72 : f32
// CHECK-NEXT:     %74 = arith.addf %49, %3 : f32
// CHECK-NEXT:     %44 = arith.addf %1, %73 : f32
// CHECK-NEXT:     %40 = arith.addf %73, %1 : f32
// CHECK-NEXT:     %75 = arith.addf %1, %49 : f32
// CHECK-NEXT:     %76 = equivalence.class %77, %75 : f32
// CHECK-NEXT:     %77 = arith.addf %49, %1 : f32
// CHECK-NEXT:     %45 = arith.addf %3, %76 : f32
// CHECK-NEXT:     %41 = arith.addf %76, %3 : f32
// CHECK-NEXT:     %46 = arith.addf %73, %20 : f32
// CHECK-NEXT:     %78 = arith.addf %49, %20 : f32
// CHECK-NEXT:     %47 = arith.addf %78, %3 : f32
// CHECK-NEXT:     %38 = arith.addf %49, %33 : f32
// CHECK-NEXT:     %37 = arith.mulf %33, %13 : f32
// CHECK-NEXT:     equivalence.yield %28 : f32
// CHECK-NEXT:   }
// CHECK-NEXT:   func.return %res : f32
// CHECK-NEXT: }
