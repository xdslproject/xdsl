// RUN: xdsl-opt --verify-diagnostic %s | filecheck %s

"builtin.module"() ({
    %0 = "arith.constant"() {"value" = 10 : index} : () -> index
    %gdmemref = "gpu.alloc"(%0, %0,%0) {"operand_segment_sizes" = array<i32: 0, 3, 0>}: () -> memref<10x10x10xf64>
}) : () -> ()

// CHECK-NEXT: Expected 0 dynamic sizes, got 3. All dynamic sizes need to be set in the alloc operation.
