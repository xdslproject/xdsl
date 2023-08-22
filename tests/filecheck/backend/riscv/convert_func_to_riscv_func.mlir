// RUN: xdsl-opt -p lower-func-to-riscv-func --split-input-file  %s | filecheck %s

builtin.module {
    func.func @main() {
        %0, %1 = "test.op"() : () -> (i32, i32)
        %2, %3 = func.call @foo(%0, %1) : (i32, i32) -> (i32, i32)
        func.return
    }

    func.func @foo(%arg0 : i32, %arg1 : i32) -> (i32, i32) {
        %res0, %res1 = "test.op"(%arg0, %arg1) : (i32, i32) -> (i32, i32)
        func.return %res0, %res1 : i32, i32
    }
}

// CHECK:       builtin.module {
// CHECK-NEXT:      riscv_func.func @main() {
// CHECK-NEXT:          %0, %1 = "test.op"() : () -> (i32, i32)
// CHECK-NEXT:          %{{.*}} = builtin.unrealized_conversion_cast %0 : i32 to !riscv.reg<>
// CHECK-NEXT:          %{{.*}} = builtin.unrealized_conversion_cast %1 : i32 to !riscv.reg<>
// CHECK-NEXT:          %{{.*}} = riscv.mv %{{.*}} : (!riscv.reg<>) -> !riscv.reg<a0>
// CHECK-NEXT:          %{{.*}} = riscv.mv %{{.*}} : (!riscv.reg<>) -> !riscv.reg<a1>
// CHECK-NEXT:          %{{.*}}, %{{.*}} = riscv_func.call @foo(%{{.*}}, %{{.*}}) : (!riscv.reg<a0>, !riscv.reg<a1>) -> (!riscv.reg<a0>, !riscv.reg<a1>)
// CHECK-NEXT:          %{{.*}} = riscv.mv %{{.*}} : (!riscv.reg<a0>) -> !riscv.reg<>
// CHECK-NEXT:          %{{.*}} = riscv.mv %{{.*}} : (!riscv.reg<a1>) -> !riscv.reg<>
// CHECK-NEXT:          %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : !riscv.reg<> to i32
// CHECK-NEXT:          %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : !riscv.reg<> to i32
// CHECK-NEXT:          riscv_func.return
// CHECK-NEXT:      }
// CHECK-NEXT:      riscv_func.func @foo(%arg0 : !riscv.reg<a0>, %arg1 : !riscv.reg<a1>) -> (!riscv.reg<a0>, !riscv.reg<a1>) {
// CHECK-NEXT:        %{{.*}} = riscv.mv %arg0 : (!riscv.reg<a0>) -> !riscv.reg<>
// CHECK-NEXT:        %arg0_1 = builtin.unrealized_conversion_cast %{{.*}} : !riscv.reg<> to i32
// CHECK-NEXT:        %{{.*}} = riscv.mv %arg1 : (!riscv.reg<a1>) -> !riscv.reg<>
// CHECK-NEXT:        %arg1_1 = builtin.unrealized_conversion_cast %{{.*}} : !riscv.reg<> to i32
// CHECK-NEXT:        %res0, %res1 = "test.op"(%arg0_1, %arg1_1) : (i32, i32) -> (i32, i32)
// CHECK-NEXT:        %{{.*}} = builtin.unrealized_conversion_cast %res0 : i32 to !riscv.reg<>
// CHECK-NEXT:        %{{.*}} = builtin.unrealized_conversion_cast %res1 : i32 to !riscv.reg<>
// CHECK-NEXT:        %{{.*}} = riscv.mv %{{.*}} : (!riscv.reg<>) -> !riscv.reg<a0>
// CHECK-NEXT:        %{{.*}} = riscv.mv %{{.*}} : (!riscv.reg<>) -> !riscv.reg<a1>
// CHECK-NEXT:        riscv_func.return %{{.*}}, %{{.*}} : !riscv.reg<a0>, !riscv.reg<a1>
// CHECK-NEXT:      }
// CHECK-NEXT:  }
