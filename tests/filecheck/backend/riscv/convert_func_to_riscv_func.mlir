// RUN: xdsl-opt -p convert-func-to-riscv-func --split-input-file  %s | filecheck %s

builtin.module {
    func.func @main() {
        %0, %1 = "test.op"() : () -> (i32, f32)
        %2, %3 = func.call @foo(%0, %1) : (i32, f32) -> (i64, f64)
        func.return
    }

    func.func @foo(%arg0 : i32, %arg1 : f32) -> (i64, f64) {
        %res0, %res1 = "test.op"(%arg0, %arg1) : (i32, f32) -> (i64, f64)
        func.return %res0, %res1 : i64, f64
    }

    func.func @foo_float(%farg0 : f32, %farg1 : f32) -> (f32, f32) {
        %fres0, %fres1 = "test.op"(%farg0, %farg1) : (f32, f32) -> (f32, f32)
        func.return %fres0, %fres1 : f32, f32
    }

    func.func @foo_int_float(%arg0 : i32, %farg0 : f32) -> (f32, i32) {
        %fres0, %res0 = "test.op"(%arg0, %farg0) : (i32, f32) -> (f32, i32)
        func.return %fres0, %res0 : f32, i32
    }

    func.func @foo_int_double(%arg0 : i32, %farg0 : f64) -> (f64, i32) {
        %fres0, %res0 = "test.op"(%arg0, %farg0) : (i32, f64) -> (f64, i32)
        func.return %fres0, %res0 : f64, i32
    }

    func.func private @private_func() {
        func.return
    }

    func.func @external(%arg0 : i32, %farg0 : f64) -> (f64, i32)
}

// CHECK:       builtin.module {
// CHECK-NEXT:    riscv_func.func public @main() attributes {p2align = 2 : i8} {
// CHECK-NEXT:        %0, %1 = "test.op"() : () -> (i32, f32)
// CHECK-NEXT:        %{{.*}} = builtin.unrealized_conversion_cast %0 : i32 to !riscv.reg
// CHECK-NEXT:        %{{.*}} = builtin.unrealized_conversion_cast %1 : f32 to !riscv.freg
// CHECK-NEXT:        %{{.*}} = riscv.mv %{{.*}} : (!riscv.reg) -> !riscv.reg<a0>
// CHECK-NEXT:        %{{.*}} = riscv.fmv.s %{{.*}} : (!riscv.freg) -> !riscv.freg<fa0>
// CHECK-NEXT:        %{{.*}}, %{{.*}} = riscv_func.call @foo(%{{.*}}, %{{.*}}) : (!riscv.reg<a0>, !riscv.freg<fa0>) -> (!riscv.reg<a0>, !riscv.freg<fa0>)
// CHECK-NEXT:        %{{.*}}, %{{.*}} = riscv.parallel_mov %{{.*}}, %{{.*}} : (!riscv.reg<a0>, !riscv.freg<fa0>) -> (!riscv.reg, !riscv.freg)
// CHECK-NEXT:        %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : !riscv.reg to i64
// CHECK-NEXT:        %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : !riscv.freg to f64
// CHECK-NEXT:        riscv_func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    riscv_func.func public @foo(%arg0 : !riscv.reg<a0>, %arg1 : !riscv.freg<fa0>) -> (!riscv.reg<a0>, !riscv.freg<fa0>) attributes {p2align = 2 : i8} {
// CHECK-NEXT:      %{{.*}} = riscv.mv %arg0 : (!riscv.reg<a0>) -> !riscv.reg
// CHECK-NEXT:      %arg0_1 = builtin.unrealized_conversion_cast %{{.*}} : !riscv.reg to i32
// CHECK-NEXT:      %{{.*}} = riscv.fmv.s %arg1 : (!riscv.freg<fa0>) -> !riscv.freg
// CHECK-NEXT:      %arg1_1 = builtin.unrealized_conversion_cast %{{.*}} : !riscv.freg to f32
// CHECK-NEXT:      %res0, %res1 = "test.op"(%arg0_1, %arg1_1) : (i32, f32) -> (i64, f64)
// CHECK-NEXT:      %{{.*}} = builtin.unrealized_conversion_cast %res0 : i64 to !riscv.reg
// CHECK-NEXT:      %{{.*}} = builtin.unrealized_conversion_cast %res1 : f64 to !riscv.freg
// CHECK-NEXT:      %{{.*}} = riscv.mv %{{.*}} : (!riscv.reg) -> !riscv.reg<a0>
// CHECK-NEXT:      %{{.*}} = riscv.fmv.d %{{.*}} : (!riscv.freg) -> !riscv.freg<fa0>
// CHECK-NEXT:      riscv_func.return %{{.*}}, %{{.*}} : !riscv.reg<a0>, !riscv.freg<fa0>
// CHECK-NEXT:    }
// CHECK-NEXT:    riscv_func.func public @foo_float(%farg0 : !riscv.freg<fa0>, %farg1 : !riscv.freg<fa1>) -> (!riscv.freg<fa0>, !riscv.freg<fa1>) attributes {p2align = 2 : i8} {
// CHECK-NEXT:      %{{.*}} = riscv.fmv.s %farg0 : (!riscv.freg<fa0>) -> !riscv.freg
// CHECK-NEXT:      %farg0_1 = builtin.unrealized_conversion_cast %{{.*}} : !riscv.freg to f32
// CHECK-NEXT:      %{{.*}} = riscv.fmv.s %farg1 : (!riscv.freg<fa1>) -> !riscv.freg
// CHECK-NEXT:      %farg1_1 = builtin.unrealized_conversion_cast %{{.*}} : !riscv.freg to f32
// CHECK-NEXT:      %fres0, %fres1 = "test.op"(%farg0_1, %farg1_1) : (f32, f32) -> (f32, f32)
// CHECK-NEXT:      %{{.*}} = builtin.unrealized_conversion_cast %fres0 : f32 to !riscv.freg
// CHECK-NEXT:      %{{.*}} = builtin.unrealized_conversion_cast %fres1 : f32 to !riscv.freg
// CHECK-NEXT:      %{{.*}} = riscv.fmv.s %{{.*}} : (!riscv.freg) -> !riscv.freg<fa0>
// CHECK-NEXT:      %{{.*}} = riscv.fmv.s %{{.*}} : (!riscv.freg) -> !riscv.freg<fa1>
// CHECK-NEXT:      riscv_func.return %{{.*}}, %{{.*}} : !riscv.freg<fa0>, !riscv.freg<fa1>
// CHECK-NEXT:    }
// CHECK-NEXT:    riscv_func.func public @foo_int_float(%arg0 : !riscv.reg<a0>, %farg0 : !riscv.freg<fa0>) -> (!riscv.freg<fa0>, !riscv.reg<a0>) attributes {p2align = 2 : i8} {
// CHECK-NEXT:      %{{.*}} = riscv.mv %arg0 : (!riscv.reg<a0>) -> !riscv.reg
// CHECK-NEXT:      %arg0_1 = builtin.unrealized_conversion_cast %{{.*}} : !riscv.reg to i32
// CHECK-NEXT:      %{{.*}} = riscv.fmv.s %farg0 : (!riscv.freg<fa0>) -> !riscv.freg
// CHECK-NEXT:      %farg0_1 = builtin.unrealized_conversion_cast %{{.*}} : !riscv.freg to f32
// CHECK-NEXT:      %fres0, %res0 = "test.op"(%arg0_1, %farg0_1) : (i32, f32) -> (f32, i32)
// CHECK-NEXT:      %{{.*}} = builtin.unrealized_conversion_cast %fres0 : f32 to !riscv.freg
// CHECK-NEXT:      %{{.*}} = builtin.unrealized_conversion_cast %res0 : i32 to !riscv.reg
// CHECK-NEXT:      %{{.*}} = riscv.fmv.s %{{.*}} : (!riscv.freg) -> !riscv.freg<fa0>
// CHECK-NEXT:      %{{.*}} = riscv.mv %{{.*}} : (!riscv.reg) -> !riscv.reg<a0>
// CHECK-NEXT:      riscv_func.return %{{.*}}, %{{.*}} : !riscv.freg<fa0>, !riscv.reg<a0>
// CHECK-NEXT:    }
// CHECK-NEXT:    riscv_func.func public @foo_int_double(%arg0 : !riscv.reg<a0>, %farg0 : !riscv.freg<fa0>) -> (!riscv.freg<fa0>, !riscv.reg<a0>) attributes {p2align = 2 : i8} {
// CHECK-NEXT:      %{{.*}} = riscv.mv %arg0 : (!riscv.reg<a0>) -> !riscv.reg
// CHECK-NEXT:      %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : !riscv.reg to i32
// CHECK-NEXT:      %{{.*}} = riscv.fmv.d %{{.*}} : (!riscv.freg<fa0>) -> !riscv.freg
// CHECK-NEXT:      %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : !riscv.freg to f64
// CHECK-NEXT:      %{{.*}}, %{{.*}} = "test.op"(%{{.*}}, %{{.*}}) : (i32, f64) -> (f64, i32)
// CHECK-NEXT:      %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : f64 to !riscv.freg
// CHECK-NEXT:      %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : i32 to !riscv.reg
// CHECK-NEXT:      %{{.*}} = riscv.fmv.d %{{.*}} : (!riscv.freg) -> !riscv.freg<fa0>
// CHECK-NEXT:      %{{.*}} = riscv.mv %{{.*}} : (!riscv.reg) -> !riscv.reg<a0>
// CHECK-NEXT:      riscv_func.return %{{.*}}, %{{.*}} : !riscv.freg<fa0>, !riscv.reg<a0>
// CHECK-NEXT:    }
// CHECK-NEXT:    riscv_func.func private @private_func() attributes {p2align = 2 : i8} {
// CHECK-NEXT:      riscv_func.return
// CHECK-NEXT:    }
// CHECK-NEXT:    riscv_func.func public @external(!riscv.reg<a0>, !riscv.freg<fa0>) -> (!riscv.freg<fa0>, !riscv.reg<a0>) attributes {p2align = 2 : i8}
// CHECK-NEXT:  }
