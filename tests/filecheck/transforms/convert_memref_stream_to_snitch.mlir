// RUN: xdsl-opt -p convert-memref-stream-to-snitch %s | filecheck %s

%readable, %writable = "test.op"() : () -> (!stream.readable<f32>, !stream.writable<f32>)

%val = memref_stream.read from %readable : f32
memref_stream.write %val to %writable : f32

// CHECK:       builtin.module {
// CHECK-NEXT:    %readable, %writable = "test.op"() : () -> (!stream.readable<f32>, !stream.writable<f32>)
// CHECK-NEXT:    %val = builtin.unrealized_conversion_cast %readable : !stream.readable<f32> to !stream.readable<!riscv.freg<>>
// CHECK-NEXT:    %{{.*}} = riscv_snitch.read from %val : !riscv.freg<>
// CHECK-NEXT:    %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : !riscv.freg<> to f32
// CHECK-NEXT:    %{{.*}} = builtin.unrealized_conversion_cast %writable : !stream.writable<f32> to !stream.writable<!riscv.freg<>>
// CHECK-NEXT:    %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : f32 to !riscv.freg<>
// CHECK-NEXT:    riscv_snitch.write %{{.*}} to %{{.*}} : !riscv.freg<>
// CHECK-NEXT:  }
