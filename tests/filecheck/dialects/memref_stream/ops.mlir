// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

%readable, %writable = "test.op"() : () -> (!stream.readable<f32>, !stream.writable<f32>)

%val = memref_stream.read from %readable : f32
memref_stream.write %val to %writable : f32

// CHECK:       builtin.module {
// CHECK-NEXT:    %readable, %writable = "test.op"() : () -> (!stream.readable<f32>, !stream.writable<f32>)
// CHECK-NEXT:    %val = memref_stream.read from %readable : f32
// CHECK-NEXT:    memref_stream.write %val to %writable : f32
// CHECK-NEXT:  }

// CHECK-GENERIC:       "builtin.module"() ({
// CHECK-GENERIC-NEXT:    %readable, %writable = "test.op"() : () -> (!stream.readable<f32>, !stream.writable<f32>)
// CHECK-GENERIC-NEXT:    %val = "memref_stream.read"(%readable) : (!stream.readable<f32>) -> f32
// CHECK-GENERIC-NEXT:    "memref_stream.write"(%val, %writable) : (f32, !stream.writable<f32>) -> ()
// CHECK-GENERIC-NEXT:  }) : () -> ()
