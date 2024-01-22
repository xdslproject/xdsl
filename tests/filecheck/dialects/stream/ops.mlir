// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

// CHECK: builtin.module {


%readable_stream = "test.op"() : () -> !stream.readable<index>
// CHECK-NEXT: %readable_stream = "test.op"() : () -> !stream.readable<index>

%writable_stream = "test.op"() : () -> !stream.writable<index>
// CHECK-NEXT: %writable_stream = "test.op"() : () -> !stream.writable<index>

%value = stream.read from %readable_stream : index
// CHECK-NEXT: %value = stream.read from %readable_stream : index

stream.write %value to %writable_stream : index
// CHECK-NEXT: stream.write %value to %writable_stream : index


// CHECK-NEXT: }



// CHECK-GENERIC:       "builtin.module"() ({
// CHECK-NEXT-GENERIC:    %readable_stream = "test.op"() : () -> !stream.readable<index>
// CHECK-NEXT-GENERIC:    %writable_stream = "test.op"() : () -> !stream.writable<index>
// CHECK-NEXT-GENERIC:    "stream.read"(%readable_stream) : (!stream.readable<index>) -> index
// CHECK-NEXT-GENERIC:    "stream.write"(%value, %writable_stream) : (index, !stream.writable<index>) -> ()
// CHECK-NEXT-GENERIC:  }) : () -> ()
