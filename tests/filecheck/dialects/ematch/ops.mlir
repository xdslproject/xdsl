// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP


// CHECK:      func.func @main(%val : !pdl.value, %valrange : !pdl.range<value>, %op : !pdl.operation) {
// CHECK-NEXT:   %class_vals = ematch.get_class_vals %val
// CHECK-NEXT:   %representative = ematch.get_class_representative %val
// CHECK-NEXT:   %result = ematch.get_class_result %val
// CHECK-NEXT:   %results = ematch.get_class_results %valrange
// CHECK-NEXT:   ematch.union %val : !pdl.value, %val : !pdl.value
// CHECK-NEXT:   ematch.union %op : !pdl.operation, %valrange : !pdl.range<value>
// CHECK-NEXT:   ematch.union %valrange : !pdl.range<value>, %valrange : !pdl.range<value>
// CHECK-NEXT:   %newop = ematch.dedup %op
// CHECK-NEXT:   func.return
// CHECK-NEXT: }

// CHECK-GENERIC:     "func.func"() <{sym_name = "main", function_type = (!pdl.value, !pdl.range<value>, !pdl.operation) -> ()}> ({
// CHECK-GENERIC-NEXT: ^bb0(%val : !pdl.value, %valrange : !pdl.range<value>, %op : !pdl.operation):
// CHECK-GENERIC-NEXT:   %class_vals = "ematch.get_class_vals"(%val) : (!pdl.value) -> !pdl.range<value>
// CHECK-GENERIC-NEXT:   %representative = "ematch.get_class_representative"(%val) : (!pdl.value) -> !pdl.value
// CHECK-GENERIC-NEXT:   %result = "ematch.get_class_result"(%val) : (!pdl.value) -> !pdl.value
// CHECK-GENERIC-NEXT:   %results = "ematch.get_class_results"(%valrange) : (!pdl.range<value>) -> !pdl.range<value>
// CHECK-GENERIC-NEXT:   "ematch.union"(%val, %val) : (!pdl.value, !pdl.value) -> ()
// CHECK-GENERIC-NEXT:   "ematch.union"(%op, %valrange) : (!pdl.operation, !pdl.range<value>) -> ()
// CHECK-GENERIC-NEXT:   "ematch.union"(%valrange, %valrange) : (!pdl.range<value>, !pdl.range<value>) -> ()
// CHECK-GENERIC-NEXT:   %newop = "ematch.dedup"(%op) : (!pdl.operation) -> !pdl.operation
// CHECK-GENERIC-NEXT:   "func.return"() : () -> ()
// CHECK-GENERIC-NEXT: }) : () -> ()



func.func @main(%val: !pdl.value, %valrange: !pdl.range<value>, %op: !pdl.operation) -> ()
{
    %class_vals = ematch.get_class_vals %val
    
    %representative = ematch.get_class_representative %val
    
    %result = ematch.get_class_result %val
    
    %results = ematch.get_class_results %valrange
    
    ematch.union %val : !pdl.value, %val : !pdl.value
    ematch.union %op : !pdl.operation, %valrange : !pdl.range<value>
    ematch.union %valrange : !pdl.range<value>, %valrange : !pdl.range<value>
    
    %newop = ematch.dedup %op
    
    func.return
}
