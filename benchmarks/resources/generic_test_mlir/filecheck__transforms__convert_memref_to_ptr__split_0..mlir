"builtin.module"() ({
  %0:3 = "test.op"() : () -> (i32, index, memref<10xi32>)
  "memref.store"(%0#0, %0#2, %0#1) <{nontemporal = false}> : (i32, memref<10xi32>, index) -> ()
  %1:3 = "test.op"() : () -> (index, index, memref<10x10xi32>)
  "memref.store"(%0#0, %1#2, %1#0, %1#1) <{nontemporal = false}> : (i32, memref<10x10xi32>, index, index) -> ()
  %2 = "memref.load"(%0#2, %0#1) <{nontemporal = false}> : (memref<10xi32>, index) -> i32
  %3 = "memref.load"(%1#2, %1#0, %1#1) <{nontemporal = false}> : (memref<10x10xi32>, index, index) -> i32
  %4:2 = "test.op"() : () -> (f64, memref<10xf64>)
  "memref.store"(%4#0, %4#1, %0#1) <{nontemporal = false}> : (f64, memref<10xf64>, index) -> ()
  %5 = "memref.load"(%4#1, %0#1) <{nontemporal = false}> : (memref<10xf64>, index) -> f64
  %6 = "test.op"() : () -> memref<f64>
  %7 = "memref.load"(%6) <{nontemporal = false}> : (memref<f64>) -> f64
}) : () -> ()
