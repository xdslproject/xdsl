"builtin.module"() ({
  %0:3 = "test.op"() : () -> (f64, index, memref<2xf64, strided<[?]>>)
  "memref.store"(%0#0, %0#2, %0#1) <{nontemporal = false}> : (f64, memref<2xf64, strided<[?]>>, index) -> ()
}) : () -> ()
