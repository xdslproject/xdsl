// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

// CHECK:       builtin.module {
// CHECK-GENERIC:       "builtin.module"() ({


wasm.module()
// CHECK-NEXT:    wasm.module()
// CHECK-GENERIC-NEXT:    "wasm.module"() <{"tables" = [], "mems" = [], "imports" = [], "exports" = []}> : () -> ()

wasm.module() attributes {"hello" = "world"}
// CHECK-NEXT:    wasm.module() attributes {"hello" = "world"}
// CHECK-GENERIC-NEXT:    "wasm.module"() <{"tables" = [], "mems" = [], "imports" = [], "exports" = []}> {"hello" = "world"} : () -> ()

// CHECK-NEXT: wasm.module(
// CHECK-NEXT:    tables [#wasm.table<#wasm<reftype funcref>, #wasm.limits<0 : i32, 8 : i32>>]
// CHECK-NEXT:    mems [#wasm.limits<0 : i32, unit>, #wasm.limits<0 : i32, 4096 : i32>]
// CHECK-NEXT:    imports [#wasm.import<"test_im", #wasm.import_desc_func<#wasm.functype<(i32) -> (i32)>>>]
// CHECK-NEXT:    exports [#wasm.export<"test_ex", #wasm.export_desc_mem<0 : i32>>]
// CHECK-NEXT:  )
// CHECK-GENERIC-NEXT: "wasm.module"() <{"tables" = [#wasm.table<#wasm<reftype funcref>, #wasm.limits<0 : i32, 8 : i32>>], "mems" = [#wasm.limits<0 : i32, unit>, #wasm.limits<0 : i32, 4096 : i32>], "imports" = [#wasm.import<"test_im", #wasm.import_desc_func<#wasm.functype<(i32) -> (i32)>>>], "exports" = [#wasm.export<"test_ex", #wasm.export_desc_mem<0 : i32>>]}> : () -> ()
wasm.module(
  tables [#wasm.table<#wasm<reftype funcref>, #wasm.limits<0 : i32, 8 : i32>>]
  mems [#wasm.limits<0 : i32, unit>, #wasm.limits<0 : i32, 4096 : i32>]
  imports [#wasm.import<"test_im", #wasm.import_desc_func<#wasm.functype<(i32) -> (i32)>>>]
  exports [#wasm.export<"test_ex", #wasm.export_desc_mem<0 : i32>>]
)

// CHECK-NEXT: wasm.module(
// CHECK-NEXT:    tables [#wasm.table<#wasm<reftype funcref>, #wasm.limits<0 : i32, 8 : i32>>]
// CHECK-NEXT:    mems [#wasm.limits<0 : i32, unit>, #wasm.limits<0 : i32, 4096 : i32>]
// CHECK-NEXT:    imports [#wasm.import<"test_im", #wasm.import_desc_func<#wasm.functype<(i32) -> (i32)>>>]
// CHECK-NEXT:    exports [#wasm.export<"test_ex", #wasm.export_desc_mem<0 : i32>>]
// CHECK-NEXT:  )
// CHECK-GENERIC-NEXT: "wasm.module"() <{"tables" = [#wasm.table<#wasm<reftype funcref>, #wasm.limits<0 : i32, 8 : i32>>], "mems" = [#wasm.limits<0 : i32, unit>, #wasm.limits<0 : i32, 4096 : i32>], "imports" = [#wasm.import<"test_im", #wasm.import_desc_func<#wasm.functype<(i32) -> (i32)>>>], "exports" = [#wasm.export<"test_ex", #wasm.export_desc_mem<0 : i32>>]}> : () -> ()
wasm.module(
  imports [#wasm.import<"test_im", #wasm.import_desc_func<#wasm.functype<(i32) -> (i32)>>>]
  tables [#wasm.table<#wasm<reftype funcref>, #wasm.limits<0 : i32, 8 : i32>>] mems [#wasm.limits<0 : i32, unit>, #wasm.limits<0 : i32, 4096 : i32>]
  exports [#wasm.export<"test_ex", #wasm.export_desc_mem<0 : i32>>])

// CHECK-NEXT:  }
// CHECK-GENERIC-NEXT:  }) : () -> ()
