// RUN: xdsl-opt %s -p 'builtin-to-csl{pe_program="pe.csl"}' | filecheck %s

builtin.module {
  func.func @some_function(%x: i32, %y: f32) -> i32 {
    func.return %x : i32
  }
}

// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   "csl.module"() <{"kind" = #csl<module_kind program>}> ({
// CHECK-NEXT:     csl.func @some_function(%x : i32, %y : f32) -> i32 {
// CHECK-NEXT:       csl.return %x : i32
// CHECK-NEXT:     }
// CHECK-NEXT:   }) {"sym_name" = "program"} : () -> ()
// CHECK-NEXT:   "csl.module"() <{"kind" = #csl<module_kind layout>}> ({
// CHECK-NEXT:     %LAUNCH = "csl.get_color"() <{"id" = 0 : i5}> : () -> !csl.color
// CHECK-NEXT:     %memcpy_init_params = "csl.const_struct"(%LAUNCH) <{"items" = {"width" = 1 : i32, "height" = 1 : i32}, "ssa_fields" = ["LAUNCH"]}> : (!csl.color) -> !csl.comptime_struct
// CHECK-NEXT:     %memcpy = "csl.import_module"(%memcpy_init_params) <{"module" = "<memcpy/get_params>"}> : (!csl.comptime_struct) -> !csl.imported_module
// CHECK-NEXT:     csl.layout {
// CHECK-NEXT:       %x_dim_idx = arith.constant 1 : index
// CHECK-NEXT:       %y_dim_idx = arith.constant 1 : index
// CHECK-NEXT:       %x_dim = arith.index_cast %x_dim_idx : index to i32
// CHECK-NEXT:       %y_dim = arith.index_cast %y_dim_idx : index to i32
// CHECK-NEXT:       "csl.set_rectangle"(%x_dim, %y_dim) : (i32, i32) -> ()
// CHECK-NEXT:       %c0 = arith.constant 0 : index
// CHECK-NEXT:       %c1 = arith.constant 1 : index
// CHECK-NEXT:       scf.for %x_coord_idx = %c0 to %x_dim_idx step %c1 {
// CHECK-NEXT:         scf.for %y_coord_idx = %c0 to %y_dim_idx step %c1 {
// CHECK-NEXT:           %x_coord = arith.index_cast %x_coord_idx : index to i32
// CHECK-NEXT:           %y_coord = arith.index_cast %y_coord_idx : index to i32
// CHECK-NEXT:           %memcpy_params = "csl.member_call"(%memcpy, %x_coord) <{"field" = "get_params"}> : (!csl.imported_module, i32) -> !csl.comptime_struct
// CHECK-NEXT:           %tile_code_params = "csl.const_struct"(%memcpy_params) <{"ssa_fields" = ["memcpy_params"]}> : (!csl.comptime_struct) -> !csl.comptime_struct
// CHECK-NEXT:           "csl.set_tile_code"(%x_coord, %y_coord, %tile_code_params) <{"file" = "pe.csl"}> : (i32, i32, !csl.comptime_struct) -> ()
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }) {"sym_name" = "layout"} : () -> ()
// CHECK-NEXT: }
