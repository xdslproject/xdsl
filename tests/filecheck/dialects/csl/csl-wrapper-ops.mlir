// RUN: XDSL_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP

builtin.module {
    "csl_wrapper.module"() <{"width"=10 : i16, "height"=10: i16, target = "wse2", "params" = [
        #csl_wrapper.param<"z_dim" default=4: i16>, #csl_wrapper.param<"pattern" : i16>
    ]}> ({
        ^bb0(%x: i16, %y: i16, %width: i16, %height: i16, %z_dim: i16, %pattern: i16):
            %0 = arith.constant 0 : i16
            %1 = "csl.get_color"(%0) : (i16) -> !csl.color

            %routes = "csl_wrapper.import"(%pattern, %width, %height) <{"module" = "routes.csl", "fields" = ["pattern", "peWidth", "peHeight"]}> : (i16, i16, i16) -> !csl.imported_module
            %memcpy = "csl_wrapper.import"(%width, %height, %1) <{"module" = "<memcpy/get_params>", "fields" = ["width", "height", "LAUNCH"]}> : (i16, i16, !csl.color) -> !csl.imported_module

            %compute_all_routes = "csl.member_call"(%routes, %x, %y, %height, %width, %pattern) <{"field" = "computeAllRoutes"}> : (!csl.imported_module, i16, i16, i16, i16, i16) -> !csl.comptime_struct
            %memcpy_params = "csl.member_call"(%memcpy, %x) <{"field" = "get_params"}> : (!csl.imported_module, i16) -> !csl.comptime_struct

            %2 = arith.constant 1 : i16
            %3 = arith.minsi %pattern, %2 : i16
            %4 = arith.minsi %width, %x : i16
            %5 = arith.minsi %height, %y : i16
            %6 = arith.cmpi slt, %x, %3 : i16
            %7 = arith.cmpi slt, %y, %3 : i16
            %8 = arith.cmpi slt, %4, %pattern : i16
            %9 = arith.cmpi slt, %5, %pattern : i16
            %10 = arith.ori %6, %7 : i1
            %11 = arith.ori %10, %8 : i1
            %is_border_region_pe = arith.ori %11, %9 : i1

            "csl_wrapper.yield"(%memcpy_params, %compute_all_routes, %is_border_region_pe) <{"fields" = ["memcpy_params", "stencil_comms_params", "isBorderRegionPE"]}> : (!csl.comptime_struct, !csl.comptime_struct, i1) -> ()
    }, {
        ^bb0(%width: i16, %height: i16, %z_dim: i16, %pattern: i16, %memcpy_params: !csl.comptime_struct, %stencil_comms_params: !csl.comptime_struct, %is_border_region_pe: i1):

            func.func @gauss_seidel () {
                func.return
            }
            "csl_wrapper.yield"() <{"fields" = []}> : () -> ()
    }) : () -> ()
}


// CHECK:      builtin.module {
// CHECK-NEXT:   "csl_wrapper.module"() <{width = 10 : i16, height = 10 : i16, target = "wse2", params = [#csl_wrapper.param<"z_dim" default=4 : i16>, #csl_wrapper.param<"pattern" : i16>]}> ({
// CHECK-NEXT:   ^bb0(%x : i16, %y : i16, %width : i16, %height : i16, %z_dim : i16, %pattern : i16):
// CHECK-NEXT:     %0 = arith.constant 0 : i16
// CHECK-NEXT:     %1 = "csl.get_color"(%0) : (i16) -> !csl.color
// CHECK-NEXT:     %routes = "csl_wrapper.import"(%pattern, %width, %height) <{module = "routes.csl", fields = ["pattern", "peWidth", "peHeight"]}> : (i16, i16, i16) -> !csl.imported_module
// CHECK-NEXT:     %memcpy = "csl_wrapper.import"(%width, %height, %1) <{module = "<memcpy/get_params>", fields = ["width", "height", "LAUNCH"]}> : (i16, i16, !csl.color) -> !csl.imported_module
// CHECK-NEXT:     %compute_all_routes = "csl.member_call"(%routes, %x, %y, %height, %width, %pattern) <{field = "computeAllRoutes"}> : (!csl.imported_module, i16, i16, i16, i16, i16) -> !csl.comptime_struct
// CHECK-NEXT:     %memcpy_params = "csl.member_call"(%memcpy, %x) <{field = "get_params"}> : (!csl.imported_module, i16) -> !csl.comptime_struct
// CHECK-NEXT:     %2 = arith.constant 1 : i16
// CHECK-NEXT:     %3 = arith.minsi %pattern, %2 : i16
// CHECK-NEXT:     %4 = arith.minsi %width, %x : i16
// CHECK-NEXT:     %5 = arith.minsi %height, %y : i16
// CHECK-NEXT:     %6 = arith.cmpi slt, %x, %3 : i16
// CHECK-NEXT:     %7 = arith.cmpi slt, %y, %3 : i16
// CHECK-NEXT:     %8 = arith.cmpi slt, %4, %pattern : i16
// CHECK-NEXT:     %9 = arith.cmpi slt, %5, %pattern : i16
// CHECK-NEXT:     %10 = arith.ori %6, %7 : i1
// CHECK-NEXT:     %11 = arith.ori %10, %8 : i1
// CHECK-NEXT:     %is_border_region_pe = arith.ori %11, %9 : i1
// CHECK-NEXT:     "csl_wrapper.yield"(%memcpy_params, %compute_all_routes, %is_border_region_pe) <{fields = ["memcpy_params", "stencil_comms_params", "isBorderRegionPE"]}> : (!csl.comptime_struct, !csl.comptime_struct, i1) -> ()
// CHECK-NEXT:   }, {
// CHECK-NEXT:   ^bb1(%width_1 : i16, %height_1 : i16, %z_dim_1 : i16, %pattern_1 : i16, %memcpy_params_1 : !csl.comptime_struct, %stencil_comms_params : !csl.comptime_struct, %is_border_region_pe_1 : i1):
// CHECK-NEXT:     func.func @gauss_seidel() {
// CHECK-NEXT:       func.return
// CHECK-NEXT:     }
// CHECK-NEXT:     "csl_wrapper.yield"() <{fields = []}> : () -> ()
// CHECK-NEXT:   }) : () -> ()
// CHECK-NEXT: }


// CHECK-GENERIC:      "builtin.module"() ({
// CHECK-GENERIC-NEXT:   "csl_wrapper.module"() <{width = 10 : i16, height = 10 : i16, target = "wse2", params = [#csl_wrapper.param<"z_dim" default=4 : i16>, #csl_wrapper.param<"pattern" : i16>]}> ({
// CHECK-GENERIC-NEXT:   ^bb0(%x : i16, %y : i16, %width : i16, %height : i16, %z_dim : i16, %pattern : i16):
// CHECK-GENERIC-NEXT:     %0 = "arith.constant"() <{value = 0 : i16}> : () -> i16
// CHECK-GENERIC-NEXT:     %1 = "csl.get_color"(%0) : (i16) -> !csl.color
// CHECK-GENERIC-NEXT:     %routes = "csl_wrapper.import"(%pattern, %width, %height) <{module = "routes.csl", fields = ["pattern", "peWidth", "peHeight"]}> : (i16, i16, i16) -> !csl.imported_module
// CHECK-GENERIC-NEXT:     %memcpy = "csl_wrapper.import"(%width, %height, %1) <{module = "<memcpy/get_params>", fields = ["width", "height", "LAUNCH"]}> : (i16, i16, !csl.color) -> !csl.imported_module
// CHECK-GENERIC-NEXT:     %compute_all_routes = "csl.member_call"(%routes, %x, %y, %height, %width, %pattern) <{field = "computeAllRoutes"}> : (!csl.imported_module, i16, i16, i16, i16, i16) -> !csl.comptime_struct
// CHECK-GENERIC-NEXT:     %memcpy_params = "csl.member_call"(%memcpy, %x) <{field = "get_params"}> : (!csl.imported_module, i16) -> !csl.comptime_struct
// CHECK-GENERIC-NEXT:     %2 = "arith.constant"() <{value = 1 : i16}> : () -> i16
// CHECK-GENERIC-NEXT:     %3 = "arith.minsi"(%pattern, %2) : (i16, i16) -> i16
// CHECK-GENERIC-NEXT:     %4 = "arith.minsi"(%width, %x) : (i16, i16) -> i16
// CHECK-GENERIC-NEXT:     %5 = "arith.minsi"(%height, %y) : (i16, i16) -> i16
// CHECK-GENERIC-NEXT:     %6 = "arith.cmpi"(%x, %3) <{predicate = 2 : i64}> : (i16, i16) -> i1
// CHECK-GENERIC-NEXT:     %7 = "arith.cmpi"(%y, %3) <{predicate = 2 : i64}> : (i16, i16) -> i1
// CHECK-GENERIC-NEXT:     %8 = "arith.cmpi"(%4, %pattern) <{predicate = 2 : i64}> : (i16, i16) -> i1
// CHECK-GENERIC-NEXT:     %9 = "arith.cmpi"(%5, %pattern) <{predicate = 2 : i64}> : (i16, i16) -> i1
// CHECK-GENERIC-NEXT:     %10 = "arith.ori"(%6, %7) : (i1, i1) -> i1
// CHECK-GENERIC-NEXT:     %11 = "arith.ori"(%10, %8) : (i1, i1) -> i1
// CHECK-GENERIC-NEXT:     %is_border_region_pe = "arith.ori"(%11, %9) : (i1, i1) -> i1
// CHECK-GENERIC-NEXT:     "csl_wrapper.yield"(%memcpy_params, %compute_all_routes, %is_border_region_pe) <{fields = ["memcpy_params", "stencil_comms_params", "isBorderRegionPE"]}> : (!csl.comptime_struct, !csl.comptime_struct, i1) -> ()
// CHECK-GENERIC-NEXT:   }, {
// CHECK-GENERIC-NEXT:   ^bb1(%width_1 : i16, %height_1 : i16, %z_dim_1 : i16, %pattern_1 : i16, %memcpy_params_1 : !csl.comptime_struct, %stencil_comms_params : !csl.comptime_struct, %is_border_region_pe_1 : i1):
// CHECK-GENERIC-NEXT:     "func.func"() <{sym_name = "gauss_seidel", function_type = () -> ()}> ({
// CHECK-GENERIC-NEXT:       "func.return"() : () -> ()
// CHECK-GENERIC-NEXT:     }) : () -> ()
// CHECK-GENERIC-NEXT:     "csl_wrapper.yield"() <{fields = []}> : () -> ()
// CHECK-GENERIC-NEXT:   }) : () -> ()
// CHECK-GENERIC-NEXT: }) : () -> ()
