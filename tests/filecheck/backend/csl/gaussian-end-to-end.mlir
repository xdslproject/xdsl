// RUN: xdsl-opt -t csl %f | filecheck %f

builtin.module {
  // program:
  "csl.module"() <{kind=#csl<module_kind program>}> ({
    %math = "csl.import_module"() <{module = "<math>"}> : () -> !csl.imported_module
  }) {sym_name = "pe.csl"} : () -> ()

  // layout
  "csl.module"() <{kind=#csl<module_kind layout>}> ({
    %LAUNCH_ID = arith.constant 0 : i16

    %LAUNCH = "csl.get_color"(%LAUNCH_ID) : (i16) -> !csl.color

    %width = "csl.param"() <{param_name = "width"}> : () -> ui16
    %height = "csl.param"() <{param_name = "height"}> : () -> ui16
    %zDim = "csl.param"() <{param_name = "zDim"}> : () -> ui16



    %task_three = "test.op"() {"comment" = "@get_local_task_id(3)"} : () -> i32  // is supposed to be task_id
    %iterationTaskId = "csl.param"(%task_three) <{param_name = "iterationTaskId"}> : (i32) -> i32
    %pattern = "csl.param"() <{param_name = "pattern"}> : () -> ui16

    %invariants = "csl.const_struct"(%iterationTaskId, %zDim, %pattern) <{ssa_fields = ["iterationTaskId","zDim","pattern"]}> : (i32, ui16, ui16) -> !csl.comptime_struct

    %util = "csl.import_module"() <{module = "util.csl"}> : () -> !csl.imported_module

    %memcpy_call_params = "csl.const_struct"(%width, %height, %LAUNCH) <{ssa_fields = ["width","height","LAUNCH"]}> : (ui16, ui16, !csl.color) -> !csl.comptime_struct

    %memcpy = "csl.import_module"(%memcpy_call_params) <{module = "<memcpy/get_params>"}> : (!csl.comptime_struct) -> !csl.imported_module


    %routes_params = "csl.const_struct"(%width, %height, %pattern) <{ssa_fields = ["peWidth","peHeight","pattern"]}> : (ui16, ui16, ui16) -> !csl.comptime_struct

    %routes = "csl.import_module"(%routes_params) <{module = "routes.csl"}> : (!csl.comptime_struct) -> !csl.imported_module

    csl.layout {
        "csl.set_rectangle"(%width, %height) : (ui16, ui16) -> ()

        %width_i16 = csl.mlir.signedness_cast %width : ui16 to i16
        %height_i16 = csl.mlir.signedness_cast %height : ui16 to i16
        %pattern_i16 = csl.mlir.signedness_cast %pattern : ui16 to i16
        %cst0 = arith.constant 0 : i16
        %cst1 = arith.constant 1 : i16

        %pattern_sub_one = arith.subi %pattern_i16, %cst1 : i16
        %width_sub_pattern = arith.subi %width_i16, %pattern_i16 : i16
        %height_sub_pattern = arith.subi %width_i16, %pattern_i16 : i16

        scf.for %xId = %cst0 to %width_i16 step %cst1 : i16 {
            scf.for %yId = %cst0 to %height_i16 step %cst1 : i16 {
                // compute this: (xId < pattern - 1) or (yId < pattern - 1) or (width - xId < pattern) or (height - yId < pattern)
                // we rewrite (width - xId < pattern) as (width - pattern < xId)
                %c1 = arith.cmpi slt, %xId, %pattern_sub_one : i16
                %c2 = arith.cmpi slt, %yId, %pattern_sub_one : i16
                %c3 = arith.cmpi slt, %width_sub_pattern, %xId : i16
                %c4 = arith.cmpi slt, %height_sub_pattern, %yId : i16
                %c1_c2 = arith.andi %c1, %c2 : i1
                %c3_c4 = arith.andi %c3, %c4 : i1
                %is_border_region = arith.andi %c1_c2, %c3_c4 : i1

                %tmp_struct = "csl.const_struct"(%is_border_region) <{ssa_fields = ["isBorderRegionPE"]}> : (i1) -> !csl.comptime_struct

                %params_task = "csl.concat_struct"(%invariants, %tmp_struct) : (!csl.comptime_struct, !csl.comptime_struct) -> !csl.comptime_struct

                %memcpy_params = "csl.member_call"(%memcpy, %xId) <{field = "get_params"}> : (!csl.imported_module, i16) -> !csl.comptime_struct

                %route_params = "csl.member_call"(%routes, %xId, %yId, %width, %height, %pattern) <{field = "computeAllRoutes"}> : (!csl.imported_module, i16, i16, ui16, ui16, ui16) -> !csl.comptime_struct

                %additional_params = "csl.const_struct"(%memcpy_params, %route_params) <{ssa_fields = ["memcpyParams", "stencilCommsParams"]}> : (!csl.comptime_struct, !csl.comptime_struct) -> !csl.comptime_struct

                %concat_params = "csl.concat_struct"(%params_task, %additional_params) : (!csl.comptime_struct, !csl.comptime_struct) -> !csl.comptime_struct

                "csl.set_tile_code"(%xId, %yId, %concat_params) <{file = "pe.csl"}> : (i16, i16, !csl.comptime_struct) -> ()
            }
        }

    }


  }) {sym_name = "layout.csl"} : () -> ()

}