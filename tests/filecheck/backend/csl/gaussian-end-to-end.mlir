// RUN: xdsl-opt -t csl %f | filecheck %f

builtin.module {
  // program:
  "csl.module"() <{kind=#csl<module_kind program>}> ({
    %memcpyParams = "csl.param"() <{param_name = "memcpyParams"}> : () -> !csl.comptime_struct
    %stencilCommsParams = "csl.param"() <{param_name = "stencilCommsParams"}> : () -> !csl.comptime_struct
    %iterationTaskId = "csl.param"() <{param_name = "iterationTaskId"}> : () -> i32  // is supposed to be task_id

    %zDim = "csl.param"() <{param_name = "zDim"}> : () ->  i16
    %pattern = "csl.param"() <{param_name = "pattern"}> : () ->  ui16
    %isBorderRegionPE = "csl.param"() <{param_name = "isBorderRegionPE"}> : () ->  i1

    %memcpy = "csl.import_module"() <{module = "<memcpy/memcpy>"}> : () -> !csl.imported_module
    %time = "csl.import_module"() <{module = "<time>"}> : () -> !csl.imported_module
    %util = "csl.import_module"() <{module = "util.csl"}> : () -> !csl.imported_module

    %directionCount = arith.constant 4 : i16

    %numChunks = "csl.member_call"(%util, %zDim) <{field = "computeChunks"}> : (!csl.imported_module, i16) -> i32
    %chunkSize = "csl.member_call"(%util, %zDim, %numChunks) <{field = "computeChunkSize"}> : (!csl.imported_module, i16, i32) -> i32
    %paddedZDim = arith.muli %chunkSize, %numChunks : i32

    %zero_u16 = arith.constant 0 : ui16
    %zero_f32 = arith.constant 0 : f32
    %one_f32 = arith.constant 1 : f32

    %tsc_size_words = "csl.member_access"(%time) <{field = "tsc_size_words"}> : (!csl.imported_module) -> i32

    %tscEndBuffer = "csl.constants"(%tsc_size_words, %zero_u16) : (i32, ui16) -> memref<?xui16>
    %tscStartBuffer = "csl.constants"(%tsc_size_words, %zero_u16) : (i32, ui16) -> memref<?xui16>
    %0 = arith.constant 2 : ui16
    %1 = arith.muli %0, %pattern : ui16
    %2 = arith.constant 1 : ui16
    %3 = arith.subi %1, %2 : ui16

    %zCoeffs = "csl.constants"(%3, %one_f32) <{is_const}> : (ui16, f32) -> memref<?xf32>

    %zValuesA = "csl.constants"(%paddedZDim, %zero_f32) : (i32, f32) -> memref<?xf32>
    %zValuesB = "csl.constants"(%paddedZDim, %zero_f32) : (i32, f32) -> memref<?xf32>
    %accumulator = "csl.constants"(%paddedZDim, %zero_f32) : (i32, f32) -> memref<?xf32>

    %dummy = "csl.constants"(%zDim, %zero_f32) : (i16, f32) -> memref<?xf32>

    %c5_i32 = arith.constant 5 : i32
    %time_buf_f32 = "csl.constants"(%c5_i32, %zero_f32) : (i32, f32) -> memref<?xf32>

    // TODO: add these three:
    // var ptr_time_buf_f32 : [*]f32 = &time_buf_f32;
    // var ptr_z_buffer_a: [*]f32 = &zValuesA;
    // var ptr_z_buffer_b: [*]f32 = &zValuesB;

    %mem_z_buf_dsd = "csl.get_mem_dsd"(%dummy, %zDim) : (memref<?xf32>, i16) -> !csl<dsd mem1d_dsd>
    // var mem_z_buf_dsd = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{zDim} -> dummy[i] });

    // @concat_structs(.{
    //   .pattern = pattern,
    //   .chunkSize = chunkSize,
    // }, stencilCommsParams
    %4 = "csl.const_struct"(%pattern, %chunkSize) <{ssa_fields = ["pattern","chunkSize"]}> : (ui16, i32) -> !csl.comptime_struct



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

                %params_task = "csl.concat_structs"(%invariants, %tmp_struct) : (!csl.comptime_struct, !csl.comptime_struct) -> !csl.comptime_struct

                %memcpy_params = "csl.member_call"(%memcpy, %xId) <{field = "get_params"}> : (!csl.imported_module, i16) -> !csl.comptime_struct

                %route_params = "csl.member_call"(%routes, %xId, %yId, %width, %height, %pattern) <{field = "computeAllRoutes"}> : (!csl.imported_module, i16, i16, ui16, ui16, ui16) -> !csl.comptime_struct

                %additional_params = "csl.const_struct"(%memcpy_params, %route_params) <{ssa_fields = ["memcpyParams", "stencilCommsParams"]}> : (!csl.comptime_struct, !csl.comptime_struct) -> !csl.comptime_struct

                %concat_params = "csl.concat_structs"(%params_task, %additional_params) : (!csl.comptime_struct, !csl.comptime_struct) -> !csl.comptime_struct

                "csl.set_tile_code"(%xId, %yId, %concat_params) <{file = "pe.csl"}> : (i16, i16, !csl.comptime_struct) -> ()
            }
        }

    }


  }) {sym_name = "layout.csl"} : () -> ()

}


// CHECK-NEXT: // FILE: pe.csl
// CHECK-NEXT: const math : imported_module = @import_module("<math>");
// CHECK-NEXT: // -----
// CHECK-NEXT: // FILE: layout.csl
// CHECK-NEXT: const LAUNCH_ID : i16 = 0;
// CHECK-NEXT: const LAUNCH : color = @get_color(LAUNCH_ID);
// CHECK-NEXT: param width : u16;
// CHECK-NEXT: param height : u16;
// CHECK-NEXT: param zDim : u16;
// CHECK-NEXT: //unknown op TestOp(%task_three = "test.op"() {"comment" = "@get_local_task_id(3)"} : () -> i32)
// CHECK-NEXT: param iterationTaskId : i32 = task_three;
// CHECK-NEXT: param pattern : u16;
// CHECK-NEXT: const invariants : comptime_struct = .{
// CHECK-NEXT:   .iterationTaskId = iterationTaskId,
// CHECK-NEXT:   .zDim = zDim,
// CHECK-NEXT:   .pattern = pattern,
// CHECK-NEXT: };
// CHECK-NEXT: const util : imported_module = @import_module("util.csl");
// CHECK-NEXT: const memcpy_call_params : comptime_struct = .{
// CHECK-NEXT:   .width = width,
// CHECK-NEXT:   .height = height,
// CHECK-NEXT:   .LAUNCH = LAUNCH,
// CHECK-NEXT: };
// CHECK-NEXT: const memcpy : imported_module = @import_module("<memcpy/get_params>", memcpy_call_params);
// CHECK-NEXT: const routes_params : comptime_struct = .{
// CHECK-NEXT:   .peWidth = width,
// CHECK-NEXT:   .peHeight = height,
// CHECK-NEXT:   .pattern = pattern,
// CHECK-NEXT: };
// CHECK-NEXT: const routes : imported_module = @import_module("routes.csl", routes_params);
// CHECK-NEXT: layout {
// CHECK-NEXT:   @set_rectangle(width, height);
// CHECK-NEXT:   const width_i16 : i16 = @as(i16, width);
// CHECK-NEXT:   const height_i16 : i16 = @as(i16, height);
// CHECK-NEXT:   const pattern_i16 : i16 = @as(i16, pattern);
// CHECK-NEXT:   const cst0 : i16 = 0;
// CHECK-NEXT:   const cst1 : i16 = 1;
// CHECK-NEXT:   //unknown op Subi(%pattern_sub_one = arith.subi %pattern_i16, %cst1 : i16)
// CHECK-NEXT:   //unknown op Subi(%width_sub_pattern = arith.subi %width_i16, %pattern_i16 : i16)
// CHECK-NEXT:   //unknown op Subi(%height_sub_pattern = arith.subi %width_i16, %pattern_i16 : i16)
// CHECK-NEXT:
// CHECK-NEXT:   for(@range(i16, cst0, width_i16, cst1)) |xId| {
// CHECK-NEXT:
// CHECK-NEXT:     for(@range(i16, cst0, height_i16, cst1)) |yId| {
// CHECK-NEXT:       //unknown op Cmpi(%c1 = arith.cmpi slt, %xId, %pattern_sub_one : i16)
// CHECK-NEXT:       //unknown op Cmpi(%c2 = arith.cmpi slt, %yId, %pattern_sub_one : i16)
// CHECK-NEXT:       //unknown op Cmpi(%c3 = arith.cmpi slt, %width_sub_pattern, %xId : i16)
// CHECK-NEXT:       //unknown op Cmpi(%c4 = arith.cmpi slt, %height_sub_pattern, %yId : i16)
// CHECK-NEXT:       //unknown op AndI(%c1_c2 = arith.andi %c1, %c2 : i1)
// CHECK-NEXT:       //unknown op AndI(%c3_c4 = arith.andi %c3, %c4 : i1)
// CHECK-NEXT:       //unknown op AndI(%is_border_region = arith.andi %c1_c2, %c3_c4 : i1)
// CHECK-NEXT:       const tmp_struct : comptime_struct = .{
// CHECK-NEXT:         .isBorderRegionPE = is_border_region,
// CHECK-NEXT:       };
// CHECK-NEXT:       const params_task : comptime_struct = @concat_structs(invariants, tmp_struct);
// CHECK-NEXT:       const memcpy_params : comptime_struct = memcpy.get_params(xId);
// CHECK-NEXT:       const route_params : comptime_struct = routes.computeAllRoutes(xId, yId, width, height, pattern);
// CHECK-NEXT:       const additional_params : comptime_struct = .{
// CHECK-NEXT:         .memcpyParams = memcpy_params,
// CHECK-NEXT:         .stencilCommsParams = route_params,
// CHECK-NEXT:       };
// CHECK-NEXT:       const concat_params : comptime_struct = @concat_structs(params_task, additional_params);
// CHECK-NEXT:       @set_tile_code(xId, yId, "pe.csl", concat_params);
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
