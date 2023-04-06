// RUN: mlir-opt %s -cse | filecheck %s

"builtin.module"() ({
  "func.func"() ({
  ^0(%data : memref<2xmemref<?x?x?xf32>>):
    %time_m = "arith.constant"() {"value" = 0 : index} : () -> index
    %time_M = "arith.constant"() {"value" = 1000 : index} : () -> index
    %step = "arith.constant"() {"value" = 1 : index} : () -> index
    %0 = "arith.constant"() {"value" = 1 : index} : () -> index
    %time_M_1 = "arith.addi"(%time_M, %0) : (index, index) -> index
    "scf.for"(%time_m, %time_M_1, %step) ({
    ^1(%time : index):
      %time_1 = "arith.index_cast"(%time) : (index) -> i64
      %1 = "arith.constant"() {"value" = 2 : i64} : () -> i64
      %2 = "arith.constant"() {"value" = 0 : i64} : () -> i64
      %3 = "arith.addi"(%time_1, %2) : (i64, i64) -> i64
      %t0 = "arith.remsi"(%3, %1) : (i64, i64) -> i64
      %4 = "arith.constant"() {"value" = 1 : i64} : () -> i64
      %5 = "arith.addi"(%time_1, %4) : (i64, i64) -> i64
      %t1 = "arith.remsi"(%5, %1) : (i64, i64) -> i64
      %t0_w_size = "arith.index_cast"(%t0) : (i64) -> index
      %t0_w_size_1 = "memref.load"(%data, %t0_w_size) : (memref<2xmemref<?x?x?xf32>>, index) -> memref<?x?x?xf32>
      %t0_w_size_3 = "memref.cast"(%t0_w_size_1) : (memref<?x?x?xf32>) -> memref<58x88x48xf32>
      %t1_w_size = "arith.index_cast"(%t1) : (i64) -> index
      %t1_w_size_1 = "memref.load"(%data, %t1_w_size) : (memref<2xmemref<?x?x?xf32>>, index) -> memref<?x?x?xf32>
      %t1_w_size_3 = "memref.cast"(%t1_w_size_1) : (memref<?x?x?xf32>) -> memref<58x88x48xf32>
      %6 = "arith.constant"() {"value" = 0 : index} : () -> index
      %7 = "arith.constant"() {"value" = 1 : index} : () -> index
      %8 = "arith.constant"() {"value" = 50 : index} : () -> index
      %9 = "arith.constant"() {"value" = 80 : index} : () -> index
      %10 = "arith.constant"() {"value" = 40 : index} : () -> index
      "scf.parallel"(%6, %6, %6, %8, %9, %10, %7, %7, %7) ({
      ^2(%11 : index, %12 : index, %13 : index):
        %14 = "arith.constant"() {"value" = 4 : index} : () -> index
        %15 = "arith.constant"() {"value" = 4 : index} : () -> index
        %16 = "arith.constant"() {"value" = 4 : index} : () -> index
        %17 = "arith.addi"(%11, %14) : (index, index) -> index
        %18 = "arith.addi"(%12, %15) : (index, index) -> index
        %19 = "arith.addi"(%13, %16) : (index, index) -> index
        %20 = "memref.load"(%t0_w_size_3, %17, %18, %19) : (memref<58x88x48xf32>, index, index, index) -> f32
        %21 = "arith.constant"() {"value" = 3 : index} : () -> index
        %22 = "arith.constant"() {"value" = 4 : index} : () -> index
        %23 = "arith.constant"() {"value" = 4 : index} : () -> index
        %24 = "arith.addi"(%11, %21) : (index, index) -> index
        %25 = "arith.addi"(%12, %22) : (index, index) -> index
        %26 = "arith.addi"(%13, %23) : (index, index) -> index
        %27 = "memref.load"(%t0_w_size_3, %24, %25, %26) : (memref<58x88x48xf32>, index, index, index) -> f32
        %28 = "arith.constant"() {"value" = 5 : index} : () -> index
        %29 = "arith.constant"() {"value" = 4 : index} : () -> index
        %30 = "arith.constant"() {"value" = 4 : index} : () -> index
        %31 = "arith.addi"(%11, %28) : (index, index) -> index
        %32 = "arith.addi"(%12, %29) : (index, index) -> index
        %33 = "arith.addi"(%13, %30) : (index, index) -> index
        %34 = "memref.load"(%t0_w_size_3, %31, %32, %33) : (memref<58x88x48xf32>, index, index, index) -> f32
        %35 = "arith.constant"() {"value" = 2 : index} : () -> index
        %36 = "arith.constant"() {"value" = 4 : index} : () -> index
        %37 = "arith.constant"() {"value" = 4 : index} : () -> index
        %38 = "arith.addi"(%11, %35) : (index, index) -> index
        %39 = "arith.addi"(%12, %36) : (index, index) -> index
        %40 = "arith.addi"(%13, %37) : (index, index) -> index
        %41 = "memref.load"(%t0_w_size_3, %38, %39, %40) : (memref<58x88x48xf32>, index, index, index) -> f32
        %42 = "arith.constant"() {"value" = 6 : index} : () -> index
        %43 = "arith.constant"() {"value" = 4 : index} : () -> index
        %44 = "arith.constant"() {"value" = 4 : index} : () -> index
        %45 = "arith.addi"(%11, %42) : (index, index) -> index
        %46 = "arith.addi"(%12, %43) : (index, index) -> index
        %47 = "arith.addi"(%13, %44) : (index, index) -> index
        %48 = "memref.load"(%t0_w_size_3, %45, %46, %47) : (memref<58x88x48xf32>, index, index, index) -> f32
        %49 = "arith.constant"() {"value" = 4 : index} : () -> index
        %50 = "arith.constant"() {"value" = 3 : index} : () -> index
        %51 = "arith.constant"() {"value" = 4 : index} : () -> index
        %52 = "arith.addi"(%11, %49) : (index, index) -> index
        %53 = "arith.addi"(%12, %50) : (index, index) -> index
        %54 = "arith.addi"(%13, %51) : (index, index) -> index
        %55 = "memref.load"(%t0_w_size_3, %52, %53, %54) : (memref<58x88x48xf32>, index, index, index) -> f32
        %56 = "arith.constant"() {"value" = 4 : index} : () -> index
        %57 = "arith.constant"() {"value" = 5 : index} : () -> index
        %58 = "arith.constant"() {"value" = 4 : index} : () -> index
        %59 = "arith.addi"(%11, %56) : (index, index) -> index
        %60 = "arith.addi"(%12, %57) : (index, index) -> index
        %61 = "arith.addi"(%13, %58) : (index, index) -> index
        %62 = "memref.load"(%t0_w_size_3, %59, %60, %61) : (memref<58x88x48xf32>, index, index, index) -> f32
        %63 = "arith.constant"() {"value" = 4 : index} : () -> index
        %64 = "arith.constant"() {"value" = 2 : index} : () -> index
        %65 = "arith.constant"() {"value" = 4 : index} : () -> index
        %66 = "arith.addi"(%11, %63) : (index, index) -> index
        %67 = "arith.addi"(%12, %64) : (index, index) -> index
        %68 = "arith.addi"(%13, %65) : (index, index) -> index
        %69 = "memref.load"(%t0_w_size_3, %66, %67, %68) : (memref<58x88x48xf32>, index, index, index) -> f32
        %70 = "arith.constant"() {"value" = 4 : index} : () -> index
        %71 = "arith.constant"() {"value" = 6 : index} : () -> index
        %72 = "arith.constant"() {"value" = 4 : index} : () -> index
        %73 = "arith.addi"(%11, %70) : (index, index) -> index
        %74 = "arith.addi"(%12, %71) : (index, index) -> index
        %75 = "arith.addi"(%13, %72) : (index, index) -> index
        %76 = "memref.load"(%t0_w_size_3, %73, %74, %75) : (memref<58x88x48xf32>, index, index, index) -> f32
        %77 = "arith.constant"() {"value" = 4 : index} : () -> index
        %78 = "arith.constant"() {"value" = 4 : index} : () -> index
        %79 = "arith.constant"() {"value" = 3 : index} : () -> index
        %80 = "arith.addi"(%11, %77) : (index, index) -> index
        %81 = "arith.addi"(%12, %78) : (index, index) -> index
        %82 = "arith.addi"(%13, %79) : (index, index) -> index
        %83 = "memref.load"(%t0_w_size_3, %80, %81, %82) : (memref<58x88x48xf32>, index, index, index) -> f32
        %84 = "arith.constant"() {"value" = 4 : index} : () -> index
        %85 = "arith.constant"() {"value" = 4 : index} : () -> index
        %86 = "arith.constant"() {"value" = 5 : index} : () -> index
        %87 = "arith.addi"(%11, %84) : (index, index) -> index
        %88 = "arith.addi"(%12, %85) : (index, index) -> index
        %89 = "arith.addi"(%13, %86) : (index, index) -> index
        %90 = "memref.load"(%t0_w_size_3, %87, %88, %89) : (memref<58x88x48xf32>, index, index, index) -> f32
        %91 = "arith.constant"() {"value" = 4 : index} : () -> index
        %92 = "arith.constant"() {"value" = 4 : index} : () -> index
        %93 = "arith.constant"() {"value" = 2 : index} : () -> index
        %94 = "arith.addi"(%11, %91) : (index, index) -> index
        %95 = "arith.addi"(%12, %92) : (index, index) -> index
        %96 = "arith.addi"(%13, %93) : (index, index) -> index
        %97 = "memref.load"(%t0_w_size_3, %94, %95, %96) : (memref<58x88x48xf32>, index, index, index) -> f32
        %98 = "arith.constant"() {"value" = 4 : index} : () -> index
        %99 = "arith.constant"() {"value" = 4 : index} : () -> index
        %100 = "arith.constant"() {"value" = 6 : index} : () -> index
        %101 = "arith.addi"(%11, %98) : (index, index) -> index
        %102 = "arith.addi"(%12, %99) : (index, index) -> index
        %103 = "arith.addi"(%13, %100) : (index, index) -> index
        %104 = "memref.load"(%t0_w_size_3, %101, %102, %103) : (memref<58x88x48xf32>, index, index, index) -> f32
        %dt = "arith.constant"() {"value" = 4.122440608513459e-06 : f32} : () -> f32
        %105 = "arith.constant"() {"value" = -1 : i64} : () -> i64
        %106 = "math.fpowi"(%dt, %105) : (f32, i64) -> f32
        %107 = "arith.mulf"(%106, %20) : (f32, f32) -> f32
        %108 = "arith.constant"() {"value" = 1.3333333332557231 : f32} : () -> f32
        %h_x = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
        %109 = "arith.constant"() {"value" = -2 : i64} : () -> i64
        %110 = "math.fpowi"(%h_x, %109) : (f32, i64) -> f32
        %111 = "arith.mulf"(%108, %110) : (f32, f32) -> f32
        %112 = "arith.mulf"(%111, %27) : (f32, f32) -> f32
        %113 = "arith.constant"() {"value" = 1.3333333332557231 : f32} : () -> f32
        %h_x_1 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
        %114 = "arith.constant"() {"value" = -2 : i64} : () -> i64
        %115 = "math.fpowi"(%h_x_1, %114) : (f32, i64) -> f32
        %116 = "arith.mulf"(%113, %115) : (f32, f32) -> f32
        %117 = "arith.mulf"(%116, %34) : (f32, f32) -> f32
        %118 = "arith.constant"() {"value" = -2.5 : f32} : () -> f32
        %h_x_2 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
        %119 = "arith.constant"() {"value" = -2 : i64} : () -> i64
        %120 = "math.fpowi"(%h_x_2, %119) : (f32, i64) -> f32
        %121 = "arith.mulf"(%118, %120) : (f32, f32) -> f32
        %122 = "arith.mulf"(%121, %20) : (f32, f32) -> f32
        %123 = "arith.constant"() {"value" = -0.0833333333284827 : f32} : () -> f32
        %h_x_3 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
        %124 = "arith.constant"() {"value" = -2 : i64} : () -> i64
        %125 = "math.fpowi"(%h_x_3, %124) : (f32, i64) -> f32
        %126 = "arith.mulf"(%123, %125) : (f32, f32) -> f32
        %127 = "arith.mulf"(%126, %41) : (f32, f32) -> f32
        %128 = "arith.constant"() {"value" = -0.0833333333284827 : f32} : () -> f32
        %h_x_4 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
        %129 = "arith.constant"() {"value" = -2 : i64} : () -> i64
        %130 = "math.fpowi"(%h_x_4, %129) : (f32, i64) -> f32
        %131 = "arith.mulf"(%128, %130) : (f32, f32) -> f32
        %132 = "arith.mulf"(%131, %48) : (f32, f32) -> f32
        %133 = "arith.addf"(%112, %117) : (f32, f32) -> f32
        %134 = "arith.addf"(%133, %122) : (f32, f32) -> f32
        %135 = "arith.addf"(%134, %127) : (f32, f32) -> f32
        %136 = "arith.addf"(%135, %132) : (f32, f32) -> f32
        %137 = "arith.constant"() {"value" = 1.3333333332557231 : f32} : () -> f32
        %h_y = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
        %138 = "arith.constant"() {"value" = -2 : i64} : () -> i64
        %139 = "math.fpowi"(%h_y, %138) : (f32, i64) -> f32
        %140 = "arith.mulf"(%137, %139) : (f32, f32) -> f32
        %141 = "arith.mulf"(%140, %55) : (f32, f32) -> f32
        %142 = "arith.constant"() {"value" = 1.3333333332557231 : f32} : () -> f32
        %h_y_1 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
        %143 = "arith.constant"() {"value" = -2 : i64} : () -> i64
        %144 = "math.fpowi"(%h_y_1, %143) : (f32, i64) -> f32
        %145 = "arith.mulf"(%142, %144) : (f32, f32) -> f32
        %146 = "arith.mulf"(%145, %62) : (f32, f32) -> f32
        %147 = "arith.constant"() {"value" = -2.5 : f32} : () -> f32
        %h_y_2 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
        %148 = "arith.constant"() {"value" = -2 : i64} : () -> i64
        %149 = "math.fpowi"(%h_y_2, %148) : (f32, i64) -> f32
        %150 = "arith.mulf"(%147, %149) : (f32, f32) -> f32
        %151 = "arith.mulf"(%150, %20) : (f32, f32) -> f32
        %152 = "arith.constant"() {"value" = -0.0833333333284827 : f32} : () -> f32
        %h_y_3 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
        %153 = "arith.constant"() {"value" = -2 : i64} : () -> i64
        %154 = "math.fpowi"(%h_y_3, %153) : (f32, i64) -> f32
        %155 = "arith.mulf"(%152, %154) : (f32, f32) -> f32
        %156 = "arith.mulf"(%155, %69) : (f32, f32) -> f32
        %157 = "arith.constant"() {"value" = -0.0833333333284827 : f32} : () -> f32
        %h_y_4 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
        %158 = "arith.constant"() {"value" = -2 : i64} : () -> i64
        %159 = "math.fpowi"(%h_y_4, %158) : (f32, i64) -> f32
        %160 = "arith.mulf"(%157, %159) : (f32, f32) -> f32
        %161 = "arith.mulf"(%160, %76) : (f32, f32) -> f32
        %162 = "arith.addf"(%141, %146) : (f32, f32) -> f32
        %163 = "arith.addf"(%162, %151) : (f32, f32) -> f32
        %164 = "arith.addf"(%163, %156) : (f32, f32) -> f32
        %165 = "arith.addf"(%164, %161) : (f32, f32) -> f32
        %166 = "arith.constant"() {"value" = 1.3333333332557231 : f32} : () -> f32
        %h_z = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
        %167 = "arith.constant"() {"value" = -2 : i64} : () -> i64
        %168 = "math.fpowi"(%h_z, %167) : (f32, i64) -> f32
        %169 = "arith.mulf"(%166, %168) : (f32, f32) -> f32
        %170 = "arith.mulf"(%169, %83) : (f32, f32) -> f32
        %171 = "arith.constant"() {"value" = 1.3333333332557231 : f32} : () -> f32
        %h_z_1 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
        %172 = "arith.constant"() {"value" = -2 : i64} : () -> i64
        %173 = "math.fpowi"(%h_z_1, %172) : (f32, i64) -> f32
        %174 = "arith.mulf"(%171, %173) : (f32, f32) -> f32
        %175 = "arith.mulf"(%174, %90) : (f32, f32) -> f32
        %176 = "arith.constant"() {"value" = -2.5 : f32} : () -> f32
        %h_z_2 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
        %177 = "arith.constant"() {"value" = -2 : i64} : () -> i64
        %178 = "math.fpowi"(%h_z_2, %177) : (f32, i64) -> f32
        %179 = "arith.mulf"(%176, %178) : (f32, f32) -> f32
        %180 = "arith.mulf"(%179, %20) : (f32, f32) -> f32
        %181 = "arith.constant"() {"value" = -0.0833333333284827 : f32} : () -> f32
        %h_z_3 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
        %182 = "arith.constant"() {"value" = -2 : i64} : () -> i64
        %183 = "math.fpowi"(%h_z_3, %182) : (f32, i64) -> f32
        %184 = "arith.mulf"(%181, %183) : (f32, f32) -> f32
        %185 = "arith.mulf"(%184, %97) : (f32, f32) -> f32
        %186 = "arith.constant"() {"value" = -0.0833333333284827 : f32} : () -> f32
        %h_z_4 = "arith.constant"() {"value" = 0.020202020183205605 : f32} : () -> f32
        %187 = "arith.constant"() {"value" = -2 : i64} : () -> i64
        %188 = "math.fpowi"(%h_z_4, %187) : (f32, i64) -> f32
        %189 = "arith.mulf"(%186, %188) : (f32, f32) -> f32
        %190 = "arith.mulf"(%189, %104) : (f32, f32) -> f32
        %191 = "arith.addf"(%170, %175) : (f32, f32) -> f32
        %192 = "arith.addf"(%191, %180) : (f32, f32) -> f32
        %193 = "arith.addf"(%192, %185) : (f32, f32) -> f32
        %194 = "arith.addf"(%193, %190) : (f32, f32) -> f32
        %195 = "arith.addf"(%136, %165) : (f32, f32) -> f32
        %196 = "arith.addf"(%195, %194) : (f32, f32) -> f32
        %a = "arith.constant"() {"value" = 0.5 : f32} : () -> f32
        %197 = "arith.mulf"(%196, %a) : (f32, f32) -> f32
        %198 = "arith.addf"(%107, %197) : (f32, f32) -> f32
        %dt_1 = "arith.constant"() {"value" = 4.122440608513459e-06 : f32} : () -> f32
        %199 = "arith.mulf"(%198, %dt_1) : (f32, f32) -> f32
        %200 = "arith.constant"() {"value" = 4 : index} : () -> index
        %201 = "arith.constant"() {"value" = 4 : index} : () -> index
        %202 = "arith.constant"() {"value" = 4 : index} : () -> index
        %203 = "arith.addi"(%11, %200) : (index, index) -> index
        %204 = "arith.addi"(%12, %201) : (index, index) -> index
        %205 = "arith.addi"(%13, %202) : (index, index) -> index
        "memref.store"(%199, %t1_w_size_3, %203, %204, %205) : (f32, memref<58x88x48xf32>, index, index, index) -> ()
        "scf.yield"() : () -> ()
      }) {"operand_segment_sizes" = array<i32: 3, 3, 3, 0>} : (index, index, index, index, index, index, index, index, index) -> ()
      "scf.yield"() : () -> ()
    }) : (index, index, index) -> ()
    "func.return"() : () -> ()
  }) {"sym_name" = "myfunc", "function_type" = (memref<2xmemref<?x?x?xf32>>) -> (), "sym_visibility" = "private", "param_names" = ["data"]} : () -> ()
}) : () -> ()



// CHECK:      module {
// CHECK-NEXT:   func.func private @myfunc(%arg0: memref<2xmemref<?x?x?xf32>>) attributes {param_names = ["data"]} {
// CHECK-NEXT:   %c0 = arith.constant 0 : index
// CHECK-NEXT:   %c1000 = arith.constant 1000 : index
// CHECK-NEXT:   %c1 = arith.constant 1 : index
// CHECK-NEXT:   %0 = arith.addi %c1000, %c1 : index
// CHECK-NEXT:   scf.for %arg1 = %c0 to %0 step %c1 {
// CHECK-NEXT:     %1 = arith.index_cast %arg1 : index to i64
// CHECK-NEXT:     %c2_i64 = arith.constant 2 : i64
// CHECK-NEXT:     %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:     %2 = arith.addi %1, %c0_i64 : i64
// CHECK-NEXT:     %3 = arith.remsi %2, %c2_i64 : i64
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %4 = arith.addi %1, %c1_i64 : i64
// CHECK-NEXT:     %5 = arith.remsi %4, %c2_i64 : i64
// CHECK-NEXT:     %6 = arith.index_cast %3 : i64 to index
// CHECK-NEXT:     %7 = memref.load %arg0[%6] : memref<2xmemref<?x?x?xf32>>
// CHECK-NEXT:     %cast = memref.cast %7 : memref<?x?x?xf32> to memref<58x88x48xf32>
// CHECK-NEXT:     %8 = arith.index_cast %5 : i64 to index
// CHECK-NEXT:     %9 = memref.load %arg0[%8] : memref<2xmemref<?x?x?xf32>>
// CHECK-NEXT:     %cast_0 = memref.cast %9 : memref<?x?x?xf32> to memref<58x88x48xf32>
// CHECK-NEXT:     %c50 = arith.constant 50 : index
// CHECK-NEXT:     %c80 = arith.constant 80 : index
// CHECK-NEXT:     %c40 = arith.constant 40 : index
// CHECK-NEXT:     scf.parallel (%arg2, %arg3, %arg4) = (%c0, %c0, %c0) to (%c50, %c80, %c40) step (%c1, %c1, %c1) {
// CHECK-NEXT:       %c4 = arith.constant 4 : index
// CHECK-NEXT:       %10 = arith.addi %arg2, %c4 : index
// CHECK-NEXT:       %11 = arith.addi %arg3, %c4 : index
// CHECK-NEXT:       %12 = arith.addi %arg4, %c4 : index
// CHECK-NEXT:       %13 = memref.load %cast[%10, %11, %12] : memref<58x88x48xf32>
// CHECK-NEXT:       %c3 = arith.constant 3 : index
// CHECK-NEXT:       %14 = arith.addi %arg2, %c3 : index
// CHECK-NEXT:       %15 = memref.load %cast[%14, %11, %12] : memref<58x88x48xf32>
// CHECK-NEXT:       %c5 = arith.constant 5 : index
// CHECK-NEXT:       %16 = arith.addi %arg2, %c5 : index
// CHECK-NEXT:       %17 = memref.load %cast[%16, %11, %12] : memref<58x88x48xf32>
// CHECK-NEXT:       %c2 = arith.constant 2 : index
// CHECK-NEXT:       %18 = arith.addi %arg2, %c2 : index
// CHECK-NEXT:       %19 = memref.load %cast[%18, %11, %12] : memref<58x88x48xf32>
// CHECK-NEXT:       %c6 = arith.constant 6 : index
// CHECK-NEXT:       %20 = arith.addi %arg2, %c6 : index
// CHECK-NEXT:       %21 = memref.load %cast[%20, %11, %12] : memref<58x88x48xf32>
// CHECK-NEXT:       %22 = arith.addi %arg3, %c3 : index
// CHECK-NEXT:       %23 = memref.load %cast[%10, %22, %12] : memref<58x88x48xf32>
// CHECK-NEXT:       %24 = arith.addi %arg3, %c5 : index
// CHECK-NEXT:       %25 = memref.load %cast[%10, %24, %12] : memref<58x88x48xf32>
// CHECK-NEXT:       %26 = arith.addi %arg3, %c2 : index
// CHECK-NEXT:       %27 = memref.load %cast[%10, %26, %12] : memref<58x88x48xf32>
// CHECK-NEXT:       %28 = arith.addi %arg3, %c6 : index
// CHECK-NEXT:       %29 = memref.load %cast[%10, %28, %12] : memref<58x88x48xf32>
// CHECK-NEXT:       %30 = arith.addi %arg4, %c3 : index
// CHECK-NEXT:       %31 = memref.load %cast[%10, %11, %30] : memref<58x88x48xf32>
// CHECK-NEXT:       %32 = arith.addi %arg4, %c5 : index
// CHECK-NEXT:       %33 = memref.load %cast[%10, %11, %32] : memref<58x88x48xf32>
// CHECK-NEXT:       %34 = arith.addi %arg4, %c2 : index
// CHECK-NEXT:       %35 = memref.load %cast[%10, %11, %34] : memref<58x88x48xf32>
// CHECK-NEXT:       %36 = arith.addi %arg4, %c6 : index
// CHECK-NEXT:       %37 = memref.load %cast[%10, %11, %36] : memref<58x88x48xf32>
// CHECK-NEXT:       %cst = arith.constant 4.12244071E-6 : f32
// CHECK-NEXT:       %c-1_i64 = arith.constant -1 : i64
// CHECK-NEXT:       %38 = math.fpowi %cst, %c-1_i64 : f32, i64
// CHECK-NEXT:       %39 = arith.mulf %38, %13 : f32
// CHECK-NEXT:       %cst_1 = arith.constant 1.33333337 : f32
// CHECK-NEXT:       %cst_2 = arith.constant 0.0202020202 : f32
// CHECK-NEXT:       %c-2_i64 = arith.constant -2 : i64
// CHECK-NEXT:       %40 = math.fpowi %cst_2, %c-2_i64 : f32, i64
// CHECK-NEXT:       %41 = arith.mulf %cst_1, %40 : f32
// CHECK-NEXT:       %42 = arith.mulf %41, %15 : f32
// CHECK-NEXT:       %43 = arith.mulf %41, %17 : f32
// CHECK-NEXT:       %cst_3 = arith.constant -2.500000e+00 : f32
// CHECK-NEXT:       %44 = arith.mulf %cst_3, %40 : f32
// CHECK-NEXT:       %45 = arith.mulf %44, %13 : f32
// CHECK-NEXT:       %cst_4 = arith.constant -0.0833333358 : f32
// CHECK-NEXT:       %46 = arith.mulf %cst_4, %40 : f32
// CHECK-NEXT:       %47 = arith.mulf %46, %19 : f32
// CHECK-NEXT:       %48 = arith.mulf %46, %21 : f32
// CHECK-NEXT:       %49 = arith.addf %42, %43 : f32
// CHECK-NEXT:       %50 = arith.addf %49, %45 : f32
// CHECK-NEXT:       %51 = arith.addf %50, %47 : f32
// CHECK-NEXT:       %52 = arith.addf %51, %48 : f32
// CHECK-NEXT:       %53 = arith.mulf %41, %23 : f32
// CHECK-NEXT:       %54 = arith.mulf %41, %25 : f32
// CHECK-NEXT:       %55 = arith.mulf %46, %27 : f32
// CHECK-NEXT:       %56 = arith.mulf %46, %29 : f32
// CHECK-NEXT:       %57 = arith.addf %53, %54 : f32
// CHECK-NEXT:       %58 = arith.addf %57, %45 : f32
// CHECK-NEXT:       %59 = arith.addf %58, %55 : f32
// CHECK-NEXT:       %60 = arith.addf %59, %56 : f32
// CHECK-NEXT:       %61 = arith.mulf %41, %31 : f32
// CHECK-NEXT:       %62 = arith.mulf %41, %33 : f32
// CHECK-NEXT:       %63 = arith.mulf %46, %35 : f32
// CHECK-NEXT:       %64 = arith.mulf %46, %37 : f32
// CHECK-NEXT:       %65 = arith.addf %61, %62 : f32
// CHECK-NEXT:       %66 = arith.addf %65, %45 : f32
// CHECK-NEXT:       %67 = arith.addf %66, %63 : f32
// CHECK-NEXT:       %68 = arith.addf %67, %64 : f32
// CHECK-NEXT:       %69 = arith.addf %52, %60 : f32
// CHECK-NEXT:       %70 = arith.addf %69, %68 : f32
// CHECK-NEXT:       %cst_5 = arith.constant 5.000000e-01 : f32
// CHECK-NEXT:       %71 = arith.mulf %70, %cst_5 : f32
// CHECK-NEXT:       %72 = arith.addf %39, %71 : f32
// CHECK-NEXT:       %73 = arith.mulf %72, %cst : f32
// CHECK-NEXT:       memref.store %73, %cast_0[%10, %11, %12] : memref<58x88x48xf32>
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT:   }
// CHECK-NEXT: }
