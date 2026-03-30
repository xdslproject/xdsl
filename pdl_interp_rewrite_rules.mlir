builtin.module {
  pdl_interp.func @matcher(%0 : !pdl.operation) {
    %1 = pdl_interp.get_result 0 of %0
    pdl_interp.is_not_null %1 : !pdl.value -> ^bb0, ^bb1
  ^bb1:
    pdl_interp.finalize
  ^bb0:
    pdl_interp.switch_operation_name of %0 to ["arith.addf", "arith.mulf", "arith.subf", "arith.divf", "arith.constant", "math.absf", "math.sqrt", "math.copysign", "math.powf", "fmin", "fmax", "math.exp", "math.cbrt", "math.cos", "math.tan", "math.sinh", "math.cosh", "math.tanh", "math.asinh", "math.acosh", "math.atanh"](^bb2, ^bb3, ^bb4, ^bb5, ^bb6, ^bb7, ^bb8, ^bb9, ^bb10, ^bb11, ^bb12, ^bb13, ^bb14, ^bb15, ^bb16, ^bb17, ^bb18, ^bb19, ^bb20, ^bb21, ^bb22) -> ^bb23
  ^bb23:
    %2 = pdl_interp.get_operand 0 of %0
    %3 = pdl_interp.get_defining_op of %2 : !pdl.value {position = "root.operand[0].defining_op"}
    pdl_interp.is_not_null %3 : !pdl.operation -> ^bb24, ^bb1
  ^bb24:
    pdl_interp.switch_operation_name of %0 to ["arith.addf", "arith.subf", "arith.mulf", "arith.divf", "arith.negf", "math.powf", "math.sqrt", "math.cbrt", "math.absf", "math.copysign", "math.exp", "math.log", "math.sin", "math.cos", "math.tan", "math.atan2", "math.asin", "math.atan", "math.cosh", "math.sinh", "math.tanh", "math.acosh"](^bb25, ^bb26, ^bb27, ^bb28, ^bb29, ^bb30, ^bb31, ^bb32, ^bb33, ^bb34, ^bb35, ^bb36, ^bb37, ^bb38, ^bb39, ^bb40, ^bb41, ^bb42, ^bb43, ^bb44, ^bb45, ^bb46) -> ^bb1
  ^bb25:
    pdl_interp.check_operand_count of %0 is 2 -> ^bb47, ^bb1
  ^bb47:
    pdl_interp.check_result_count of %0 is 1 -> ^bb48, ^bb1
  ^bb48:
    pdl_interp.is_not_null %2 : !pdl.value -> ^bb49, ^bb50
  ^bb50:
    %4 = pdl_interp.get_operand 1 of %0
    %5 = pdl_interp.get_defining_op of %4 : !pdl.value {position = "root.operand[1].defining_op"}
    pdl_interp.is_not_null %5 : !pdl.operation -> ^bb51, ^bb1
  ^bb51:
    pdl_interp.is_not_null %2 : !pdl.value -> ^bb52, ^bb1
  ^bb52:
    pdl_interp.switch_operation_name of %3 to ["arith.mulf", "arith.negf", "arith.addf", "arith.subf", "arith.divf", "math.powf", "math.log", "math.sin", "math.cos", "math.atan", "arith.constant", "math.cosh", "math.exp", "math.sinh"](^bb53, ^bb54, ^bb55, ^bb56, ^bb57, ^bb58, ^bb59, ^bb60, ^bb61, ^bb62, ^bb63, ^bb64, ^bb65, ^bb66) -> ^bb1
  ^bb53:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb67, ^bb1
  ^bb67:
    pdl_interp.check_result_count of %3 is 1 -> ^bb68, ^bb1
  ^bb68:
    %6 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %6 : !pdl.value -> ^bb69, ^bb1
  ^bb69:
    pdl_interp.are_equal %6, %2 : !pdl.value -> ^bb70, ^bb1
  ^bb70:
    %7 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %7 : !pdl.value -> ^bb71, ^bb1
  ^bb71:
    pdl_interp.is_not_null %4 : !pdl.value -> ^bb72, ^bb1
  ^bb72:
    %8 = pdl_interp.get_value_type of %7 : !pdl.type
    %9 = pdl_interp.get_value_type of %6 : !pdl.type
    pdl_interp.are_equal %8, %9 : !pdl.type -> ^bb73, ^bb74
  ^bb74:
    pdl_interp.switch_operation_name of %5 to ["arith.mulf", "arith.constant"](^bb75, ^bb76) -> ^bb1
  ^bb75:
    pdl_interp.check_operand_count of %5 is 2 -> ^bb77, ^bb1
  ^bb77:
    pdl_interp.check_result_count of %5 is 1 -> ^bb78, ^bb1
  ^bb78:
    %10 = pdl_interp.get_result 0 of %5
    pdl_interp.is_not_null %10 : !pdl.value -> ^bb79, ^bb1
  ^bb79:
    pdl_interp.are_equal %10, %4 : !pdl.value -> ^bb80, ^bb1
  ^bb80:
    %11 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %11 : !pdl.value -> ^bb81, ^bb1
  ^bb81:
    %12 = pdl_interp.get_defining_op of %11 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %12 : !pdl.operation -> ^bb82, ^bb1
  ^bb82:
    %13 = pdl_interp.get_defining_op of %7 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %13 : !pdl.operation -> ^bb83, ^bb1
  ^bb83:
    %14 = pdl_interp.get_operand 0 of %5
    pdl_interp.is_not_null %14 : !pdl.value -> ^bb84, ^bb1
  ^bb84:
    %15 = pdl_interp.get_defining_op of %14 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %15 : !pdl.operation -> ^bb85, ^bb1
  ^bb85:
    %16 = pdl_interp.get_operand 1 of %5
    %17 = pdl_interp.get_defining_op of %16 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %17 : !pdl.operation -> ^bb86, ^bb1
  ^bb86:
    pdl_interp.is_not_null %16 : !pdl.value -> ^bb87, ^bb1
  ^bb87:
    pdl_interp.switch_operation_name of %12 to ["math.cos", "math.sinh", "math.cosh"](^bb88, ^bb89, ^bb90) -> ^bb1
  ^bb88:
    pdl_interp.check_operand_count of %12 is 1 -> ^bb91, ^bb1
  ^bb91:
    pdl_interp.check_result_count of %12 is 1 -> ^bb92, ^bb1
  ^bb92:
    %18 = pdl_interp.get_result 0 of %12
    pdl_interp.is_not_null %18 : !pdl.value -> ^bb93, ^bb1
  ^bb93:
    pdl_interp.are_equal %18, %11 : !pdl.value -> ^bb94, ^bb1
  ^bb94:
    pdl_interp.switch_operation_name of %13 to ["math.cos", "math.sin"](^bb95, ^bb96) -> ^bb1
  ^bb95:
    pdl_interp.check_operand_count of %13 is 1 -> ^bb97, ^bb1
  ^bb97:
    pdl_interp.check_result_count of %13 is 1 -> ^bb98, ^bb1
  ^bb98:
    %19 = pdl_interp.get_result 0 of %13
    pdl_interp.is_not_null %19 : !pdl.value -> ^bb99, ^bb1
  ^bb99:
    pdl_interp.are_equal %19, %7 : !pdl.value -> ^bb100, ^bb1
  ^bb100:
    pdl_interp.check_operation_name of %15 is "math.sin" -> ^bb101, ^bb1
  ^bb101:
    pdl_interp.check_operand_count of %15 is 1 -> ^bb102, ^bb1
  ^bb102:
    pdl_interp.check_result_count of %15 is 1 -> ^bb103, ^bb1
  ^bb103:
    %20 = pdl_interp.get_result 0 of %15
    pdl_interp.is_not_null %20 : !pdl.value -> ^bb104, ^bb1
  ^bb104:
    pdl_interp.are_equal %20, %14 : !pdl.value -> ^bb105, ^bb1
  ^bb105:
    pdl_interp.check_operation_name of %17 is "math.sin" -> ^bb106, ^bb1
  ^bb106:
    pdl_interp.check_operand_count of %17 is 1 -> ^bb107, ^bb1
  ^bb107:
    pdl_interp.check_result_count of %17 is 1 -> ^bb108, ^bb1
  ^bb108:
    %21 = pdl_interp.get_result 0 of %17
    pdl_interp.is_not_null %21 : !pdl.value -> ^bb109, ^bb1
  ^bb109:
    pdl_interp.are_equal %21, %16 : !pdl.value -> ^bb110, ^bb1
  ^bb110:
    %22 = pdl_interp.get_operand 0 of %13
    pdl_interp.is_not_null %22 : !pdl.value -> ^bb111, ^bb1
  ^bb111:
    %23 = pdl_interp.get_value_type of %22 : !pdl.type
    %24 = pdl_interp.get_value_type of %19 : !pdl.type
    pdl_interp.are_equal %23, %24 : !pdl.type -> ^bb112, ^bb1
  ^bb112:
    %25 = pdl_interp.get_value_type of %6 : !pdl.type
    pdl_interp.are_equal %23, %25 : !pdl.type -> ^bb113, ^bb1
  ^bb113:
    %26 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %23, %26 : !pdl.type -> ^bb114, ^bb1
  ^bb114:
    pdl_interp.check_type %23 is f32 -> ^bb115, ^bb1
  ^bb115:
    %27 = pdl_interp.get_value_type of %18 : !pdl.type
    pdl_interp.are_equal %23, %27 : !pdl.type -> ^bb116, ^bb117
  ^bb117:
    %28 = pdl_interp.get_operand 0 of %12
    pdl_interp.is_not_null %28 : !pdl.value -> ^bb118, ^bb1
  ^bb118:
    %29 = pdl_interp.get_value_type of %18 : !pdl.type
    pdl_interp.are_equal %23, %29 : !pdl.type -> ^bb119, ^bb1
  ^bb119:
    %30 = pdl_interp.get_value_type of %10 : !pdl.type
    pdl_interp.are_equal %23, %30 : !pdl.type -> ^bb120, ^bb1
  ^bb120:
    %31 = pdl_interp.get_value_type of %21 : !pdl.type
    pdl_interp.are_equal %23, %31 : !pdl.type -> ^bb121, ^bb1
  ^bb121:
    %32 = pdl_interp.get_value_type of %20 : !pdl.type
    pdl_interp.are_equal %23, %32 : !pdl.type -> ^bb122, ^bb1
  ^bb122:
    %33 = pdl_interp.get_operand 0 of %15
    pdl_interp.are_equal %22, %33 : !pdl.value -> ^bb123, ^bb1
  ^bb123:
    %34 = pdl_interp.get_value_type of %28 : !pdl.type
    pdl_interp.are_equal %23, %34 : !pdl.type -> ^bb124, ^bb1
  ^bb124:
    %35 = pdl_interp.get_operand 0 of %17
    pdl_interp.are_equal %28, %35 : !pdl.value -> ^bb125, ^bb1
  ^bb125:
    pdl_interp.record_match @rewriters::@cos_diff_rev(%22, %28, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb1
  ^bb116:
    %36 = pdl_interp.get_value_type of %10 : !pdl.type
    pdl_interp.are_equal %23, %36 : !pdl.type -> ^bb126, ^bb117
  ^bb126:
    %37 = pdl_interp.get_value_type of %21 : !pdl.type
    pdl_interp.are_equal %23, %37 : !pdl.type -> ^bb127, ^bb117
  ^bb127:
    %38 = pdl_interp.get_value_type of %20 : !pdl.type
    pdl_interp.are_equal %23, %38 : !pdl.type -> ^bb128, ^bb117
  ^bb128:
    %39 = pdl_interp.get_operand 0 of %15
    pdl_interp.are_equal %22, %39 : !pdl.value -> ^bb129, ^bb117
  ^bb129:
    %40 = pdl_interp.get_operand 0 of %12
    pdl_interp.are_equal %22, %40 : !pdl.value -> ^bb130, ^bb117
  ^bb130:
    %41 = pdl_interp.get_operand 0 of %17
    pdl_interp.are_equal %22, %41 : !pdl.value -> ^bb131, ^bb117
  ^bb131:
    pdl_interp.record_match @rewriters::@cos_sin_sum(%0 : !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb117
  ^bb96:
    pdl_interp.check_operand_count of %13 is 1 -> ^bb132, ^bb1
  ^bb132:
    pdl_interp.check_result_count of %13 is 1 -> ^bb133, ^bb1
  ^bb133:
    %42 = pdl_interp.get_result 0 of %13
    pdl_interp.is_not_null %42 : !pdl.value -> ^bb134, ^bb1
  ^bb134:
    pdl_interp.are_equal %42, %7 : !pdl.value -> ^bb135, ^bb1
  ^bb135:
    pdl_interp.check_operation_name of %15 is "math.cos" -> ^bb136, ^bb1
  ^bb136:
    pdl_interp.check_operand_count of %15 is 1 -> ^bb137, ^bb1
  ^bb137:
    pdl_interp.check_result_count of %15 is 1 -> ^bb138, ^bb1
  ^bb138:
    %43 = pdl_interp.get_result 0 of %15
    pdl_interp.is_not_null %43 : !pdl.value -> ^bb139, ^bb1
  ^bb139:
    pdl_interp.are_equal %43, %14 : !pdl.value -> ^bb140, ^bb1
  ^bb140:
    pdl_interp.check_operation_name of %17 is "math.sin" -> ^bb141, ^bb1
  ^bb141:
    pdl_interp.check_operand_count of %17 is 1 -> ^bb142, ^bb1
  ^bb142:
    pdl_interp.check_result_count of %17 is 1 -> ^bb143, ^bb1
  ^bb143:
    %44 = pdl_interp.get_result 0 of %17
    pdl_interp.is_not_null %44 : !pdl.value -> ^bb144, ^bb1
  ^bb144:
    pdl_interp.are_equal %44, %16 : !pdl.value -> ^bb145, ^bb1
  ^bb145:
    %45 = pdl_interp.get_operand 0 of %13
    pdl_interp.is_not_null %45 : !pdl.value -> ^bb146, ^bb1
  ^bb146:
    %46 = pdl_interp.get_value_type of %45 : !pdl.type
    %47 = pdl_interp.get_value_type of %42 : !pdl.type
    pdl_interp.are_equal %46, %47 : !pdl.type -> ^bb147, ^bb1
  ^bb147:
    %48 = pdl_interp.get_value_type of %6 : !pdl.type
    pdl_interp.are_equal %46, %48 : !pdl.type -> ^bb148, ^bb1
  ^bb148:
    %49 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %46, %49 : !pdl.type -> ^bb149, ^bb1
  ^bb149:
    pdl_interp.check_type %46 is f32 -> ^bb150, ^bb1
  ^bb150:
    %50 = pdl_interp.get_operand 0 of %12
    pdl_interp.is_not_null %50 : !pdl.value -> ^bb151, ^bb1
  ^bb151:
    %51 = pdl_interp.get_value_type of %18 : !pdl.type
    pdl_interp.are_equal %46, %51 : !pdl.type -> ^bb152, ^bb1
  ^bb152:
    %52 = pdl_interp.get_value_type of %10 : !pdl.type
    pdl_interp.are_equal %46, %52 : !pdl.type -> ^bb153, ^bb1
  ^bb153:
    %53 = pdl_interp.get_value_type of %44 : !pdl.type
    pdl_interp.are_equal %46, %53 : !pdl.type -> ^bb154, ^bb1
  ^bb154:
    %54 = pdl_interp.get_value_type of %43 : !pdl.type
    pdl_interp.are_equal %46, %54 : !pdl.type -> ^bb155, ^bb1
  ^bb155:
    %55 = pdl_interp.get_operand 0 of %15
    pdl_interp.are_equal %45, %55 : !pdl.value -> ^bb156, ^bb1
  ^bb156:
    %56 = pdl_interp.get_value_type of %50 : !pdl.type
    pdl_interp.are_equal %46, %56 : !pdl.type -> ^bb157, ^bb1
  ^bb157:
    %57 = pdl_interp.get_operand 0 of %17
    pdl_interp.are_equal %50, %57 : !pdl.value -> ^bb158, ^bb1
  ^bb158:
    pdl_interp.record_match @rewriters::@sin_sum_rev(%45, %50, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb1
  ^bb89:
    pdl_interp.check_operand_count of %12 is 1 -> ^bb159, ^bb1
  ^bb159:
    pdl_interp.check_result_count of %12 is 1 -> ^bb160, ^bb1
  ^bb160:
    %58 = pdl_interp.get_result 0 of %12
    pdl_interp.is_not_null %58 : !pdl.value -> ^bb161, ^bb1
  ^bb161:
    pdl_interp.are_equal %58, %11 : !pdl.value -> ^bb162, ^bb1
  ^bb162:
    pdl_interp.check_operation_name of %13 is "math.sinh" -> ^bb163, ^bb1
  ^bb163:
    pdl_interp.check_operand_count of %13 is 1 -> ^bb164, ^bb1
  ^bb164:
    pdl_interp.check_result_count of %13 is 1 -> ^bb165, ^bb1
  ^bb165:
    %59 = pdl_interp.get_result 0 of %13
    pdl_interp.is_not_null %59 : !pdl.value -> ^bb166, ^bb1
  ^bb166:
    pdl_interp.are_equal %59, %7 : !pdl.value -> ^bb167, ^bb1
  ^bb167:
    pdl_interp.check_operation_name of %15 is "math.cosh" -> ^bb168, ^bb1
  ^bb168:
    pdl_interp.check_operand_count of %15 is 1 -> ^bb169, ^bb1
  ^bb169:
    pdl_interp.check_result_count of %15 is 1 -> ^bb170, ^bb1
  ^bb170:
    %60 = pdl_interp.get_result 0 of %15
    pdl_interp.is_not_null %60 : !pdl.value -> ^bb171, ^bb1
  ^bb171:
    pdl_interp.are_equal %60, %14 : !pdl.value -> ^bb172, ^bb1
  ^bb172:
    pdl_interp.check_operation_name of %17 is "math.cosh" -> ^bb173, ^bb1
  ^bb173:
    pdl_interp.check_operand_count of %17 is 1 -> ^bb174, ^bb1
  ^bb174:
    pdl_interp.check_result_count of %17 is 1 -> ^bb175, ^bb1
  ^bb175:
    %61 = pdl_interp.get_result 0 of %17
    pdl_interp.is_not_null %61 : !pdl.value -> ^bb176, ^bb1
  ^bb176:
    pdl_interp.are_equal %61, %16 : !pdl.value -> ^bb177, ^bb1
  ^bb177:
    %62 = pdl_interp.get_operand 0 of %13
    pdl_interp.is_not_null %62 : !pdl.value -> ^bb178, ^bb1
  ^bb178:
    %63 = pdl_interp.get_value_type of %62 : !pdl.type
    %64 = pdl_interp.get_value_type of %59 : !pdl.type
    pdl_interp.are_equal %63, %64 : !pdl.type -> ^bb179, ^bb1
  ^bb179:
    %65 = pdl_interp.get_value_type of %6 : !pdl.type
    pdl_interp.are_equal %63, %65 : !pdl.type -> ^bb180, ^bb1
  ^bb180:
    %66 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %63, %66 : !pdl.type -> ^bb181, ^bb1
  ^bb181:
    pdl_interp.check_type %63 is f32 -> ^bb182, ^bb1
  ^bb182:
    %67 = pdl_interp.get_value_type of %58 : !pdl.type
    pdl_interp.are_equal %63, %67 : !pdl.type -> ^bb183, ^bb1
  ^bb183:
    %68 = pdl_interp.get_value_type of %10 : !pdl.type
    pdl_interp.are_equal %63, %68 : !pdl.type -> ^bb184, ^bb1
  ^bb184:
    %69 = pdl_interp.get_value_type of %61 : !pdl.type
    pdl_interp.are_equal %63, %69 : !pdl.type -> ^bb185, ^bb1
  ^bb185:
    %70 = pdl_interp.get_value_type of %60 : !pdl.type
    pdl_interp.are_equal %63, %70 : !pdl.type -> ^bb186, ^bb1
  ^bb186:
    %71 = pdl_interp.get_operand 0 of %15
    pdl_interp.are_equal %62, %71 : !pdl.value -> ^bb187, ^bb1
  ^bb187:
    %72 = pdl_interp.get_operand 0 of %12
    pdl_interp.are_equal %62, %72 : !pdl.value -> ^bb188, ^bb1
  ^bb188:
    %73 = pdl_interp.get_operand 0 of %17
    pdl_interp.are_equal %62, %73 : !pdl.value -> ^bb189, ^bb1
  ^bb189:
    pdl_interp.record_match @rewriters::@cosh_2_rev(%62, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb1
  ^bb90:
    pdl_interp.check_operand_count of %12 is 1 -> ^bb190, ^bb1
  ^bb190:
    pdl_interp.check_result_count of %12 is 1 -> ^bb191, ^bb1
  ^bb191:
    %74 = pdl_interp.get_result 0 of %12
    pdl_interp.is_not_null %74 : !pdl.value -> ^bb192, ^bb1
  ^bb192:
    pdl_interp.are_equal %74, %11 : !pdl.value -> ^bb193, ^bb1
  ^bb193:
    pdl_interp.switch_operation_name of %13 to ["math.sinh", "math.cosh"](^bb194, ^bb195) -> ^bb1
  ^bb194:
    pdl_interp.check_operand_count of %13 is 1 -> ^bb196, ^bb1
  ^bb196:
    pdl_interp.check_result_count of %13 is 1 -> ^bb197, ^bb1
  ^bb197:
    %75 = pdl_interp.get_result 0 of %13
    pdl_interp.is_not_null %75 : !pdl.value -> ^bb198, ^bb1
  ^bb198:
    pdl_interp.are_equal %75, %7 : !pdl.value -> ^bb199, ^bb1
  ^bb199:
    pdl_interp.check_operation_name of %15 is "math.cosh" -> ^bb200, ^bb1
  ^bb200:
    pdl_interp.check_operand_count of %15 is 1 -> ^bb201, ^bb1
  ^bb201:
    pdl_interp.check_result_count of %15 is 1 -> ^bb202, ^bb1
  ^bb202:
    %76 = pdl_interp.get_result 0 of %15
    pdl_interp.is_not_null %76 : !pdl.value -> ^bb203, ^bb1
  ^bb203:
    pdl_interp.are_equal %76, %14 : !pdl.value -> ^bb204, ^bb1
  ^bb204:
    pdl_interp.check_operation_name of %17 is "math.sinh" -> ^bb205, ^bb1
  ^bb205:
    pdl_interp.check_operand_count of %17 is 1 -> ^bb206, ^bb1
  ^bb206:
    pdl_interp.check_result_count of %17 is 1 -> ^bb207, ^bb1
  ^bb207:
    %77 = pdl_interp.get_result 0 of %17
    pdl_interp.is_not_null %77 : !pdl.value -> ^bb208, ^bb1
  ^bb208:
    pdl_interp.are_equal %77, %16 : !pdl.value -> ^bb209, ^bb1
  ^bb209:
    %78 = pdl_interp.get_operand 0 of %13
    pdl_interp.is_not_null %78 : !pdl.value -> ^bb210, ^bb1
  ^bb210:
    %79 = pdl_interp.get_value_type of %78 : !pdl.type
    %80 = pdl_interp.get_value_type of %75 : !pdl.type
    pdl_interp.are_equal %79, %80 : !pdl.type -> ^bb211, ^bb1
  ^bb211:
    %81 = pdl_interp.get_value_type of %6 : !pdl.type
    pdl_interp.are_equal %79, %81 : !pdl.type -> ^bb212, ^bb1
  ^bb212:
    %82 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %79, %82 : !pdl.type -> ^bb213, ^bb1
  ^bb213:
    pdl_interp.check_type %79 is f32 -> ^bb214, ^bb1
  ^bb214:
    %83 = pdl_interp.get_operand 0 of %12
    pdl_interp.is_not_null %83 : !pdl.value -> ^bb215, ^bb1
  ^bb215:
    %84 = pdl_interp.get_value_type of %74 : !pdl.type
    pdl_interp.are_equal %79, %84 : !pdl.type -> ^bb216, ^bb1
  ^bb216:
    %85 = pdl_interp.get_value_type of %10 : !pdl.type
    pdl_interp.are_equal %79, %85 : !pdl.type -> ^bb217, ^bb1
  ^bb217:
    %86 = pdl_interp.get_value_type of %77 : !pdl.type
    pdl_interp.are_equal %79, %86 : !pdl.type -> ^bb218, ^bb1
  ^bb218:
    %87 = pdl_interp.get_value_type of %76 : !pdl.type
    pdl_interp.are_equal %79, %87 : !pdl.type -> ^bb219, ^bb1
  ^bb219:
    %88 = pdl_interp.get_operand 0 of %15
    pdl_interp.are_equal %78, %88 : !pdl.value -> ^bb220, ^bb1
  ^bb220:
    %89 = pdl_interp.get_value_type of %83 : !pdl.type
    pdl_interp.are_equal %79, %89 : !pdl.type -> ^bb221, ^bb1
  ^bb221:
    %90 = pdl_interp.get_operand 0 of %17
    pdl_interp.are_equal %83, %90 : !pdl.value -> ^bb222, ^bb1
  ^bb222:
    pdl_interp.record_match @rewriters::@sinh_sum_rev(%78, %83, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb1
  ^bb195:
    pdl_interp.check_operand_count of %13 is 1 -> ^bb223, ^bb1
  ^bb223:
    pdl_interp.check_result_count of %13 is 1 -> ^bb224, ^bb1
  ^bb224:
    %91 = pdl_interp.get_result 0 of %13
    pdl_interp.is_not_null %91 : !pdl.value -> ^bb225, ^bb1
  ^bb225:
    pdl_interp.are_equal %91, %7 : !pdl.value -> ^bb226, ^bb1
  ^bb226:
    pdl_interp.check_operation_name of %15 is "math.sinh" -> ^bb227, ^bb1
  ^bb227:
    pdl_interp.check_operand_count of %15 is 1 -> ^bb228, ^bb1
  ^bb228:
    pdl_interp.check_result_count of %15 is 1 -> ^bb229, ^bb1
  ^bb229:
    %92 = pdl_interp.get_result 0 of %15
    pdl_interp.is_not_null %92 : !pdl.value -> ^bb230, ^bb1
  ^bb230:
    pdl_interp.are_equal %92, %14 : !pdl.value -> ^bb231, ^bb1
  ^bb231:
    pdl_interp.check_operation_name of %17 is "math.sinh" -> ^bb232, ^bb1
  ^bb232:
    pdl_interp.check_operand_count of %17 is 1 -> ^bb233, ^bb1
  ^bb233:
    pdl_interp.check_result_count of %17 is 1 -> ^bb234, ^bb1
  ^bb234:
    %93 = pdl_interp.get_result 0 of %17
    pdl_interp.is_not_null %93 : !pdl.value -> ^bb235, ^bb1
  ^bb235:
    pdl_interp.are_equal %93, %16 : !pdl.value -> ^bb236, ^bb1
  ^bb236:
    %94 = pdl_interp.get_operand 0 of %13
    pdl_interp.is_not_null %94 : !pdl.value -> ^bb237, ^bb1
  ^bb237:
    %95 = pdl_interp.get_value_type of %94 : !pdl.type
    %96 = pdl_interp.get_value_type of %91 : !pdl.type
    pdl_interp.are_equal %95, %96 : !pdl.type -> ^bb238, ^bb1
  ^bb238:
    %97 = pdl_interp.get_value_type of %6 : !pdl.type
    pdl_interp.are_equal %95, %97 : !pdl.type -> ^bb239, ^bb1
  ^bb239:
    %98 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %95, %98 : !pdl.type -> ^bb240, ^bb1
  ^bb240:
    pdl_interp.check_type %95 is f32 -> ^bb241, ^bb1
  ^bb241:
    %99 = pdl_interp.get_operand 0 of %12
    pdl_interp.is_not_null %99 : !pdl.value -> ^bb242, ^bb1
  ^bb242:
    %100 = pdl_interp.get_value_type of %74 : !pdl.type
    pdl_interp.are_equal %95, %100 : !pdl.type -> ^bb243, ^bb1
  ^bb243:
    %101 = pdl_interp.get_value_type of %10 : !pdl.type
    pdl_interp.are_equal %95, %101 : !pdl.type -> ^bb244, ^bb1
  ^bb244:
    %102 = pdl_interp.get_value_type of %93 : !pdl.type
    pdl_interp.are_equal %95, %102 : !pdl.type -> ^bb245, ^bb1
  ^bb245:
    %103 = pdl_interp.get_value_type of %92 : !pdl.type
    pdl_interp.are_equal %95, %103 : !pdl.type -> ^bb246, ^bb1
  ^bb246:
    %104 = pdl_interp.get_operand 0 of %15
    pdl_interp.are_equal %94, %104 : !pdl.value -> ^bb247, ^bb1
  ^bb247:
    %105 = pdl_interp.get_value_type of %99 : !pdl.type
    pdl_interp.are_equal %95, %105 : !pdl.type -> ^bb248, ^bb1
  ^bb248:
    %106 = pdl_interp.get_operand 0 of %17
    pdl_interp.are_equal %99, %106 : !pdl.value -> ^bb249, ^bb1
  ^bb249:
    pdl_interp.record_match @rewriters::@cosh_sum_rev(%94, %99, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb1
  ^bb76:
    pdl_interp.check_operand_count of %5 is 0 -> ^bb250, ^bb1
  ^bb250:
    pdl_interp.check_result_count of %5 is 1 -> ^bb251, ^bb1
  ^bb251:
    %107 = pdl_interp.get_result 0 of %5
    pdl_interp.is_not_null %107 : !pdl.value -> ^bb252, ^bb1
  ^bb252:
    pdl_interp.are_equal %107, %4 : !pdl.value -> ^bb253, ^bb1
  ^bb253:
    %108 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %108 : !pdl.value -> ^bb254, ^bb1
  ^bb254:
    %109 = pdl_interp.get_defining_op of %108 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %109 : !pdl.operation -> ^bb255, ^bb1
  ^bb255:
    %110 = pdl_interp.get_defining_op of %7 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %110 : !pdl.operation -> ^bb256, ^bb1
  ^bb256:
    pdl_interp.switch_operation_name of %109 to ["math.cos", "math.sin"](^bb257, ^bb258) -> ^bb1
  ^bb257:
    pdl_interp.check_operand_count of %109 is 1 -> ^bb259, ^bb1
  ^bb259:
    pdl_interp.check_result_count of %109 is 1 -> ^bb260, ^bb1
  ^bb260:
    %111 = pdl_interp.get_result 0 of %109
    pdl_interp.is_not_null %111 : !pdl.value -> ^bb261, ^bb1
  ^bb261:
    pdl_interp.are_equal %111, %108 : !pdl.value -> ^bb262, ^bb1
  ^bb262:
    pdl_interp.check_operation_name of %110 is "math.cos" -> ^bb263, ^bb1
  ^bb263:
    pdl_interp.check_operand_count of %110 is 1 -> ^bb264, ^bb1
  ^bb264:
    pdl_interp.check_result_count of %110 is 1 -> ^bb265, ^bb1
  ^bb265:
    %112 = pdl_interp.get_result 0 of %110
    pdl_interp.is_not_null %112 : !pdl.value -> ^bb266, ^bb1
  ^bb266:
    pdl_interp.are_equal %112, %7 : !pdl.value -> ^bb267, ^bb1
  ^bb267:
    %113 = pdl_interp.get_operand 0 of %110
    pdl_interp.is_not_null %113 : !pdl.value -> ^bb268, ^bb1
  ^bb268:
    %114 = pdl_interp.get_value_type of %113 : !pdl.type
    %115 = pdl_interp.get_value_type of %112 : !pdl.type
    pdl_interp.are_equal %114, %115 : !pdl.type -> ^bb269, ^bb1
  ^bb269:
    %116 = pdl_interp.get_value_type of %6 : !pdl.type
    pdl_interp.are_equal %114, %116 : !pdl.type -> ^bb270, ^bb1
  ^bb270:
    %117 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %114, %117 : !pdl.type -> ^bb271, ^bb1
  ^bb271:
    pdl_interp.check_type %114 is f32 -> ^bb272, ^bb1
  ^bb272:
    %118 = pdl_interp.get_value_type of %111 : !pdl.type
    pdl_interp.are_equal %114, %118 : !pdl.type -> ^bb273, ^bb1
  ^bb273:
    %119 = pdl_interp.get_attribute "value" of %5
    pdl_interp.is_not_null %119 : !pdl.attribute -> ^bb274, ^bb1
  ^bb274:
    pdl_interp.check_attribute %119 is -1.000000e+00 : f32 -> ^bb275, ^bb1
  ^bb275:
    %120 = pdl_interp.get_value_type of %107 : !pdl.type
    pdl_interp.are_equal %114, %120 : !pdl.type -> ^bb276, ^bb1
  ^bb276:
    %121 = pdl_interp.get_operand 0 of %109
    pdl_interp.are_equal %113, %121 : !pdl.value -> ^bb277, ^bb1
  ^bb277:
    pdl_interp.record_match @rewriters::@_1_add_cos(%113, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb1
  ^bb258:
    pdl_interp.check_operand_count of %109 is 1 -> ^bb278, ^bb1
  ^bb278:
    pdl_interp.check_result_count of %109 is 1 -> ^bb279, ^bb1
  ^bb279:
    %122 = pdl_interp.get_result 0 of %109
    pdl_interp.is_not_null %122 : !pdl.value -> ^bb280, ^bb1
  ^bb280:
    pdl_interp.are_equal %122, %108 : !pdl.value -> ^bb281, ^bb1
  ^bb281:
    pdl_interp.check_operation_name of %110 is "math.sin" -> ^bb282, ^bb1
  ^bb282:
    pdl_interp.check_operand_count of %110 is 1 -> ^bb283, ^bb1
  ^bb283:
    pdl_interp.check_result_count of %110 is 1 -> ^bb284, ^bb1
  ^bb284:
    %123 = pdl_interp.get_result 0 of %110
    pdl_interp.is_not_null %123 : !pdl.value -> ^bb285, ^bb1
  ^bb285:
    pdl_interp.are_equal %123, %7 : !pdl.value -> ^bb286, ^bb1
  ^bb286:
    %124 = pdl_interp.get_operand 0 of %110
    pdl_interp.is_not_null %124 : !pdl.value -> ^bb287, ^bb1
  ^bb287:
    %125 = pdl_interp.get_value_type of %124 : !pdl.type
    %126 = pdl_interp.get_value_type of %123 : !pdl.type
    pdl_interp.are_equal %125, %126 : !pdl.type -> ^bb288, ^bb1
  ^bb288:
    %127 = pdl_interp.get_value_type of %6 : !pdl.type
    pdl_interp.are_equal %125, %127 : !pdl.type -> ^bb289, ^bb1
  ^bb289:
    %128 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %125, %128 : !pdl.type -> ^bb290, ^bb1
  ^bb290:
    pdl_interp.check_type %125 is f32 -> ^bb291, ^bb1
  ^bb291:
    %129 = pdl_interp.get_value_type of %122 : !pdl.type
    pdl_interp.are_equal %125, %129 : !pdl.type -> ^bb292, ^bb1
  ^bb292:
    %130 = pdl_interp.get_attribute "value" of %5
    pdl_interp.is_not_null %130 : !pdl.attribute -> ^bb293, ^bb1
  ^bb293:
    pdl_interp.check_attribute %130 is -1.000000e+00 : f32 -> ^bb294, ^bb1
  ^bb294:
    %131 = pdl_interp.get_value_type of %107 : !pdl.type
    pdl_interp.are_equal %125, %131 : !pdl.type -> ^bb295, ^bb1
  ^bb295:
    %132 = pdl_interp.get_operand 0 of %109
    pdl_interp.are_equal %124, %132 : !pdl.value -> ^bb296, ^bb1
  ^bb296:
    pdl_interp.record_match @rewriters::@_1_add_sin(%124, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb1
  ^bb73:
    %133 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %8, %133 : !pdl.type -> ^bb297, ^bb74
  ^bb297:
    pdl_interp.check_type %8 is f32 -> ^bb298, ^bb74
  ^bb298:
    pdl_interp.switch_operation_name of %5 to ["arith.mulf", "arith.constant"](^bb299, ^bb300) -> ^bb74
  ^bb299:
    pdl_interp.check_operand_count of %5 is 2 -> ^bb301, ^bb74
  ^bb301:
    pdl_interp.check_result_count of %5 is 1 -> ^bb302, ^bb74
  ^bb302:
    %134 = pdl_interp.get_result 0 of %5
    pdl_interp.is_not_null %134 : !pdl.value -> ^bb303, ^bb74
  ^bb303:
    pdl_interp.are_equal %134, %4 : !pdl.value -> ^bb304, ^bb74
  ^bb304:
    %135 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %135 : !pdl.value -> ^bb305, ^bb74
  ^bb305:
    %136 = pdl_interp.get_operand 1 of %5
    pdl_interp.is_not_null %136 : !pdl.value -> ^bb306, ^bb307
  ^bb307:
    %137 = pdl_interp.get_operand 0 of %5
    pdl_interp.is_not_null %137 : !pdl.value -> ^bb308, ^bb74
  ^bb308:
    %138 = pdl_interp.get_value_type of %134 : !pdl.type
    pdl_interp.are_equal %8, %138 : !pdl.type -> ^bb309, ^bb74
  ^bb309:
    %139 = pdl_interp.get_value_type of %135 : !pdl.type
    pdl_interp.are_equal %8, %139 : !pdl.type -> ^bb310, ^bb74
  ^bb310:
    %140 = pdl_interp.get_value_type of %137 : !pdl.type
    pdl_interp.are_equal %8, %140 : !pdl.type -> ^bb311, ^bb74
  ^bb311:
    %141 = pdl_interp.get_operand 1 of %5
    pdl_interp.are_equal %135, %141 : !pdl.value -> ^bb312, ^bb74
  ^bb312:
    pdl_interp.record_match @rewriters::@distribute_rgt_out(%7, %137, %135, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb74
  ^bb306:
    %142 = pdl_interp.get_value_type of %134 : !pdl.type
    pdl_interp.are_equal %8, %142 : !pdl.type -> ^bb313, ^bb307
  ^bb313:
    %143 = pdl_interp.get_value_type of %135 : !pdl.type
    pdl_interp.are_equal %8, %143 : !pdl.type -> ^bb314, ^bb307
  ^bb314:
    %144 = pdl_interp.get_operand 0 of %5
    pdl_interp.are_equal %7, %144 : !pdl.value -> ^bb315, ^bb307
  ^bb315:
    %145 = pdl_interp.get_value_type of %136 : !pdl.type
    pdl_interp.are_equal %8, %145 : !pdl.type -> ^bb316, ^bb307
  ^bb316:
    pdl_interp.record_match @rewriters::@distribute_lft_out(%135, %136, %7, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb307
  ^bb300:
    pdl_interp.check_operand_count of %5 is 0 -> ^bb317, ^bb74
  ^bb317:
    pdl_interp.check_result_count of %5 is 1 -> ^bb318, ^bb74
  ^bb318:
    %146 = pdl_interp.get_result 0 of %5
    pdl_interp.is_not_null %146 : !pdl.value -> ^bb319, ^bb74
  ^bb319:
    pdl_interp.are_equal %146, %4 : !pdl.value -> ^bb320, ^bb74
  ^bb320:
    %147 = pdl_interp.get_value_type of %146 : !pdl.type
    pdl_interp.are_equal %8, %147 : !pdl.type -> ^bb321, ^bb74
  ^bb321:
    %148 = pdl_interp.get_attribute "value" of %5
    pdl_interp.is_not_null %148 : !pdl.attribute -> ^bb322, ^bb74
  ^bb322:
    pdl_interp.check_attribute %148 is -1.000000e+00 : f32 -> ^bb323, ^bb74
  ^bb323:
    %149 = pdl_interp.get_operand 1 of %3
    pdl_interp.are_equal %7, %149 : !pdl.value -> ^bb324, ^bb74
  ^bb324:
    pdl_interp.record_match @rewriters::@difference_of_sqrsub_1(%7, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb74
  ^bb54:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb325, ^bb1
  ^bb325:
    pdl_interp.check_result_count of %3 is 1 -> ^bb326, ^bb1
  ^bb326:
    %150 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %150 : !pdl.value -> ^bb327, ^bb1
  ^bb327:
    pdl_interp.are_equal %150, %2 : !pdl.value -> ^bb328, ^bb1
  ^bb328:
    %151 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %151 : !pdl.value -> ^bb329, ^bb1
  ^bb329:
    pdl_interp.is_not_null %4 : !pdl.value -> ^bb330, ^bb1
  ^bb330:
    %152 = pdl_interp.get_value_type of %151 : !pdl.type
    %153 = pdl_interp.get_value_type of %150 : !pdl.type
    pdl_interp.are_equal %152, %153 : !pdl.type -> ^bb331, ^bb1
  ^bb331:
    %154 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %152, %154 : !pdl.type -> ^bb332, ^bb1
  ^bb332:
    pdl_interp.check_type %152 is f32 -> ^bb333, ^bb1
  ^bb333:
    pdl_interp.check_operation_name of %5 is "arith.negf" -> ^bb334, ^bb1
  ^bb334:
    pdl_interp.check_operand_count of %5 is 1 -> ^bb335, ^bb1
  ^bb335:
    pdl_interp.check_result_count of %5 is 1 -> ^bb336, ^bb1
  ^bb336:
    %155 = pdl_interp.get_result 0 of %5
    pdl_interp.is_not_null %155 : !pdl.value -> ^bb337, ^bb1
  ^bb337:
    pdl_interp.are_equal %155, %4 : !pdl.value -> ^bb338, ^bb1
  ^bb338:
    %156 = pdl_interp.get_operand 0 of %5
    pdl_interp.is_not_null %156 : !pdl.value -> ^bb339, ^bb1
  ^bb339:
    %157 = pdl_interp.get_value_type of %155 : !pdl.type
    pdl_interp.are_equal %152, %157 : !pdl.type -> ^bb340, ^bb1
  ^bb340:
    %158 = pdl_interp.get_value_type of %156 : !pdl.type
    pdl_interp.are_equal %152, %158 : !pdl.type -> ^bb341, ^bb1
  ^bb341:
    pdl_interp.record_match @rewriters::@distribute_neg_out(%151, %156, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb1
  ^bb55:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb342, ^bb1
  ^bb342:
    pdl_interp.check_result_count of %3 is 1 -> ^bb343, ^bb1
  ^bb343:
    %159 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %159 : !pdl.value -> ^bb344, ^bb1
  ^bb344:
    pdl_interp.are_equal %159, %2 : !pdl.value -> ^bb345, ^bb1
  ^bb345:
    %160 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %160 : !pdl.value -> ^bb346, ^bb1
  ^bb346:
    pdl_interp.is_not_null %4 : !pdl.value -> ^bb347, ^bb1
  ^bb347:
    pdl_interp.check_operation_name of %5 is "math.powf" -> ^bb348, ^bb1
  ^bb348:
    pdl_interp.check_operand_count of %5 is 2 -> ^bb349, ^bb1
  ^bb349:
    pdl_interp.check_result_count of %5 is 1 -> ^bb350, ^bb1
  ^bb350:
    %161 = pdl_interp.get_result 0 of %5
    pdl_interp.is_not_null %161 : !pdl.value -> ^bb351, ^bb1
  ^bb351:
    pdl_interp.are_equal %161, %4 : !pdl.value -> ^bb352, ^bb1
  ^bb352:
    %162 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %162 : !pdl.value -> ^bb353, ^bb1
  ^bb353:
    %163 = pdl_interp.get_defining_op of %162 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %163 : !pdl.operation -> ^bb354, ^bb1
  ^bb354:
    %164 = pdl_interp.get_defining_op of %160 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %164 : !pdl.operation -> ^bb355, ^bb1
  ^bb355:
    %165 = pdl_interp.get_operand 1 of %5
    %166 = pdl_interp.get_defining_op of %165 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %166 : !pdl.operation -> ^bb356, ^bb1
  ^bb356:
    pdl_interp.is_not_null %165 : !pdl.value -> ^bb357, ^bb1
  ^bb357:
    pdl_interp.check_operation_name of %163 is "arith.mulf" -> ^bb358, ^bb1
  ^bb358:
    pdl_interp.check_operand_count of %163 is 2 -> ^bb359, ^bb1
  ^bb359:
    pdl_interp.check_result_count of %163 is 1 -> ^bb360, ^bb1
  ^bb360:
    %167 = pdl_interp.get_result 0 of %163
    pdl_interp.is_not_null %167 : !pdl.value -> ^bb361, ^bb1
  ^bb361:
    pdl_interp.are_equal %167, %162 : !pdl.value -> ^bb362, ^bb1
  ^bb362:
    pdl_interp.check_operation_name of %164 is "math.powf" -> ^bb363, ^bb1
  ^bb363:
    pdl_interp.check_operand_count of %164 is 2 -> ^bb364, ^bb1
  ^bb364:
    pdl_interp.check_result_count of %164 is 1 -> ^bb365, ^bb1
  ^bb365:
    %168 = pdl_interp.get_result 0 of %164
    pdl_interp.is_not_null %168 : !pdl.value -> ^bb366, ^bb1
  ^bb366:
    pdl_interp.are_equal %168, %160 : !pdl.value -> ^bb367, ^bb1
  ^bb367:
    pdl_interp.check_operation_name of %166 is "arith.constant" -> ^bb368, ^bb1
  ^bb368:
    pdl_interp.check_operand_count of %166 is 0 -> ^bb369, ^bb1
  ^bb369:
    pdl_interp.check_result_count of %166 is 1 -> ^bb370, ^bb1
  ^bb370:
    %169 = pdl_interp.get_result 0 of %166
    pdl_interp.is_not_null %169 : !pdl.value -> ^bb371, ^bb1
  ^bb371:
    pdl_interp.are_equal %169, %165 : !pdl.value -> ^bb372, ^bb1
  ^bb372:
    %170 = pdl_interp.get_operand 0 of %164
    pdl_interp.is_not_null %170 : !pdl.value -> ^bb373, ^bb1
  ^bb373:
    %171 = pdl_interp.get_value_type of %170 : !pdl.type
    %172 = pdl_interp.get_value_type of %168 : !pdl.type
    pdl_interp.are_equal %171, %172 : !pdl.type -> ^bb374, ^bb1
  ^bb374:
    %173 = pdl_interp.get_value_type of %159 : !pdl.type
    pdl_interp.are_equal %171, %173 : !pdl.type -> ^bb375, ^bb1
  ^bb375:
    %174 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %171, %174 : !pdl.type -> ^bb376, ^bb1
  ^bb376:
    pdl_interp.check_type %171 is f32 -> ^bb377, ^bb1
  ^bb377:
    %175 = pdl_interp.get_operand 0 of %163
    pdl_interp.is_not_null %175 : !pdl.value -> ^bb378, ^bb1
  ^bb378:
    %176 = pdl_interp.get_value_type of %167 : !pdl.type
    pdl_interp.are_equal %171, %176 : !pdl.type -> ^bb379, ^bb1
  ^bb379:
    %177 = pdl_interp.get_value_type of %161 : !pdl.type
    pdl_interp.are_equal %171, %177 : !pdl.type -> ^bb380, ^bb1
  ^bb380:
    %178 = pdl_interp.get_defining_op of %175 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %178 : !pdl.operation -> ^bb381, ^bb1
  ^bb381:
    %179 = pdl_interp.get_value_type of %169 : !pdl.type
    pdl_interp.are_equal %171, %179 : !pdl.type -> ^bb382, ^bb1
  ^bb382:
    pdl_interp.check_operation_name of %178 is "arith.constant" -> ^bb383, ^bb1
  ^bb383:
    pdl_interp.check_operand_count of %178 is 0 -> ^bb384, ^bb1
  ^bb384:
    pdl_interp.check_result_count of %178 is 1 -> ^bb385, ^bb1
  ^bb385:
    %180 = pdl_interp.get_result 0 of %178
    pdl_interp.is_not_null %180 : !pdl.value -> ^bb386, ^bb1
  ^bb386:
    pdl_interp.are_equal %180, %175 : !pdl.value -> ^bb387, ^bb1
  ^bb387:
    %181 = pdl_interp.get_operand 1 of %164
    %182 = pdl_interp.get_defining_op of %181 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %182 : !pdl.operation -> ^bb388, ^bb1
  ^bb388:
    %183 = pdl_interp.get_attribute "value" of %166
    pdl_interp.is_not_null %183 : !pdl.attribute -> ^bb389, ^bb1
  ^bb389:
    pdl_interp.check_attribute %183 is 2.000000e+00 : f32 -> ^bb390, ^bb1
  ^bb390:
    %184 = pdl_interp.get_operand 1 of %163
    %185 = pdl_interp.get_defining_op of %184 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %185 : !pdl.operation -> ^bb391, ^bb1
  ^bb391:
    pdl_interp.is_not_null %181 : !pdl.value -> ^bb392, ^bb1
  ^bb392:
    %186 = pdl_interp.get_value_type of %180 : !pdl.type
    pdl_interp.are_equal %186, %171 : !pdl.type -> ^bb393, ^bb1
  ^bb393:
    pdl_interp.is_not_null %184 : !pdl.value -> ^bb394, ^bb1
  ^bb394:
    pdl_interp.check_operation_name of %182 is "arith.constant" -> ^bb395, ^bb1
  ^bb395:
    pdl_interp.check_operand_count of %182 is 0 -> ^bb396, ^bb1
  ^bb396:
    pdl_interp.check_result_count of %182 is 1 -> ^bb397, ^bb1
  ^bb397:
    %187 = pdl_interp.get_result 0 of %182
    pdl_interp.is_not_null %187 : !pdl.value -> ^bb398, ^bb1
  ^bb398:
    pdl_interp.are_equal %187, %181 : !pdl.value -> ^bb399, ^bb1
  ^bb399:
    pdl_interp.check_operation_name of %185 is "arith.mulf" -> ^bb400, ^bb1
  ^bb400:
    pdl_interp.check_operand_count of %185 is 2 -> ^bb401, ^bb1
  ^bb401:
    pdl_interp.check_result_count of %185 is 1 -> ^bb402, ^bb1
  ^bb402:
    %188 = pdl_interp.get_result 0 of %185
    pdl_interp.is_not_null %188 : !pdl.value -> ^bb403, ^bb1
  ^bb403:
    pdl_interp.are_equal %188, %184 : !pdl.value -> ^bb404, ^bb1
  ^bb404:
    %189 = pdl_interp.get_operand 0 of %185
    pdl_interp.are_equal %189, %170 : !pdl.value -> ^bb405, ^bb1
  ^bb405:
    %190 = pdl_interp.get_value_type of %188 : !pdl.type
    pdl_interp.are_equal %190, %171 : !pdl.type -> ^bb406, ^bb1
  ^bb406:
    %191 = pdl_interp.get_attribute "value" of %182
    pdl_interp.is_not_null %191 : !pdl.attribute -> ^bb407, ^bb1
  ^bb407:
    pdl_interp.check_attribute %191 is 2.000000e+00 : f32 -> ^bb408, ^bb1
  ^bb408:
    %192 = pdl_interp.get_operand 1 of %185
    pdl_interp.is_not_null %192 : !pdl.value -> ^bb409, ^bb1
  ^bb409:
    %193 = pdl_interp.get_operand 0 of %5
    pdl_interp.are_equal %192, %193 : !pdl.value -> ^bb410, ^bb1
  ^bb410:
    %194 = pdl_interp.get_attribute "value" of %178
    pdl_interp.is_not_null %194 : !pdl.attribute -> ^bb411, ^bb1
  ^bb411:
    pdl_interp.check_attribute %194 is 2.000000e+00 : f32 -> ^bb412, ^bb1
  ^bb412:
    %195 = pdl_interp.get_value_type of %187 : !pdl.type
    pdl_interp.are_equal %195, %171 : !pdl.type -> ^bb413, ^bb1
  ^bb413:
    %196 = pdl_interp.get_value_type of %192 : !pdl.type
    pdl_interp.are_equal %196, %171 : !pdl.type -> ^bb414, ^bb1
  ^bb414:
    pdl_interp.record_match @rewriters::@sum_square_pow_rev(%170, %192, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb1
  ^bb56:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb415, ^bb1
  ^bb415:
    pdl_interp.check_result_count of %3 is 1 -> ^bb416, ^bb1
  ^bb416:
    %197 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %197 : !pdl.value -> ^bb417, ^bb1
  ^bb417:
    pdl_interp.are_equal %197, %2 : !pdl.value -> ^bb418, ^bb1
  ^bb418:
    %198 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %198 : !pdl.value -> ^bb419, ^bb1
  ^bb419:
    pdl_interp.is_not_null %4 : !pdl.value -> ^bb420, ^bb1
  ^bb420:
    pdl_interp.check_operation_name of %5 is "math.powf" -> ^bb421, ^bb1
  ^bb421:
    pdl_interp.check_operand_count of %5 is 2 -> ^bb422, ^bb1
  ^bb422:
    pdl_interp.check_result_count of %5 is 1 -> ^bb423, ^bb1
  ^bb423:
    %199 = pdl_interp.get_result 0 of %5
    pdl_interp.is_not_null %199 : !pdl.value -> ^bb424, ^bb1
  ^bb424:
    pdl_interp.are_equal %199, %4 : !pdl.value -> ^bb425, ^bb1
  ^bb425:
    %200 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %200 : !pdl.value -> ^bb426, ^bb1
  ^bb426:
    %201 = pdl_interp.get_defining_op of %200 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %201 : !pdl.operation -> ^bb427, ^bb1
  ^bb427:
    %202 = pdl_interp.get_defining_op of %198 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %202 : !pdl.operation -> ^bb428, ^bb1
  ^bb428:
    %203 = pdl_interp.get_operand 1 of %5
    %204 = pdl_interp.get_defining_op of %203 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %204 : !pdl.operation -> ^bb429, ^bb1
  ^bb429:
    pdl_interp.is_not_null %203 : !pdl.value -> ^bb430, ^bb1
  ^bb430:
    pdl_interp.check_operation_name of %201 is "arith.mulf" -> ^bb431, ^bb1
  ^bb431:
    pdl_interp.check_operand_count of %201 is 2 -> ^bb432, ^bb1
  ^bb432:
    pdl_interp.check_result_count of %201 is 1 -> ^bb433, ^bb1
  ^bb433:
    %205 = pdl_interp.get_result 0 of %201
    pdl_interp.is_not_null %205 : !pdl.value -> ^bb434, ^bb1
  ^bb434:
    pdl_interp.are_equal %205, %200 : !pdl.value -> ^bb435, ^bb1
  ^bb435:
    pdl_interp.check_operation_name of %202 is "math.powf" -> ^bb436, ^bb1
  ^bb436:
    pdl_interp.check_operand_count of %202 is 2 -> ^bb437, ^bb1
  ^bb437:
    pdl_interp.check_result_count of %202 is 1 -> ^bb438, ^bb1
  ^bb438:
    %206 = pdl_interp.get_result 0 of %202
    pdl_interp.is_not_null %206 : !pdl.value -> ^bb439, ^bb1
  ^bb439:
    pdl_interp.are_equal %206, %198 : !pdl.value -> ^bb440, ^bb1
  ^bb440:
    pdl_interp.check_operation_name of %204 is "arith.constant" -> ^bb441, ^bb1
  ^bb441:
    pdl_interp.check_operand_count of %204 is 0 -> ^bb442, ^bb1
  ^bb442:
    pdl_interp.check_result_count of %204 is 1 -> ^bb443, ^bb1
  ^bb443:
    %207 = pdl_interp.get_result 0 of %204
    pdl_interp.is_not_null %207 : !pdl.value -> ^bb444, ^bb1
  ^bb444:
    pdl_interp.are_equal %207, %203 : !pdl.value -> ^bb445, ^bb1
  ^bb445:
    %208 = pdl_interp.get_operand 0 of %202
    pdl_interp.is_not_null %208 : !pdl.value -> ^bb446, ^bb1
  ^bb446:
    %209 = pdl_interp.get_value_type of %208 : !pdl.type
    %210 = pdl_interp.get_value_type of %206 : !pdl.type
    pdl_interp.are_equal %209, %210 : !pdl.type -> ^bb447, ^bb1
  ^bb447:
    %211 = pdl_interp.get_value_type of %197 : !pdl.type
    pdl_interp.are_equal %209, %211 : !pdl.type -> ^bb448, ^bb1
  ^bb448:
    %212 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %209, %212 : !pdl.type -> ^bb449, ^bb1
  ^bb449:
    pdl_interp.check_type %209 is f32 -> ^bb450, ^bb1
  ^bb450:
    %213 = pdl_interp.get_operand 0 of %201
    pdl_interp.is_not_null %213 : !pdl.value -> ^bb451, ^bb1
  ^bb451:
    %214 = pdl_interp.get_value_type of %205 : !pdl.type
    pdl_interp.are_equal %209, %214 : !pdl.type -> ^bb452, ^bb1
  ^bb452:
    %215 = pdl_interp.get_value_type of %199 : !pdl.type
    pdl_interp.are_equal %209, %215 : !pdl.type -> ^bb453, ^bb1
  ^bb453:
    %216 = pdl_interp.get_defining_op of %213 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %216 : !pdl.operation -> ^bb454, ^bb1
  ^bb454:
    %217 = pdl_interp.get_value_type of %207 : !pdl.type
    pdl_interp.are_equal %209, %217 : !pdl.type -> ^bb455, ^bb1
  ^bb455:
    pdl_interp.check_operation_name of %216 is "arith.constant" -> ^bb456, ^bb1
  ^bb456:
    pdl_interp.check_operand_count of %216 is 0 -> ^bb457, ^bb1
  ^bb457:
    pdl_interp.check_result_count of %216 is 1 -> ^bb458, ^bb1
  ^bb458:
    %218 = pdl_interp.get_result 0 of %216
    pdl_interp.is_not_null %218 : !pdl.value -> ^bb459, ^bb1
  ^bb459:
    pdl_interp.are_equal %218, %213 : !pdl.value -> ^bb460, ^bb1
  ^bb460:
    %219 = pdl_interp.get_operand 1 of %202
    %220 = pdl_interp.get_defining_op of %219 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %220 : !pdl.operation -> ^bb461, ^bb1
  ^bb461:
    %221 = pdl_interp.get_attribute "value" of %204
    pdl_interp.is_not_null %221 : !pdl.attribute -> ^bb462, ^bb1
  ^bb462:
    pdl_interp.check_attribute %221 is 2.000000e+00 : f32 -> ^bb463, ^bb1
  ^bb463:
    %222 = pdl_interp.get_operand 1 of %201
    %223 = pdl_interp.get_defining_op of %222 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %223 : !pdl.operation -> ^bb464, ^bb1
  ^bb464:
    pdl_interp.is_not_null %219 : !pdl.value -> ^bb465, ^bb1
  ^bb465:
    %224 = pdl_interp.get_value_type of %218 : !pdl.type
    pdl_interp.are_equal %224, %209 : !pdl.type -> ^bb466, ^bb1
  ^bb466:
    pdl_interp.is_not_null %222 : !pdl.value -> ^bb467, ^bb1
  ^bb467:
    pdl_interp.check_operation_name of %220 is "arith.constant" -> ^bb468, ^bb1
  ^bb468:
    pdl_interp.check_operand_count of %220 is 0 -> ^bb469, ^bb1
  ^bb469:
    pdl_interp.check_result_count of %220 is 1 -> ^bb470, ^bb1
  ^bb470:
    %225 = pdl_interp.get_result 0 of %220
    pdl_interp.is_not_null %225 : !pdl.value -> ^bb471, ^bb1
  ^bb471:
    pdl_interp.are_equal %225, %219 : !pdl.value -> ^bb472, ^bb1
  ^bb472:
    pdl_interp.check_operation_name of %223 is "arith.mulf" -> ^bb473, ^bb1
  ^bb473:
    pdl_interp.check_operand_count of %223 is 2 -> ^bb474, ^bb1
  ^bb474:
    pdl_interp.check_result_count of %223 is 1 -> ^bb475, ^bb1
  ^bb475:
    %226 = pdl_interp.get_result 0 of %223
    pdl_interp.is_not_null %226 : !pdl.value -> ^bb476, ^bb1
  ^bb476:
    pdl_interp.are_equal %226, %222 : !pdl.value -> ^bb477, ^bb1
  ^bb477:
    %227 = pdl_interp.get_operand 0 of %223
    pdl_interp.are_equal %227, %208 : !pdl.value -> ^bb478, ^bb1
  ^bb478:
    %228 = pdl_interp.get_value_type of %226 : !pdl.type
    pdl_interp.are_equal %228, %209 : !pdl.type -> ^bb479, ^bb1
  ^bb479:
    %229 = pdl_interp.get_attribute "value" of %220
    pdl_interp.is_not_null %229 : !pdl.attribute -> ^bb480, ^bb1
  ^bb480:
    pdl_interp.check_attribute %229 is 2.000000e+00 : f32 -> ^bb481, ^bb1
  ^bb481:
    %230 = pdl_interp.get_operand 1 of %223
    pdl_interp.is_not_null %230 : !pdl.value -> ^bb482, ^bb1
  ^bb482:
    %231 = pdl_interp.get_operand 0 of %5
    pdl_interp.are_equal %230, %231 : !pdl.value -> ^bb483, ^bb1
  ^bb483:
    %232 = pdl_interp.get_attribute "value" of %216
    pdl_interp.is_not_null %232 : !pdl.attribute -> ^bb484, ^bb1
  ^bb484:
    pdl_interp.check_attribute %232 is 2.000000e+00 : f32 -> ^bb485, ^bb1
  ^bb485:
    %233 = pdl_interp.get_value_type of %225 : !pdl.type
    pdl_interp.are_equal %233, %209 : !pdl.type -> ^bb486, ^bb1
  ^bb486:
    %234 = pdl_interp.get_value_type of %230 : !pdl.type
    pdl_interp.are_equal %234, %209 : !pdl.type -> ^bb487, ^bb1
  ^bb487:
    pdl_interp.record_match @rewriters::@sub_square_pow_rev(%208, %230, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb1
  ^bb57:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb488, ^bb1
  ^bb488:
    pdl_interp.check_result_count of %3 is 1 -> ^bb489, ^bb1
  ^bb489:
    %235 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %235 : !pdl.value -> ^bb490, ^bb1
  ^bb490:
    pdl_interp.are_equal %235, %2 : !pdl.value -> ^bb491, ^bb1
  ^bb491:
    %236 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %236 : !pdl.value -> ^bb492, ^bb1
  ^bb492:
    pdl_interp.is_not_null %4 : !pdl.value -> ^bb493, ^bb1
  ^bb493:
    %237 = pdl_interp.get_value_type of %236 : !pdl.type
    %238 = pdl_interp.get_value_type of %235 : !pdl.type
    pdl_interp.are_equal %237, %238 : !pdl.type -> ^bb494, ^bb1
  ^bb494:
    %239 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %237, %239 : !pdl.type -> ^bb495, ^bb1
  ^bb495:
    pdl_interp.check_type %237 is f32 -> ^bb496, ^bb1
  ^bb496:
    pdl_interp.check_operation_name of %5 is "arith.divf" -> ^bb497, ^bb1
  ^bb497:
    pdl_interp.check_operand_count of %5 is 2 -> ^bb498, ^bb1
  ^bb498:
    pdl_interp.check_result_count of %5 is 1 -> ^bb499, ^bb1
  ^bb499:
    %240 = pdl_interp.get_result 0 of %5
    pdl_interp.is_not_null %240 : !pdl.value -> ^bb500, ^bb1
  ^bb500:
    pdl_interp.are_equal %240, %4 : !pdl.value -> ^bb501, ^bb1
  ^bb501:
    %241 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %241 : !pdl.value -> ^bb502, ^bb1
  ^bb502:
    %242 = pdl_interp.get_operand 0 of %5
    pdl_interp.is_not_null %242 : !pdl.value -> ^bb503, ^bb1
  ^bb503:
    %243 = pdl_interp.get_operand 1 of %5
    pdl_interp.is_not_null %243 : !pdl.value -> ^bb504, ^bb505
  ^bb505:
    %244 = pdl_interp.get_value_type of %240 : !pdl.type
    pdl_interp.are_equal %237, %244 : !pdl.type -> ^bb506, ^bb1
  ^bb506:
    %245 = pdl_interp.get_value_type of %241 : !pdl.type
    pdl_interp.are_equal %237, %245 : !pdl.type -> ^bb507, ^bb1
  ^bb507:
    %246 = pdl_interp.get_value_type of %242 : !pdl.type
    pdl_interp.are_equal %237, %246 : !pdl.type -> ^bb508, ^bb1
  ^bb508:
    %247 = pdl_interp.get_operand 1 of %5
    pdl_interp.are_equal %241, %247 : !pdl.value -> ^bb509, ^bb1
  ^bb509:
    pdl_interp.record_match @rewriters::@div_add_rev(%236, %242, %241, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb1
  ^bb504:
    %248 = pdl_interp.get_value_type of %240 : !pdl.type
    pdl_interp.are_equal %237, %248 : !pdl.type -> ^bb510, ^bb505
  ^bb510:
    %249 = pdl_interp.get_value_type of %241 : !pdl.type
    pdl_interp.are_equal %237, %249 : !pdl.type -> ^bb511, ^bb505
  ^bb511:
    %250 = pdl_interp.get_value_type of %242 : !pdl.type
    pdl_interp.are_equal %237, %250 : !pdl.type -> ^bb512, ^bb505
  ^bb512:
    %251 = pdl_interp.get_value_type of %243 : !pdl.type
    pdl_interp.are_equal %237, %251 : !pdl.type -> ^bb513, ^bb505
  ^bb513:
    pdl_interp.record_match @rewriters::@frac_add(%236, %243, %241, %242, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb514
  ^bb514:
    pdl_interp.record_match @rewriters::@common_denominator(%236, %243, %242, %241, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb505
  ^bb58:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb515, ^bb1
  ^bb515:
    pdl_interp.check_result_count of %3 is 1 -> ^bb516, ^bb1
  ^bb516:
    %252 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %252 : !pdl.value -> ^bb517, ^bb1
  ^bb517:
    pdl_interp.are_equal %252, %2 : !pdl.value -> ^bb518, ^bb1
  ^bb518:
    %253 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %253 : !pdl.value -> ^bb519, ^bb1
  ^bb519:
    pdl_interp.is_not_null %4 : !pdl.value -> ^bb520, ^bb1
  ^bb520:
    %254 = pdl_interp.get_value_type of %253 : !pdl.type
    %255 = pdl_interp.get_value_type of %252 : !pdl.type
    pdl_interp.are_equal %254, %255 : !pdl.type -> ^bb521, ^bb1
  ^bb521:
    %256 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %254, %256 : !pdl.type -> ^bb522, ^bb1
  ^bb522:
    pdl_interp.check_type %254 is f32 -> ^bb523, ^bb1
  ^bb523:
    pdl_interp.check_operation_name of %5 is "math.powf" -> ^bb524, ^bb1
  ^bb524:
    pdl_interp.check_operand_count of %5 is 2 -> ^bb525, ^bb1
  ^bb525:
    pdl_interp.check_result_count of %5 is 1 -> ^bb526, ^bb1
  ^bb526:
    %257 = pdl_interp.get_result 0 of %5
    pdl_interp.is_not_null %257 : !pdl.value -> ^bb527, ^bb1
  ^bb527:
    pdl_interp.are_equal %257, %4 : !pdl.value -> ^bb528, ^bb1
  ^bb528:
    %258 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %258 : !pdl.value -> ^bb529, ^bb1
  ^bb529:
    %259 = pdl_interp.get_defining_op of %258 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %259 : !pdl.operation -> ^bb530, ^bb1
  ^bb530:
    %260 = pdl_interp.get_operand 0 of %5
    pdl_interp.is_not_null %260 : !pdl.value -> ^bb531, ^bb1
  ^bb531:
    %261 = pdl_interp.get_operand 1 of %5
    %262 = pdl_interp.get_defining_op of %261 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %262 : !pdl.operation -> ^bb532, ^bb1
  ^bb532:
    pdl_interp.is_not_null %261 : !pdl.value -> ^bb533, ^bb1
  ^bb533:
    %263 = pdl_interp.get_value_type of %257 : !pdl.type
    pdl_interp.are_equal %254, %263 : !pdl.type -> ^bb534, ^bb1
  ^bb534:
    pdl_interp.check_operation_name of %259 is "arith.constant" -> ^bb535, ^bb1
  ^bb535:
    pdl_interp.check_operand_count of %259 is 0 -> ^bb536, ^bb1
  ^bb536:
    pdl_interp.check_result_count of %259 is 1 -> ^bb537, ^bb1
  ^bb537:
    %264 = pdl_interp.get_result 0 of %259
    pdl_interp.is_not_null %264 : !pdl.value -> ^bb538, ^bb1
  ^bb538:
    pdl_interp.are_equal %264, %258 : !pdl.value -> ^bb539, ^bb1
  ^bb539:
    pdl_interp.check_operation_name of %262 is "arith.constant" -> ^bb540, ^bb1
  ^bb540:
    pdl_interp.check_operand_count of %262 is 0 -> ^bb541, ^bb1
  ^bb541:
    pdl_interp.check_result_count of %262 is 1 -> ^bb542, ^bb1
  ^bb542:
    %265 = pdl_interp.get_result 0 of %262
    pdl_interp.is_not_null %265 : !pdl.value -> ^bb543, ^bb1
  ^bb543:
    pdl_interp.are_equal %265, %261 : !pdl.value -> ^bb544, ^bb1
  ^bb544:
    %266 = pdl_interp.get_value_type of %260 : !pdl.type
    pdl_interp.are_equal %254, %266 : !pdl.type -> ^bb545, ^bb1
  ^bb545:
    %267 = pdl_interp.get_attribute "value" of %259
    pdl_interp.is_not_null %267 : !pdl.attribute -> ^bb546, ^bb1
  ^bb546:
    pdl_interp.check_attribute %267 is 3.000000e+00 : f32 -> ^bb547, ^bb1
  ^bb547:
    %268 = pdl_interp.get_value_type of %264 : !pdl.type
    pdl_interp.are_equal %268, %254 : !pdl.type -> ^bb548, ^bb1
  ^bb548:
    %269 = pdl_interp.get_value_type of %265 : !pdl.type
    pdl_interp.are_equal %269, %254 : !pdl.type -> ^bb549, ^bb1
  ^bb549:
    %270 = pdl_interp.get_attribute "value" of %262
    pdl_interp.is_not_null %270 : !pdl.attribute -> ^bb550, ^bb1
  ^bb550:
    pdl_interp.check_attribute %270 is 3.000000e+00 : f32 -> ^bb551, ^bb1
  ^bb551:
    pdl_interp.record_match @rewriters::@sum_cubes(%253, %260, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb1
  ^bb59:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb552, ^bb1
  ^bb552:
    pdl_interp.check_result_count of %3 is 1 -> ^bb553, ^bb1
  ^bb553:
    %271 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %271 : !pdl.value -> ^bb554, ^bb1
  ^bb554:
    pdl_interp.are_equal %271, %2 : !pdl.value -> ^bb555, ^bb1
  ^bb555:
    %272 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %272 : !pdl.value -> ^bb556, ^bb1
  ^bb556:
    pdl_interp.is_not_null %4 : !pdl.value -> ^bb557, ^bb1
  ^bb557:
    %273 = pdl_interp.get_value_type of %272 : !pdl.type
    %274 = pdl_interp.get_value_type of %271 : !pdl.type
    pdl_interp.are_equal %273, %274 : !pdl.type -> ^bb558, ^bb1
  ^bb558:
    %275 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %273, %275 : !pdl.type -> ^bb559, ^bb1
  ^bb559:
    pdl_interp.check_type %273 is f32 -> ^bb560, ^bb1
  ^bb560:
    pdl_interp.check_operation_name of %5 is "math.log" -> ^bb561, ^bb1
  ^bb561:
    pdl_interp.check_operand_count of %5 is 1 -> ^bb562, ^bb1
  ^bb562:
    pdl_interp.check_result_count of %5 is 1 -> ^bb563, ^bb1
  ^bb563:
    %276 = pdl_interp.get_result 0 of %5
    pdl_interp.is_not_null %276 : !pdl.value -> ^bb564, ^bb1
  ^bb564:
    pdl_interp.are_equal %276, %4 : !pdl.value -> ^bb565, ^bb1
  ^bb565:
    %277 = pdl_interp.get_operand 0 of %5
    pdl_interp.is_not_null %277 : !pdl.value -> ^bb566, ^bb1
  ^bb566:
    %278 = pdl_interp.get_value_type of %276 : !pdl.type
    pdl_interp.are_equal %273, %278 : !pdl.type -> ^bb567, ^bb1
  ^bb567:
    %279 = pdl_interp.get_value_type of %277 : !pdl.type
    pdl_interp.are_equal %273, %279 : !pdl.type -> ^bb568, ^bb1
  ^bb568:
    pdl_interp.record_match @rewriters::@sum_log(%272, %277, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb1
  ^bb60:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb569, ^bb1
  ^bb569:
    pdl_interp.check_result_count of %3 is 1 -> ^bb570, ^bb1
  ^bb570:
    %280 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %280 : !pdl.value -> ^bb571, ^bb1
  ^bb571:
    pdl_interp.are_equal %280, %2 : !pdl.value -> ^bb572, ^bb1
  ^bb572:
    %281 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %281 : !pdl.value -> ^bb573, ^bb1
  ^bb573:
    pdl_interp.is_not_null %4 : !pdl.value -> ^bb574, ^bb1
  ^bb574:
    %282 = pdl_interp.get_value_type of %281 : !pdl.type
    %283 = pdl_interp.get_value_type of %280 : !pdl.type
    pdl_interp.are_equal %282, %283 : !pdl.type -> ^bb575, ^bb1
  ^bb575:
    %284 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %282, %284 : !pdl.type -> ^bb576, ^bb1
  ^bb576:
    pdl_interp.check_type %282 is f32 -> ^bb577, ^bb1
  ^bb577:
    pdl_interp.check_operation_name of %5 is "math.sin" -> ^bb578, ^bb1
  ^bb578:
    pdl_interp.check_operand_count of %5 is 1 -> ^bb579, ^bb1
  ^bb579:
    pdl_interp.check_result_count of %5 is 1 -> ^bb580, ^bb1
  ^bb580:
    %285 = pdl_interp.get_result 0 of %5
    pdl_interp.is_not_null %285 : !pdl.value -> ^bb581, ^bb1
  ^bb581:
    pdl_interp.are_equal %285, %4 : !pdl.value -> ^bb582, ^bb1
  ^bb582:
    %286 = pdl_interp.get_operand 0 of %5
    pdl_interp.is_not_null %286 : !pdl.value -> ^bb583, ^bb1
  ^bb583:
    %287 = pdl_interp.get_value_type of %285 : !pdl.type
    pdl_interp.are_equal %282, %287 : !pdl.type -> ^bb584, ^bb1
  ^bb584:
    %288 = pdl_interp.get_value_type of %286 : !pdl.type
    pdl_interp.are_equal %282, %288 : !pdl.type -> ^bb585, ^bb1
  ^bb585:
    pdl_interp.record_match @rewriters::@sum_sin(%281, %286, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb1
  ^bb61:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb586, ^bb1
  ^bb586:
    pdl_interp.check_result_count of %3 is 1 -> ^bb587, ^bb1
  ^bb587:
    %289 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %289 : !pdl.value -> ^bb588, ^bb1
  ^bb588:
    pdl_interp.are_equal %289, %2 : !pdl.value -> ^bb589, ^bb1
  ^bb589:
    %290 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %290 : !pdl.value -> ^bb590, ^bb1
  ^bb590:
    pdl_interp.is_not_null %4 : !pdl.value -> ^bb591, ^bb1
  ^bb591:
    %291 = pdl_interp.get_value_type of %290 : !pdl.type
    %292 = pdl_interp.get_value_type of %289 : !pdl.type
    pdl_interp.are_equal %291, %292 : !pdl.type -> ^bb592, ^bb1
  ^bb592:
    %293 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %291, %293 : !pdl.type -> ^bb593, ^bb1
  ^bb593:
    pdl_interp.check_type %291 is f32 -> ^bb594, ^bb1
  ^bb594:
    pdl_interp.check_operation_name of %5 is "math.cos" -> ^bb595, ^bb1
  ^bb595:
    pdl_interp.check_operand_count of %5 is 1 -> ^bb596, ^bb1
  ^bb596:
    pdl_interp.check_result_count of %5 is 1 -> ^bb597, ^bb1
  ^bb597:
    %294 = pdl_interp.get_result 0 of %5
    pdl_interp.is_not_null %294 : !pdl.value -> ^bb598, ^bb1
  ^bb598:
    pdl_interp.are_equal %294, %4 : !pdl.value -> ^bb599, ^bb1
  ^bb599:
    %295 = pdl_interp.get_operand 0 of %5
    pdl_interp.is_not_null %295 : !pdl.value -> ^bb600, ^bb1
  ^bb600:
    %296 = pdl_interp.get_value_type of %294 : !pdl.type
    pdl_interp.are_equal %291, %296 : !pdl.type -> ^bb601, ^bb1
  ^bb601:
    %297 = pdl_interp.get_value_type of %295 : !pdl.type
    pdl_interp.are_equal %291, %297 : !pdl.type -> ^bb602, ^bb1
  ^bb602:
    pdl_interp.record_match @rewriters::@sum_cos(%290, %295, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb1
  ^bb62:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb603, ^bb1
  ^bb603:
    pdl_interp.check_result_count of %3 is 1 -> ^bb604, ^bb1
  ^bb604:
    %298 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %298 : !pdl.value -> ^bb605, ^bb1
  ^bb605:
    pdl_interp.are_equal %298, %2 : !pdl.value -> ^bb606, ^bb1
  ^bb606:
    %299 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %299 : !pdl.value -> ^bb607, ^bb1
  ^bb607:
    pdl_interp.is_not_null %4 : !pdl.value -> ^bb608, ^bb1
  ^bb608:
    %300 = pdl_interp.get_value_type of %299 : !pdl.type
    %301 = pdl_interp.get_value_type of %298 : !pdl.type
    pdl_interp.are_equal %300, %301 : !pdl.type -> ^bb609, ^bb1
  ^bb609:
    %302 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %300, %302 : !pdl.type -> ^bb610, ^bb1
  ^bb610:
    pdl_interp.check_type %300 is f32 -> ^bb611, ^bb1
  ^bb611:
    pdl_interp.check_operation_name of %5 is "math.atan" -> ^bb612, ^bb1
  ^bb612:
    pdl_interp.check_operand_count of %5 is 1 -> ^bb613, ^bb1
  ^bb613:
    pdl_interp.check_result_count of %5 is 1 -> ^bb614, ^bb1
  ^bb614:
    %303 = pdl_interp.get_result 0 of %5
    pdl_interp.is_not_null %303 : !pdl.value -> ^bb615, ^bb1
  ^bb615:
    pdl_interp.are_equal %303, %4 : !pdl.value -> ^bb616, ^bb1
  ^bb616:
    %304 = pdl_interp.get_operand 0 of %5
    pdl_interp.is_not_null %304 : !pdl.value -> ^bb617, ^bb1
  ^bb617:
    %305 = pdl_interp.get_value_type of %303 : !pdl.type
    pdl_interp.are_equal %300, %305 : !pdl.type -> ^bb618, ^bb1
  ^bb618:
    %306 = pdl_interp.get_value_type of %304 : !pdl.type
    pdl_interp.are_equal %300, %306 : !pdl.type -> ^bb619, ^bb1
  ^bb619:
    pdl_interp.record_match @rewriters::@sum_atan(%299, %304, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb1
  ^bb63:
    pdl_interp.check_operand_count of %3 is 0 -> ^bb620, ^bb1
  ^bb620:
    pdl_interp.check_result_count of %3 is 1 -> ^bb621, ^bb1
  ^bb621:
    %307 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %307 : !pdl.value -> ^bb622, ^bb1
  ^bb622:
    pdl_interp.are_equal %307, %2 : !pdl.value -> ^bb623, ^bb1
  ^bb623:
    pdl_interp.is_not_null %4 : !pdl.value -> ^bb624, ^bb1
  ^bb624:
    pdl_interp.check_operation_name of %5 is "arith.mulf" -> ^bb625, ^bb1
  ^bb625:
    pdl_interp.check_operand_count of %5 is 2 -> ^bb626, ^bb1
  ^bb626:
    pdl_interp.check_result_count of %5 is 1 -> ^bb627, ^bb1
  ^bb627:
    %308 = pdl_interp.get_result 0 of %5
    pdl_interp.is_not_null %308 : !pdl.value -> ^bb628, ^bb1
  ^bb628:
    pdl_interp.are_equal %308, %4 : !pdl.value -> ^bb629, ^bb1
  ^bb629:
    %309 = pdl_interp.get_operand 0 of %5
    pdl_interp.is_not_null %309 : !pdl.value -> ^bb630, ^bb1
  ^bb630:
    %310 = pdl_interp.get_defining_op of %309 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %310 : !pdl.operation -> ^bb631, ^bb1
  ^bb631:
    %311 = pdl_interp.get_operand 1 of %5
    %312 = pdl_interp.get_defining_op of %311 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %312 : !pdl.operation -> ^bb632, ^bb1
  ^bb632:
    pdl_interp.is_not_null %311 : !pdl.value -> ^bb633, ^bb1
  ^bb633:
    pdl_interp.check_operation_name of %310 is "arith.constant" -> ^bb634, ^bb1
  ^bb634:
    pdl_interp.check_operand_count of %310 is 0 -> ^bb635, ^bb1
  ^bb635:
    pdl_interp.check_result_count of %310 is 1 -> ^bb636, ^bb1
  ^bb636:
    %313 = pdl_interp.get_result 0 of %310
    pdl_interp.is_not_null %313 : !pdl.value -> ^bb637, ^bb1
  ^bb637:
    pdl_interp.are_equal %313, %309 : !pdl.value -> ^bb638, ^bb1
  ^bb638:
    pdl_interp.check_operation_name of %312 is "math.cos" -> ^bb639, ^bb1
  ^bb639:
    pdl_interp.check_operand_count of %312 is 1 -> ^bb640, ^bb1
  ^bb640:
    pdl_interp.check_result_count of %312 is 1 -> ^bb641, ^bb1
  ^bb641:
    %314 = pdl_interp.get_result 0 of %312
    pdl_interp.is_not_null %314 : !pdl.value -> ^bb642, ^bb1
  ^bb642:
    pdl_interp.are_equal %314, %311 : !pdl.value -> ^bb643, ^bb1
  ^bb643:
    %315 = pdl_interp.get_attribute "value" of %3
    pdl_interp.is_not_null %315 : !pdl.attribute -> ^bb644, ^bb1
  ^bb644:
    pdl_interp.check_attribute %315 is 5.000000e-01 : f32 -> ^bb645, ^bb1
  ^bb645:
    %316 = pdl_interp.get_value_type of %307 : !pdl.type
    %317 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %316, %317 : !pdl.type -> ^bb646, ^bb1
  ^bb646:
    pdl_interp.check_type %316 is f32 -> ^bb647, ^bb1
  ^bb647:
    %318 = pdl_interp.get_operand 0 of %312
    %319 = pdl_interp.get_defining_op of %318 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %319 : !pdl.operation -> ^bb648, ^bb1
  ^bb648:
    %320 = pdl_interp.get_value_type of %308 : !pdl.type
    pdl_interp.are_equal %316, %320 : !pdl.type -> ^bb649, ^bb1
  ^bb649:
    %321 = pdl_interp.get_operand 0 of %319
    %322 = pdl_interp.get_defining_op of %321 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %322 : !pdl.operation -> ^bb650, ^bb1
  ^bb650:
    %323 = pdl_interp.get_value_type of %313 : !pdl.type
    pdl_interp.are_equal %323, %316 : !pdl.type -> ^bb651, ^bb1
  ^bb651:
    pdl_interp.is_not_null %318 : !pdl.value -> ^bb652, ^bb1
  ^bb652:
    pdl_interp.check_operation_name of %319 is "arith.mulf" -> ^bb653, ^bb1
  ^bb653:
    pdl_interp.check_operand_count of %319 is 2 -> ^bb654, ^bb1
  ^bb654:
    pdl_interp.check_result_count of %319 is 1 -> ^bb655, ^bb1
  ^bb655:
    %324 = pdl_interp.get_result 0 of %319
    pdl_interp.is_not_null %324 : !pdl.value -> ^bb656, ^bb1
  ^bb656:
    pdl_interp.are_equal %324, %318 : !pdl.value -> ^bb657, ^bb1
  ^bb657:
    %325 = pdl_interp.get_attribute "value" of %310
    pdl_interp.is_not_null %325 : !pdl.attribute -> ^bb658, ^bb1
  ^bb658:
    pdl_interp.check_attribute %325 is 5.000000e-01 : f32 -> ^bb659, ^bb1
  ^bb659:
    %326 = pdl_interp.get_value_type of %314 : !pdl.type
    pdl_interp.are_equal %326, %316 : !pdl.type -> ^bb660, ^bb1
  ^bb660:
    pdl_interp.is_not_null %321 : !pdl.value -> ^bb661, ^bb1
  ^bb661:
    pdl_interp.check_operation_name of %322 is "arith.constant" -> ^bb662, ^bb1
  ^bb662:
    pdl_interp.check_operand_count of %322 is 0 -> ^bb663, ^bb1
  ^bb663:
    pdl_interp.check_result_count of %322 is 1 -> ^bb664, ^bb1
  ^bb664:
    %327 = pdl_interp.get_result 0 of %322
    pdl_interp.is_not_null %327 : !pdl.value -> ^bb665, ^bb1
  ^bb665:
    pdl_interp.are_equal %327, %321 : !pdl.value -> ^bb666, ^bb1
  ^bb666:
    %328 = pdl_interp.get_operand 1 of %319
    pdl_interp.is_not_null %328 : !pdl.value -> ^bb667, ^bb1
  ^bb667:
    %329 = pdl_interp.get_value_type of %324 : !pdl.type
    pdl_interp.are_equal %329, %316 : !pdl.type -> ^bb668, ^bb1
  ^bb668:
    %330 = pdl_interp.get_value_type of %327 : !pdl.type
    pdl_interp.are_equal %330, %316 : !pdl.type -> ^bb669, ^bb1
  ^bb669:
    %331 = pdl_interp.get_attribute "value" of %322
    pdl_interp.is_not_null %331 : !pdl.attribute -> ^bb670, ^bb1
  ^bb670:
    pdl_interp.check_attribute %331 is 2.000000e+00 : f32 -> ^bb671, ^bb1
  ^bb671:
    %332 = pdl_interp.get_value_type of %328 : !pdl.type
    pdl_interp.are_equal %332, %316 : !pdl.type -> ^bb672, ^bb1
  ^bb672:
    pdl_interp.record_match @rewriters::@sqr_cos_a_rev(%328, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb1
  ^bb64:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb673, ^bb1
  ^bb673:
    pdl_interp.check_result_count of %3 is 1 -> ^bb674, ^bb1
  ^bb674:
    %333 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %333 : !pdl.value -> ^bb675, ^bb1
  ^bb675:
    pdl_interp.are_equal %333, %2 : !pdl.value -> ^bb676, ^bb1
  ^bb676:
    %334 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %334 : !pdl.value -> ^bb677, ^bb1
  ^bb677:
    pdl_interp.is_not_null %4 : !pdl.value -> ^bb678, ^bb1
  ^bb678:
    %335 = pdl_interp.get_value_type of %334 : !pdl.type
    %336 = pdl_interp.get_value_type of %333 : !pdl.type
    pdl_interp.are_equal %335, %336 : !pdl.type -> ^bb679, ^bb1
  ^bb679:
    %337 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %335, %337 : !pdl.type -> ^bb680, ^bb1
  ^bb680:
    pdl_interp.check_type %335 is f32 -> ^bb681, ^bb1
  ^bb681:
    pdl_interp.switch_operation_name of %5 to ["math.sinh", "math.cosh"](^bb682, ^bb683) -> ^bb1
  ^bb682:
    pdl_interp.check_operand_count of %5 is 1 -> ^bb684, ^bb1
  ^bb684:
    pdl_interp.check_result_count of %5 is 1 -> ^bb685, ^bb1
  ^bb685:
    %338 = pdl_interp.get_result 0 of %5
    pdl_interp.is_not_null %338 : !pdl.value -> ^bb686, ^bb1
  ^bb686:
    pdl_interp.are_equal %338, %4 : !pdl.value -> ^bb687, ^bb1
  ^bb687:
    %339 = pdl_interp.get_value_type of %338 : !pdl.type
    pdl_interp.are_equal %335, %339 : !pdl.type -> ^bb688, ^bb1
  ^bb688:
    %340 = pdl_interp.get_operand 0 of %5
    pdl_interp.are_equal %334, %340 : !pdl.value -> ^bb689, ^bb1
  ^bb689:
    pdl_interp.record_match @rewriters::@sinh_add_cosh(%334, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb1
  ^bb683:
    pdl_interp.check_operand_count of %5 is 1 -> ^bb690, ^bb1
  ^bb690:
    pdl_interp.check_result_count of %5 is 1 -> ^bb691, ^bb1
  ^bb691:
    %341 = pdl_interp.get_result 0 of %5
    pdl_interp.is_not_null %341 : !pdl.value -> ^bb692, ^bb1
  ^bb692:
    pdl_interp.are_equal %341, %4 : !pdl.value -> ^bb693, ^bb1
  ^bb693:
    %342 = pdl_interp.get_operand 0 of %5
    pdl_interp.is_not_null %342 : !pdl.value -> ^bb694, ^bb1
  ^bb694:
    %343 = pdl_interp.get_value_type of %341 : !pdl.type
    pdl_interp.are_equal %335, %343 : !pdl.type -> ^bb695, ^bb1
  ^bb695:
    %344 = pdl_interp.get_value_type of %342 : !pdl.type
    pdl_interp.are_equal %335, %344 : !pdl.type -> ^bb696, ^bb1
  ^bb696:
    pdl_interp.record_match @rewriters::@sum_cosh(%334, %342, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb1
  ^bb65:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb697, ^bb1
  ^bb697:
    pdl_interp.check_result_count of %3 is 1 -> ^bb698, ^bb1
  ^bb698:
    %345 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %345 : !pdl.value -> ^bb699, ^bb1
  ^bb699:
    pdl_interp.are_equal %345, %2 : !pdl.value -> ^bb700, ^bb1
  ^bb700:
    %346 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %346 : !pdl.value -> ^bb701, ^bb1
  ^bb701:
    pdl_interp.is_not_null %4 : !pdl.value -> ^bb702, ^bb1
  ^bb702:
    %347 = pdl_interp.get_value_type of %346 : !pdl.type
    %348 = pdl_interp.get_value_type of %345 : !pdl.type
    pdl_interp.are_equal %347, %348 : !pdl.type -> ^bb703, ^bb1
  ^bb703:
    %349 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %347, %349 : !pdl.type -> ^bb704, ^bb1
  ^bb704:
    pdl_interp.check_type %347 is f32 -> ^bb705, ^bb1
  ^bb705:
    pdl_interp.check_operation_name of %5 is "math.exp" -> ^bb706, ^bb1
  ^bb706:
    pdl_interp.check_operand_count of %5 is 1 -> ^bb707, ^bb1
  ^bb707:
    pdl_interp.check_result_count of %5 is 1 -> ^bb708, ^bb1
  ^bb708:
    %350 = pdl_interp.get_result 0 of %5
    pdl_interp.is_not_null %350 : !pdl.value -> ^bb709, ^bb1
  ^bb709:
    pdl_interp.are_equal %350, %4 : !pdl.value -> ^bb710, ^bb1
  ^bb710:
    %351 = pdl_interp.get_operand 0 of %5
    pdl_interp.is_not_null %351 : !pdl.value -> ^bb711, ^bb1
  ^bb711:
    %352 = pdl_interp.get_defining_op of %351 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %352 : !pdl.operation -> ^bb712, ^bb1
  ^bb712:
    %353 = pdl_interp.get_value_type of %350 : !pdl.type
    pdl_interp.are_equal %347, %353 : !pdl.type -> ^bb713, ^bb1
  ^bb713:
    pdl_interp.check_operation_name of %352 is "arith.negf" -> ^bb714, ^bb1
  ^bb714:
    pdl_interp.check_operand_count of %352 is 1 -> ^bb715, ^bb1
  ^bb715:
    pdl_interp.check_result_count of %352 is 1 -> ^bb716, ^bb1
  ^bb716:
    %354 = pdl_interp.get_result 0 of %352
    pdl_interp.is_not_null %354 : !pdl.value -> ^bb717, ^bb1
  ^bb717:
    pdl_interp.are_equal %354, %351 : !pdl.value -> ^bb718, ^bb1
  ^bb718:
    %355 = pdl_interp.get_value_type of %354 : !pdl.type
    pdl_interp.are_equal %355, %347 : !pdl.type -> ^bb719, ^bb1
  ^bb719:
    %356 = pdl_interp.get_operand 0 of %352
    pdl_interp.are_equal %356, %346 : !pdl.value -> ^bb720, ^bb1
  ^bb720:
    pdl_interp.record_match @rewriters::@cosh_undef(%346, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb1
  ^bb66:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb721, ^bb1
  ^bb721:
    pdl_interp.check_result_count of %3 is 1 -> ^bb722, ^bb1
  ^bb722:
    %357 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %357 : !pdl.value -> ^bb723, ^bb1
  ^bb723:
    pdl_interp.are_equal %357, %2 : !pdl.value -> ^bb724, ^bb1
  ^bb724:
    %358 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %358 : !pdl.value -> ^bb725, ^bb1
  ^bb725:
    pdl_interp.is_not_null %4 : !pdl.value -> ^bb726, ^bb1
  ^bb726:
    %359 = pdl_interp.get_value_type of %358 : !pdl.type
    %360 = pdl_interp.get_value_type of %357 : !pdl.type
    pdl_interp.are_equal %359, %360 : !pdl.type -> ^bb727, ^bb1
  ^bb727:
    %361 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %359, %361 : !pdl.type -> ^bb728, ^bb1
  ^bb728:
    pdl_interp.check_type %359 is f32 -> ^bb729, ^bb1
  ^bb729:
    pdl_interp.check_operation_name of %5 is "math.sinh" -> ^bb730, ^bb1
  ^bb730:
    pdl_interp.check_operand_count of %5 is 1 -> ^bb731, ^bb1
  ^bb731:
    pdl_interp.check_result_count of %5 is 1 -> ^bb732, ^bb1
  ^bb732:
    %362 = pdl_interp.get_result 0 of %5
    pdl_interp.is_not_null %362 : !pdl.value -> ^bb733, ^bb1
  ^bb733:
    pdl_interp.are_equal %362, %4 : !pdl.value -> ^bb734, ^bb1
  ^bb734:
    %363 = pdl_interp.get_operand 0 of %5
    pdl_interp.is_not_null %363 : !pdl.value -> ^bb735, ^bb1
  ^bb735:
    %364 = pdl_interp.get_value_type of %362 : !pdl.type
    pdl_interp.are_equal %359, %364 : !pdl.type -> ^bb736, ^bb1
  ^bb736:
    %365 = pdl_interp.get_value_type of %363 : !pdl.type
    pdl_interp.are_equal %359, %365 : !pdl.type -> ^bb737, ^bb1
  ^bb737:
    pdl_interp.record_match @rewriters::@sum_sinh(%358, %363, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb1
  ^bb49:
    pdl_interp.switch_operation_name of %3 to ["arith.addf", "arith.subf", "arith.constant", "arith.mulf"](^bb738, ^bb739, ^bb740, ^bb741) -> ^bb50
  ^bb738:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb742, ^bb50
  ^bb742:
    pdl_interp.check_result_count of %3 is 1 -> ^bb743, ^bb50
  ^bb743:
    %366 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %366 : !pdl.value -> ^bb744, ^bb50
  ^bb744:
    pdl_interp.are_equal %366, %2 : !pdl.value -> ^bb745, ^bb50
  ^bb745:
    %367 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %367 : !pdl.value -> ^bb746, ^bb50
  ^bb746:
    %368 = pdl_interp.get_operand 1 of %0
    pdl_interp.is_not_null %368 : !pdl.value -> ^bb747, ^bb50
  ^bb747:
    %369 = pdl_interp.get_value_type of %367 : !pdl.type
    %370 = pdl_interp.get_value_type of %366 : !pdl.type
    pdl_interp.are_equal %369, %370 : !pdl.type -> ^bb748, ^bb50
  ^bb748:
    %371 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %369, %371 : !pdl.type -> ^bb749, ^bb50
  ^bb749:
    pdl_interp.check_type %369 is f32 -> ^bb750, ^bb50
  ^bb750:
    %372 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %372 : !pdl.value -> ^bb751, ^bb50
  ^bb751:
    %373 = pdl_interp.get_value_type of %372 : !pdl.type
    pdl_interp.are_equal %369, %373 : !pdl.type -> ^bb752, ^bb50
  ^bb752:
    %374 = pdl_interp.get_value_type of %368 : !pdl.type
    pdl_interp.are_equal %369, %374 : !pdl.type -> ^bb753, ^bb50
  ^bb753:
    pdl_interp.record_match @rewriters::@associate_addladd(%372, %368, %367, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb50
  ^bb739:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb754, ^bb50
  ^bb754:
    pdl_interp.check_result_count of %3 is 1 -> ^bb755, ^bb50
  ^bb755:
    %375 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %375 : !pdl.value -> ^bb756, ^bb50
  ^bb756:
    pdl_interp.are_equal %375, %2 : !pdl.value -> ^bb757, ^bb50
  ^bb757:
    %376 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %376 : !pdl.value -> ^bb758, ^bb50
  ^bb758:
    %377 = pdl_interp.get_operand 1 of %0
    pdl_interp.is_not_null %377 : !pdl.value -> ^bb759, ^bb50
  ^bb759:
    %378 = pdl_interp.get_value_type of %376 : !pdl.type
    %379 = pdl_interp.get_value_type of %375 : !pdl.type
    pdl_interp.are_equal %378, %379 : !pdl.type -> ^bb760, ^bb50
  ^bb760:
    %380 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %378, %380 : !pdl.type -> ^bb761, ^bb50
  ^bb761:
    pdl_interp.check_type %378 is f32 -> ^bb762, ^bb50
  ^bb762:
    %381 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %381 : !pdl.value -> ^bb763, ^bb50
  ^bb763:
    %382 = pdl_interp.get_value_type of %381 : !pdl.type
    pdl_interp.are_equal %378, %382 : !pdl.type -> ^bb764, ^bb50
  ^bb764:
    %383 = pdl_interp.get_value_type of %377 : !pdl.type
    pdl_interp.are_equal %378, %383 : !pdl.type -> ^bb765, ^bb50
  ^bb765:
    pdl_interp.record_match @rewriters::@associate_addl_(%381, %377, %376, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb50
  ^bb740:
    pdl_interp.check_operand_count of %3 is 0 -> ^bb766, ^bb50
  ^bb766:
    pdl_interp.check_result_count of %3 is 1 -> ^bb767, ^bb50
  ^bb767:
    %384 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %384 : !pdl.value -> ^bb768, ^bb50
  ^bb768:
    pdl_interp.are_equal %384, %2 : !pdl.value -> ^bb769, ^bb50
  ^bb769:
    %385 = pdl_interp.get_operand 1 of %0
    pdl_interp.is_not_null %385 : !pdl.value -> ^bb770, ^bb50
  ^bb770:
    %386 = pdl_interp.get_attribute "value" of %3
    pdl_interp.is_not_null %386 : !pdl.attribute -> ^bb771, ^bb50
  ^bb771:
    pdl_interp.check_attribute %386 is 0.000000e+00 : f32 -> ^bb772, ^bb50
  ^bb772:
    %387 = pdl_interp.get_value_type of %384 : !pdl.type
    %388 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %387, %388 : !pdl.type -> ^bb773, ^bb50
  ^bb773:
    pdl_interp.check_type %387 is f32 -> ^bb774, ^bb50
  ^bb774:
    %389 = pdl_interp.get_value_type of %385 : !pdl.type
    pdl_interp.are_equal %387, %389 : !pdl.type -> ^bb775, ^bb50
  ^bb775:
    pdl_interp.record_match @rewriters::@add_lft_identity(%385, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb50
  ^bb741:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb776, ^bb50
  ^bb776:
    pdl_interp.check_result_count of %3 is 1 -> ^bb777, ^bb50
  ^bb777:
    %390 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %390 : !pdl.value -> ^bb778, ^bb50
  ^bb778:
    pdl_interp.are_equal %390, %2 : !pdl.value -> ^bb779, ^bb50
  ^bb779:
    %391 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %391 : !pdl.value -> ^bb780, ^bb50
  ^bb780:
    %392 = pdl_interp.get_value_type of %391 : !pdl.type
    %393 = pdl_interp.get_value_type of %390 : !pdl.type
    pdl_interp.are_equal %392, %393 : !pdl.type -> ^bb781, ^bb50
  ^bb781:
    %394 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %392, %394 : !pdl.type -> ^bb782, ^bb50
  ^bb782:
    pdl_interp.check_type %392 is f32 -> ^bb783, ^bb50
  ^bb783:
    %395 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %395 : !pdl.value -> ^bb784, ^bb50
  ^bb784:
    %396 = pdl_interp.get_value_type of %395 : !pdl.type
    pdl_interp.are_equal %392, %396 : !pdl.type -> ^bb785, ^bb50
  ^bb785:
    %397 = pdl_interp.get_operand 1 of %0
    pdl_interp.are_equal %395, %397 : !pdl.value -> ^bb786, ^bb50
  ^bb786:
    pdl_interp.record_match @rewriters::@distribute_lft1_in(%391, %395, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb50
  ^bb26:
    pdl_interp.check_operand_count of %0 is 2 -> ^bb787, ^bb1
  ^bb787:
    pdl_interp.check_result_count of %0 is 1 -> ^bb788, ^bb1
  ^bb788:
    pdl_interp.is_not_null %2 : !pdl.value -> ^bb789, ^bb790
  ^bb790:
    %398 = pdl_interp.get_operand 1 of %0
    %399 = pdl_interp.get_defining_op of %398 : !pdl.value {position = "root.operand[1].defining_op"}
    pdl_interp.is_not_null %399 : !pdl.operation -> ^bb791, ^bb1
  ^bb791:
    pdl_interp.is_not_null %2 : !pdl.value -> ^bb792, ^bb1
  ^bb792:
    pdl_interp.switch_operation_name of %3 to ["arith.mulf", "math.powf", "arith.divf", "math.log", "arith.constant", "math.sin", "math.cos", "math.atan", "math.cosh", "math.exp", "math.sinh"](^bb793, ^bb794, ^bb795, ^bb796, ^bb797, ^bb798, ^bb799, ^bb800, ^bb801, ^bb802, ^bb803) -> ^bb1
  ^bb793:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb804, ^bb1
  ^bb804:
    pdl_interp.check_result_count of %3 is 1 -> ^bb805, ^bb1
  ^bb805:
    %400 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %400 : !pdl.value -> ^bb806, ^bb1
  ^bb806:
    pdl_interp.are_equal %400, %2 : !pdl.value -> ^bb807, ^bb1
  ^bb807:
    %401 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %401 : !pdl.value -> ^bb808, ^bb1
  ^bb808:
    pdl_interp.is_not_null %398 : !pdl.value -> ^bb809, ^bb1
  ^bb809:
    %402 = pdl_interp.get_value_type of %401 : !pdl.type
    %403 = pdl_interp.get_value_type of %400 : !pdl.type
    pdl_interp.are_equal %402, %403 : !pdl.type -> ^bb810, ^bb811
  ^bb811:
    pdl_interp.switch_operation_name of %399 to ["arith.constant", "arith.mulf"](^bb812, ^bb813) -> ^bb1
  ^bb812:
    pdl_interp.check_operand_count of %399 is 0 -> ^bb814, ^bb1
  ^bb814:
    pdl_interp.check_result_count of %399 is 1 -> ^bb815, ^bb1
  ^bb815:
    %404 = pdl_interp.get_result 0 of %399
    pdl_interp.is_not_null %404 : !pdl.value -> ^bb816, ^bb1
  ^bb816:
    pdl_interp.are_equal %404, %398 : !pdl.value -> ^bb817, ^bb1
  ^bb817:
    %405 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %405 : !pdl.value -> ^bb818, ^bb1
  ^bb818:
    %406 = pdl_interp.get_defining_op of %405 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %406 : !pdl.operation -> ^bb819, ^bb1
  ^bb819:
    %407 = pdl_interp.get_defining_op of %401 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %407 : !pdl.operation -> ^bb820, ^bb1
  ^bb820:
    pdl_interp.switch_operation_name of %406 to ["math.cos", "math.sin"](^bb821, ^bb822) -> ^bb1
  ^bb821:
    pdl_interp.check_operand_count of %406 is 1 -> ^bb823, ^bb1
  ^bb823:
    pdl_interp.check_result_count of %406 is 1 -> ^bb824, ^bb1
  ^bb824:
    %408 = pdl_interp.get_result 0 of %406
    pdl_interp.is_not_null %408 : !pdl.value -> ^bb825, ^bb1
  ^bb825:
    pdl_interp.are_equal %408, %405 : !pdl.value -> ^bb826, ^bb1
  ^bb826:
    pdl_interp.check_operation_name of %407 is "math.cos" -> ^bb827, ^bb1
  ^bb827:
    pdl_interp.check_operand_count of %407 is 1 -> ^bb828, ^bb1
  ^bb828:
    pdl_interp.check_result_count of %407 is 1 -> ^bb829, ^bb1
  ^bb829:
    %409 = pdl_interp.get_result 0 of %407
    pdl_interp.is_not_null %409 : !pdl.value -> ^bb830, ^bb1
  ^bb830:
    pdl_interp.are_equal %409, %401 : !pdl.value -> ^bb831, ^bb1
  ^bb831:
    %410 = pdl_interp.get_operand 0 of %407
    pdl_interp.is_not_null %410 : !pdl.value -> ^bb832, ^bb1
  ^bb832:
    %411 = pdl_interp.get_value_type of %410 : !pdl.type
    %412 = pdl_interp.get_value_type of %409 : !pdl.type
    pdl_interp.are_equal %411, %412 : !pdl.type -> ^bb833, ^bb1
  ^bb833:
    %413 = pdl_interp.get_value_type of %400 : !pdl.type
    pdl_interp.are_equal %411, %413 : !pdl.type -> ^bb834, ^bb1
  ^bb834:
    %414 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %411, %414 : !pdl.type -> ^bb835, ^bb1
  ^bb835:
    pdl_interp.check_type %411 is f32 -> ^bb836, ^bb1
  ^bb836:
    %415 = pdl_interp.get_value_type of %408 : !pdl.type
    pdl_interp.are_equal %411, %415 : !pdl.type -> ^bb837, ^bb1
  ^bb837:
    %416 = pdl_interp.get_attribute "value" of %399
    pdl_interp.is_not_null %416 : !pdl.attribute -> ^bb838, ^bb1
  ^bb838:
    pdl_interp.check_attribute %416 is 1.000000e+00 : f32 -> ^bb839, ^bb1
  ^bb839:
    %417 = pdl_interp.get_value_type of %404 : !pdl.type
    pdl_interp.are_equal %411, %417 : !pdl.type -> ^bb840, ^bb1
  ^bb840:
    %418 = pdl_interp.get_operand 0 of %406
    pdl_interp.are_equal %410, %418 : !pdl.value -> ^bb841, ^bb1
  ^bb841:
    pdl_interp.record_match @rewriters::@sub_1_cos(%410, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb1
  ^bb822:
    pdl_interp.check_operand_count of %406 is 1 -> ^bb842, ^bb1
  ^bb842:
    pdl_interp.check_result_count of %406 is 1 -> ^bb843, ^bb1
  ^bb843:
    %419 = pdl_interp.get_result 0 of %406
    pdl_interp.is_not_null %419 : !pdl.value -> ^bb844, ^bb1
  ^bb844:
    pdl_interp.are_equal %419, %405 : !pdl.value -> ^bb845, ^bb1
  ^bb845:
    pdl_interp.check_operation_name of %407 is "math.sin" -> ^bb846, ^bb1
  ^bb846:
    pdl_interp.check_operand_count of %407 is 1 -> ^bb847, ^bb1
  ^bb847:
    pdl_interp.check_result_count of %407 is 1 -> ^bb848, ^bb1
  ^bb848:
    %420 = pdl_interp.get_result 0 of %407
    pdl_interp.is_not_null %420 : !pdl.value -> ^bb849, ^bb1
  ^bb849:
    pdl_interp.are_equal %420, %401 : !pdl.value -> ^bb850, ^bb1
  ^bb850:
    %421 = pdl_interp.get_operand 0 of %407
    pdl_interp.is_not_null %421 : !pdl.value -> ^bb851, ^bb1
  ^bb851:
    %422 = pdl_interp.get_value_type of %421 : !pdl.type
    %423 = pdl_interp.get_value_type of %420 : !pdl.type
    pdl_interp.are_equal %422, %423 : !pdl.type -> ^bb852, ^bb1
  ^bb852:
    %424 = pdl_interp.get_value_type of %400 : !pdl.type
    pdl_interp.are_equal %422, %424 : !pdl.type -> ^bb853, ^bb1
  ^bb853:
    %425 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %422, %425 : !pdl.type -> ^bb854, ^bb1
  ^bb854:
    pdl_interp.check_type %422 is f32 -> ^bb855, ^bb1
  ^bb855:
    %426 = pdl_interp.get_value_type of %419 : !pdl.type
    pdl_interp.are_equal %422, %426 : !pdl.type -> ^bb856, ^bb1
  ^bb856:
    %427 = pdl_interp.get_attribute "value" of %399
    pdl_interp.is_not_null %427 : !pdl.attribute -> ^bb857, ^bb1
  ^bb857:
    pdl_interp.check_attribute %427 is 1.000000e+00 : f32 -> ^bb858, ^bb1
  ^bb858:
    %428 = pdl_interp.get_value_type of %404 : !pdl.type
    pdl_interp.are_equal %422, %428 : !pdl.type -> ^bb859, ^bb1
  ^bb859:
    %429 = pdl_interp.get_operand 0 of %406
    pdl_interp.are_equal %421, %429 : !pdl.value -> ^bb860, ^bb1
  ^bb860:
    pdl_interp.record_match @rewriters::@sub_1_sin(%421, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb1
  ^bb813:
    pdl_interp.check_operand_count of %399 is 2 -> ^bb861, ^bb1
  ^bb861:
    pdl_interp.check_result_count of %399 is 1 -> ^bb862, ^bb1
  ^bb862:
    %430 = pdl_interp.get_result 0 of %399
    pdl_interp.is_not_null %430 : !pdl.value -> ^bb863, ^bb1
  ^bb863:
    pdl_interp.are_equal %430, %398 : !pdl.value -> ^bb864, ^bb1
  ^bb864:
    %431 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %431 : !pdl.value -> ^bb865, ^bb1
  ^bb865:
    %432 = pdl_interp.get_defining_op of %431 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %432 : !pdl.operation -> ^bb866, ^bb1
  ^bb866:
    %433 = pdl_interp.get_defining_op of %401 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %433 : !pdl.operation -> ^bb867, ^bb1
  ^bb867:
    %434 = pdl_interp.get_operand 0 of %399
    pdl_interp.is_not_null %434 : !pdl.value -> ^bb868, ^bb1
  ^bb868:
    %435 = pdl_interp.get_defining_op of %434 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %435 : !pdl.operation -> ^bb869, ^bb1
  ^bb869:
    %436 = pdl_interp.get_operand 1 of %399
    %437 = pdl_interp.get_defining_op of %436 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %437 : !pdl.operation -> ^bb870, ^bb1
  ^bb870:
    pdl_interp.is_not_null %436 : !pdl.value -> ^bb871, ^bb1
  ^bb871:
    pdl_interp.switch_operation_name of %432 to ["math.sin", "math.cos", "math.powf", "math.cosh"](^bb872, ^bb873, ^bb874, ^bb875) -> ^bb1
  ^bb872:
    pdl_interp.check_operand_count of %432 is 1 -> ^bb876, ^bb1
  ^bb876:
    pdl_interp.check_result_count of %432 is 1 -> ^bb877, ^bb1
  ^bb877:
    %438 = pdl_interp.get_result 0 of %432
    pdl_interp.is_not_null %438 : !pdl.value -> ^bb878, ^bb1
  ^bb878:
    pdl_interp.are_equal %438, %431 : !pdl.value -> ^bb879, ^bb1
  ^bb879:
    pdl_interp.check_operation_name of %433 is "arith.constant" -> ^bb880, ^bb1
  ^bb880:
    pdl_interp.check_operand_count of %433 is 0 -> ^bb881, ^bb1
  ^bb881:
    pdl_interp.check_result_count of %433 is 1 -> ^bb882, ^bb1
  ^bb882:
    %439 = pdl_interp.get_result 0 of %433
    pdl_interp.is_not_null %439 : !pdl.value -> ^bb883, ^bb1
  ^bb883:
    pdl_interp.are_equal %439, %401 : !pdl.value -> ^bb884, ^bb1
  ^bb884:
    pdl_interp.check_operation_name of %435 is "arith.constant" -> ^bb885, ^bb1
  ^bb885:
    pdl_interp.check_operand_count of %435 is 0 -> ^bb886, ^bb1
  ^bb886:
    pdl_interp.check_result_count of %435 is 1 -> ^bb887, ^bb1
  ^bb887:
    %440 = pdl_interp.get_result 0 of %435
    pdl_interp.is_not_null %440 : !pdl.value -> ^bb888, ^bb1
  ^bb888:
    pdl_interp.are_equal %440, %434 : !pdl.value -> ^bb889, ^bb1
  ^bb889:
    pdl_interp.check_operation_name of %437 is "math.powf" -> ^bb890, ^bb1
  ^bb890:
    pdl_interp.check_operand_count of %437 is 2 -> ^bb891, ^bb1
  ^bb891:
    pdl_interp.check_result_count of %437 is 1 -> ^bb892, ^bb1
  ^bb892:
    %441 = pdl_interp.get_result 0 of %437
    pdl_interp.is_not_null %441 : !pdl.value -> ^bb893, ^bb1
  ^bb893:
    pdl_interp.are_equal %441, %436 : !pdl.value -> ^bb894, ^bb1
  ^bb894:
    %442 = pdl_interp.get_operand 0 of %432
    pdl_interp.is_not_null %442 : !pdl.value -> ^bb895, ^bb1
  ^bb895:
    %443 = pdl_interp.get_operand 0 of %437
    %444 = pdl_interp.get_defining_op of %443 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %444 : !pdl.operation -> ^bb896, ^bb1
  ^bb896:
    %445 = pdl_interp.get_attribute "value" of %433
    pdl_interp.is_not_null %445 : !pdl.attribute -> ^bb897, ^bb1
  ^bb897:
    pdl_interp.check_attribute %445 is 3.000000e+00 : f32 -> ^bb898, ^bb1
  ^bb898:
    %446 = pdl_interp.get_value_type of %439 : !pdl.type
    %447 = pdl_interp.get_value_type of %400 : !pdl.type
    pdl_interp.are_equal %446, %447 : !pdl.type -> ^bb899, ^bb1
  ^bb899:
    %448 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %446, %448 : !pdl.type -> ^bb900, ^bb1
  ^bb900:
    pdl_interp.check_type %446 is f32 -> ^bb901, ^bb1
  ^bb901:
    pdl_interp.is_not_null %443 : !pdl.value -> ^bb902, ^bb1
  ^bb902:
    pdl_interp.check_operation_name of %444 is "math.sin" -> ^bb903, ^bb1
  ^bb903:
    pdl_interp.check_operand_count of %444 is 1 -> ^bb904, ^bb1
  ^bb904:
    pdl_interp.check_result_count of %444 is 1 -> ^bb905, ^bb1
  ^bb905:
    %449 = pdl_interp.get_result 0 of %444
    pdl_interp.is_not_null %449 : !pdl.value -> ^bb906, ^bb1
  ^bb906:
    pdl_interp.are_equal %449, %443 : !pdl.value -> ^bb907, ^bb1
  ^bb907:
    %450 = pdl_interp.get_attribute "value" of %435
    pdl_interp.is_not_null %450 : !pdl.attribute -> ^bb908, ^bb1
  ^bb908:
    pdl_interp.check_attribute %450 is 4.000000e+00 : f32 -> ^bb909, ^bb1
  ^bb909:
    %451 = pdl_interp.get_value_type of %438 : !pdl.type
    pdl_interp.are_equal %446, %451 : !pdl.type -> ^bb910, ^bb1
  ^bb910:
    %452 = pdl_interp.get_operand 1 of %437
    %453 = pdl_interp.get_defining_op of %452 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %453 : !pdl.operation -> ^bb911, ^bb1
  ^bb911:
    %454 = pdl_interp.get_value_type of %442 : !pdl.type
    pdl_interp.are_equal %446, %454 : !pdl.type -> ^bb912, ^bb1
  ^bb912:
    %455 = pdl_interp.get_value_type of %430 : !pdl.type
    pdl_interp.are_equal %446, %455 : !pdl.type -> ^bb913, ^bb1
  ^bb913:
    %456 = pdl_interp.get_value_type of %440 : !pdl.type
    pdl_interp.are_equal %446, %456 : !pdl.type -> ^bb914, ^bb1
  ^bb914:
    pdl_interp.is_not_null %452 : !pdl.value -> ^bb915, ^bb1
  ^bb915:
    pdl_interp.check_operation_name of %453 is "arith.constant" -> ^bb916, ^bb1
  ^bb916:
    pdl_interp.check_operand_count of %453 is 0 -> ^bb917, ^bb1
  ^bb917:
    pdl_interp.check_result_count of %453 is 1 -> ^bb918, ^bb1
  ^bb918:
    %457 = pdl_interp.get_result 0 of %453
    pdl_interp.is_not_null %457 : !pdl.value -> ^bb919, ^bb1
  ^bb919:
    pdl_interp.are_equal %457, %452 : !pdl.value -> ^bb920, ^bb1
  ^bb920:
    %458 = pdl_interp.get_value_type of %441 : !pdl.type
    pdl_interp.are_equal %446, %458 : !pdl.type -> ^bb921, ^bb1
  ^bb921:
    %459 = pdl_interp.get_value_type of %449 : !pdl.type
    pdl_interp.are_equal %459, %446 : !pdl.type -> ^bb922, ^bb1
  ^bb922:
    %460 = pdl_interp.get_operand 0 of %444
    pdl_interp.are_equal %460, %442 : !pdl.value -> ^bb923, ^bb1
  ^bb923:
    %461 = pdl_interp.get_value_type of %457 : !pdl.type
    pdl_interp.are_equal %461, %446 : !pdl.type -> ^bb924, ^bb1
  ^bb924:
    %462 = pdl_interp.get_attribute "value" of %453
    pdl_interp.is_not_null %462 : !pdl.attribute -> ^bb925, ^bb1
  ^bb925:
    pdl_interp.check_attribute %462 is 3.000000e+00 : f32 -> ^bb926, ^bb1
  ^bb926:
    pdl_interp.record_match @rewriters::@_3_sin(%442, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb1
  ^bb873:
    pdl_interp.check_operand_count of %432 is 1 -> ^bb927, ^bb1
  ^bb927:
    pdl_interp.check_result_count of %432 is 1 -> ^bb928, ^bb1
  ^bb928:
    %463 = pdl_interp.get_result 0 of %432
    pdl_interp.is_not_null %463 : !pdl.value -> ^bb929, ^bb1
  ^bb929:
    pdl_interp.are_equal %463, %431 : !pdl.value -> ^bb930, ^bb1
  ^bb930:
    pdl_interp.switch_operation_name of %433 to ["math.cos", "math.sin"](^bb931, ^bb932) -> ^bb1
  ^bb931:
    pdl_interp.check_operand_count of %433 is 1 -> ^bb933, ^bb1
  ^bb933:
    pdl_interp.check_result_count of %433 is 1 -> ^bb934, ^bb1
  ^bb934:
    %464 = pdl_interp.get_result 0 of %433
    pdl_interp.is_not_null %464 : !pdl.value -> ^bb935, ^bb1
  ^bb935:
    pdl_interp.are_equal %464, %401 : !pdl.value -> ^bb936, ^bb1
  ^bb936:
    pdl_interp.check_operation_name of %435 is "math.sin" -> ^bb937, ^bb1
  ^bb937:
    pdl_interp.check_operand_count of %435 is 1 -> ^bb938, ^bb1
  ^bb938:
    pdl_interp.check_result_count of %435 is 1 -> ^bb939, ^bb1
  ^bb939:
    %465 = pdl_interp.get_result 0 of %435
    pdl_interp.is_not_null %465 : !pdl.value -> ^bb940, ^bb1
  ^bb940:
    pdl_interp.are_equal %465, %434 : !pdl.value -> ^bb941, ^bb1
  ^bb941:
    pdl_interp.check_operation_name of %437 is "math.sin" -> ^bb942, ^bb1
  ^bb942:
    pdl_interp.check_operand_count of %437 is 1 -> ^bb943, ^bb1
  ^bb943:
    pdl_interp.check_result_count of %437 is 1 -> ^bb944, ^bb1
  ^bb944:
    %466 = pdl_interp.get_result 0 of %437
    pdl_interp.is_not_null %466 : !pdl.value -> ^bb945, ^bb1
  ^bb945:
    pdl_interp.are_equal %466, %436 : !pdl.value -> ^bb946, ^bb1
  ^bb946:
    %467 = pdl_interp.get_operand 0 of %433
    pdl_interp.is_not_null %467 : !pdl.value -> ^bb947, ^bb1
  ^bb947:
    %468 = pdl_interp.get_value_type of %467 : !pdl.type
    %469 = pdl_interp.get_value_type of %464 : !pdl.type
    pdl_interp.are_equal %468, %469 : !pdl.type -> ^bb948, ^bb1
  ^bb948:
    %470 = pdl_interp.get_value_type of %400 : !pdl.type
    pdl_interp.are_equal %468, %470 : !pdl.type -> ^bb949, ^bb1
  ^bb949:
    %471 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %468, %471 : !pdl.type -> ^bb950, ^bb1
  ^bb950:
    pdl_interp.check_type %468 is f32 -> ^bb951, ^bb1
  ^bb951:
    %472 = pdl_interp.get_value_type of %463 : !pdl.type
    pdl_interp.are_equal %468, %472 : !pdl.type -> ^bb952, ^bb953
  ^bb953:
    %473 = pdl_interp.get_operand 0 of %432
    pdl_interp.is_not_null %473 : !pdl.value -> ^bb954, ^bb1
  ^bb954:
    %474 = pdl_interp.get_value_type of %463 : !pdl.type
    pdl_interp.are_equal %468, %474 : !pdl.type -> ^bb955, ^bb1
  ^bb955:
    %475 = pdl_interp.get_value_type of %430 : !pdl.type
    pdl_interp.are_equal %468, %475 : !pdl.type -> ^bb956, ^bb1
  ^bb956:
    %476 = pdl_interp.get_value_type of %466 : !pdl.type
    pdl_interp.are_equal %468, %476 : !pdl.type -> ^bb957, ^bb1
  ^bb957:
    %477 = pdl_interp.get_value_type of %465 : !pdl.type
    pdl_interp.are_equal %468, %477 : !pdl.type -> ^bb958, ^bb1
  ^bb958:
    %478 = pdl_interp.get_operand 0 of %435
    pdl_interp.are_equal %467, %478 : !pdl.value -> ^bb959, ^bb1
  ^bb959:
    %479 = pdl_interp.get_value_type of %473 : !pdl.type
    pdl_interp.are_equal %468, %479 : !pdl.type -> ^bb960, ^bb1
  ^bb960:
    %480 = pdl_interp.get_operand 0 of %437
    pdl_interp.are_equal %473, %480 : !pdl.value -> ^bb961, ^bb1
  ^bb961:
    pdl_interp.record_match @rewriters::@cos_sum_rev(%467, %473, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb1
  ^bb952:
    %481 = pdl_interp.get_value_type of %430 : !pdl.type
    pdl_interp.are_equal %468, %481 : !pdl.type -> ^bb962, ^bb953
  ^bb962:
    %482 = pdl_interp.get_value_type of %466 : !pdl.type
    pdl_interp.are_equal %468, %482 : !pdl.type -> ^bb963, ^bb953
  ^bb963:
    %483 = pdl_interp.get_value_type of %465 : !pdl.type
    pdl_interp.are_equal %468, %483 : !pdl.type -> ^bb964, ^bb953
  ^bb964:
    %484 = pdl_interp.get_operand 0 of %435
    pdl_interp.are_equal %467, %484 : !pdl.value -> ^bb965, ^bb953
  ^bb965:
    %485 = pdl_interp.get_operand 0 of %432
    pdl_interp.are_equal %467, %485 : !pdl.value -> ^bb966, ^bb953
  ^bb966:
    %486 = pdl_interp.get_operand 0 of %437
    pdl_interp.are_equal %467, %486 : !pdl.value -> ^bb967, ^bb953
  ^bb967:
    pdl_interp.record_match @rewriters::@_2_cos(%467, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb953
  ^bb932:
    pdl_interp.check_operand_count of %433 is 1 -> ^bb968, ^bb1
  ^bb968:
    pdl_interp.check_result_count of %433 is 1 -> ^bb969, ^bb1
  ^bb969:
    %487 = pdl_interp.get_result 0 of %433
    pdl_interp.is_not_null %487 : !pdl.value -> ^bb970, ^bb1
  ^bb970:
    pdl_interp.are_equal %487, %401 : !pdl.value -> ^bb971, ^bb1
  ^bb971:
    pdl_interp.check_operation_name of %435 is "math.cos" -> ^bb972, ^bb1
  ^bb972:
    pdl_interp.check_operand_count of %435 is 1 -> ^bb973, ^bb1
  ^bb973:
    pdl_interp.check_result_count of %435 is 1 -> ^bb974, ^bb1
  ^bb974:
    %488 = pdl_interp.get_result 0 of %435
    pdl_interp.is_not_null %488 : !pdl.value -> ^bb975, ^bb1
  ^bb975:
    pdl_interp.are_equal %488, %434 : !pdl.value -> ^bb976, ^bb1
  ^bb976:
    pdl_interp.check_operation_name of %437 is "math.sin" -> ^bb977, ^bb1
  ^bb977:
    pdl_interp.check_operand_count of %437 is 1 -> ^bb978, ^bb1
  ^bb978:
    pdl_interp.check_result_count of %437 is 1 -> ^bb979, ^bb1
  ^bb979:
    %489 = pdl_interp.get_result 0 of %437
    pdl_interp.is_not_null %489 : !pdl.value -> ^bb980, ^bb1
  ^bb980:
    pdl_interp.are_equal %489, %436 : !pdl.value -> ^bb981, ^bb1
  ^bb981:
    %490 = pdl_interp.get_operand 0 of %433
    pdl_interp.is_not_null %490 : !pdl.value -> ^bb982, ^bb1
  ^bb982:
    %491 = pdl_interp.get_value_type of %490 : !pdl.type
    %492 = pdl_interp.get_value_type of %487 : !pdl.type
    pdl_interp.are_equal %491, %492 : !pdl.type -> ^bb983, ^bb1
  ^bb983:
    %493 = pdl_interp.get_value_type of %400 : !pdl.type
    pdl_interp.are_equal %491, %493 : !pdl.type -> ^bb984, ^bb1
  ^bb984:
    %494 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %491, %494 : !pdl.type -> ^bb985, ^bb1
  ^bb985:
    pdl_interp.check_type %491 is f32 -> ^bb986, ^bb1
  ^bb986:
    %495 = pdl_interp.get_operand 0 of %432
    pdl_interp.is_not_null %495 : !pdl.value -> ^bb987, ^bb1
  ^bb987:
    %496 = pdl_interp.get_value_type of %463 : !pdl.type
    pdl_interp.are_equal %491, %496 : !pdl.type -> ^bb988, ^bb1
  ^bb988:
    %497 = pdl_interp.get_value_type of %430 : !pdl.type
    pdl_interp.are_equal %491, %497 : !pdl.type -> ^bb989, ^bb1
  ^bb989:
    %498 = pdl_interp.get_value_type of %489 : !pdl.type
    pdl_interp.are_equal %491, %498 : !pdl.type -> ^bb990, ^bb1
  ^bb990:
    %499 = pdl_interp.get_value_type of %488 : !pdl.type
    pdl_interp.are_equal %491, %499 : !pdl.type -> ^bb991, ^bb1
  ^bb991:
    %500 = pdl_interp.get_operand 0 of %435
    pdl_interp.are_equal %490, %500 : !pdl.value -> ^bb992, ^bb1
  ^bb992:
    %501 = pdl_interp.get_value_type of %495 : !pdl.type
    pdl_interp.are_equal %491, %501 : !pdl.type -> ^bb993, ^bb1
  ^bb993:
    %502 = pdl_interp.get_operand 0 of %437
    pdl_interp.are_equal %495, %502 : !pdl.value -> ^bb994, ^bb1
  ^bb994:
    pdl_interp.record_match @rewriters::@sin_diff_rev(%490, %495, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb1
  ^bb874:
    pdl_interp.check_operand_count of %432 is 2 -> ^bb995, ^bb1
  ^bb995:
    pdl_interp.check_result_count of %432 is 1 -> ^bb996, ^bb1
  ^bb996:
    %503 = pdl_interp.get_result 0 of %432
    pdl_interp.is_not_null %503 : !pdl.value -> ^bb997, ^bb1
  ^bb997:
    pdl_interp.are_equal %503, %431 : !pdl.value -> ^bb998, ^bb1
  ^bb998:
    pdl_interp.check_operation_name of %433 is "arith.constant" -> ^bb999, ^bb1
  ^bb999:
    pdl_interp.check_operand_count of %433 is 0 -> ^bb1000, ^bb1
  ^bb1000:
    pdl_interp.check_result_count of %433 is 1 -> ^bb1001, ^bb1
  ^bb1001:
    %504 = pdl_interp.get_result 0 of %433
    pdl_interp.is_not_null %504 : !pdl.value -> ^bb1002, ^bb1
  ^bb1002:
    pdl_interp.are_equal %504, %401 : !pdl.value -> ^bb1003, ^bb1
  ^bb1003:
    pdl_interp.check_operation_name of %435 is "arith.constant" -> ^bb1004, ^bb1
  ^bb1004:
    pdl_interp.check_operand_count of %435 is 0 -> ^bb1005, ^bb1
  ^bb1005:
    pdl_interp.check_result_count of %435 is 1 -> ^bb1006, ^bb1
  ^bb1006:
    %505 = pdl_interp.get_result 0 of %435
    pdl_interp.is_not_null %505 : !pdl.value -> ^bb1007, ^bb1
  ^bb1007:
    pdl_interp.are_equal %505, %434 : !pdl.value -> ^bb1008, ^bb1
  ^bb1008:
    pdl_interp.check_operation_name of %437 is "math.cos" -> ^bb1009, ^bb1
  ^bb1009:
    pdl_interp.check_operand_count of %437 is 1 -> ^bb1010, ^bb1
  ^bb1010:
    pdl_interp.check_result_count of %437 is 1 -> ^bb1011, ^bb1
  ^bb1011:
    %506 = pdl_interp.get_result 0 of %437
    pdl_interp.is_not_null %506 : !pdl.value -> ^bb1012, ^bb1
  ^bb1012:
    pdl_interp.are_equal %506, %436 : !pdl.value -> ^bb1013, ^bb1
  ^bb1013:
    %507 = pdl_interp.get_operand 0 of %432
    pdl_interp.is_not_null %507 : !pdl.value -> ^bb1014, ^bb1
  ^bb1014:
    %508 = pdl_interp.get_defining_op of %507 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %508 : !pdl.operation -> ^bb1015, ^bb1
  ^bb1015:
    %509 = pdl_interp.get_attribute "value" of %433
    pdl_interp.is_not_null %509 : !pdl.attribute -> ^bb1016, ^bb1
  ^bb1016:
    pdl_interp.check_attribute %509 is 4.000000e+00 : f32 -> ^bb1017, ^bb1
  ^bb1017:
    %510 = pdl_interp.get_value_type of %504 : !pdl.type
    %511 = pdl_interp.get_value_type of %400 : !pdl.type
    pdl_interp.are_equal %510, %511 : !pdl.type -> ^bb1018, ^bb1
  ^bb1018:
    %512 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %510, %512 : !pdl.type -> ^bb1019, ^bb1
  ^bb1019:
    pdl_interp.check_type %510 is f32 -> ^bb1020, ^bb1
  ^bb1020:
    %513 = pdl_interp.get_attribute "value" of %435
    pdl_interp.is_not_null %513 : !pdl.attribute -> ^bb1021, ^bb1
  ^bb1021:
    pdl_interp.check_attribute %513 is 3.000000e+00 : f32 -> ^bb1022, ^bb1
  ^bb1022:
    pdl_interp.check_operation_name of %508 is "math.cos" -> ^bb1023, ^bb1
  ^bb1023:
    pdl_interp.check_operand_count of %508 is 1 -> ^bb1024, ^bb1
  ^bb1024:
    pdl_interp.check_result_count of %508 is 1 -> ^bb1025, ^bb1
  ^bb1025:
    %514 = pdl_interp.get_result 0 of %508
    pdl_interp.is_not_null %514 : !pdl.value -> ^bb1026, ^bb1
  ^bb1026:
    pdl_interp.are_equal %514, %507 : !pdl.value -> ^bb1027, ^bb1
  ^bb1027:
    %515 = pdl_interp.get_value_type of %503 : !pdl.type
    pdl_interp.are_equal %510, %515 : !pdl.type -> ^bb1028, ^bb1
  ^bb1028:
    %516 = pdl_interp.get_operand 1 of %432
    %517 = pdl_interp.get_defining_op of %516 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %517 : !pdl.operation -> ^bb1029, ^bb1
  ^bb1029:
    %518 = pdl_interp.get_value_type of %430 : !pdl.type
    pdl_interp.are_equal %510, %518 : !pdl.type -> ^bb1030, ^bb1
  ^bb1030:
    pdl_interp.is_not_null %516 : !pdl.value -> ^bb1031, ^bb1
  ^bb1031:
    %519 = pdl_interp.get_value_type of %505 : !pdl.type
    pdl_interp.are_equal %510, %519 : !pdl.type -> ^bb1032, ^bb1
  ^bb1032:
    %520 = pdl_interp.get_operand 0 of %508
    pdl_interp.is_not_null %520 : !pdl.value -> ^bb1033, ^bb1
  ^bb1033:
    %521 = pdl_interp.get_value_type of %506 : !pdl.type
    pdl_interp.are_equal %510, %521 : !pdl.type -> ^bb1034, ^bb1
  ^bb1034:
    pdl_interp.check_operation_name of %517 is "arith.constant" -> ^bb1035, ^bb1
  ^bb1035:
    pdl_interp.check_operand_count of %517 is 0 -> ^bb1036, ^bb1
  ^bb1036:
    pdl_interp.check_result_count of %517 is 1 -> ^bb1037, ^bb1
  ^bb1037:
    %522 = pdl_interp.get_result 0 of %517
    pdl_interp.is_not_null %522 : !pdl.value -> ^bb1038, ^bb1
  ^bb1038:
    pdl_interp.are_equal %522, %516 : !pdl.value -> ^bb1039, ^bb1
  ^bb1039:
    %523 = pdl_interp.get_value_type of %514 : !pdl.type
    pdl_interp.are_equal %523, %510 : !pdl.type -> ^bb1040, ^bb1
  ^bb1040:
    %524 = pdl_interp.get_operand 0 of %437
    pdl_interp.are_equal %520, %524 : !pdl.value -> ^bb1041, ^bb1
  ^bb1041:
    %525 = pdl_interp.get_attribute "value" of %517
    pdl_interp.is_not_null %525 : !pdl.attribute -> ^bb1042, ^bb1
  ^bb1042:
    pdl_interp.check_attribute %525 is 3.000000e+00 : f32 -> ^bb1043, ^bb1
  ^bb1043:
    %526 = pdl_interp.get_value_type of %520 : !pdl.type
    pdl_interp.are_equal %526, %510 : !pdl.type -> ^bb1044, ^bb1
  ^bb1044:
    %527 = pdl_interp.get_value_type of %522 : !pdl.type
    pdl_interp.are_equal %527, %510 : !pdl.type -> ^bb1045, ^bb1
  ^bb1045:
    pdl_interp.record_match @rewriters::@_3_cos(%520, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb1
  ^bb875:
    pdl_interp.check_operand_count of %432 is 1 -> ^bb1046, ^bb1
  ^bb1046:
    pdl_interp.check_result_count of %432 is 1 -> ^bb1047, ^bb1
  ^bb1047:
    %528 = pdl_interp.get_result 0 of %432
    pdl_interp.is_not_null %528 : !pdl.value -> ^bb1048, ^bb1
  ^bb1048:
    pdl_interp.are_equal %528, %431 : !pdl.value -> ^bb1049, ^bb1
  ^bb1049:
    pdl_interp.switch_operation_name of %433 to ["math.cosh", "math.sinh"](^bb1050, ^bb1051) -> ^bb1
  ^bb1050:
    pdl_interp.check_operand_count of %433 is 1 -> ^bb1052, ^bb1
  ^bb1052:
    pdl_interp.check_result_count of %433 is 1 -> ^bb1053, ^bb1
  ^bb1053:
    %529 = pdl_interp.get_result 0 of %433
    pdl_interp.is_not_null %529 : !pdl.value -> ^bb1054, ^bb1
  ^bb1054:
    pdl_interp.are_equal %529, %401 : !pdl.value -> ^bb1055, ^bb1
  ^bb1055:
    pdl_interp.check_operation_name of %435 is "math.sinh" -> ^bb1056, ^bb1
  ^bb1056:
    pdl_interp.check_operand_count of %435 is 1 -> ^bb1057, ^bb1
  ^bb1057:
    pdl_interp.check_result_count of %435 is 1 -> ^bb1058, ^bb1
  ^bb1058:
    %530 = pdl_interp.get_result 0 of %435
    pdl_interp.is_not_null %530 : !pdl.value -> ^bb1059, ^bb1
  ^bb1059:
    pdl_interp.are_equal %530, %434 : !pdl.value -> ^bb1060, ^bb1
  ^bb1060:
    pdl_interp.check_operation_name of %437 is "math.sinh" -> ^bb1061, ^bb1
  ^bb1061:
    pdl_interp.check_operand_count of %437 is 1 -> ^bb1062, ^bb1
  ^bb1062:
    pdl_interp.check_result_count of %437 is 1 -> ^bb1063, ^bb1
  ^bb1063:
    %531 = pdl_interp.get_result 0 of %437
    pdl_interp.is_not_null %531 : !pdl.value -> ^bb1064, ^bb1
  ^bb1064:
    pdl_interp.are_equal %531, %436 : !pdl.value -> ^bb1065, ^bb1
  ^bb1065:
    %532 = pdl_interp.get_operand 0 of %433
    pdl_interp.is_not_null %532 : !pdl.value -> ^bb1066, ^bb1
  ^bb1066:
    %533 = pdl_interp.get_value_type of %532 : !pdl.type
    %534 = pdl_interp.get_value_type of %529 : !pdl.type
    pdl_interp.are_equal %533, %534 : !pdl.type -> ^bb1067, ^bb1
  ^bb1067:
    %535 = pdl_interp.get_value_type of %400 : !pdl.type
    pdl_interp.are_equal %533, %535 : !pdl.type -> ^bb1068, ^bb1
  ^bb1068:
    %536 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %533, %536 : !pdl.type -> ^bb1069, ^bb1
  ^bb1069:
    pdl_interp.check_type %533 is f32 -> ^bb1070, ^bb1
  ^bb1070:
    %537 = pdl_interp.get_value_type of %528 : !pdl.type
    pdl_interp.are_equal %533, %537 : !pdl.type -> ^bb1071, ^bb1072
  ^bb1072:
    %538 = pdl_interp.get_operand 0 of %432
    pdl_interp.is_not_null %538 : !pdl.value -> ^bb1073, ^bb1
  ^bb1073:
    %539 = pdl_interp.get_value_type of %528 : !pdl.type
    pdl_interp.are_equal %533, %539 : !pdl.type -> ^bb1074, ^bb1
  ^bb1074:
    %540 = pdl_interp.get_value_type of %430 : !pdl.type
    pdl_interp.are_equal %533, %540 : !pdl.type -> ^bb1075, ^bb1
  ^bb1075:
    %541 = pdl_interp.get_value_type of %531 : !pdl.type
    pdl_interp.are_equal %533, %541 : !pdl.type -> ^bb1076, ^bb1
  ^bb1076:
    %542 = pdl_interp.get_value_type of %530 : !pdl.type
    pdl_interp.are_equal %533, %542 : !pdl.type -> ^bb1077, ^bb1
  ^bb1077:
    %543 = pdl_interp.get_operand 0 of %435
    pdl_interp.are_equal %532, %543 : !pdl.value -> ^bb1078, ^bb1
  ^bb1078:
    %544 = pdl_interp.get_value_type of %538 : !pdl.type
    pdl_interp.are_equal %533, %544 : !pdl.type -> ^bb1079, ^bb1
  ^bb1079:
    %545 = pdl_interp.get_operand 0 of %437
    pdl_interp.are_equal %538, %545 : !pdl.value -> ^bb1080, ^bb1
  ^bb1080:
    pdl_interp.record_match @rewriters::@cosh_diff_rev(%532, %538, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb1
  ^bb1071:
    %546 = pdl_interp.get_value_type of %430 : !pdl.type
    pdl_interp.are_equal %533, %546 : !pdl.type -> ^bb1081, ^bb1072
  ^bb1081:
    %547 = pdl_interp.get_value_type of %531 : !pdl.type
    pdl_interp.are_equal %533, %547 : !pdl.type -> ^bb1082, ^bb1072
  ^bb1082:
    %548 = pdl_interp.get_value_type of %530 : !pdl.type
    pdl_interp.are_equal %533, %548 : !pdl.type -> ^bb1083, ^bb1072
  ^bb1083:
    %549 = pdl_interp.get_operand 0 of %435
    pdl_interp.are_equal %532, %549 : !pdl.value -> ^bb1084, ^bb1072
  ^bb1084:
    %550 = pdl_interp.get_operand 0 of %432
    pdl_interp.are_equal %532, %550 : !pdl.value -> ^bb1085, ^bb1072
  ^bb1085:
    %551 = pdl_interp.get_operand 0 of %437
    pdl_interp.are_equal %532, %551 : !pdl.value -> ^bb1086, ^bb1072
  ^bb1086:
    pdl_interp.record_match @rewriters::@sinh_cosh(%0 : !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb1072
  ^bb1051:
    pdl_interp.check_operand_count of %433 is 1 -> ^bb1087, ^bb1
  ^bb1087:
    pdl_interp.check_result_count of %433 is 1 -> ^bb1088, ^bb1
  ^bb1088:
    %552 = pdl_interp.get_result 0 of %433
    pdl_interp.is_not_null %552 : !pdl.value -> ^bb1089, ^bb1
  ^bb1089:
    pdl_interp.are_equal %552, %401 : !pdl.value -> ^bb1090, ^bb1
  ^bb1090:
    pdl_interp.check_operation_name of %435 is "math.cosh" -> ^bb1091, ^bb1
  ^bb1091:
    pdl_interp.check_operand_count of %435 is 1 -> ^bb1092, ^bb1
  ^bb1092:
    pdl_interp.check_result_count of %435 is 1 -> ^bb1093, ^bb1
  ^bb1093:
    %553 = pdl_interp.get_result 0 of %435
    pdl_interp.is_not_null %553 : !pdl.value -> ^bb1094, ^bb1
  ^bb1094:
    pdl_interp.are_equal %553, %434 : !pdl.value -> ^bb1095, ^bb1
  ^bb1095:
    pdl_interp.check_operation_name of %437 is "math.sinh" -> ^bb1096, ^bb1
  ^bb1096:
    pdl_interp.check_operand_count of %437 is 1 -> ^bb1097, ^bb1
  ^bb1097:
    pdl_interp.check_result_count of %437 is 1 -> ^bb1098, ^bb1
  ^bb1098:
    %554 = pdl_interp.get_result 0 of %437
    pdl_interp.is_not_null %554 : !pdl.value -> ^bb1099, ^bb1
  ^bb1099:
    pdl_interp.are_equal %554, %436 : !pdl.value -> ^bb1100, ^bb1
  ^bb1100:
    %555 = pdl_interp.get_operand 0 of %433
    pdl_interp.is_not_null %555 : !pdl.value -> ^bb1101, ^bb1
  ^bb1101:
    %556 = pdl_interp.get_value_type of %555 : !pdl.type
    %557 = pdl_interp.get_value_type of %552 : !pdl.type
    pdl_interp.are_equal %556, %557 : !pdl.type -> ^bb1102, ^bb1
  ^bb1102:
    %558 = pdl_interp.get_value_type of %400 : !pdl.type
    pdl_interp.are_equal %556, %558 : !pdl.type -> ^bb1103, ^bb1
  ^bb1103:
    %559 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %556, %559 : !pdl.type -> ^bb1104, ^bb1
  ^bb1104:
    pdl_interp.check_type %556 is f32 -> ^bb1105, ^bb1
  ^bb1105:
    %560 = pdl_interp.get_operand 0 of %432
    pdl_interp.is_not_null %560 : !pdl.value -> ^bb1106, ^bb1
  ^bb1106:
    %561 = pdl_interp.get_value_type of %528 : !pdl.type
    pdl_interp.are_equal %556, %561 : !pdl.type -> ^bb1107, ^bb1
  ^bb1107:
    %562 = pdl_interp.get_value_type of %430 : !pdl.type
    pdl_interp.are_equal %556, %562 : !pdl.type -> ^bb1108, ^bb1
  ^bb1108:
    %563 = pdl_interp.get_value_type of %554 : !pdl.type
    pdl_interp.are_equal %556, %563 : !pdl.type -> ^bb1109, ^bb1
  ^bb1109:
    %564 = pdl_interp.get_value_type of %553 : !pdl.type
    pdl_interp.are_equal %556, %564 : !pdl.type -> ^bb1110, ^bb1
  ^bb1110:
    %565 = pdl_interp.get_operand 0 of %435
    pdl_interp.are_equal %555, %565 : !pdl.value -> ^bb1111, ^bb1
  ^bb1111:
    %566 = pdl_interp.get_value_type of %560 : !pdl.type
    pdl_interp.are_equal %556, %566 : !pdl.type -> ^bb1112, ^bb1
  ^bb1112:
    %567 = pdl_interp.get_operand 0 of %437
    pdl_interp.are_equal %560, %567 : !pdl.value -> ^bb1113, ^bb1
  ^bb1113:
    pdl_interp.record_match @rewriters::@sinh_diff_rev(%555, %560, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb1
  ^bb810:
    %568 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %402, %568 : !pdl.type -> ^bb1114, ^bb811
  ^bb1114:
    pdl_interp.check_type %402 is f32 -> ^bb1115, ^bb811
  ^bb1115:
    pdl_interp.switch_operation_name of %399 to ["arith.mulf", "arith.constant"](^bb1116, ^bb1117) -> ^bb811
  ^bb1116:
    pdl_interp.check_operand_count of %399 is 2 -> ^bb1118, ^bb811
  ^bb1118:
    pdl_interp.check_result_count of %399 is 1 -> ^bb1119, ^bb811
  ^bb1119:
    %569 = pdl_interp.get_result 0 of %399
    pdl_interp.is_not_null %569 : !pdl.value -> ^bb1120, ^bb811
  ^bb1120:
    pdl_interp.are_equal %569, %398 : !pdl.value -> ^bb1121, ^bb811
  ^bb1121:
    %570 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %570 : !pdl.value -> ^bb1122, ^bb1123
  ^bb1123:
    %571 = pdl_interp.get_operand 0 of %399
    pdl_interp.is_not_null %571 : !pdl.value -> ^bb1124, ^bb811
  ^bb1124:
    %572 = pdl_interp.get_value_type of %569 : !pdl.type
    pdl_interp.are_equal %402, %572 : !pdl.type -> ^bb1125, ^bb811
  ^bb1125:
    %573 = pdl_interp.get_value_type of %571 : !pdl.type
    pdl_interp.are_equal %402, %573 : !pdl.type -> ^bb1126, ^bb811
  ^bb1126:
    %574 = pdl_interp.get_operand 1 of %3
    pdl_interp.are_equal %401, %574 : !pdl.value -> ^bb1127, ^bb811
  ^bb1127:
    %575 = pdl_interp.get_operand 1 of %399
    pdl_interp.are_equal %571, %575 : !pdl.value -> ^bb1128, ^bb811
  ^bb1128:
    pdl_interp.record_match @rewriters::@difference_of_squares(%401, %571, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb811
  ^bb1122:
    %576 = pdl_interp.get_operand 1 of %399
    pdl_interp.is_not_null %576 : !pdl.value -> ^bb1129, ^bb1130
  ^bb1130:
    %577 = pdl_interp.get_operand 0 of %399
    pdl_interp.is_not_null %577 : !pdl.value -> ^bb1131, ^bb1123
  ^bb1131:
    %578 = pdl_interp.get_value_type of %569 : !pdl.type
    pdl_interp.are_equal %402, %578 : !pdl.type -> ^bb1132, ^bb1123
  ^bb1132:
    %579 = pdl_interp.get_value_type of %570 : !pdl.type
    pdl_interp.are_equal %402, %579 : !pdl.type -> ^bb1133, ^bb1123
  ^bb1133:
    %580 = pdl_interp.get_value_type of %577 : !pdl.type
    pdl_interp.are_equal %402, %580 : !pdl.type -> ^bb1134, ^bb1123
  ^bb1134:
    %581 = pdl_interp.get_operand 1 of %399
    pdl_interp.are_equal %570, %581 : !pdl.value -> ^bb1135, ^bb1123
  ^bb1135:
    pdl_interp.record_match @rewriters::@distribute_rgt_outsub_(%401, %577, %570, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb1123
  ^bb1129:
    %582 = pdl_interp.get_value_type of %569 : !pdl.type
    pdl_interp.are_equal %402, %582 : !pdl.type -> ^bb1136, ^bb1130
  ^bb1136:
    %583 = pdl_interp.get_value_type of %570 : !pdl.type
    pdl_interp.are_equal %402, %583 : !pdl.type -> ^bb1137, ^bb1130
  ^bb1137:
    %584 = pdl_interp.get_operand 0 of %399
    pdl_interp.are_equal %401, %584 : !pdl.value -> ^bb1138, ^bb1130
  ^bb1138:
    %585 = pdl_interp.get_value_type of %576 : !pdl.type
    pdl_interp.are_equal %402, %585 : !pdl.type -> ^bb1139, ^bb1130
  ^bb1139:
    pdl_interp.record_match @rewriters::@distribute_lft_outsub_(%570, %576, %401, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb1130
  ^bb1117:
    pdl_interp.check_operand_count of %399 is 0 -> ^bb1140, ^bb811
  ^bb1140:
    pdl_interp.check_result_count of %399 is 1 -> ^bb1141, ^bb811
  ^bb1141:
    %586 = pdl_interp.get_result 0 of %399
    pdl_interp.is_not_null %586 : !pdl.value -> ^bb1142, ^bb811
  ^bb1142:
    pdl_interp.are_equal %586, %398 : !pdl.value -> ^bb1143, ^bb811
  ^bb1143:
    %587 = pdl_interp.get_value_type of %586 : !pdl.type
    pdl_interp.are_equal %402, %587 : !pdl.type -> ^bb1144, ^bb811
  ^bb1144:
    %588 = pdl_interp.get_attribute "value" of %399
    pdl_interp.is_not_null %588 : !pdl.attribute -> ^bb1145, ^bb811
  ^bb1145:
    pdl_interp.check_attribute %588 is 1.000000e+00 : f32 -> ^bb1146, ^bb811
  ^bb1146:
    %589 = pdl_interp.get_operand 1 of %3
    pdl_interp.are_equal %401, %589 : !pdl.value -> ^bb1147, ^bb811
  ^bb1147:
    pdl_interp.record_match @rewriters::@difference_of_sqr_1(%401, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb811
  ^bb794:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb1148, ^bb1
  ^bb1148:
    pdl_interp.check_result_count of %3 is 1 -> ^bb1149, ^bb1
  ^bb1149:
    %590 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %590 : !pdl.value -> ^bb1150, ^bb1
  ^bb1150:
    pdl_interp.are_equal %590, %2 : !pdl.value -> ^bb1151, ^bb1
  ^bb1151:
    %591 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %591 : !pdl.value -> ^bb1152, ^bb1
  ^bb1152:
    pdl_interp.is_not_null %398 : !pdl.value -> ^bb1153, ^bb1
  ^bb1153:
    %592 = pdl_interp.get_value_type of %591 : !pdl.type
    %593 = pdl_interp.get_value_type of %590 : !pdl.type
    pdl_interp.are_equal %592, %593 : !pdl.type -> ^bb1154, ^bb1
  ^bb1154:
    %594 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %592, %594 : !pdl.type -> ^bb1155, ^bb1
  ^bb1155:
    pdl_interp.check_type %592 is f32 -> ^bb1156, ^bb1
  ^bb1156:
    pdl_interp.check_operation_name of %399 is "math.powf" -> ^bb1157, ^bb1
  ^bb1157:
    pdl_interp.check_operand_count of %399 is 2 -> ^bb1158, ^bb1
  ^bb1158:
    pdl_interp.check_result_count of %399 is 1 -> ^bb1159, ^bb1
  ^bb1159:
    %595 = pdl_interp.get_result 0 of %399
    pdl_interp.is_not_null %595 : !pdl.value -> ^bb1160, ^bb1
  ^bb1160:
    pdl_interp.are_equal %595, %398 : !pdl.value -> ^bb1161, ^bb1
  ^bb1161:
    %596 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %596 : !pdl.value -> ^bb1162, ^bb1
  ^bb1162:
    %597 = pdl_interp.get_defining_op of %596 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %597 : !pdl.operation -> ^bb1163, ^bb1
  ^bb1163:
    %598 = pdl_interp.get_operand 0 of %399
    pdl_interp.is_not_null %598 : !pdl.value -> ^bb1164, ^bb1
  ^bb1164:
    %599 = pdl_interp.get_operand 1 of %399
    %600 = pdl_interp.get_defining_op of %599 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %600 : !pdl.operation -> ^bb1165, ^bb1
  ^bb1165:
    pdl_interp.is_not_null %599 : !pdl.value -> ^bb1166, ^bb1
  ^bb1166:
    %601 = pdl_interp.get_value_type of %595 : !pdl.type
    pdl_interp.are_equal %592, %601 : !pdl.type -> ^bb1167, ^bb1
  ^bb1167:
    pdl_interp.check_operation_name of %597 is "arith.constant" -> ^bb1168, ^bb1
  ^bb1168:
    pdl_interp.check_operand_count of %597 is 0 -> ^bb1169, ^bb1
  ^bb1169:
    pdl_interp.check_result_count of %597 is 1 -> ^bb1170, ^bb1
  ^bb1170:
    %602 = pdl_interp.get_result 0 of %597
    pdl_interp.is_not_null %602 : !pdl.value -> ^bb1171, ^bb1
  ^bb1171:
    pdl_interp.are_equal %602, %596 : !pdl.value -> ^bb1172, ^bb1
  ^bb1172:
    pdl_interp.check_operation_name of %600 is "arith.constant" -> ^bb1173, ^bb1
  ^bb1173:
    pdl_interp.check_operand_count of %600 is 0 -> ^bb1174, ^bb1
  ^bb1174:
    pdl_interp.check_result_count of %600 is 1 -> ^bb1175, ^bb1
  ^bb1175:
    %603 = pdl_interp.get_result 0 of %600
    pdl_interp.is_not_null %603 : !pdl.value -> ^bb1176, ^bb1
  ^bb1176:
    pdl_interp.are_equal %603, %599 : !pdl.value -> ^bb1177, ^bb1
  ^bb1177:
    %604 = pdl_interp.get_value_type of %598 : !pdl.type
    pdl_interp.are_equal %592, %604 : !pdl.type -> ^bb1178, ^bb1
  ^bb1178:
    %605 = pdl_interp.get_attribute "value" of %597
    pdl_interp.is_not_null %605 : !pdl.attribute -> ^bb1179, ^bb1
  ^bb1179:
    pdl_interp.check_attribute %605 is 3.000000e+00 : f32 -> ^bb1180, ^bb1
  ^bb1180:
    %606 = pdl_interp.get_value_type of %602 : !pdl.type
    pdl_interp.are_equal %606, %592 : !pdl.type -> ^bb1181, ^bb1
  ^bb1181:
    %607 = pdl_interp.get_value_type of %603 : !pdl.type
    pdl_interp.are_equal %607, %592 : !pdl.type -> ^bb1182, ^bb1
  ^bb1182:
    %608 = pdl_interp.get_attribute "value" of %600
    pdl_interp.is_not_null %608 : !pdl.attribute -> ^bb1183, ^bb1
  ^bb1183:
    pdl_interp.check_attribute %608 is 3.000000e+00 : f32 -> ^bb1184, ^bb1
  ^bb1184:
    pdl_interp.record_match @rewriters::@difference_cubes(%591, %598, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb1
  ^bb795:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb1185, ^bb1
  ^bb1185:
    pdl_interp.check_result_count of %3 is 1 -> ^bb1186, ^bb1
  ^bb1186:
    %609 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %609 : !pdl.value -> ^bb1187, ^bb1
  ^bb1187:
    pdl_interp.are_equal %609, %2 : !pdl.value -> ^bb1188, ^bb1
  ^bb1188:
    %610 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %610 : !pdl.value -> ^bb1189, ^bb1
  ^bb1189:
    pdl_interp.is_not_null %398 : !pdl.value -> ^bb1190, ^bb1
  ^bb1190:
    %611 = pdl_interp.get_value_type of %610 : !pdl.type
    %612 = pdl_interp.get_value_type of %609 : !pdl.type
    pdl_interp.are_equal %611, %612 : !pdl.type -> ^bb1191, ^bb1
  ^bb1191:
    %613 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %611, %613 : !pdl.type -> ^bb1192, ^bb1
  ^bb1192:
    pdl_interp.check_type %611 is f32 -> ^bb1193, ^bb1
  ^bb1193:
    pdl_interp.check_operation_name of %399 is "arith.divf" -> ^bb1194, ^bb1
  ^bb1194:
    pdl_interp.check_operand_count of %399 is 2 -> ^bb1195, ^bb1
  ^bb1195:
    pdl_interp.check_result_count of %399 is 1 -> ^bb1196, ^bb1
  ^bb1196:
    %614 = pdl_interp.get_result 0 of %399
    pdl_interp.is_not_null %614 : !pdl.value -> ^bb1197, ^bb1
  ^bb1197:
    pdl_interp.are_equal %614, %398 : !pdl.value -> ^bb1198, ^bb1
  ^bb1198:
    %615 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %615 : !pdl.value -> ^bb1199, ^bb1
  ^bb1199:
    %616 = pdl_interp.get_operand 0 of %399
    pdl_interp.is_not_null %616 : !pdl.value -> ^bb1200, ^bb1
  ^bb1200:
    %617 = pdl_interp.get_value_type of %614 : !pdl.type
    pdl_interp.are_equal %611, %617 : !pdl.type -> ^bb1201, ^bb1202
  ^bb1202:
    %618 = pdl_interp.get_operand 1 of %399
    pdl_interp.is_not_null %618 : !pdl.value -> ^bb1203, ^bb1
  ^bb1203:
    %619 = pdl_interp.get_value_type of %614 : !pdl.type
    pdl_interp.are_equal %611, %619 : !pdl.type -> ^bb1204, ^bb1
  ^bb1204:
    %620 = pdl_interp.get_value_type of %615 : !pdl.type
    pdl_interp.are_equal %611, %620 : !pdl.type -> ^bb1205, ^bb1
  ^bb1205:
    %621 = pdl_interp.get_value_type of %616 : !pdl.type
    pdl_interp.are_equal %611, %621 : !pdl.type -> ^bb1206, ^bb1
  ^bb1206:
    %622 = pdl_interp.get_value_type of %618 : !pdl.type
    pdl_interp.are_equal %611, %622 : !pdl.type -> ^bb1207, ^bb1
  ^bb1207:
    pdl_interp.record_match @rewriters::@frac_sub(%610, %618, %615, %616, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb1
  ^bb1201:
    %623 = pdl_interp.get_value_type of %615 : !pdl.type
    pdl_interp.are_equal %611, %623 : !pdl.type -> ^bb1208, ^bb1202
  ^bb1208:
    %624 = pdl_interp.get_value_type of %616 : !pdl.type
    pdl_interp.are_equal %611, %624 : !pdl.type -> ^bb1209, ^bb1202
  ^bb1209:
    %625 = pdl_interp.get_operand 1 of %399
    pdl_interp.are_equal %615, %625 : !pdl.value -> ^bb1210, ^bb1202
  ^bb1210:
    pdl_interp.record_match @rewriters::@sub_div(%610, %616, %615, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb1202
  ^bb796:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb1211, ^bb1
  ^bb1211:
    pdl_interp.check_result_count of %3 is 1 -> ^bb1212, ^bb1
  ^bb1212:
    %626 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %626 : !pdl.value -> ^bb1213, ^bb1
  ^bb1213:
    pdl_interp.are_equal %626, %2 : !pdl.value -> ^bb1214, ^bb1
  ^bb1214:
    %627 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %627 : !pdl.value -> ^bb1215, ^bb1
  ^bb1215:
    pdl_interp.is_not_null %398 : !pdl.value -> ^bb1216, ^bb1
  ^bb1216:
    %628 = pdl_interp.get_value_type of %627 : !pdl.type
    %629 = pdl_interp.get_value_type of %626 : !pdl.type
    pdl_interp.are_equal %628, %629 : !pdl.type -> ^bb1217, ^bb1
  ^bb1217:
    %630 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %628, %630 : !pdl.type -> ^bb1218, ^bb1
  ^bb1218:
    pdl_interp.check_type %628 is f32 -> ^bb1219, ^bb1
  ^bb1219:
    pdl_interp.check_operation_name of %399 is "math.log" -> ^bb1220, ^bb1
  ^bb1220:
    pdl_interp.check_operand_count of %399 is 1 -> ^bb1221, ^bb1
  ^bb1221:
    pdl_interp.check_result_count of %399 is 1 -> ^bb1222, ^bb1
  ^bb1222:
    %631 = pdl_interp.get_result 0 of %399
    pdl_interp.is_not_null %631 : !pdl.value -> ^bb1223, ^bb1
  ^bb1223:
    pdl_interp.are_equal %631, %398 : !pdl.value -> ^bb1224, ^bb1
  ^bb1224:
    %632 = pdl_interp.get_operand 0 of %399
    pdl_interp.is_not_null %632 : !pdl.value -> ^bb1225, ^bb1
  ^bb1225:
    %633 = pdl_interp.get_value_type of %631 : !pdl.type
    pdl_interp.are_equal %628, %633 : !pdl.type -> ^bb1226, ^bb1
  ^bb1226:
    %634 = pdl_interp.get_value_type of %632 : !pdl.type
    pdl_interp.are_equal %628, %634 : !pdl.type -> ^bb1227, ^bb1
  ^bb1227:
    pdl_interp.record_match @rewriters::@diff_log(%627, %632, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb1
  ^bb797:
    pdl_interp.check_operand_count of %3 is 0 -> ^bb1228, ^bb1
  ^bb1228:
    pdl_interp.check_result_count of %3 is 1 -> ^bb1229, ^bb1
  ^bb1229:
    %635 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %635 : !pdl.value -> ^bb1230, ^bb1
  ^bb1230:
    pdl_interp.are_equal %635, %2 : !pdl.value -> ^bb1231, ^bb1
  ^bb1231:
    pdl_interp.is_not_null %398 : !pdl.value -> ^bb1232, ^bb1
  ^bb1232:
    pdl_interp.check_operation_name of %399 is "arith.mulf" -> ^bb1233, ^bb1
  ^bb1233:
    pdl_interp.check_operand_count of %399 is 2 -> ^bb1234, ^bb1
  ^bb1234:
    pdl_interp.check_result_count of %399 is 1 -> ^bb1235, ^bb1
  ^bb1235:
    %636 = pdl_interp.get_result 0 of %399
    pdl_interp.is_not_null %636 : !pdl.value -> ^bb1236, ^bb1
  ^bb1236:
    pdl_interp.are_equal %636, %398 : !pdl.value -> ^bb1237, ^bb1
  ^bb1237:
    %637 = pdl_interp.get_operand 0 of %399
    pdl_interp.is_not_null %637 : !pdl.value -> ^bb1238, ^bb1
  ^bb1238:
    %638 = pdl_interp.get_defining_op of %637 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %638 : !pdl.operation -> ^bb1239, ^bb1
  ^bb1239:
    %639 = pdl_interp.get_operand 1 of %399
    %640 = pdl_interp.get_defining_op of %639 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %640 : !pdl.operation -> ^bb1240, ^bb1
  ^bb1240:
    pdl_interp.is_not_null %639 : !pdl.value -> ^bb1241, ^bb1
  ^bb1241:
    pdl_interp.switch_operation_name of %638 to ["math.sin", "math.cos", "arith.constant"](^bb1242, ^bb1243, ^bb1244) -> ^bb1
  ^bb1242:
    pdl_interp.check_operand_count of %638 is 1 -> ^bb1245, ^bb1
  ^bb1245:
    pdl_interp.check_result_count of %638 is 1 -> ^bb1246, ^bb1
  ^bb1246:
    %641 = pdl_interp.get_result 0 of %638
    pdl_interp.is_not_null %641 : !pdl.value -> ^bb1247, ^bb1
  ^bb1247:
    pdl_interp.are_equal %641, %637 : !pdl.value -> ^bb1248, ^bb1
  ^bb1248:
    pdl_interp.check_operation_name of %640 is "math.sin" -> ^bb1249, ^bb1
  ^bb1249:
    pdl_interp.check_operand_count of %640 is 1 -> ^bb1250, ^bb1
  ^bb1250:
    pdl_interp.check_result_count of %640 is 1 -> ^bb1251, ^bb1
  ^bb1251:
    %642 = pdl_interp.get_result 0 of %640
    pdl_interp.is_not_null %642 : !pdl.value -> ^bb1252, ^bb1
  ^bb1252:
    pdl_interp.are_equal %642, %639 : !pdl.value -> ^bb1253, ^bb1
  ^bb1253:
    %643 = pdl_interp.get_attribute "value" of %3
    pdl_interp.is_not_null %643 : !pdl.attribute -> ^bb1254, ^bb1
  ^bb1254:
    pdl_interp.check_attribute %643 is 1.000000e+00 : f32 -> ^bb1255, ^bb1
  ^bb1255:
    %644 = pdl_interp.get_value_type of %635 : !pdl.type
    %645 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %644, %645 : !pdl.type -> ^bb1256, ^bb1
  ^bb1256:
    pdl_interp.check_type %644 is f32 -> ^bb1257, ^bb1
  ^bb1257:
    %646 = pdl_interp.get_value_type of %636 : !pdl.type
    pdl_interp.are_equal %644, %646 : !pdl.type -> ^bb1258, ^bb1
  ^bb1258:
    %647 = pdl_interp.get_operand 0 of %638
    pdl_interp.is_not_null %647 : !pdl.value -> ^bb1259, ^bb1
  ^bb1259:
    %648 = pdl_interp.get_value_type of %641 : !pdl.type
    pdl_interp.are_equal %648, %644 : !pdl.type -> ^bb1260, ^bb1
  ^bb1260:
    %649 = pdl_interp.get_value_type of %642 : !pdl.type
    pdl_interp.are_equal %649, %644 : !pdl.type -> ^bb1261, ^bb1
  ^bb1261:
    %650 = pdl_interp.get_operand 0 of %640
    pdl_interp.are_equal %647, %650 : !pdl.value -> ^bb1262, ^bb1
  ^bb1262:
    %651 = pdl_interp.get_value_type of %647 : !pdl.type
    pdl_interp.are_equal %651, %644 : !pdl.type -> ^bb1263, ^bb1
  ^bb1263:
    pdl_interp.record_match @rewriters::@_1_sub_sin(%647, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb1264
  ^bb1264:
    pdl_interp.record_match @rewriters::@sqr_cos_b_rev(%647, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb1
  ^bb1243:
    pdl_interp.check_operand_count of %638 is 1 -> ^bb1265, ^bb1
  ^bb1265:
    pdl_interp.check_result_count of %638 is 1 -> ^bb1266, ^bb1
  ^bb1266:
    %652 = pdl_interp.get_result 0 of %638
    pdl_interp.is_not_null %652 : !pdl.value -> ^bb1267, ^bb1
  ^bb1267:
    pdl_interp.are_equal %652, %637 : !pdl.value -> ^bb1268, ^bb1
  ^bb1268:
    pdl_interp.check_operation_name of %640 is "math.cos" -> ^bb1269, ^bb1
  ^bb1269:
    pdl_interp.check_operand_count of %640 is 1 -> ^bb1270, ^bb1
  ^bb1270:
    pdl_interp.check_result_count of %640 is 1 -> ^bb1271, ^bb1
  ^bb1271:
    %653 = pdl_interp.get_result 0 of %640
    pdl_interp.is_not_null %653 : !pdl.value -> ^bb1272, ^bb1
  ^bb1272:
    pdl_interp.are_equal %653, %639 : !pdl.value -> ^bb1273, ^bb1
  ^bb1273:
    %654 = pdl_interp.get_attribute "value" of %3
    pdl_interp.is_not_null %654 : !pdl.attribute -> ^bb1274, ^bb1
  ^bb1274:
    pdl_interp.check_attribute %654 is 1.000000e+00 : f32 -> ^bb1275, ^bb1
  ^bb1275:
    %655 = pdl_interp.get_value_type of %635 : !pdl.type
    %656 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %655, %656 : !pdl.type -> ^bb1276, ^bb1
  ^bb1276:
    pdl_interp.check_type %655 is f32 -> ^bb1277, ^bb1
  ^bb1277:
    %657 = pdl_interp.get_value_type of %636 : !pdl.type
    pdl_interp.are_equal %655, %657 : !pdl.type -> ^bb1278, ^bb1
  ^bb1278:
    %658 = pdl_interp.get_operand 0 of %638
    pdl_interp.is_not_null %658 : !pdl.value -> ^bb1279, ^bb1
  ^bb1279:
    %659 = pdl_interp.get_value_type of %652 : !pdl.type
    pdl_interp.are_equal %659, %655 : !pdl.type -> ^bb1280, ^bb1
  ^bb1280:
    %660 = pdl_interp.get_value_type of %653 : !pdl.type
    pdl_interp.are_equal %660, %655 : !pdl.type -> ^bb1281, ^bb1
  ^bb1281:
    %661 = pdl_interp.get_operand 0 of %640
    pdl_interp.are_equal %658, %661 : !pdl.value -> ^bb1282, ^bb1
  ^bb1282:
    %662 = pdl_interp.get_value_type of %658 : !pdl.type
    pdl_interp.are_equal %662, %655 : !pdl.type -> ^bb1283, ^bb1
  ^bb1283:
    pdl_interp.record_match @rewriters::@_1_sub_cos(%658, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb1284
  ^bb1284:
    pdl_interp.record_match @rewriters::@sqr_sin_b_rev(%658, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb1
  ^bb1244:
    pdl_interp.check_operand_count of %638 is 0 -> ^bb1285, ^bb1
  ^bb1285:
    pdl_interp.check_result_count of %638 is 1 -> ^bb1286, ^bb1
  ^bb1286:
    %663 = pdl_interp.get_result 0 of %638
    pdl_interp.is_not_null %663 : !pdl.value -> ^bb1287, ^bb1
  ^bb1287:
    pdl_interp.are_equal %663, %637 : !pdl.value -> ^bb1288, ^bb1
  ^bb1288:
    pdl_interp.check_operation_name of %640 is "math.cos" -> ^bb1289, ^bb1
  ^bb1289:
    pdl_interp.check_operand_count of %640 is 1 -> ^bb1290, ^bb1
  ^bb1290:
    pdl_interp.check_result_count of %640 is 1 -> ^bb1291, ^bb1
  ^bb1291:
    %664 = pdl_interp.get_result 0 of %640
    pdl_interp.is_not_null %664 : !pdl.value -> ^bb1292, ^bb1
  ^bb1292:
    pdl_interp.are_equal %664, %639 : !pdl.value -> ^bb1293, ^bb1
  ^bb1293:
    %665 = pdl_interp.get_attribute "value" of %3
    pdl_interp.is_not_null %665 : !pdl.attribute -> ^bb1294, ^bb1
  ^bb1294:
    pdl_interp.check_attribute %665 is 5.000000e-01 : f32 -> ^bb1295, ^bb1
  ^bb1295:
    %666 = pdl_interp.get_value_type of %635 : !pdl.type
    %667 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %666, %667 : !pdl.type -> ^bb1296, ^bb1
  ^bb1296:
    pdl_interp.check_type %666 is f32 -> ^bb1297, ^bb1
  ^bb1297:
    %668 = pdl_interp.get_operand 0 of %640
    %669 = pdl_interp.get_defining_op of %668 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %669 : !pdl.operation -> ^bb1298, ^bb1
  ^bb1298:
    %670 = pdl_interp.get_value_type of %636 : !pdl.type
    pdl_interp.are_equal %666, %670 : !pdl.type -> ^bb1299, ^bb1
  ^bb1299:
    %671 = pdl_interp.get_operand 0 of %669
    %672 = pdl_interp.get_defining_op of %671 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %672 : !pdl.operation -> ^bb1300, ^bb1
  ^bb1300:
    %673 = pdl_interp.get_value_type of %663 : !pdl.type
    pdl_interp.are_equal %673, %666 : !pdl.type -> ^bb1301, ^bb1
  ^bb1301:
    pdl_interp.is_not_null %668 : !pdl.value -> ^bb1302, ^bb1
  ^bb1302:
    pdl_interp.check_operation_name of %669 is "arith.mulf" -> ^bb1303, ^bb1
  ^bb1303:
    pdl_interp.check_operand_count of %669 is 2 -> ^bb1304, ^bb1
  ^bb1304:
    pdl_interp.check_result_count of %669 is 1 -> ^bb1305, ^bb1
  ^bb1305:
    %674 = pdl_interp.get_result 0 of %669
    pdl_interp.is_not_null %674 : !pdl.value -> ^bb1306, ^bb1
  ^bb1306:
    pdl_interp.are_equal %674, %668 : !pdl.value -> ^bb1307, ^bb1
  ^bb1307:
    %675 = pdl_interp.get_attribute "value" of %638
    pdl_interp.is_not_null %675 : !pdl.attribute -> ^bb1308, ^bb1
  ^bb1308:
    pdl_interp.check_attribute %675 is 5.000000e-01 : f32 -> ^bb1309, ^bb1
  ^bb1309:
    %676 = pdl_interp.get_value_type of %664 : !pdl.type
    pdl_interp.are_equal %676, %666 : !pdl.type -> ^bb1310, ^bb1
  ^bb1310:
    pdl_interp.is_not_null %671 : !pdl.value -> ^bb1311, ^bb1
  ^bb1311:
    pdl_interp.check_operation_name of %672 is "arith.constant" -> ^bb1312, ^bb1
  ^bb1312:
    pdl_interp.check_operand_count of %672 is 0 -> ^bb1313, ^bb1
  ^bb1313:
    pdl_interp.check_result_count of %672 is 1 -> ^bb1314, ^bb1
  ^bb1314:
    %677 = pdl_interp.get_result 0 of %672
    pdl_interp.is_not_null %677 : !pdl.value -> ^bb1315, ^bb1
  ^bb1315:
    pdl_interp.are_equal %677, %671 : !pdl.value -> ^bb1316, ^bb1
  ^bb1316:
    %678 = pdl_interp.get_operand 1 of %669
    pdl_interp.is_not_null %678 : !pdl.value -> ^bb1317, ^bb1
  ^bb1317:
    %679 = pdl_interp.get_value_type of %674 : !pdl.type
    pdl_interp.are_equal %679, %666 : !pdl.type -> ^bb1318, ^bb1
  ^bb1318:
    %680 = pdl_interp.get_value_type of %677 : !pdl.type
    pdl_interp.are_equal %680, %666 : !pdl.type -> ^bb1319, ^bb1
  ^bb1319:
    %681 = pdl_interp.get_attribute "value" of %672
    pdl_interp.is_not_null %681 : !pdl.attribute -> ^bb1320, ^bb1
  ^bb1320:
    pdl_interp.check_attribute %681 is 2.000000e+00 : f32 -> ^bb1321, ^bb1
  ^bb1321:
    %682 = pdl_interp.get_value_type of %678 : !pdl.type
    pdl_interp.are_equal %682, %666 : !pdl.type -> ^bb1322, ^bb1
  ^bb1322:
    pdl_interp.record_match @rewriters::@sqr_sin_a_rev(%678, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb1
  ^bb798:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb1323, ^bb1
  ^bb1323:
    pdl_interp.check_result_count of %3 is 1 -> ^bb1324, ^bb1
  ^bb1324:
    %683 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %683 : !pdl.value -> ^bb1325, ^bb1
  ^bb1325:
    pdl_interp.are_equal %683, %2 : !pdl.value -> ^bb1326, ^bb1
  ^bb1326:
    %684 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %684 : !pdl.value -> ^bb1327, ^bb1
  ^bb1327:
    pdl_interp.is_not_null %398 : !pdl.value -> ^bb1328, ^bb1
  ^bb1328:
    %685 = pdl_interp.get_value_type of %684 : !pdl.type
    %686 = pdl_interp.get_value_type of %683 : !pdl.type
    pdl_interp.are_equal %685, %686 : !pdl.type -> ^bb1329, ^bb1
  ^bb1329:
    %687 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %685, %687 : !pdl.type -> ^bb1330, ^bb1
  ^bb1330:
    pdl_interp.check_type %685 is f32 -> ^bb1331, ^bb1
  ^bb1331:
    pdl_interp.check_operation_name of %399 is "math.sin" -> ^bb1332, ^bb1
  ^bb1332:
    pdl_interp.check_operand_count of %399 is 1 -> ^bb1333, ^bb1
  ^bb1333:
    pdl_interp.check_result_count of %399 is 1 -> ^bb1334, ^bb1
  ^bb1334:
    %688 = pdl_interp.get_result 0 of %399
    pdl_interp.is_not_null %688 : !pdl.value -> ^bb1335, ^bb1
  ^bb1335:
    pdl_interp.are_equal %688, %398 : !pdl.value -> ^bb1336, ^bb1
  ^bb1336:
    %689 = pdl_interp.get_operand 0 of %399
    pdl_interp.is_not_null %689 : !pdl.value -> ^bb1337, ^bb1
  ^bb1337:
    %690 = pdl_interp.get_value_type of %688 : !pdl.type
    pdl_interp.are_equal %685, %690 : !pdl.type -> ^bb1338, ^bb1
  ^bb1338:
    %691 = pdl_interp.get_value_type of %689 : !pdl.type
    pdl_interp.are_equal %685, %691 : !pdl.type -> ^bb1339, ^bb1
  ^bb1339:
    pdl_interp.record_match @rewriters::@diff_sin(%684, %689, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb1
  ^bb799:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb1340, ^bb1
  ^bb1340:
    pdl_interp.check_result_count of %3 is 1 -> ^bb1341, ^bb1
  ^bb1341:
    %692 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %692 : !pdl.value -> ^bb1342, ^bb1
  ^bb1342:
    pdl_interp.are_equal %692, %2 : !pdl.value -> ^bb1343, ^bb1
  ^bb1343:
    %693 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %693 : !pdl.value -> ^bb1344, ^bb1
  ^bb1344:
    pdl_interp.is_not_null %398 : !pdl.value -> ^bb1345, ^bb1
  ^bb1345:
    %694 = pdl_interp.get_value_type of %693 : !pdl.type
    %695 = pdl_interp.get_value_type of %692 : !pdl.type
    pdl_interp.are_equal %694, %695 : !pdl.type -> ^bb1346, ^bb1
  ^bb1346:
    %696 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %694, %696 : !pdl.type -> ^bb1347, ^bb1
  ^bb1347:
    pdl_interp.check_type %694 is f32 -> ^bb1348, ^bb1
  ^bb1348:
    pdl_interp.check_operation_name of %399 is "math.cos" -> ^bb1349, ^bb1
  ^bb1349:
    pdl_interp.check_operand_count of %399 is 1 -> ^bb1350, ^bb1
  ^bb1350:
    pdl_interp.check_result_count of %399 is 1 -> ^bb1351, ^bb1
  ^bb1351:
    %697 = pdl_interp.get_result 0 of %399
    pdl_interp.is_not_null %697 : !pdl.value -> ^bb1352, ^bb1
  ^bb1352:
    pdl_interp.are_equal %697, %398 : !pdl.value -> ^bb1353, ^bb1
  ^bb1353:
    %698 = pdl_interp.get_operand 0 of %399
    pdl_interp.is_not_null %698 : !pdl.value -> ^bb1354, ^bb1
  ^bb1354:
    %699 = pdl_interp.get_value_type of %697 : !pdl.type
    pdl_interp.are_equal %694, %699 : !pdl.type -> ^bb1355, ^bb1
  ^bb1355:
    %700 = pdl_interp.get_value_type of %698 : !pdl.type
    pdl_interp.are_equal %694, %700 : !pdl.type -> ^bb1356, ^bb1
  ^bb1356:
    pdl_interp.record_match @rewriters::@diff_cos(%693, %698, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb1
  ^bb800:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb1357, ^bb1
  ^bb1357:
    pdl_interp.check_result_count of %3 is 1 -> ^bb1358, ^bb1
  ^bb1358:
    %701 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %701 : !pdl.value -> ^bb1359, ^bb1
  ^bb1359:
    pdl_interp.are_equal %701, %2 : !pdl.value -> ^bb1360, ^bb1
  ^bb1360:
    %702 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %702 : !pdl.value -> ^bb1361, ^bb1
  ^bb1361:
    pdl_interp.is_not_null %398 : !pdl.value -> ^bb1362, ^bb1
  ^bb1362:
    %703 = pdl_interp.get_value_type of %702 : !pdl.type
    %704 = pdl_interp.get_value_type of %701 : !pdl.type
    pdl_interp.are_equal %703, %704 : !pdl.type -> ^bb1363, ^bb1
  ^bb1363:
    %705 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %703, %705 : !pdl.type -> ^bb1364, ^bb1
  ^bb1364:
    pdl_interp.check_type %703 is f32 -> ^bb1365, ^bb1
  ^bb1365:
    pdl_interp.check_operation_name of %399 is "math.atan" -> ^bb1366, ^bb1
  ^bb1366:
    pdl_interp.check_operand_count of %399 is 1 -> ^bb1367, ^bb1
  ^bb1367:
    pdl_interp.check_result_count of %399 is 1 -> ^bb1368, ^bb1
  ^bb1368:
    %706 = pdl_interp.get_result 0 of %399
    pdl_interp.is_not_null %706 : !pdl.value -> ^bb1369, ^bb1
  ^bb1369:
    pdl_interp.are_equal %706, %398 : !pdl.value -> ^bb1370, ^bb1
  ^bb1370:
    %707 = pdl_interp.get_operand 0 of %399
    pdl_interp.is_not_null %707 : !pdl.value -> ^bb1371, ^bb1
  ^bb1371:
    %708 = pdl_interp.get_value_type of %706 : !pdl.type
    pdl_interp.are_equal %703, %708 : !pdl.type -> ^bb1372, ^bb1
  ^bb1372:
    %709 = pdl_interp.get_value_type of %707 : !pdl.type
    pdl_interp.are_equal %703, %709 : !pdl.type -> ^bb1373, ^bb1
  ^bb1373:
    pdl_interp.record_match @rewriters::@diff_atan(%702, %707, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb1
  ^bb801:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb1374, ^bb1
  ^bb1374:
    pdl_interp.check_result_count of %3 is 1 -> ^bb1375, ^bb1
  ^bb1375:
    %710 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %710 : !pdl.value -> ^bb1376, ^bb1
  ^bb1376:
    pdl_interp.are_equal %710, %2 : !pdl.value -> ^bb1377, ^bb1
  ^bb1377:
    %711 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %711 : !pdl.value -> ^bb1378, ^bb1
  ^bb1378:
    pdl_interp.is_not_null %398 : !pdl.value -> ^bb1379, ^bb1
  ^bb1379:
    %712 = pdl_interp.get_value_type of %711 : !pdl.type
    %713 = pdl_interp.get_value_type of %710 : !pdl.type
    pdl_interp.are_equal %712, %713 : !pdl.type -> ^bb1380, ^bb1
  ^bb1380:
    %714 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %712, %714 : !pdl.type -> ^bb1381, ^bb1
  ^bb1381:
    pdl_interp.check_type %712 is f32 -> ^bb1382, ^bb1
  ^bb1382:
    pdl_interp.switch_operation_name of %399 to ["math.sinh", "math.cosh"](^bb1383, ^bb1384) -> ^bb1
  ^bb1383:
    pdl_interp.check_operand_count of %399 is 1 -> ^bb1385, ^bb1
  ^bb1385:
    pdl_interp.check_result_count of %399 is 1 -> ^bb1386, ^bb1
  ^bb1386:
    %715 = pdl_interp.get_result 0 of %399
    pdl_interp.is_not_null %715 : !pdl.value -> ^bb1387, ^bb1
  ^bb1387:
    pdl_interp.are_equal %715, %398 : !pdl.value -> ^bb1388, ^bb1
  ^bb1388:
    %716 = pdl_interp.get_value_type of %715 : !pdl.type
    pdl_interp.are_equal %712, %716 : !pdl.type -> ^bb1389, ^bb1
  ^bb1389:
    %717 = pdl_interp.get_operand 0 of %399
    pdl_interp.are_equal %711, %717 : !pdl.value -> ^bb1390, ^bb1
  ^bb1390:
    pdl_interp.record_match @rewriters::@sinhsub__cosh(%711, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb1
  ^bb1384:
    pdl_interp.check_operand_count of %399 is 1 -> ^bb1391, ^bb1
  ^bb1391:
    pdl_interp.check_result_count of %399 is 1 -> ^bb1392, ^bb1
  ^bb1392:
    %718 = pdl_interp.get_result 0 of %399
    pdl_interp.is_not_null %718 : !pdl.value -> ^bb1393, ^bb1
  ^bb1393:
    pdl_interp.are_equal %718, %398 : !pdl.value -> ^bb1394, ^bb1
  ^bb1394:
    %719 = pdl_interp.get_operand 0 of %399
    pdl_interp.is_not_null %719 : !pdl.value -> ^bb1395, ^bb1
  ^bb1395:
    %720 = pdl_interp.get_value_type of %718 : !pdl.type
    pdl_interp.are_equal %712, %720 : !pdl.type -> ^bb1396, ^bb1
  ^bb1396:
    %721 = pdl_interp.get_value_type of %719 : !pdl.type
    pdl_interp.are_equal %712, %721 : !pdl.type -> ^bb1397, ^bb1
  ^bb1397:
    pdl_interp.record_match @rewriters::@diff_cosh(%711, %719, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb1
  ^bb802:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb1398, ^bb1
  ^bb1398:
    pdl_interp.check_result_count of %3 is 1 -> ^bb1399, ^bb1
  ^bb1399:
    %722 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %722 : !pdl.value -> ^bb1400, ^bb1
  ^bb1400:
    pdl_interp.are_equal %722, %2 : !pdl.value -> ^bb1401, ^bb1
  ^bb1401:
    %723 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %723 : !pdl.value -> ^bb1402, ^bb1
  ^bb1402:
    pdl_interp.is_not_null %398 : !pdl.value -> ^bb1403, ^bb1
  ^bb1403:
    %724 = pdl_interp.get_value_type of %723 : !pdl.type
    %725 = pdl_interp.get_value_type of %722 : !pdl.type
    pdl_interp.are_equal %724, %725 : !pdl.type -> ^bb1404, ^bb1
  ^bb1404:
    %726 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %724, %726 : !pdl.type -> ^bb1405, ^bb1
  ^bb1405:
    pdl_interp.check_type %724 is f32 -> ^bb1406, ^bb1
  ^bb1406:
    pdl_interp.check_operation_name of %399 is "math.exp" -> ^bb1407, ^bb1
  ^bb1407:
    pdl_interp.check_operand_count of %399 is 1 -> ^bb1408, ^bb1
  ^bb1408:
    pdl_interp.check_result_count of %399 is 1 -> ^bb1409, ^bb1
  ^bb1409:
    %727 = pdl_interp.get_result 0 of %399
    pdl_interp.is_not_null %727 : !pdl.value -> ^bb1410, ^bb1
  ^bb1410:
    pdl_interp.are_equal %727, %398 : !pdl.value -> ^bb1411, ^bb1
  ^bb1411:
    %728 = pdl_interp.get_operand 0 of %399
    pdl_interp.is_not_null %728 : !pdl.value -> ^bb1412, ^bb1
  ^bb1412:
    %729 = pdl_interp.get_defining_op of %728 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %729 : !pdl.operation -> ^bb1413, ^bb1
  ^bb1413:
    %730 = pdl_interp.get_value_type of %727 : !pdl.type
    pdl_interp.are_equal %724, %730 : !pdl.type -> ^bb1414, ^bb1
  ^bb1414:
    pdl_interp.check_operation_name of %729 is "arith.negf" -> ^bb1415, ^bb1
  ^bb1415:
    pdl_interp.check_operand_count of %729 is 1 -> ^bb1416, ^bb1
  ^bb1416:
    pdl_interp.check_result_count of %729 is 1 -> ^bb1417, ^bb1
  ^bb1417:
    %731 = pdl_interp.get_result 0 of %729
    pdl_interp.is_not_null %731 : !pdl.value -> ^bb1418, ^bb1
  ^bb1418:
    pdl_interp.are_equal %731, %728 : !pdl.value -> ^bb1419, ^bb1
  ^bb1419:
    %732 = pdl_interp.get_value_type of %731 : !pdl.type
    pdl_interp.are_equal %732, %724 : !pdl.type -> ^bb1420, ^bb1
  ^bb1420:
    %733 = pdl_interp.get_operand 0 of %729
    pdl_interp.are_equal %733, %723 : !pdl.value -> ^bb1421, ^bb1
  ^bb1421:
    pdl_interp.record_match @rewriters::@sinh_undef(%723, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb1
  ^bb803:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb1422, ^bb1
  ^bb1422:
    pdl_interp.check_result_count of %3 is 1 -> ^bb1423, ^bb1
  ^bb1423:
    %734 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %734 : !pdl.value -> ^bb1424, ^bb1
  ^bb1424:
    pdl_interp.are_equal %734, %2 : !pdl.value -> ^bb1425, ^bb1
  ^bb1425:
    %735 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %735 : !pdl.value -> ^bb1426, ^bb1
  ^bb1426:
    pdl_interp.is_not_null %398 : !pdl.value -> ^bb1427, ^bb1
  ^bb1427:
    %736 = pdl_interp.get_value_type of %735 : !pdl.type
    %737 = pdl_interp.get_value_type of %734 : !pdl.type
    pdl_interp.are_equal %736, %737 : !pdl.type -> ^bb1428, ^bb1
  ^bb1428:
    %738 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %736, %738 : !pdl.type -> ^bb1429, ^bb1
  ^bb1429:
    pdl_interp.check_type %736 is f32 -> ^bb1430, ^bb1
  ^bb1430:
    pdl_interp.check_operation_name of %399 is "math.sinh" -> ^bb1431, ^bb1
  ^bb1431:
    pdl_interp.check_operand_count of %399 is 1 -> ^bb1432, ^bb1
  ^bb1432:
    pdl_interp.check_result_count of %399 is 1 -> ^bb1433, ^bb1
  ^bb1433:
    %739 = pdl_interp.get_result 0 of %399
    pdl_interp.is_not_null %739 : !pdl.value -> ^bb1434, ^bb1
  ^bb1434:
    pdl_interp.are_equal %739, %398 : !pdl.value -> ^bb1435, ^bb1
  ^bb1435:
    %740 = pdl_interp.get_operand 0 of %399
    pdl_interp.is_not_null %740 : !pdl.value -> ^bb1436, ^bb1
  ^bb1436:
    %741 = pdl_interp.get_value_type of %739 : !pdl.type
    pdl_interp.are_equal %736, %741 : !pdl.type -> ^bb1437, ^bb1
  ^bb1437:
    %742 = pdl_interp.get_value_type of %740 : !pdl.type
    pdl_interp.are_equal %736, %742 : !pdl.type -> ^bb1438, ^bb1
  ^bb1438:
    pdl_interp.record_match @rewriters::@diff_sinh(%735, %740, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb1
  ^bb789:
    pdl_interp.switch_operation_name of %3 to ["arith.addf", "arith.subf", "arith.constant"](^bb1439, ^bb1440, ^bb1441) -> ^bb790
  ^bb1439:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb1442, ^bb790
  ^bb1442:
    pdl_interp.check_result_count of %3 is 1 -> ^bb1443, ^bb790
  ^bb1443:
    %743 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %743 : !pdl.value -> ^bb1444, ^bb790
  ^bb1444:
    pdl_interp.are_equal %743, %2 : !pdl.value -> ^bb1445, ^bb790
  ^bb1445:
    %744 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %744 : !pdl.value -> ^bb1446, ^bb790
  ^bb1446:
    %745 = pdl_interp.get_operand 1 of %0
    pdl_interp.is_not_null %745 : !pdl.value -> ^bb1447, ^bb790
  ^bb1447:
    %746 = pdl_interp.get_value_type of %744 : !pdl.type
    %747 = pdl_interp.get_value_type of %743 : !pdl.type
    pdl_interp.are_equal %746, %747 : !pdl.type -> ^bb1448, ^bb790
  ^bb1448:
    %748 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %746, %748 : !pdl.type -> ^bb1449, ^bb790
  ^bb1449:
    pdl_interp.check_type %746 is f32 -> ^bb1450, ^bb790
  ^bb1450:
    %749 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %749 : !pdl.value -> ^bb1451, ^bb790
  ^bb1451:
    %750 = pdl_interp.get_value_type of %749 : !pdl.type
    pdl_interp.are_equal %746, %750 : !pdl.type -> ^bb1452, ^bb790
  ^bb1452:
    %751 = pdl_interp.get_value_type of %745 : !pdl.type
    pdl_interp.are_equal %746, %751 : !pdl.type -> ^bb1453, ^bb790
  ^bb1453:
    pdl_interp.record_match @rewriters::@associatesub_ladd(%749, %745, %744, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb790
  ^bb1440:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb1454, ^bb790
  ^bb1454:
    pdl_interp.check_result_count of %3 is 1 -> ^bb1455, ^bb790
  ^bb1455:
    %752 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %752 : !pdl.value -> ^bb1456, ^bb790
  ^bb1456:
    pdl_interp.are_equal %752, %2 : !pdl.value -> ^bb1457, ^bb790
  ^bb1457:
    %753 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %753 : !pdl.value -> ^bb1458, ^bb790
  ^bb1458:
    %754 = pdl_interp.get_operand 1 of %0
    pdl_interp.is_not_null %754 : !pdl.value -> ^bb1459, ^bb790
  ^bb1459:
    %755 = pdl_interp.get_value_type of %753 : !pdl.type
    %756 = pdl_interp.get_value_type of %752 : !pdl.type
    pdl_interp.are_equal %755, %756 : !pdl.type -> ^bb1460, ^bb790
  ^bb1460:
    %757 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %755, %757 : !pdl.type -> ^bb1461, ^bb790
  ^bb1461:
    pdl_interp.check_type %755 is f32 -> ^bb1462, ^bb790
  ^bb1462:
    %758 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %758 : !pdl.value -> ^bb1463, ^bb790
  ^bb1463:
    %759 = pdl_interp.get_value_type of %758 : !pdl.type
    pdl_interp.are_equal %755, %759 : !pdl.type -> ^bb1464, ^bb790
  ^bb1464:
    %760 = pdl_interp.get_value_type of %754 : !pdl.type
    pdl_interp.are_equal %755, %760 : !pdl.type -> ^bb1465, ^bb790
  ^bb1465:
    pdl_interp.record_match @rewriters::@associatesub_l_(%758, %754, %753, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb790
  ^bb1441:
    pdl_interp.check_operand_count of %3 is 0 -> ^bb1466, ^bb790
  ^bb1466:
    pdl_interp.check_result_count of %3 is 1 -> ^bb1467, ^bb790
  ^bb1467:
    %761 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %761 : !pdl.value -> ^bb1468, ^bb790
  ^bb1468:
    pdl_interp.are_equal %761, %2 : !pdl.value -> ^bb1469, ^bb790
  ^bb1469:
    %762 = pdl_interp.get_operand 1 of %0
    pdl_interp.is_not_null %762 : !pdl.value -> ^bb1470, ^bb790
  ^bb1470:
    %763 = pdl_interp.get_attribute "value" of %3
    pdl_interp.is_not_null %763 : !pdl.attribute -> ^bb1471, ^bb790
  ^bb1471:
    pdl_interp.check_attribute %763 is 0.000000e+00 : f32 -> ^bb1472, ^bb790
  ^bb1472:
    %764 = pdl_interp.get_value_type of %761 : !pdl.type
    %765 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %764, %765 : !pdl.type -> ^bb1473, ^bb790
  ^bb1473:
    pdl_interp.check_type %764 is f32 -> ^bb1474, ^bb790
  ^bb1474:
    %766 = pdl_interp.get_value_type of %762 : !pdl.type
    pdl_interp.are_equal %764, %766 : !pdl.type -> ^bb1475, ^bb790
  ^bb1475:
    pdl_interp.record_match @rewriters::@sub0_neg(%762, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb790
  ^bb27:
    pdl_interp.check_operand_count of %0 is 2 -> ^bb1476, ^bb1
  ^bb1476:
    pdl_interp.check_result_count of %0 is 1 -> ^bb1477, ^bb1
  ^bb1477:
    pdl_interp.is_not_null %2 : !pdl.value -> ^bb1478, ^bb1479
  ^bb1479:
    %767 = pdl_interp.get_operand 1 of %0
    %768 = pdl_interp.get_defining_op of %767 : !pdl.value {position = "root.operand[1].defining_op"}
    pdl_interp.is_not_null %768 : !pdl.operation -> ^bb1480, ^bb1
  ^bb1480:
    pdl_interp.is_not_null %2 : !pdl.value -> ^bb1481, ^bb1
  ^bb1481:
    pdl_interp.switch_operation_name of %3 to ["arith.mulf", "math.powf", "arith.addf", "arith.divf", "math.sqrt", "arith.negf", "math.absf", "math.cbrt", "math.exp", "math.sin", "math.cos", "arith.constant"](^bb1482, ^bb1483, ^bb1484, ^bb1485, ^bb1486, ^bb1487, ^bb1488, ^bb1489, ^bb1490, ^bb1491, ^bb1492, ^bb1493) -> ^bb1
  ^bb1482:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb1494, ^bb1
  ^bb1494:
    pdl_interp.check_result_count of %3 is 1 -> ^bb1495, ^bb1
  ^bb1495:
    %769 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %769 : !pdl.value -> ^bb1496, ^bb1
  ^bb1496:
    pdl_interp.are_equal %769, %2 : !pdl.value -> ^bb1497, ^bb1
  ^bb1497:
    %770 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %770 : !pdl.value -> ^bb1498, ^bb1
  ^bb1498:
    pdl_interp.is_not_null %767 : !pdl.value -> ^bb1499, ^bb1
  ^bb1499:
    %771 = pdl_interp.get_value_type of %770 : !pdl.type
    %772 = pdl_interp.get_value_type of %769 : !pdl.type
    pdl_interp.are_equal %771, %772 : !pdl.type -> ^bb1500, ^bb1501
  ^bb1501:
    pdl_interp.check_operation_name of %768 is "math.cbrt" -> ^bb1502, ^bb1
  ^bb1502:
    pdl_interp.check_operand_count of %768 is 1 -> ^bb1503, ^bb1
  ^bb1503:
    pdl_interp.check_result_count of %768 is 1 -> ^bb1504, ^bb1
  ^bb1504:
    %773 = pdl_interp.get_result 0 of %768
    pdl_interp.is_not_null %773 : !pdl.value -> ^bb1505, ^bb1
  ^bb1505:
    pdl_interp.are_equal %773, %767 : !pdl.value -> ^bb1506, ^bb1
  ^bb1506:
    %774 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %774 : !pdl.value -> ^bb1507, ^bb1
  ^bb1507:
    %775 = pdl_interp.get_defining_op of %774 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %775 : !pdl.operation -> ^bb1508, ^bb1
  ^bb1508:
    %776 = pdl_interp.get_defining_op of %770 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %776 : !pdl.operation -> ^bb1509, ^bb1
  ^bb1509:
    pdl_interp.check_operation_name of %775 is "math.cbrt" -> ^bb1510, ^bb1
  ^bb1510:
    pdl_interp.check_operand_count of %775 is 1 -> ^bb1511, ^bb1
  ^bb1511:
    pdl_interp.check_result_count of %775 is 1 -> ^bb1512, ^bb1
  ^bb1512:
    %777 = pdl_interp.get_result 0 of %775
    pdl_interp.is_not_null %777 : !pdl.value -> ^bb1513, ^bb1
  ^bb1513:
    pdl_interp.are_equal %777, %774 : !pdl.value -> ^bb1514, ^bb1
  ^bb1514:
    pdl_interp.check_operation_name of %776 is "math.cbrt" -> ^bb1515, ^bb1
  ^bb1515:
    pdl_interp.check_operand_count of %776 is 1 -> ^bb1516, ^bb1
  ^bb1516:
    pdl_interp.check_result_count of %776 is 1 -> ^bb1517, ^bb1
  ^bb1517:
    %778 = pdl_interp.get_result 0 of %776
    pdl_interp.is_not_null %778 : !pdl.value -> ^bb1518, ^bb1
  ^bb1518:
    pdl_interp.are_equal %778, %770 : !pdl.value -> ^bb1519, ^bb1
  ^bb1519:
    %779 = pdl_interp.get_operand 0 of %776
    pdl_interp.is_not_null %779 : !pdl.value -> ^bb1520, ^bb1
  ^bb1520:
    %780 = pdl_interp.get_value_type of %779 : !pdl.type
    %781 = pdl_interp.get_value_type of %778 : !pdl.type
    pdl_interp.are_equal %780, %781 : !pdl.type -> ^bb1521, ^bb1
  ^bb1521:
    %782 = pdl_interp.get_value_type of %769 : !pdl.type
    pdl_interp.are_equal %780, %782 : !pdl.type -> ^bb1522, ^bb1
  ^bb1522:
    %783 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %780, %783 : !pdl.type -> ^bb1523, ^bb1
  ^bb1523:
    pdl_interp.check_type %780 is f32 -> ^bb1524, ^bb1
  ^bb1524:
    %784 = pdl_interp.get_value_type of %777 : !pdl.type
    pdl_interp.are_equal %780, %784 : !pdl.type -> ^bb1525, ^bb1
  ^bb1525:
    %785 = pdl_interp.get_value_type of %773 : !pdl.type
    pdl_interp.are_equal %780, %785 : !pdl.type -> ^bb1526, ^bb1
  ^bb1526:
    %786 = pdl_interp.get_operand 0 of %775
    pdl_interp.are_equal %779, %786 : !pdl.value -> ^bb1527, ^bb1
  ^bb1527:
    %787 = pdl_interp.get_operand 0 of %768
    pdl_interp.are_equal %779, %787 : !pdl.value -> ^bb1528, ^bb1
  ^bb1528:
    pdl_interp.record_match @rewriters::@rem_3cbrt_lft(%779, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1
  ^bb1500:
    %788 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %771, %788 : !pdl.type -> ^bb1529, ^bb1501
  ^bb1529:
    pdl_interp.check_type %771 is f32 -> ^bb1530, ^bb1501
  ^bb1530:
    pdl_interp.check_operation_name of %768 is "arith.mulf" -> ^bb1531, ^bb1501
  ^bb1531:
    pdl_interp.check_operand_count of %768 is 2 -> ^bb1532, ^bb1501
  ^bb1532:
    pdl_interp.check_result_count of %768 is 1 -> ^bb1533, ^bb1501
  ^bb1533:
    %789 = pdl_interp.get_result 0 of %768
    pdl_interp.is_not_null %789 : !pdl.value -> ^bb1534, ^bb1501
  ^bb1534:
    pdl_interp.are_equal %789, %767 : !pdl.value -> ^bb1535, ^bb1501
  ^bb1535:
    %790 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %790 : !pdl.value -> ^bb1536, ^bb1537
  ^bb1537:
    %791 = pdl_interp.get_operand 0 of %768
    pdl_interp.is_not_null %791 : !pdl.value -> ^bb1538, ^bb1501
  ^bb1538:
    %792 = pdl_interp.get_value_type of %789 : !pdl.type
    pdl_interp.are_equal %771, %792 : !pdl.type -> ^bb1539, ^bb1501
  ^bb1539:
    %793 = pdl_interp.get_value_type of %791 : !pdl.type
    pdl_interp.are_equal %771, %793 : !pdl.type -> ^bb1540, ^bb1501
  ^bb1540:
    %794 = pdl_interp.get_operand 1 of %3
    pdl_interp.are_equal %770, %794 : !pdl.value -> ^bb1541, ^bb1501
  ^bb1541:
    %795 = pdl_interp.get_operand 1 of %768
    pdl_interp.are_equal %791, %795 : !pdl.value -> ^bb1542, ^bb1501
  ^bb1542:
    pdl_interp.record_match @rewriters::@unswap_sqr(%770, %791, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1501
  ^bb1536:
    %796 = pdl_interp.get_value_type of %789 : !pdl.type
    pdl_interp.are_equal %771, %796 : !pdl.type -> ^bb1543, ^bb1537
  ^bb1543:
    %797 = pdl_interp.get_value_type of %790 : !pdl.type
    pdl_interp.are_equal %771, %797 : !pdl.type -> ^bb1544, ^bb1537
  ^bb1544:
    %798 = pdl_interp.get_operand 0 of %768
    pdl_interp.are_equal %770, %798 : !pdl.value -> ^bb1545, ^bb1537
  ^bb1545:
    %799 = pdl_interp.get_operand 1 of %768
    pdl_interp.are_equal %790, %799 : !pdl.value -> ^bb1546, ^bb1537
  ^bb1546:
    pdl_interp.record_match @rewriters::@swap_sqr(%770, %790, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1537
  ^bb1483:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb1547, ^bb1
  ^bb1547:
    pdl_interp.check_result_count of %3 is 1 -> ^bb1548, ^bb1
  ^bb1548:
    %800 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %800 : !pdl.value -> ^bb1549, ^bb1
  ^bb1549:
    pdl_interp.are_equal %800, %2 : !pdl.value -> ^bb1550, ^bb1
  ^bb1550:
    %801 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %801 : !pdl.value -> ^bb1551, ^bb1
  ^bb1551:
    pdl_interp.is_not_null %767 : !pdl.value -> ^bb1552, ^bb1
  ^bb1552:
    %802 = pdl_interp.get_value_type of %801 : !pdl.type
    %803 = pdl_interp.get_value_type of %800 : !pdl.type
    pdl_interp.are_equal %802, %803 : !pdl.type -> ^bb1553, ^bb1
  ^bb1553:
    %804 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %802, %804 : !pdl.type -> ^bb1554, ^bb1
  ^bb1554:
    pdl_interp.check_type %802 is f32 -> ^bb1555, ^bb1
  ^bb1555:
    pdl_interp.check_operation_name of %768 is "math.powf" -> ^bb1556, ^bb1
  ^bb1556:
    pdl_interp.check_operand_count of %768 is 2 -> ^bb1557, ^bb1
  ^bb1557:
    pdl_interp.check_result_count of %768 is 1 -> ^bb1558, ^bb1
  ^bb1558:
    %805 = pdl_interp.get_result 0 of %768
    pdl_interp.is_not_null %805 : !pdl.value -> ^bb1559, ^bb1
  ^bb1559:
    pdl_interp.are_equal %805, %767 : !pdl.value -> ^bb1560, ^bb1
  ^bb1560:
    %806 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %806 : !pdl.value -> ^bb1561, ^bb1
  ^bb1561:
    %807 = pdl_interp.get_value_type of %805 : !pdl.type
    pdl_interp.are_equal %802, %807 : !pdl.type -> ^bb1562, ^bb1563
  ^bb1563:
    %808 = pdl_interp.get_defining_op of %806 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %808 : !pdl.operation -> ^bb1564, ^bb1565
  ^bb1565:
    %809 = pdl_interp.get_operand 0 of %768
    pdl_interp.is_not_null %809 : !pdl.value -> ^bb1566, ^bb1567
  ^bb1567:
    %810 = pdl_interp.get_operand 1 of %768
    pdl_interp.is_not_null %810 : !pdl.value -> ^bb1568, ^bb1
  ^bb1568:
    %811 = pdl_interp.get_value_type of %805 : !pdl.type
    pdl_interp.are_equal %802, %811 : !pdl.type -> ^bb1569, ^bb1
  ^bb1569:
    %812 = pdl_interp.get_value_type of %806 : !pdl.type
    pdl_interp.are_equal %802, %812 : !pdl.type -> ^bb1570, ^bb1
  ^bb1570:
    %813 = pdl_interp.get_operand 0 of %768
    pdl_interp.are_equal %801, %813 : !pdl.value -> ^bb1571, ^bb1
  ^bb1571:
    %814 = pdl_interp.get_value_type of %810 : !pdl.type
    pdl_interp.are_equal %802, %814 : !pdl.type -> ^bb1572, ^bb1
  ^bb1572:
    pdl_interp.record_match @rewriters::@pow_prod_up(%806, %810, %801, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1
  ^bb1566:
    %815 = pdl_interp.get_value_type of %805 : !pdl.type
    pdl_interp.are_equal %802, %815 : !pdl.type -> ^bb1573, ^bb1567
  ^bb1573:
    %816 = pdl_interp.get_value_type of %806 : !pdl.type
    pdl_interp.are_equal %802, %816 : !pdl.type -> ^bb1574, ^bb1567
  ^bb1574:
    %817 = pdl_interp.get_value_type of %809 : !pdl.type
    pdl_interp.are_equal %802, %817 : !pdl.type -> ^bb1575, ^bb1567
  ^bb1575:
    %818 = pdl_interp.get_operand 1 of %768
    pdl_interp.are_equal %806, %818 : !pdl.value -> ^bb1576, ^bb1567
  ^bb1576:
    pdl_interp.record_match @rewriters::@pow_prod_down(%801, %809, %806, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1567
  ^bb1564:
    %819 = pdl_interp.get_operand 0 of %768
    pdl_interp.is_not_null %819 : !pdl.value -> ^bb1577, ^bb1565
  ^bb1577:
    %820 = pdl_interp.get_operand 1 of %768
    %821 = pdl_interp.get_defining_op of %820 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %821 : !pdl.operation -> ^bb1578, ^bb1565
  ^bb1578:
    pdl_interp.is_not_null %820 : !pdl.value -> ^bb1579, ^bb1565
  ^bb1579:
    %822 = pdl_interp.get_value_type of %805 : !pdl.type
    pdl_interp.are_equal %802, %822 : !pdl.type -> ^bb1580, ^bb1565
  ^bb1580:
    pdl_interp.check_operation_name of %808 is "arith.constant" -> ^bb1581, ^bb1565
  ^bb1581:
    pdl_interp.check_operand_count of %808 is 0 -> ^bb1582, ^bb1565
  ^bb1582:
    pdl_interp.check_result_count of %808 is 1 -> ^bb1583, ^bb1565
  ^bb1583:
    %823 = pdl_interp.get_result 0 of %808
    pdl_interp.is_not_null %823 : !pdl.value -> ^bb1584, ^bb1565
  ^bb1584:
    pdl_interp.are_equal %823, %806 : !pdl.value -> ^bb1585, ^bb1565
  ^bb1585:
    pdl_interp.check_operation_name of %821 is "arith.constant" -> ^bb1586, ^bb1565
  ^bb1586:
    pdl_interp.check_operand_count of %821 is 0 -> ^bb1587, ^bb1565
  ^bb1587:
    pdl_interp.check_result_count of %821 is 1 -> ^bb1588, ^bb1565
  ^bb1588:
    %824 = pdl_interp.get_result 0 of %821
    pdl_interp.is_not_null %824 : !pdl.value -> ^bb1589, ^bb1565
  ^bb1589:
    pdl_interp.are_equal %824, %820 : !pdl.value -> ^bb1590, ^bb1565
  ^bb1590:
    %825 = pdl_interp.get_value_type of %819 : !pdl.type
    pdl_interp.are_equal %802, %825 : !pdl.type -> ^bb1591, ^bb1565
  ^bb1591:
    %826 = pdl_interp.get_attribute "value" of %808
    pdl_interp.is_not_null %826 : !pdl.attribute -> ^bb1592, ^bb1565
  ^bb1592:
    pdl_interp.check_attribute %826 is 3.000000e+00 : f32 -> ^bb1593, ^bb1565
  ^bb1593:
    %827 = pdl_interp.get_value_type of %823 : !pdl.type
    pdl_interp.are_equal %827, %802 : !pdl.type -> ^bb1594, ^bb1565
  ^bb1594:
    %828 = pdl_interp.get_value_type of %824 : !pdl.type
    pdl_interp.are_equal %828, %802 : !pdl.type -> ^bb1595, ^bb1565
  ^bb1595:
    %829 = pdl_interp.get_attribute "value" of %821
    pdl_interp.is_not_null %829 : !pdl.attribute -> ^bb1596, ^bb1565
  ^bb1596:
    pdl_interp.check_attribute %829 is 3.000000e+00 : f32 -> ^bb1597, ^bb1565
  ^bb1597:
    pdl_interp.record_match @rewriters::@cube_prod_rev(%801, %819, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1565
  ^bb1562:
    %830 = pdl_interp.get_value_type of %806 : !pdl.type
    pdl_interp.are_equal %802, %830 : !pdl.type -> ^bb1598, ^bb1563
  ^bb1598:
    %831 = pdl_interp.get_operand 0 of %768
    pdl_interp.are_equal %801, %831 : !pdl.value -> ^bb1599, ^bb1563
  ^bb1599:
    %832 = pdl_interp.get_operand 1 of %768
    pdl_interp.are_equal %806, %832 : !pdl.value -> ^bb1600, ^bb1563
  ^bb1600:
    pdl_interp.record_match @rewriters::@pow_sqr(%806, %801, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1563
  ^bb1484:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb1601, ^bb1
  ^bb1601:
    pdl_interp.check_result_count of %3 is 1 -> ^bb1602, ^bb1
  ^bb1602:
    %833 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %833 : !pdl.value -> ^bb1603, ^bb1
  ^bb1603:
    pdl_interp.are_equal %833, %2 : !pdl.value -> ^bb1604, ^bb1
  ^bb1604:
    %834 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %834 : !pdl.value -> ^bb1605, ^bb1
  ^bb1605:
    pdl_interp.is_not_null %767 : !pdl.value -> ^bb1606, ^bb1
  ^bb1606:
    %835 = pdl_interp.get_value_type of %834 : !pdl.type
    %836 = pdl_interp.get_value_type of %833 : !pdl.type
    pdl_interp.are_equal %835, %836 : !pdl.type -> ^bb1607, ^bb1608
  ^bb1608:
    pdl_interp.switch_operation_name of %768 to ["arith.subf", "arith.addf"](^bb1609, ^bb1610) -> ^bb1
  ^bb1609:
    pdl_interp.check_operand_count of %768 is 2 -> ^bb1611, ^bb1
  ^bb1611:
    pdl_interp.check_result_count of %768 is 1 -> ^bb1612, ^bb1
  ^bb1612:
    %837 = pdl_interp.get_result 0 of %768
    pdl_interp.is_not_null %837 : !pdl.value -> ^bb1613, ^bb1
  ^bb1613:
    pdl_interp.are_equal %837, %767 : !pdl.value -> ^bb1614, ^bb1
  ^bb1614:
    %838 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %838 : !pdl.value -> ^bb1615, ^bb1
  ^bb1615:
    %839 = pdl_interp.get_defining_op of %838 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %839 : !pdl.operation -> ^bb1616, ^bb1
  ^bb1616:
    %840 = pdl_interp.get_defining_op of %834 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %840 : !pdl.operation -> ^bb1617, ^bb1
  ^bb1617:
    pdl_interp.check_operation_name of %839 is "arith.addf" -> ^bb1618, ^bb1
  ^bb1618:
    pdl_interp.check_operand_count of %839 is 2 -> ^bb1619, ^bb1
  ^bb1619:
    pdl_interp.check_result_count of %839 is 1 -> ^bb1620, ^bb1
  ^bb1620:
    %841 = pdl_interp.get_result 0 of %839
    pdl_interp.is_not_null %841 : !pdl.value -> ^bb1621, ^bb1
  ^bb1621:
    pdl_interp.are_equal %841, %838 : !pdl.value -> ^bb1622, ^bb1
  ^bb1622:
    pdl_interp.check_operation_name of %840 is "arith.mulf" -> ^bb1623, ^bb1
  ^bb1623:
    pdl_interp.check_operand_count of %840 is 2 -> ^bb1624, ^bb1
  ^bb1624:
    pdl_interp.check_result_count of %840 is 1 -> ^bb1625, ^bb1
  ^bb1625:
    %842 = pdl_interp.get_result 0 of %840
    pdl_interp.is_not_null %842 : !pdl.value -> ^bb1626, ^bb1
  ^bb1626:
    pdl_interp.are_equal %842, %834 : !pdl.value -> ^bb1627, ^bb1
  ^bb1627:
    %843 = pdl_interp.get_operand 0 of %840
    pdl_interp.is_not_null %843 : !pdl.value -> ^bb1628, ^bb1
  ^bb1628:
    %844 = pdl_interp.get_value_type of %843 : !pdl.type
    %845 = pdl_interp.get_value_type of %842 : !pdl.type
    pdl_interp.are_equal %844, %845 : !pdl.type -> ^bb1629, ^bb1
  ^bb1629:
    %846 = pdl_interp.get_value_type of %833 : !pdl.type
    pdl_interp.are_equal %844, %846 : !pdl.type -> ^bb1630, ^bb1
  ^bb1630:
    %847 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %844, %847 : !pdl.type -> ^bb1631, ^bb1
  ^bb1631:
    pdl_interp.check_type %844 is f32 -> ^bb1632, ^bb1
  ^bb1632:
    %848 = pdl_interp.get_operand 0 of %839
    pdl_interp.is_not_null %848 : !pdl.value -> ^bb1633, ^bb1
  ^bb1633:
    %849 = pdl_interp.get_value_type of %841 : !pdl.type
    pdl_interp.are_equal %844, %849 : !pdl.type -> ^bb1634, ^bb1
  ^bb1634:
    %850 = pdl_interp.get_value_type of %837 : !pdl.type
    pdl_interp.are_equal %844, %850 : !pdl.type -> ^bb1635, ^bb1
  ^bb1635:
    %851 = pdl_interp.get_defining_op of %848 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %851 : !pdl.operation -> ^bb1636, ^bb1
  ^bb1636:
    pdl_interp.check_operation_name of %851 is "arith.mulf" -> ^bb1637, ^bb1
  ^bb1637:
    pdl_interp.check_operand_count of %851 is 2 -> ^bb1638, ^bb1
  ^bb1638:
    pdl_interp.check_result_count of %851 is 1 -> ^bb1639, ^bb1
  ^bb1639:
    %852 = pdl_interp.get_result 0 of %851
    pdl_interp.is_not_null %852 : !pdl.value -> ^bb1640, ^bb1
  ^bb1640:
    pdl_interp.are_equal %852, %848 : !pdl.value -> ^bb1641, ^bb1
  ^bb1641:
    %853 = pdl_interp.get_operand 1 of %839
    %854 = pdl_interp.get_defining_op of %853 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %854 : !pdl.operation -> ^bb1642, ^bb1
  ^bb1642:
    %855 = pdl_interp.get_value_type of %852 : !pdl.type
    pdl_interp.are_equal %855, %844 : !pdl.type -> ^bb1643, ^bb1
  ^bb1643:
    pdl_interp.is_not_null %853 : !pdl.value -> ^bb1644, ^bb1
  ^bb1644:
    %856 = pdl_interp.get_operand 0 of %851
    pdl_interp.is_not_null %856 : !pdl.value -> ^bb1645, ^bb1
  ^bb1645:
    pdl_interp.check_operation_name of %854 is "arith.mulf" -> ^bb1646, ^bb1
  ^bb1646:
    pdl_interp.check_operand_count of %854 is 2 -> ^bb1647, ^bb1
  ^bb1647:
    pdl_interp.check_result_count of %854 is 1 -> ^bb1648, ^bb1
  ^bb1648:
    %857 = pdl_interp.get_result 0 of %854
    pdl_interp.is_not_null %857 : !pdl.value -> ^bb1649, ^bb1
  ^bb1649:
    pdl_interp.are_equal %857, %853 : !pdl.value -> ^bb1650, ^bb1
  ^bb1650:
    %858 = pdl_interp.get_operand 0 of %768
    pdl_interp.are_equal %843, %858 : !pdl.value -> ^bb1651, ^bb1
  ^bb1651:
    %859 = pdl_interp.get_operand 0 of %854
    pdl_interp.are_equal %859, %843 : !pdl.value -> ^bb1652, ^bb1
  ^bb1652:
    %860 = pdl_interp.get_value_type of %857 : !pdl.type
    pdl_interp.are_equal %860, %844 : !pdl.type -> ^bb1653, ^bb1
  ^bb1653:
    %861 = pdl_interp.get_operand 1 of %840
    pdl_interp.are_equal %843, %861 : !pdl.value -> ^bb1654, ^bb1
  ^bb1654:
    %862 = pdl_interp.get_operand 1 of %851
    pdl_interp.are_equal %856, %862 : !pdl.value -> ^bb1655, ^bb1
  ^bb1655:
    %863 = pdl_interp.get_operand 1 of %854
    pdl_interp.are_equal %856, %863 : !pdl.value -> ^bb1656, ^bb1
  ^bb1656:
    %864 = pdl_interp.get_operand 1 of %768
    pdl_interp.are_equal %856, %864 : !pdl.value -> ^bb1657, ^bb1
  ^bb1657:
    %865 = pdl_interp.get_value_type of %856 : !pdl.type
    pdl_interp.are_equal %865, %844 : !pdl.type -> ^bb1658, ^bb1
  ^bb1658:
    pdl_interp.record_match @rewriters::@difference_cubes_rev(%843, %856, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1
  ^bb1610:
    pdl_interp.check_operand_count of %768 is 2 -> ^bb1659, ^bb1
  ^bb1659:
    pdl_interp.check_result_count of %768 is 1 -> ^bb1660, ^bb1
  ^bb1660:
    %866 = pdl_interp.get_result 0 of %768
    pdl_interp.is_not_null %866 : !pdl.value -> ^bb1661, ^bb1
  ^bb1661:
    pdl_interp.are_equal %866, %767 : !pdl.value -> ^bb1662, ^bb1
  ^bb1662:
    %867 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %867 : !pdl.value -> ^bb1663, ^bb1
  ^bb1663:
    %868 = pdl_interp.get_defining_op of %867 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %868 : !pdl.operation -> ^bb1664, ^bb1
  ^bb1664:
    %869 = pdl_interp.get_defining_op of %834 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %869 : !pdl.operation -> ^bb1665, ^bb1
  ^bb1665:
    pdl_interp.check_operation_name of %868 is "arith.subf" -> ^bb1666, ^bb1
  ^bb1666:
    pdl_interp.check_operand_count of %868 is 2 -> ^bb1667, ^bb1
  ^bb1667:
    pdl_interp.check_result_count of %868 is 1 -> ^bb1668, ^bb1
  ^bb1668:
    %870 = pdl_interp.get_result 0 of %868
    pdl_interp.is_not_null %870 : !pdl.value -> ^bb1669, ^bb1
  ^bb1669:
    pdl_interp.are_equal %870, %867 : !pdl.value -> ^bb1670, ^bb1
  ^bb1670:
    pdl_interp.check_operation_name of %869 is "arith.mulf" -> ^bb1671, ^bb1
  ^bb1671:
    pdl_interp.check_operand_count of %869 is 2 -> ^bb1672, ^bb1
  ^bb1672:
    pdl_interp.check_result_count of %869 is 1 -> ^bb1673, ^bb1
  ^bb1673:
    %871 = pdl_interp.get_result 0 of %869
    pdl_interp.is_not_null %871 : !pdl.value -> ^bb1674, ^bb1
  ^bb1674:
    pdl_interp.are_equal %871, %834 : !pdl.value -> ^bb1675, ^bb1
  ^bb1675:
    %872 = pdl_interp.get_operand 0 of %869
    pdl_interp.is_not_null %872 : !pdl.value -> ^bb1676, ^bb1
  ^bb1676:
    %873 = pdl_interp.get_value_type of %872 : !pdl.type
    %874 = pdl_interp.get_value_type of %871 : !pdl.type
    pdl_interp.are_equal %873, %874 : !pdl.type -> ^bb1677, ^bb1
  ^bb1677:
    %875 = pdl_interp.get_value_type of %833 : !pdl.type
    pdl_interp.are_equal %873, %875 : !pdl.type -> ^bb1678, ^bb1
  ^bb1678:
    %876 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %873, %876 : !pdl.type -> ^bb1679, ^bb1
  ^bb1679:
    pdl_interp.check_type %873 is f32 -> ^bb1680, ^bb1
  ^bb1680:
    %877 = pdl_interp.get_operand 0 of %868
    pdl_interp.is_not_null %877 : !pdl.value -> ^bb1681, ^bb1
  ^bb1681:
    %878 = pdl_interp.get_value_type of %870 : !pdl.type
    pdl_interp.are_equal %873, %878 : !pdl.type -> ^bb1682, ^bb1
  ^bb1682:
    %879 = pdl_interp.get_value_type of %866 : !pdl.type
    pdl_interp.are_equal %873, %879 : !pdl.type -> ^bb1683, ^bb1
  ^bb1683:
    %880 = pdl_interp.get_defining_op of %877 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %880 : !pdl.operation -> ^bb1684, ^bb1
  ^bb1684:
    pdl_interp.check_operation_name of %880 is "arith.mulf" -> ^bb1685, ^bb1
  ^bb1685:
    pdl_interp.check_operand_count of %880 is 2 -> ^bb1686, ^bb1
  ^bb1686:
    pdl_interp.check_result_count of %880 is 1 -> ^bb1687, ^bb1
  ^bb1687:
    %881 = pdl_interp.get_result 0 of %880
    pdl_interp.is_not_null %881 : !pdl.value -> ^bb1688, ^bb1
  ^bb1688:
    pdl_interp.are_equal %881, %877 : !pdl.value -> ^bb1689, ^bb1
  ^bb1689:
    %882 = pdl_interp.get_operand 1 of %868
    %883 = pdl_interp.get_defining_op of %882 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %883 : !pdl.operation -> ^bb1690, ^bb1
  ^bb1690:
    %884 = pdl_interp.get_value_type of %881 : !pdl.type
    pdl_interp.are_equal %884, %873 : !pdl.type -> ^bb1691, ^bb1
  ^bb1691:
    pdl_interp.is_not_null %882 : !pdl.value -> ^bb1692, ^bb1
  ^bb1692:
    %885 = pdl_interp.get_operand 0 of %880
    pdl_interp.is_not_null %885 : !pdl.value -> ^bb1693, ^bb1
  ^bb1693:
    pdl_interp.check_operation_name of %883 is "arith.mulf" -> ^bb1694, ^bb1
  ^bb1694:
    pdl_interp.check_operand_count of %883 is 2 -> ^bb1695, ^bb1
  ^bb1695:
    pdl_interp.check_result_count of %883 is 1 -> ^bb1696, ^bb1
  ^bb1696:
    %886 = pdl_interp.get_result 0 of %883
    pdl_interp.is_not_null %886 : !pdl.value -> ^bb1697, ^bb1
  ^bb1697:
    pdl_interp.are_equal %886, %882 : !pdl.value -> ^bb1698, ^bb1
  ^bb1698:
    %887 = pdl_interp.get_operand 0 of %768
    pdl_interp.are_equal %872, %887 : !pdl.value -> ^bb1699, ^bb1
  ^bb1699:
    %888 = pdl_interp.get_operand 0 of %883
    pdl_interp.are_equal %888, %872 : !pdl.value -> ^bb1700, ^bb1
  ^bb1700:
    %889 = pdl_interp.get_value_type of %886 : !pdl.type
    pdl_interp.are_equal %889, %873 : !pdl.type -> ^bb1701, ^bb1
  ^bb1701:
    %890 = pdl_interp.get_operand 1 of %869
    pdl_interp.are_equal %872, %890 : !pdl.value -> ^bb1702, ^bb1
  ^bb1702:
    %891 = pdl_interp.get_operand 1 of %880
    pdl_interp.are_equal %885, %891 : !pdl.value -> ^bb1703, ^bb1
  ^bb1703:
    %892 = pdl_interp.get_operand 1 of %883
    pdl_interp.are_equal %885, %892 : !pdl.value -> ^bb1704, ^bb1
  ^bb1704:
    %893 = pdl_interp.get_operand 1 of %768
    pdl_interp.are_equal %885, %893 : !pdl.value -> ^bb1705, ^bb1
  ^bb1705:
    %894 = pdl_interp.get_value_type of %885 : !pdl.type
    pdl_interp.are_equal %894, %873 : !pdl.type -> ^bb1706, ^bb1
  ^bb1706:
    pdl_interp.record_match @rewriters::@sum_cubes_rev(%872, %885, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1
  ^bb1607:
    %895 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %835, %895 : !pdl.type -> ^bb1707, ^bb1608
  ^bb1707:
    pdl_interp.check_type %835 is f32 -> ^bb1708, ^bb1608
  ^bb1708:
    pdl_interp.check_operation_name of %768 is "arith.subf" -> ^bb1709, ^bb1608
  ^bb1709:
    pdl_interp.check_operand_count of %768 is 2 -> ^bb1710, ^bb1608
  ^bb1710:
    pdl_interp.check_result_count of %768 is 1 -> ^bb1711, ^bb1608
  ^bb1711:
    %896 = pdl_interp.get_result 0 of %768
    pdl_interp.is_not_null %896 : !pdl.value -> ^bb1712, ^bb1608
  ^bb1712:
    pdl_interp.are_equal %896, %767 : !pdl.value -> ^bb1713, ^bb1608
  ^bb1713:
    %897 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %897 : !pdl.value -> ^bb1714, ^bb1608
  ^bb1714:
    %898 = pdl_interp.get_defining_op of %897 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %898 : !pdl.operation -> ^bb1715, ^bb1716
  ^bb1716:
    %899 = pdl_interp.get_value_type of %896 : !pdl.type
    pdl_interp.are_equal %835, %899 : !pdl.type -> ^bb1717, ^bb1608
  ^bb1717:
    %900 = pdl_interp.get_value_type of %897 : !pdl.type
    pdl_interp.are_equal %835, %900 : !pdl.type -> ^bb1718, ^bb1608
  ^bb1718:
    %901 = pdl_interp.get_operand 0 of %768
    pdl_interp.are_equal %834, %901 : !pdl.value -> ^bb1719, ^bb1608
  ^bb1719:
    %902 = pdl_interp.get_operand 1 of %768
    pdl_interp.are_equal %897, %902 : !pdl.value -> ^bb1720, ^bb1608
  ^bb1720:
    pdl_interp.record_match @rewriters::@difference_of_squares_rev(%834, %897, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1608
  ^bb1715:
    %903 = pdl_interp.get_operand 1 of %768
    %904 = pdl_interp.get_defining_op of %903 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %904 : !pdl.operation -> ^bb1721, ^bb1716
  ^bb1721:
    pdl_interp.is_not_null %903 : !pdl.value -> ^bb1722, ^bb1716
  ^bb1722:
    %905 = pdl_interp.get_value_type of %896 : !pdl.type
    pdl_interp.are_equal %835, %905 : !pdl.type -> ^bb1723, ^bb1716
  ^bb1723:
    pdl_interp.check_operation_name of %898 is "arith.constant" -> ^bb1724, ^bb1716
  ^bb1724:
    pdl_interp.check_operand_count of %898 is 0 -> ^bb1725, ^bb1716
  ^bb1725:
    pdl_interp.check_result_count of %898 is 1 -> ^bb1726, ^bb1716
  ^bb1726:
    %906 = pdl_interp.get_result 0 of %898
    pdl_interp.is_not_null %906 : !pdl.value -> ^bb1727, ^bb1716
  ^bb1727:
    pdl_interp.are_equal %906, %897 : !pdl.value -> ^bb1728, ^bb1716
  ^bb1728:
    pdl_interp.check_operation_name of %904 is "arith.constant" -> ^bb1729, ^bb1716
  ^bb1729:
    pdl_interp.check_operand_count of %904 is 0 -> ^bb1730, ^bb1716
  ^bb1730:
    pdl_interp.check_result_count of %904 is 1 -> ^bb1731, ^bb1716
  ^bb1731:
    %907 = pdl_interp.get_result 0 of %904
    pdl_interp.is_not_null %907 : !pdl.value -> ^bb1732, ^bb1716
  ^bb1732:
    pdl_interp.are_equal %907, %903 : !pdl.value -> ^bb1733, ^bb1716
  ^bb1733:
    %908 = pdl_interp.get_attribute "value" of %898
    pdl_interp.is_not_null %908 : !pdl.attribute -> ^bb1734, ^bb1716
  ^bb1734:
    pdl_interp.check_attribute %908 is 1.000000e+00 : f32 -> ^bb1735, ^bb1716
  ^bb1735:
    %909 = pdl_interp.get_operand 0 of %768
    pdl_interp.are_equal %834, %909 : !pdl.value -> ^bb1736, ^bb1716
  ^bb1736:
    %910 = pdl_interp.get_value_type of %906 : !pdl.type
    pdl_interp.are_equal %910, %835 : !pdl.type -> ^bb1737, ^bb1716
  ^bb1737:
    %911 = pdl_interp.get_value_type of %907 : !pdl.type
    pdl_interp.are_equal %911, %835 : !pdl.type -> ^bb1738, ^bb1716
  ^bb1738:
    %912 = pdl_interp.get_attribute "value" of %904
    pdl_interp.is_not_null %912 : !pdl.attribute -> ^bb1739, ^bb1716
  ^bb1739:
    pdl_interp.check_attribute %912 is 1.000000e+00 : f32 -> ^bb1740, ^bb1716
  ^bb1740:
    pdl_interp.record_match @rewriters::@difference_of_sqrsub_1_rev(%834, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1741
  ^bb1741:
    pdl_interp.record_match @rewriters::@difference_of_sqr_1_rev(%834, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1716
  ^bb1485:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb1742, ^bb1
  ^bb1742:
    pdl_interp.check_result_count of %3 is 1 -> ^bb1743, ^bb1
  ^bb1743:
    %913 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %913 : !pdl.value -> ^bb1744, ^bb1
  ^bb1744:
    pdl_interp.are_equal %913, %2 : !pdl.value -> ^bb1745, ^bb1
  ^bb1745:
    %914 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %914 : !pdl.value -> ^bb1746, ^bb1
  ^bb1746:
    pdl_interp.is_not_null %767 : !pdl.value -> ^bb1747, ^bb1
  ^bb1747:
    %915 = pdl_interp.get_value_type of %914 : !pdl.type
    %916 = pdl_interp.get_value_type of %913 : !pdl.type
    pdl_interp.are_equal %915, %916 : !pdl.type -> ^bb1748, ^bb1
  ^bb1748:
    %917 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %915, %917 : !pdl.type -> ^bb1749, ^bb1
  ^bb1749:
    pdl_interp.check_type %915 is f32 -> ^bb1750, ^bb1
  ^bb1750:
    pdl_interp.check_operation_name of %768 is "arith.divf" -> ^bb1751, ^bb1
  ^bb1751:
    pdl_interp.check_operand_count of %768 is 2 -> ^bb1752, ^bb1
  ^bb1752:
    pdl_interp.check_result_count of %768 is 1 -> ^bb1753, ^bb1
  ^bb1753:
    %918 = pdl_interp.get_result 0 of %768
    pdl_interp.is_not_null %918 : !pdl.value -> ^bb1754, ^bb1
  ^bb1754:
    pdl_interp.are_equal %918, %767 : !pdl.value -> ^bb1755, ^bb1
  ^bb1755:
    %919 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %919 : !pdl.value -> ^bb1756, ^bb1
  ^bb1756:
    %920 = pdl_interp.get_operand 0 of %768
    pdl_interp.is_not_null %920 : !pdl.value -> ^bb1757, ^bb1
  ^bb1757:
    %921 = pdl_interp.get_operand 1 of %768
    pdl_interp.is_not_null %921 : !pdl.value -> ^bb1758, ^bb1
  ^bb1758:
    %922 = pdl_interp.get_value_type of %918 : !pdl.type
    pdl_interp.are_equal %915, %922 : !pdl.type -> ^bb1759, ^bb1
  ^bb1759:
    %923 = pdl_interp.get_value_type of %919 : !pdl.type
    pdl_interp.are_equal %915, %923 : !pdl.type -> ^bb1760, ^bb1
  ^bb1760:
    %924 = pdl_interp.get_value_type of %920 : !pdl.type
    pdl_interp.are_equal %915, %924 : !pdl.type -> ^bb1761, ^bb1
  ^bb1761:
    %925 = pdl_interp.get_value_type of %921 : !pdl.type
    pdl_interp.are_equal %915, %925 : !pdl.type -> ^bb1762, ^bb1
  ^bb1762:
    pdl_interp.record_match @rewriters::@frac_times(%914, %920, %919, %921, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1
  ^bb1486:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb1763, ^bb1
  ^bb1763:
    pdl_interp.check_result_count of %3 is 1 -> ^bb1764, ^bb1
  ^bb1764:
    %926 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %926 : !pdl.value -> ^bb1765, ^bb1
  ^bb1765:
    pdl_interp.are_equal %926, %2 : !pdl.value -> ^bb1766, ^bb1
  ^bb1766:
    %927 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %927 : !pdl.value -> ^bb1767, ^bb1
  ^bb1767:
    pdl_interp.is_not_null %767 : !pdl.value -> ^bb1768, ^bb1
  ^bb1768:
    %928 = pdl_interp.get_value_type of %927 : !pdl.type
    %929 = pdl_interp.get_value_type of %926 : !pdl.type
    pdl_interp.are_equal %928, %929 : !pdl.type -> ^bb1769, ^bb1
  ^bb1769:
    %930 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %928, %930 : !pdl.type -> ^bb1770, ^bb1
  ^bb1770:
    pdl_interp.check_type %928 is f32 -> ^bb1771, ^bb1
  ^bb1771:
    pdl_interp.check_operation_name of %768 is "math.sqrt" -> ^bb1772, ^bb1
  ^bb1772:
    pdl_interp.check_operand_count of %768 is 1 -> ^bb1773, ^bb1
  ^bb1773:
    pdl_interp.check_result_count of %768 is 1 -> ^bb1774, ^bb1
  ^bb1774:
    %931 = pdl_interp.get_result 0 of %768
    pdl_interp.is_not_null %931 : !pdl.value -> ^bb1775, ^bb1
  ^bb1775:
    pdl_interp.are_equal %931, %767 : !pdl.value -> ^bb1776, ^bb1
  ^bb1776:
    %932 = pdl_interp.get_value_type of %931 : !pdl.type
    pdl_interp.are_equal %928, %932 : !pdl.type -> ^bb1777, ^bb1778
  ^bb1778:
    %933 = pdl_interp.get_operand 0 of %768
    pdl_interp.is_not_null %933 : !pdl.value -> ^bb1779, ^bb1
  ^bb1779:
    %934 = pdl_interp.get_value_type of %931 : !pdl.type
    pdl_interp.are_equal %928, %934 : !pdl.type -> ^bb1780, ^bb1
  ^bb1780:
    %935 = pdl_interp.get_value_type of %933 : !pdl.type
    pdl_interp.are_equal %928, %935 : !pdl.type -> ^bb1781, ^bb1
  ^bb1781:
    pdl_interp.record_match @rewriters::@sqrt_unprod(%927, %933, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1
  ^bb1777:
    %936 = pdl_interp.get_operand 0 of %768
    pdl_interp.are_equal %927, %936 : !pdl.value -> ^bb1782, ^bb1778
  ^bb1782:
    pdl_interp.record_match @rewriters::@rem_square_sqrt(%927, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1778
  ^bb1487:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb1783, ^bb1
  ^bb1783:
    pdl_interp.check_result_count of %3 is 1 -> ^bb1784, ^bb1
  ^bb1784:
    %937 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %937 : !pdl.value -> ^bb1785, ^bb1
  ^bb1785:
    pdl_interp.are_equal %937, %2 : !pdl.value -> ^bb1786, ^bb1
  ^bb1786:
    %938 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %938 : !pdl.value -> ^bb1787, ^bb1
  ^bb1787:
    pdl_interp.is_not_null %767 : !pdl.value -> ^bb1788, ^bb1
  ^bb1788:
    %939 = pdl_interp.get_value_type of %938 : !pdl.type
    %940 = pdl_interp.get_value_type of %937 : !pdl.type
    pdl_interp.are_equal %939, %940 : !pdl.type -> ^bb1789, ^bb1
  ^bb1789:
    %941 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %939, %941 : !pdl.type -> ^bb1790, ^bb1
  ^bb1790:
    pdl_interp.check_type %939 is f32 -> ^bb1791, ^bb1
  ^bb1791:
    pdl_interp.check_operation_name of %768 is "arith.negf" -> ^bb1792, ^bb1
  ^bb1792:
    pdl_interp.check_operand_count of %768 is 1 -> ^bb1793, ^bb1
  ^bb1793:
    pdl_interp.check_result_count of %768 is 1 -> ^bb1794, ^bb1
  ^bb1794:
    %942 = pdl_interp.get_result 0 of %768
    pdl_interp.is_not_null %942 : !pdl.value -> ^bb1795, ^bb1
  ^bb1795:
    pdl_interp.are_equal %942, %767 : !pdl.value -> ^bb1796, ^bb1
  ^bb1796:
    %943 = pdl_interp.get_value_type of %942 : !pdl.type
    pdl_interp.are_equal %939, %943 : !pdl.type -> ^bb1797, ^bb1
  ^bb1797:
    %944 = pdl_interp.get_operand 0 of %768
    pdl_interp.are_equal %938, %944 : !pdl.value -> ^bb1798, ^bb1
  ^bb1798:
    pdl_interp.record_match @rewriters::@sqr_neg(%938, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1
  ^bb1488:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb1799, ^bb1
  ^bb1799:
    pdl_interp.check_result_count of %3 is 1 -> ^bb1800, ^bb1
  ^bb1800:
    %945 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %945 : !pdl.value -> ^bb1801, ^bb1
  ^bb1801:
    pdl_interp.are_equal %945, %2 : !pdl.value -> ^bb1802, ^bb1
  ^bb1802:
    %946 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %946 : !pdl.value -> ^bb1803, ^bb1
  ^bb1803:
    pdl_interp.is_not_null %767 : !pdl.value -> ^bb1804, ^bb1
  ^bb1804:
    %947 = pdl_interp.get_value_type of %946 : !pdl.type
    %948 = pdl_interp.get_value_type of %945 : !pdl.type
    pdl_interp.are_equal %947, %948 : !pdl.type -> ^bb1805, ^bb1
  ^bb1805:
    %949 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %947, %949 : !pdl.type -> ^bb1806, ^bb1
  ^bb1806:
    pdl_interp.check_type %947 is f32 -> ^bb1807, ^bb1
  ^bb1807:
    pdl_interp.check_operation_name of %768 is "math.absf" -> ^bb1808, ^bb1
  ^bb1808:
    pdl_interp.check_operand_count of %768 is 1 -> ^bb1809, ^bb1
  ^bb1809:
    pdl_interp.check_result_count of %768 is 1 -> ^bb1810, ^bb1
  ^bb1810:
    %950 = pdl_interp.get_result 0 of %768
    pdl_interp.is_not_null %950 : !pdl.value -> ^bb1811, ^bb1
  ^bb1811:
    pdl_interp.are_equal %950, %767 : !pdl.value -> ^bb1812, ^bb1
  ^bb1812:
    %951 = pdl_interp.get_value_type of %950 : !pdl.type
    pdl_interp.are_equal %947, %951 : !pdl.type -> ^bb1813, ^bb1814
  ^bb1814:
    %952 = pdl_interp.get_operand 0 of %768
    pdl_interp.is_not_null %952 : !pdl.value -> ^bb1815, ^bb1
  ^bb1815:
    %953 = pdl_interp.get_value_type of %950 : !pdl.type
    pdl_interp.are_equal %947, %953 : !pdl.type -> ^bb1816, ^bb1
  ^bb1816:
    %954 = pdl_interp.get_value_type of %952 : !pdl.type
    pdl_interp.are_equal %947, %954 : !pdl.type -> ^bb1817, ^bb1
  ^bb1817:
    pdl_interp.record_match @rewriters::@mul_fabs(%946, %952, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1
  ^bb1813:
    %955 = pdl_interp.get_operand 0 of %768
    pdl_interp.are_equal %946, %955 : !pdl.value -> ^bb1818, ^bb1814
  ^bb1818:
    pdl_interp.record_match @rewriters::@sqr_abs(%946, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1814
  ^bb1489:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb1819, ^bb1
  ^bb1819:
    pdl_interp.check_result_count of %3 is 1 -> ^bb1820, ^bb1
  ^bb1820:
    %956 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %956 : !pdl.value -> ^bb1821, ^bb1
  ^bb1821:
    pdl_interp.are_equal %956, %2 : !pdl.value -> ^bb1822, ^bb1
  ^bb1822:
    %957 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %957 : !pdl.value -> ^bb1823, ^bb1
  ^bb1823:
    pdl_interp.is_not_null %767 : !pdl.value -> ^bb1824, ^bb1
  ^bb1824:
    %958 = pdl_interp.get_value_type of %957 : !pdl.type
    %959 = pdl_interp.get_value_type of %956 : !pdl.type
    pdl_interp.are_equal %958, %959 : !pdl.type -> ^bb1825, ^bb1
  ^bb1825:
    %960 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %958, %960 : !pdl.type -> ^bb1826, ^bb1
  ^bb1826:
    pdl_interp.check_type %958 is f32 -> ^bb1827, ^bb1
  ^bb1827:
    pdl_interp.switch_operation_name of %768 to ["arith.mulf", "math.cbrt"](^bb1828, ^bb1829) -> ^bb1
  ^bb1828:
    pdl_interp.check_operand_count of %768 is 2 -> ^bb1830, ^bb1
  ^bb1830:
    pdl_interp.check_result_count of %768 is 1 -> ^bb1831, ^bb1
  ^bb1831:
    %961 = pdl_interp.get_result 0 of %768
    pdl_interp.is_not_null %961 : !pdl.value -> ^bb1832, ^bb1
  ^bb1832:
    pdl_interp.are_equal %961, %767 : !pdl.value -> ^bb1833, ^bb1
  ^bb1833:
    %962 = pdl_interp.get_operand 0 of %768
    pdl_interp.is_not_null %962 : !pdl.value -> ^bb1834, ^bb1
  ^bb1834:
    %963 = pdl_interp.get_defining_op of %962 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %963 : !pdl.operation -> ^bb1835, ^bb1
  ^bb1835:
    %964 = pdl_interp.get_operand 1 of %768
    %965 = pdl_interp.get_defining_op of %964 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %965 : !pdl.operation -> ^bb1836, ^bb1
  ^bb1836:
    pdl_interp.is_not_null %964 : !pdl.value -> ^bb1837, ^bb1
  ^bb1837:
    %966 = pdl_interp.get_value_type of %961 : !pdl.type
    pdl_interp.are_equal %958, %966 : !pdl.type -> ^bb1838, ^bb1
  ^bb1838:
    pdl_interp.check_operation_name of %963 is "math.cbrt" -> ^bb1839, ^bb1
  ^bb1839:
    pdl_interp.check_operand_count of %963 is 1 -> ^bb1840, ^bb1
  ^bb1840:
    pdl_interp.check_result_count of %963 is 1 -> ^bb1841, ^bb1
  ^bb1841:
    %967 = pdl_interp.get_result 0 of %963
    pdl_interp.is_not_null %967 : !pdl.value -> ^bb1842, ^bb1
  ^bb1842:
    pdl_interp.are_equal %967, %962 : !pdl.value -> ^bb1843, ^bb1
  ^bb1843:
    pdl_interp.check_operation_name of %965 is "math.cbrt" -> ^bb1844, ^bb1
  ^bb1844:
    pdl_interp.check_operand_count of %965 is 1 -> ^bb1845, ^bb1
  ^bb1845:
    pdl_interp.check_result_count of %965 is 1 -> ^bb1846, ^bb1
  ^bb1846:
    %968 = pdl_interp.get_result 0 of %965
    pdl_interp.is_not_null %968 : !pdl.value -> ^bb1847, ^bb1
  ^bb1847:
    pdl_interp.are_equal %968, %964 : !pdl.value -> ^bb1848, ^bb1
  ^bb1848:
    %969 = pdl_interp.get_value_type of %968 : !pdl.type
    pdl_interp.are_equal %969, %958 : !pdl.type -> ^bb1849, ^bb1
  ^bb1849:
    %970 = pdl_interp.get_value_type of %967 : !pdl.type
    pdl_interp.are_equal %970, %958 : !pdl.type -> ^bb1850, ^bb1
  ^bb1850:
    %971 = pdl_interp.get_operand 0 of %963
    pdl_interp.are_equal %971, %957 : !pdl.value -> ^bb1851, ^bb1
  ^bb1851:
    %972 = pdl_interp.get_operand 0 of %965
    pdl_interp.are_equal %972, %957 : !pdl.value -> ^bb1852, ^bb1
  ^bb1852:
    pdl_interp.record_match @rewriters::@rem_3cbrt_rft(%957, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1
  ^bb1829:
    pdl_interp.check_operand_count of %768 is 1 -> ^bb1853, ^bb1
  ^bb1853:
    pdl_interp.check_result_count of %768 is 1 -> ^bb1854, ^bb1
  ^bb1854:
    %973 = pdl_interp.get_result 0 of %768
    pdl_interp.is_not_null %973 : !pdl.value -> ^bb1855, ^bb1
  ^bb1855:
    pdl_interp.are_equal %973, %767 : !pdl.value -> ^bb1856, ^bb1
  ^bb1856:
    %974 = pdl_interp.get_operand 0 of %768
    pdl_interp.is_not_null %974 : !pdl.value -> ^bb1857, ^bb1
  ^bb1857:
    %975 = pdl_interp.get_value_type of %973 : !pdl.type
    pdl_interp.are_equal %958, %975 : !pdl.type -> ^bb1858, ^bb1
  ^bb1858:
    %976 = pdl_interp.get_value_type of %974 : !pdl.type
    pdl_interp.are_equal %958, %976 : !pdl.type -> ^bb1859, ^bb1
  ^bb1859:
    pdl_interp.record_match @rewriters::@cbrt_unprod(%957, %974, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1
  ^bb1490:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb1860, ^bb1
  ^bb1860:
    pdl_interp.check_result_count of %3 is 1 -> ^bb1861, ^bb1
  ^bb1861:
    %977 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %977 : !pdl.value -> ^bb1862, ^bb1
  ^bb1862:
    pdl_interp.are_equal %977, %2 : !pdl.value -> ^bb1863, ^bb1
  ^bb1863:
    %978 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %978 : !pdl.value -> ^bb1864, ^bb1
  ^bb1864:
    pdl_interp.is_not_null %767 : !pdl.value -> ^bb1865, ^bb1
  ^bb1865:
    %979 = pdl_interp.get_value_type of %978 : !pdl.type
    %980 = pdl_interp.get_value_type of %977 : !pdl.type
    pdl_interp.are_equal %979, %980 : !pdl.type -> ^bb1866, ^bb1
  ^bb1866:
    %981 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %979, %981 : !pdl.type -> ^bb1867, ^bb1
  ^bb1867:
    pdl_interp.check_type %979 is f32 -> ^bb1868, ^bb1
  ^bb1868:
    pdl_interp.check_operation_name of %768 is "math.exp" -> ^bb1869, ^bb1
  ^bb1869:
    pdl_interp.check_operand_count of %768 is 1 -> ^bb1870, ^bb1
  ^bb1870:
    pdl_interp.check_result_count of %768 is 1 -> ^bb1871, ^bb1
  ^bb1871:
    %982 = pdl_interp.get_result 0 of %768
    pdl_interp.is_not_null %982 : !pdl.value -> ^bb1872, ^bb1
  ^bb1872:
    pdl_interp.are_equal %982, %767 : !pdl.value -> ^bb1873, ^bb1
  ^bb1873:
    %983 = pdl_interp.get_operand 0 of %768
    pdl_interp.is_not_null %983 : !pdl.value -> ^bb1874, ^bb1875
  ^bb1875:
    %984 = pdl_interp.get_value_type of %982 : !pdl.type
    pdl_interp.are_equal %979, %984 : !pdl.type -> ^bb1876, ^bb1
  ^bb1876:
    %985 = pdl_interp.get_operand 0 of %768
    pdl_interp.are_equal %978, %985 : !pdl.value -> ^bb1877, ^bb1
  ^bb1877:
    pdl_interp.record_match @rewriters::@exp_lft_sqr_rev(%978, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1
  ^bb1874:
    %986 = pdl_interp.get_value_type of %982 : !pdl.type
    pdl_interp.are_equal %979, %986 : !pdl.type -> ^bb1878, ^bb1875
  ^bb1878:
    %987 = pdl_interp.get_value_type of %983 : !pdl.type
    pdl_interp.are_equal %979, %987 : !pdl.type -> ^bb1879, ^bb1875
  ^bb1879:
    pdl_interp.record_match @rewriters::@prod_exp(%978, %983, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1875
  ^bb1491:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb1880, ^bb1
  ^bb1880:
    pdl_interp.check_result_count of %3 is 1 -> ^bb1881, ^bb1
  ^bb1881:
    %988 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %988 : !pdl.value -> ^bb1882, ^bb1
  ^bb1882:
    pdl_interp.are_equal %988, %2 : !pdl.value -> ^bb1883, ^bb1
  ^bb1883:
    %989 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %989 : !pdl.value -> ^bb1884, ^bb1
  ^bb1884:
    pdl_interp.is_not_null %767 : !pdl.value -> ^bb1885, ^bb1
  ^bb1885:
    %990 = pdl_interp.get_value_type of %989 : !pdl.type
    %991 = pdl_interp.get_value_type of %988 : !pdl.type
    pdl_interp.are_equal %990, %991 : !pdl.type -> ^bb1886, ^bb1
  ^bb1886:
    %992 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %990, %992 : !pdl.type -> ^bb1887, ^bb1
  ^bb1887:
    pdl_interp.check_type %990 is f32 -> ^bb1888, ^bb1
  ^bb1888:
    pdl_interp.switch_operation_name of %768 to ["math.sin", "math.cos"](^bb1889, ^bb1890) -> ^bb1
  ^bb1889:
    pdl_interp.check_operand_count of %768 is 1 -> ^bb1891, ^bb1
  ^bb1891:
    pdl_interp.check_result_count of %768 is 1 -> ^bb1892, ^bb1
  ^bb1892:
    %993 = pdl_interp.get_result 0 of %768
    pdl_interp.is_not_null %993 : !pdl.value -> ^bb1893, ^bb1
  ^bb1893:
    pdl_interp.are_equal %993, %767 : !pdl.value -> ^bb1894, ^bb1
  ^bb1894:
    %994 = pdl_interp.get_value_type of %993 : !pdl.type
    pdl_interp.are_equal %990, %994 : !pdl.type -> ^bb1895, ^bb1896
  ^bb1896:
    %995 = pdl_interp.get_operand 0 of %768
    pdl_interp.is_not_null %995 : !pdl.value -> ^bb1897, ^bb1
  ^bb1897:
    %996 = pdl_interp.get_value_type of %993 : !pdl.type
    pdl_interp.are_equal %990, %996 : !pdl.type -> ^bb1898, ^bb1
  ^bb1898:
    %997 = pdl_interp.get_value_type of %995 : !pdl.type
    pdl_interp.are_equal %990, %997 : !pdl.type -> ^bb1899, ^bb1
  ^bb1899:
    pdl_interp.record_match @rewriters::@sin_mult(%989, %995, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1
  ^bb1895:
    %998 = pdl_interp.get_operand 0 of %768
    pdl_interp.are_equal %989, %998 : !pdl.value -> ^bb1900, ^bb1896
  ^bb1900:
    pdl_interp.record_match @rewriters::@sqr_sin_a(%989, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1901
  ^bb1901:
    pdl_interp.record_match @rewriters::@sqr_sin_b(%989, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1896
  ^bb1890:
    pdl_interp.check_operand_count of %768 is 1 -> ^bb1902, ^bb1
  ^bb1902:
    pdl_interp.check_result_count of %768 is 1 -> ^bb1903, ^bb1
  ^bb1903:
    %999 = pdl_interp.get_result 0 of %768
    pdl_interp.is_not_null %999 : !pdl.value -> ^bb1904, ^bb1
  ^bb1904:
    pdl_interp.are_equal %999, %767 : !pdl.value -> ^bb1905, ^bb1
  ^bb1905:
    %1000 = pdl_interp.get_operand 0 of %768
    pdl_interp.is_not_null %1000 : !pdl.value -> ^bb1906, ^bb1
  ^bb1906:
    %1001 = pdl_interp.get_value_type of %999 : !pdl.type
    pdl_interp.are_equal %990, %1001 : !pdl.type -> ^bb1907, ^bb1
  ^bb1907:
    %1002 = pdl_interp.get_value_type of %1000 : !pdl.type
    pdl_interp.are_equal %990, %1002 : !pdl.type -> ^bb1908, ^bb1
  ^bb1908:
    pdl_interp.record_match @rewriters::@sin_cos_mult(%989, %1000, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1
  ^bb1492:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb1909, ^bb1
  ^bb1909:
    pdl_interp.check_result_count of %3 is 1 -> ^bb1910, ^bb1
  ^bb1910:
    %1003 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %1003 : !pdl.value -> ^bb1911, ^bb1
  ^bb1911:
    pdl_interp.are_equal %1003, %2 : !pdl.value -> ^bb1912, ^bb1
  ^bb1912:
    %1004 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %1004 : !pdl.value -> ^bb1913, ^bb1
  ^bb1913:
    pdl_interp.is_not_null %767 : !pdl.value -> ^bb1914, ^bb1
  ^bb1914:
    %1005 = pdl_interp.get_value_type of %1004 : !pdl.type
    %1006 = pdl_interp.get_value_type of %1003 : !pdl.type
    pdl_interp.are_equal %1005, %1006 : !pdl.type -> ^bb1915, ^bb1
  ^bb1915:
    %1007 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1005, %1007 : !pdl.type -> ^bb1916, ^bb1
  ^bb1916:
    pdl_interp.check_type %1005 is f32 -> ^bb1917, ^bb1
  ^bb1917:
    pdl_interp.check_operation_name of %768 is "math.cos" -> ^bb1918, ^bb1
  ^bb1918:
    pdl_interp.check_operand_count of %768 is 1 -> ^bb1919, ^bb1
  ^bb1919:
    pdl_interp.check_result_count of %768 is 1 -> ^bb1920, ^bb1
  ^bb1920:
    %1008 = pdl_interp.get_result 0 of %768
    pdl_interp.is_not_null %1008 : !pdl.value -> ^bb1921, ^bb1
  ^bb1921:
    pdl_interp.are_equal %1008, %767 : !pdl.value -> ^bb1922, ^bb1
  ^bb1922:
    %1009 = pdl_interp.get_value_type of %1008 : !pdl.type
    pdl_interp.are_equal %1005, %1009 : !pdl.type -> ^bb1923, ^bb1924
  ^bb1924:
    %1010 = pdl_interp.get_operand 0 of %768
    pdl_interp.is_not_null %1010 : !pdl.value -> ^bb1925, ^bb1
  ^bb1925:
    %1011 = pdl_interp.get_value_type of %1008 : !pdl.type
    pdl_interp.are_equal %1005, %1011 : !pdl.type -> ^bb1926, ^bb1
  ^bb1926:
    %1012 = pdl_interp.get_value_type of %1010 : !pdl.type
    pdl_interp.are_equal %1005, %1012 : !pdl.type -> ^bb1927, ^bb1
  ^bb1927:
    pdl_interp.record_match @rewriters::@cos_mult(%1004, %1010, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1
  ^bb1923:
    %1013 = pdl_interp.get_operand 0 of %768
    pdl_interp.are_equal %1004, %1013 : !pdl.value -> ^bb1928, ^bb1924
  ^bb1928:
    pdl_interp.record_match @rewriters::@sqr_cos_a(%1004, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1929
  ^bb1929:
    pdl_interp.record_match @rewriters::@_1_sub_sin_rev(%1004, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1930
  ^bb1930:
    pdl_interp.record_match @rewriters::@sqr_cos_b(%1004, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1924
  ^bb1493:
    pdl_interp.check_operand_count of %3 is 0 -> ^bb1931, ^bb1
  ^bb1931:
    pdl_interp.check_result_count of %3 is 1 -> ^bb1932, ^bb1
  ^bb1932:
    %1014 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %1014 : !pdl.value -> ^bb1933, ^bb1
  ^bb1933:
    pdl_interp.are_equal %1014, %2 : !pdl.value -> ^bb1934, ^bb1
  ^bb1934:
    pdl_interp.is_not_null %767 : !pdl.value -> ^bb1935, ^bb1
  ^bb1935:
    pdl_interp.switch_operation_name of %768 to ["arith.mulf", "math.sinh", "math.cosh", "math.acosh"](^bb1936, ^bb1937, ^bb1938, ^bb1939) -> ^bb1
  ^bb1936:
    pdl_interp.check_operand_count of %768 is 2 -> ^bb1940, ^bb1
  ^bb1940:
    pdl_interp.check_result_count of %768 is 1 -> ^bb1941, ^bb1
  ^bb1941:
    %1015 = pdl_interp.get_result 0 of %768
    pdl_interp.is_not_null %1015 : !pdl.value -> ^bb1942, ^bb1
  ^bb1942:
    pdl_interp.are_equal %1015, %767 : !pdl.value -> ^bb1943, ^bb1
  ^bb1943:
    %1016 = pdl_interp.get_operand 0 of %768
    pdl_interp.is_not_null %1016 : !pdl.value -> ^bb1944, ^bb1
  ^bb1944:
    %1017 = pdl_interp.get_defining_op of %1016 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1017 : !pdl.operation -> ^bb1945, ^bb1
  ^bb1945:
    %1018 = pdl_interp.get_operand 1 of %768
    %1019 = pdl_interp.get_defining_op of %1018 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %1019 : !pdl.operation -> ^bb1946, ^bb1
  ^bb1946:
    pdl_interp.is_not_null %1018 : !pdl.value -> ^bb1947, ^bb1
  ^bb1947:
    pdl_interp.switch_operation_name of %1017 to ["math.sin", "math.cos", "math.sinh", "math.cosh"](^bb1948, ^bb1949, ^bb1950, ^bb1951) -> ^bb1
  ^bb1948:
    pdl_interp.check_operand_count of %1017 is 1 -> ^bb1952, ^bb1
  ^bb1952:
    pdl_interp.check_result_count of %1017 is 1 -> ^bb1953, ^bb1
  ^bb1953:
    %1020 = pdl_interp.get_result 0 of %1017
    pdl_interp.is_not_null %1020 : !pdl.value -> ^bb1954, ^bb1
  ^bb1954:
    pdl_interp.are_equal %1020, %1016 : !pdl.value -> ^bb1955, ^bb1
  ^bb1955:
    pdl_interp.switch_operation_name of %1019 to ["math.cos", "math.sin"](^bb1956, ^bb1957) -> ^bb1
  ^bb1956:
    pdl_interp.check_operand_count of %1019 is 1 -> ^bb1958, ^bb1
  ^bb1958:
    pdl_interp.check_result_count of %1019 is 1 -> ^bb1959, ^bb1
  ^bb1959:
    %1021 = pdl_interp.get_result 0 of %1019
    pdl_interp.is_not_null %1021 : !pdl.value -> ^bb1960, ^bb1
  ^bb1960:
    pdl_interp.are_equal %1021, %1018 : !pdl.value -> ^bb1961, ^bb1
  ^bb1961:
    %1022 = pdl_interp.get_attribute "value" of %3
    pdl_interp.is_not_null %1022 : !pdl.attribute -> ^bb1962, ^bb1
  ^bb1962:
    pdl_interp.check_attribute %1022 is 2.000000e+00 : f32 -> ^bb1963, ^bb1
  ^bb1963:
    %1023 = pdl_interp.get_value_type of %1014 : !pdl.type
    %1024 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1023, %1024 : !pdl.type -> ^bb1964, ^bb1
  ^bb1964:
    pdl_interp.check_type %1023 is f32 -> ^bb1965, ^bb1
  ^bb1965:
    %1025 = pdl_interp.get_value_type of %1015 : !pdl.type
    pdl_interp.are_equal %1023, %1025 : !pdl.type -> ^bb1966, ^bb1967
  ^bb1967:
    %1026 = pdl_interp.get_operand 0 of %1019
    %1027 = pdl_interp.get_defining_op of %1026 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1027 : !pdl.operation -> ^bb1968, ^bb1
  ^bb1968:
    %1028 = pdl_interp.get_operand 0 of %1017
    %1029 = pdl_interp.get_defining_op of %1028 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1029 : !pdl.operation -> ^bb1969, ^bb1
  ^bb1969:
    %1030 = pdl_interp.get_value_type of %1015 : !pdl.type
    pdl_interp.are_equal %1023, %1030 : !pdl.type -> ^bb1970, ^bb1
  ^bb1970:
    pdl_interp.is_not_null %1028 : !pdl.value -> ^bb1971, ^bb1
  ^bb1971:
    %1031 = pdl_interp.get_operand 0 of %1027
    %1032 = pdl_interp.get_defining_op of %1031 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1032 : !pdl.operation -> ^bb1972, ^bb1
  ^bb1972:
    %1033 = pdl_interp.get_value_type of %1020 : !pdl.type
    pdl_interp.are_equal %1033, %1023 : !pdl.type -> ^bb1973, ^bb1
  ^bb1973:
    %1034 = pdl_interp.get_operand 0 of %1029
    %1035 = pdl_interp.get_defining_op of %1034 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1035 : !pdl.operation -> ^bb1974, ^bb1
  ^bb1974:
    pdl_interp.is_not_null %1026 : !pdl.value -> ^bb1975, ^bb1
  ^bb1975:
    pdl_interp.check_operation_name of %1027 is "arith.divf" -> ^bb1976, ^bb1
  ^bb1976:
    pdl_interp.check_operand_count of %1027 is 2 -> ^bb1977, ^bb1
  ^bb1977:
    pdl_interp.check_result_count of %1027 is 1 -> ^bb1978, ^bb1
  ^bb1978:
    %1036 = pdl_interp.get_result 0 of %1027
    pdl_interp.is_not_null %1036 : !pdl.value -> ^bb1979, ^bb1
  ^bb1979:
    pdl_interp.are_equal %1036, %1026 : !pdl.value -> ^bb1980, ^bb1
  ^bb1980:
    %1037 = pdl_interp.get_value_type of %1021 : !pdl.type
    pdl_interp.are_equal %1037, %1023 : !pdl.type -> ^bb1981, ^bb1
  ^bb1981:
    pdl_interp.check_operation_name of %1029 is "arith.divf" -> ^bb1982, ^bb1
  ^bb1982:
    pdl_interp.check_operand_count of %1029 is 2 -> ^bb1983, ^bb1
  ^bb1983:
    pdl_interp.check_result_count of %1029 is 1 -> ^bb1984, ^bb1
  ^bb1984:
    %1038 = pdl_interp.get_result 0 of %1029
    pdl_interp.is_not_null %1038 : !pdl.value -> ^bb1985, ^bb1
  ^bb1985:
    pdl_interp.are_equal %1038, %1028 : !pdl.value -> ^bb1986, ^bb1
  ^bb1986:
    %1039 = pdl_interp.get_operand 1 of %1029
    %1040 = pdl_interp.get_defining_op of %1039 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %1040 : !pdl.operation -> ^bb1987, ^bb1
  ^bb1987:
    %1041 = pdl_interp.get_operand 1 of %1027
    %1042 = pdl_interp.get_defining_op of %1041 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %1042 : !pdl.operation -> ^bb1988, ^bb1
  ^bb1988:
    pdl_interp.is_not_null %1031 : !pdl.value -> ^bb1989, ^bb1
  ^bb1989:
    pdl_interp.switch_operation_name of %1032 to ["arith.addf", "arith.subf"](^bb1990, ^bb1991) -> ^bb1
  ^bb1990:
    pdl_interp.check_operand_count of %1032 is 2 -> ^bb1992, ^bb1
  ^bb1992:
    pdl_interp.check_result_count of %1032 is 1 -> ^bb1993, ^bb1
  ^bb1993:
    %1043 = pdl_interp.get_result 0 of %1032
    pdl_interp.is_not_null %1043 : !pdl.value -> ^bb1994, ^bb1
  ^bb1994:
    pdl_interp.are_equal %1043, %1031 : !pdl.value -> ^bb1995, ^bb1
  ^bb1995:
    pdl_interp.is_not_null %1041 : !pdl.value -> ^bb1996, ^bb1
  ^bb1996:
    %1044 = pdl_interp.get_value_type of %1036 : !pdl.type
    pdl_interp.are_equal %1044, %1023 : !pdl.type -> ^bb1997, ^bb1
  ^bb1997:
    %1045 = pdl_interp.get_value_type of %1043 : !pdl.type
    pdl_interp.are_equal %1045, %1023 : !pdl.type -> ^bb1998, ^bb1
  ^bb1998:
    %1046 = pdl_interp.get_value_type of %1038 : !pdl.type
    pdl_interp.are_equal %1046, %1023 : !pdl.type -> ^bb1999, ^bb1
  ^bb1999:
    pdl_interp.is_not_null %1034 : !pdl.value -> ^bb2000, ^bb1
  ^bb2000:
    pdl_interp.check_operation_name of %1035 is "arith.subf" -> ^bb2001, ^bb1
  ^bb2001:
    pdl_interp.check_operand_count of %1035 is 2 -> ^bb2002, ^bb1
  ^bb2002:
    pdl_interp.check_result_count of %1035 is 1 -> ^bb2003, ^bb1
  ^bb2003:
    %1047 = pdl_interp.get_result 0 of %1035
    pdl_interp.is_not_null %1047 : !pdl.value -> ^bb2004, ^bb1
  ^bb2004:
    pdl_interp.are_equal %1047, %1034 : !pdl.value -> ^bb2005, ^bb1
  ^bb2005:
    pdl_interp.is_not_null %1039 : !pdl.value -> ^bb2006, ^bb1
  ^bb2006:
    pdl_interp.check_operation_name of %1040 is "arith.constant" -> ^bb2007, ^bb1
  ^bb2007:
    pdl_interp.check_operation_name of %1042 is "arith.constant" -> ^bb2008, ^bb1
  ^bb2008:
    pdl_interp.check_operand_count of %1040 is 0 -> ^bb2009, ^bb1
  ^bb2009:
    pdl_interp.check_operand_count of %1042 is 0 -> ^bb2010, ^bb1
  ^bb2010:
    pdl_interp.check_result_count of %1040 is 1 -> ^bb2011, ^bb1
  ^bb2011:
    pdl_interp.check_result_count of %1042 is 1 -> ^bb2012, ^bb1
  ^bb2012:
    %1048 = pdl_interp.get_operand 0 of %1035
    pdl_interp.is_not_null %1048 : !pdl.value -> ^bb2013, ^bb1
  ^bb2013:
    %1049 = pdl_interp.get_operand 1 of %1035
    pdl_interp.is_not_null %1049 : !pdl.value -> ^bb2014, ^bb1
  ^bb2014:
    %1050 = pdl_interp.get_operand 0 of %1032
    pdl_interp.are_equal %1048, %1050 : !pdl.value -> ^bb2015, ^bb1
  ^bb2015:
    %1051 = pdl_interp.get_operand 1 of %1032
    pdl_interp.are_equal %1049, %1051 : !pdl.value -> ^bb2016, ^bb1
  ^bb2016:
    %1052 = pdl_interp.get_attribute "value" of %1040
    pdl_interp.is_not_null %1052 : !pdl.attribute -> ^bb2017, ^bb1
  ^bb2017:
    %1053 = pdl_interp.get_attribute "value" of %1042
    pdl_interp.is_not_null %1053 : !pdl.attribute -> ^bb2018, ^bb1
  ^bb2018:
    pdl_interp.check_attribute %1052 is 2.000000e+00 : f32 -> ^bb2019, ^bb1
  ^bb2019:
    pdl_interp.check_attribute %1053 is 2.000000e+00 : f32 -> ^bb2020, ^bb1
  ^bb2020:
    %1054 = pdl_interp.get_result 0 of %1040
    pdl_interp.is_not_null %1054 : !pdl.value -> ^bb2021, ^bb1
  ^bb2021:
    %1055 = pdl_interp.get_result 0 of %1042
    pdl_interp.is_not_null %1055 : !pdl.value -> ^bb2022, ^bb1
  ^bb2022:
    pdl_interp.are_equal %1054, %1039 : !pdl.value -> ^bb2023, ^bb1
  ^bb2023:
    pdl_interp.are_equal %1055, %1041 : !pdl.value -> ^bb2024, ^bb1
  ^bb2024:
    %1056 = pdl_interp.get_value_type of %1048 : !pdl.type
    pdl_interp.are_equal %1056, %1023 : !pdl.type -> ^bb2025, ^bb1
  ^bb2025:
    %1057 = pdl_interp.get_value_type of %1049 : !pdl.type
    pdl_interp.are_equal %1057, %1023 : !pdl.type -> ^bb2026, ^bb1
  ^bb2026:
    %1058 = pdl_interp.get_value_type of %1047 : !pdl.type
    pdl_interp.are_equal %1058, %1023 : !pdl.type -> ^bb2027, ^bb1
  ^bb2027:
    %1059 = pdl_interp.get_value_type of %1054 : !pdl.type
    pdl_interp.are_equal %1059, %1023 : !pdl.type -> ^bb2028, ^bb1
  ^bb2028:
    %1060 = pdl_interp.get_value_type of %1055 : !pdl.type
    pdl_interp.are_equal %1060, %1023 : !pdl.type -> ^bb2029, ^bb1
  ^bb2029:
    pdl_interp.record_match @rewriters::@diff_sin_rev(%1048, %1049, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1
  ^bb1991:
    pdl_interp.check_operand_count of %1032 is 2 -> ^bb2030, ^bb1
  ^bb2030:
    pdl_interp.check_result_count of %1032 is 1 -> ^bb2031, ^bb1
  ^bb2031:
    %1061 = pdl_interp.get_result 0 of %1032
    pdl_interp.is_not_null %1061 : !pdl.value -> ^bb2032, ^bb1
  ^bb2032:
    pdl_interp.are_equal %1061, %1031 : !pdl.value -> ^bb2033, ^bb1
  ^bb2033:
    pdl_interp.is_not_null %1041 : !pdl.value -> ^bb2034, ^bb1
  ^bb2034:
    %1062 = pdl_interp.get_value_type of %1036 : !pdl.type
    pdl_interp.are_equal %1062, %1023 : !pdl.type -> ^bb2035, ^bb1
  ^bb2035:
    %1063 = pdl_interp.get_value_type of %1061 : !pdl.type
    pdl_interp.are_equal %1063, %1023 : !pdl.type -> ^bb2036, ^bb1
  ^bb2036:
    %1064 = pdl_interp.get_value_type of %1038 : !pdl.type
    pdl_interp.are_equal %1064, %1023 : !pdl.type -> ^bb2037, ^bb1
  ^bb2037:
    pdl_interp.is_not_null %1034 : !pdl.value -> ^bb2038, ^bb1
  ^bb2038:
    pdl_interp.check_operation_name of %1035 is "arith.addf" -> ^bb2039, ^bb1
  ^bb2039:
    pdl_interp.check_operand_count of %1035 is 2 -> ^bb2040, ^bb1
  ^bb2040:
    pdl_interp.check_result_count of %1035 is 1 -> ^bb2041, ^bb1
  ^bb2041:
    %1065 = pdl_interp.get_result 0 of %1035
    pdl_interp.is_not_null %1065 : !pdl.value -> ^bb2042, ^bb1
  ^bb2042:
    pdl_interp.are_equal %1065, %1034 : !pdl.value -> ^bb2043, ^bb1
  ^bb2043:
    pdl_interp.is_not_null %1039 : !pdl.value -> ^bb2044, ^bb1
  ^bb2044:
    pdl_interp.check_operation_name of %1040 is "arith.constant" -> ^bb2045, ^bb1
  ^bb2045:
    pdl_interp.check_operation_name of %1042 is "arith.constant" -> ^bb2046, ^bb1
  ^bb2046:
    pdl_interp.check_operand_count of %1040 is 0 -> ^bb2047, ^bb1
  ^bb2047:
    pdl_interp.check_operand_count of %1042 is 0 -> ^bb2048, ^bb1
  ^bb2048:
    pdl_interp.check_result_count of %1040 is 1 -> ^bb2049, ^bb1
  ^bb2049:
    pdl_interp.check_result_count of %1042 is 1 -> ^bb2050, ^bb1
  ^bb2050:
    %1066 = pdl_interp.get_operand 0 of %1035
    pdl_interp.is_not_null %1066 : !pdl.value -> ^bb2051, ^bb1
  ^bb2051:
    %1067 = pdl_interp.get_operand 1 of %1035
    pdl_interp.is_not_null %1067 : !pdl.value -> ^bb2052, ^bb1
  ^bb2052:
    %1068 = pdl_interp.get_operand 0 of %1032
    pdl_interp.are_equal %1066, %1068 : !pdl.value -> ^bb2053, ^bb1
  ^bb2053:
    %1069 = pdl_interp.get_operand 1 of %1032
    pdl_interp.are_equal %1067, %1069 : !pdl.value -> ^bb2054, ^bb1
  ^bb2054:
    %1070 = pdl_interp.get_attribute "value" of %1040
    pdl_interp.is_not_null %1070 : !pdl.attribute -> ^bb2055, ^bb1
  ^bb2055:
    %1071 = pdl_interp.get_attribute "value" of %1042
    pdl_interp.is_not_null %1071 : !pdl.attribute -> ^bb2056, ^bb1
  ^bb2056:
    pdl_interp.check_attribute %1070 is 2.000000e+00 : f32 -> ^bb2057, ^bb1
  ^bb2057:
    pdl_interp.check_attribute %1071 is 2.000000e+00 : f32 -> ^bb2058, ^bb1
  ^bb2058:
    %1072 = pdl_interp.get_result 0 of %1040
    pdl_interp.is_not_null %1072 : !pdl.value -> ^bb2059, ^bb1
  ^bb2059:
    %1073 = pdl_interp.get_result 0 of %1042
    pdl_interp.is_not_null %1073 : !pdl.value -> ^bb2060, ^bb1
  ^bb2060:
    pdl_interp.are_equal %1072, %1039 : !pdl.value -> ^bb2061, ^bb1
  ^bb2061:
    pdl_interp.are_equal %1073, %1041 : !pdl.value -> ^bb2062, ^bb1
  ^bb2062:
    %1074 = pdl_interp.get_value_type of %1066 : !pdl.type
    pdl_interp.are_equal %1074, %1023 : !pdl.type -> ^bb2063, ^bb1
  ^bb2063:
    %1075 = pdl_interp.get_value_type of %1067 : !pdl.type
    pdl_interp.are_equal %1075, %1023 : !pdl.type -> ^bb2064, ^bb1
  ^bb2064:
    %1076 = pdl_interp.get_value_type of %1065 : !pdl.type
    pdl_interp.are_equal %1076, %1023 : !pdl.type -> ^bb2065, ^bb1
  ^bb2065:
    %1077 = pdl_interp.get_value_type of %1072 : !pdl.type
    pdl_interp.are_equal %1077, %1023 : !pdl.type -> ^bb2066, ^bb1
  ^bb2066:
    %1078 = pdl_interp.get_value_type of %1073 : !pdl.type
    pdl_interp.are_equal %1078, %1023 : !pdl.type -> ^bb2067, ^bb1
  ^bb2067:
    pdl_interp.record_match @rewriters::@sum_sin_rev(%1066, %1067, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1
  ^bb1966:
    %1079 = pdl_interp.get_operand 0 of %1017
    pdl_interp.is_not_null %1079 : !pdl.value -> ^bb2068, ^bb1967
  ^bb2068:
    %1080 = pdl_interp.get_value_type of %1020 : !pdl.type
    pdl_interp.are_equal %1080, %1023 : !pdl.type -> ^bb2069, ^bb1967
  ^bb2069:
    %1081 = pdl_interp.get_value_type of %1021 : !pdl.type
    pdl_interp.are_equal %1081, %1023 : !pdl.type -> ^bb2070, ^bb1967
  ^bb2070:
    %1082 = pdl_interp.get_operand 0 of %1019
    pdl_interp.are_equal %1079, %1082 : !pdl.value -> ^bb2071, ^bb1967
  ^bb2071:
    %1083 = pdl_interp.get_value_type of %1079 : !pdl.type
    pdl_interp.are_equal %1083, %1023 : !pdl.type -> ^bb2072, ^bb1967
  ^bb2072:
    pdl_interp.record_match @rewriters::@_2_sin(%1079, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1967
  ^bb1957:
    pdl_interp.check_operand_count of %1019 is 1 -> ^bb2073, ^bb1
  ^bb2073:
    pdl_interp.check_result_count of %1019 is 1 -> ^bb2074, ^bb1
  ^bb2074:
    %1084 = pdl_interp.get_result 0 of %1019
    pdl_interp.is_not_null %1084 : !pdl.value -> ^bb2075, ^bb1
  ^bb2075:
    pdl_interp.are_equal %1084, %1018 : !pdl.value -> ^bb2076, ^bb1
  ^bb2076:
    %1085 = pdl_interp.get_attribute "value" of %3
    pdl_interp.is_not_null %1085 : !pdl.attribute -> ^bb2077, ^bb1
  ^bb2077:
    pdl_interp.check_attribute %1085 is -2.000000e+00 : f32 -> ^bb2078, ^bb1
  ^bb2078:
    %1086 = pdl_interp.get_value_type of %1014 : !pdl.type
    %1087 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1086, %1087 : !pdl.type -> ^bb2079, ^bb1
  ^bb2079:
    pdl_interp.check_type %1086 is f32 -> ^bb2080, ^bb1
  ^bb2080:
    %1088 = pdl_interp.get_operand 0 of %1019
    %1089 = pdl_interp.get_defining_op of %1088 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1089 : !pdl.operation -> ^bb2081, ^bb1
  ^bb2081:
    %1090 = pdl_interp.get_operand 0 of %1017
    %1091 = pdl_interp.get_defining_op of %1090 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1091 : !pdl.operation -> ^bb2082, ^bb1
  ^bb2082:
    %1092 = pdl_interp.get_value_type of %1015 : !pdl.type
    pdl_interp.are_equal %1086, %1092 : !pdl.type -> ^bb2083, ^bb1
  ^bb2083:
    pdl_interp.is_not_null %1090 : !pdl.value -> ^bb2084, ^bb1
  ^bb2084:
    %1093 = pdl_interp.get_operand 0 of %1089
    %1094 = pdl_interp.get_defining_op of %1093 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1094 : !pdl.operation -> ^bb2085, ^bb1
  ^bb2085:
    %1095 = pdl_interp.get_value_type of %1020 : !pdl.type
    pdl_interp.are_equal %1095, %1086 : !pdl.type -> ^bb2086, ^bb1
  ^bb2086:
    %1096 = pdl_interp.get_operand 0 of %1091
    %1097 = pdl_interp.get_defining_op of %1096 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1097 : !pdl.operation -> ^bb2087, ^bb1
  ^bb2087:
    pdl_interp.is_not_null %1088 : !pdl.value -> ^bb2088, ^bb1
  ^bb2088:
    pdl_interp.check_operation_name of %1089 is "arith.divf" -> ^bb2089, ^bb1
  ^bb2089:
    pdl_interp.check_operand_count of %1089 is 2 -> ^bb2090, ^bb1
  ^bb2090:
    pdl_interp.check_result_count of %1089 is 1 -> ^bb2091, ^bb1
  ^bb2091:
    %1098 = pdl_interp.get_result 0 of %1089
    pdl_interp.is_not_null %1098 : !pdl.value -> ^bb2092, ^bb1
  ^bb2092:
    pdl_interp.are_equal %1098, %1088 : !pdl.value -> ^bb2093, ^bb1
  ^bb2093:
    %1099 = pdl_interp.get_value_type of %1084 : !pdl.type
    pdl_interp.are_equal %1099, %1086 : !pdl.type -> ^bb2094, ^bb1
  ^bb2094:
    pdl_interp.check_operation_name of %1091 is "arith.divf" -> ^bb2095, ^bb1
  ^bb2095:
    pdl_interp.check_operand_count of %1091 is 2 -> ^bb2096, ^bb1
  ^bb2096:
    pdl_interp.check_result_count of %1091 is 1 -> ^bb2097, ^bb1
  ^bb2097:
    %1100 = pdl_interp.get_result 0 of %1091
    pdl_interp.is_not_null %1100 : !pdl.value -> ^bb2098, ^bb1
  ^bb2098:
    pdl_interp.are_equal %1100, %1090 : !pdl.value -> ^bb2099, ^bb1
  ^bb2099:
    %1101 = pdl_interp.get_operand 1 of %1091
    %1102 = pdl_interp.get_defining_op of %1101 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %1102 : !pdl.operation -> ^bb2100, ^bb1
  ^bb2100:
    %1103 = pdl_interp.get_operand 1 of %1089
    %1104 = pdl_interp.get_defining_op of %1103 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %1104 : !pdl.operation -> ^bb2101, ^bb1
  ^bb2101:
    pdl_interp.is_not_null %1093 : !pdl.value -> ^bb2102, ^bb1
  ^bb2102:
    pdl_interp.check_operation_name of %1094 is "arith.addf" -> ^bb2103, ^bb1
  ^bb2103:
    pdl_interp.check_operand_count of %1094 is 2 -> ^bb2104, ^bb1
  ^bb2104:
    pdl_interp.check_result_count of %1094 is 1 -> ^bb2105, ^bb1
  ^bb2105:
    %1105 = pdl_interp.get_result 0 of %1094
    pdl_interp.is_not_null %1105 : !pdl.value -> ^bb2106, ^bb1
  ^bb2106:
    pdl_interp.are_equal %1105, %1093 : !pdl.value -> ^bb2107, ^bb1
  ^bb2107:
    pdl_interp.is_not_null %1103 : !pdl.value -> ^bb2108, ^bb1
  ^bb2108:
    %1106 = pdl_interp.get_value_type of %1098 : !pdl.type
    pdl_interp.are_equal %1106, %1086 : !pdl.type -> ^bb2109, ^bb1
  ^bb2109:
    %1107 = pdl_interp.get_value_type of %1105 : !pdl.type
    pdl_interp.are_equal %1107, %1086 : !pdl.type -> ^bb2110, ^bb1
  ^bb2110:
    %1108 = pdl_interp.get_value_type of %1100 : !pdl.type
    pdl_interp.are_equal %1108, %1086 : !pdl.type -> ^bb2111, ^bb1
  ^bb2111:
    pdl_interp.is_not_null %1096 : !pdl.value -> ^bb2112, ^bb1
  ^bb2112:
    pdl_interp.check_operation_name of %1097 is "arith.subf" -> ^bb2113, ^bb1
  ^bb2113:
    pdl_interp.check_operand_count of %1097 is 2 -> ^bb2114, ^bb1
  ^bb2114:
    pdl_interp.check_result_count of %1097 is 1 -> ^bb2115, ^bb1
  ^bb2115:
    %1109 = pdl_interp.get_result 0 of %1097
    pdl_interp.is_not_null %1109 : !pdl.value -> ^bb2116, ^bb1
  ^bb2116:
    pdl_interp.are_equal %1109, %1096 : !pdl.value -> ^bb2117, ^bb1
  ^bb2117:
    pdl_interp.is_not_null %1101 : !pdl.value -> ^bb2118, ^bb1
  ^bb2118:
    pdl_interp.check_operation_name of %1102 is "arith.constant" -> ^bb2119, ^bb1
  ^bb2119:
    pdl_interp.check_operation_name of %1104 is "arith.constant" -> ^bb2120, ^bb1
  ^bb2120:
    pdl_interp.check_operand_count of %1102 is 0 -> ^bb2121, ^bb1
  ^bb2121:
    pdl_interp.check_operand_count of %1104 is 0 -> ^bb2122, ^bb1
  ^bb2122:
    pdl_interp.check_result_count of %1102 is 1 -> ^bb2123, ^bb1
  ^bb2123:
    pdl_interp.check_result_count of %1104 is 1 -> ^bb2124, ^bb1
  ^bb2124:
    %1110 = pdl_interp.get_operand 0 of %1097
    pdl_interp.is_not_null %1110 : !pdl.value -> ^bb2125, ^bb1
  ^bb2125:
    %1111 = pdl_interp.get_operand 1 of %1097
    pdl_interp.is_not_null %1111 : !pdl.value -> ^bb2126, ^bb1
  ^bb2126:
    %1112 = pdl_interp.get_operand 0 of %1094
    pdl_interp.are_equal %1110, %1112 : !pdl.value -> ^bb2127, ^bb1
  ^bb2127:
    %1113 = pdl_interp.get_operand 1 of %1094
    pdl_interp.are_equal %1111, %1113 : !pdl.value -> ^bb2128, ^bb1
  ^bb2128:
    %1114 = pdl_interp.get_attribute "value" of %1102
    pdl_interp.is_not_null %1114 : !pdl.attribute -> ^bb2129, ^bb1
  ^bb2129:
    %1115 = pdl_interp.get_attribute "value" of %1104
    pdl_interp.is_not_null %1115 : !pdl.attribute -> ^bb2130, ^bb1
  ^bb2130:
    pdl_interp.check_attribute %1114 is 2.000000e+00 : f32 -> ^bb2131, ^bb1
  ^bb2131:
    pdl_interp.check_attribute %1115 is 2.000000e+00 : f32 -> ^bb2132, ^bb1
  ^bb2132:
    %1116 = pdl_interp.get_result 0 of %1102
    pdl_interp.is_not_null %1116 : !pdl.value -> ^bb2133, ^bb1
  ^bb2133:
    %1117 = pdl_interp.get_result 0 of %1104
    pdl_interp.is_not_null %1117 : !pdl.value -> ^bb2134, ^bb1
  ^bb2134:
    pdl_interp.are_equal %1116, %1101 : !pdl.value -> ^bb2135, ^bb1
  ^bb2135:
    pdl_interp.are_equal %1117, %1103 : !pdl.value -> ^bb2136, ^bb1
  ^bb2136:
    %1118 = pdl_interp.get_value_type of %1110 : !pdl.type
    pdl_interp.are_equal %1118, %1086 : !pdl.type -> ^bb2137, ^bb1
  ^bb2137:
    %1119 = pdl_interp.get_value_type of %1111 : !pdl.type
    pdl_interp.are_equal %1119, %1086 : !pdl.type -> ^bb2138, ^bb1
  ^bb2138:
    %1120 = pdl_interp.get_value_type of %1109 : !pdl.type
    pdl_interp.are_equal %1120, %1086 : !pdl.type -> ^bb2139, ^bb1
  ^bb2139:
    %1121 = pdl_interp.get_value_type of %1116 : !pdl.type
    pdl_interp.are_equal %1121, %1086 : !pdl.type -> ^bb2140, ^bb1
  ^bb2140:
    %1122 = pdl_interp.get_value_type of %1117 : !pdl.type
    pdl_interp.are_equal %1122, %1086 : !pdl.type -> ^bb2141, ^bb1
  ^bb2141:
    pdl_interp.record_match @rewriters::@diff_cos_rev(%1110, %1111, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1
  ^bb1949:
    pdl_interp.check_operand_count of %1017 is 1 -> ^bb2142, ^bb1
  ^bb2142:
    pdl_interp.check_result_count of %1017 is 1 -> ^bb2143, ^bb1
  ^bb2143:
    %1123 = pdl_interp.get_result 0 of %1017
    pdl_interp.is_not_null %1123 : !pdl.value -> ^bb2144, ^bb1
  ^bb2144:
    pdl_interp.are_equal %1123, %1016 : !pdl.value -> ^bb2145, ^bb1
  ^bb2145:
    pdl_interp.check_operation_name of %1019 is "math.cos" -> ^bb2146, ^bb1
  ^bb2146:
    pdl_interp.check_operand_count of %1019 is 1 -> ^bb2147, ^bb1
  ^bb2147:
    pdl_interp.check_result_count of %1019 is 1 -> ^bb2148, ^bb1
  ^bb2148:
    %1124 = pdl_interp.get_result 0 of %1019
    pdl_interp.is_not_null %1124 : !pdl.value -> ^bb2149, ^bb1
  ^bb2149:
    pdl_interp.are_equal %1124, %1018 : !pdl.value -> ^bb2150, ^bb1
  ^bb2150:
    %1125 = pdl_interp.get_attribute "value" of %3
    pdl_interp.is_not_null %1125 : !pdl.attribute -> ^bb2151, ^bb1
  ^bb2151:
    pdl_interp.check_attribute %1125 is 2.000000e+00 : f32 -> ^bb2152, ^bb1
  ^bb2152:
    %1126 = pdl_interp.get_value_type of %1014 : !pdl.type
    %1127 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1126, %1127 : !pdl.type -> ^bb2153, ^bb1
  ^bb2153:
    pdl_interp.check_type %1126 is f32 -> ^bb2154, ^bb1
  ^bb2154:
    %1128 = pdl_interp.get_operand 0 of %1019
    %1129 = pdl_interp.get_defining_op of %1128 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1129 : !pdl.operation -> ^bb2155, ^bb1
  ^bb2155:
    %1130 = pdl_interp.get_operand 0 of %1017
    %1131 = pdl_interp.get_defining_op of %1130 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1131 : !pdl.operation -> ^bb2156, ^bb1
  ^bb2156:
    %1132 = pdl_interp.get_value_type of %1015 : !pdl.type
    pdl_interp.are_equal %1126, %1132 : !pdl.type -> ^bb2157, ^bb1
  ^bb2157:
    pdl_interp.is_not_null %1130 : !pdl.value -> ^bb2158, ^bb1
  ^bb2158:
    %1133 = pdl_interp.get_operand 0 of %1129
    %1134 = pdl_interp.get_defining_op of %1133 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1134 : !pdl.operation -> ^bb2159, ^bb1
  ^bb2159:
    %1135 = pdl_interp.get_value_type of %1123 : !pdl.type
    pdl_interp.are_equal %1135, %1126 : !pdl.type -> ^bb2160, ^bb1
  ^bb2160:
    %1136 = pdl_interp.get_operand 0 of %1131
    %1137 = pdl_interp.get_defining_op of %1136 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1137 : !pdl.operation -> ^bb2161, ^bb1
  ^bb2161:
    pdl_interp.is_not_null %1128 : !pdl.value -> ^bb2162, ^bb1
  ^bb2162:
    pdl_interp.check_operation_name of %1129 is "arith.divf" -> ^bb2163, ^bb1
  ^bb2163:
    pdl_interp.check_operand_count of %1129 is 2 -> ^bb2164, ^bb1
  ^bb2164:
    pdl_interp.check_result_count of %1129 is 1 -> ^bb2165, ^bb1
  ^bb2165:
    %1138 = pdl_interp.get_result 0 of %1129
    pdl_interp.is_not_null %1138 : !pdl.value -> ^bb2166, ^bb1
  ^bb2166:
    pdl_interp.are_equal %1138, %1128 : !pdl.value -> ^bb2167, ^bb1
  ^bb2167:
    %1139 = pdl_interp.get_value_type of %1124 : !pdl.type
    pdl_interp.are_equal %1139, %1126 : !pdl.type -> ^bb2168, ^bb1
  ^bb2168:
    pdl_interp.check_operation_name of %1131 is "arith.divf" -> ^bb2169, ^bb1
  ^bb2169:
    pdl_interp.check_operand_count of %1131 is 2 -> ^bb2170, ^bb1
  ^bb2170:
    pdl_interp.check_result_count of %1131 is 1 -> ^bb2171, ^bb1
  ^bb2171:
    %1140 = pdl_interp.get_result 0 of %1131
    pdl_interp.is_not_null %1140 : !pdl.value -> ^bb2172, ^bb1
  ^bb2172:
    pdl_interp.are_equal %1140, %1130 : !pdl.value -> ^bb2173, ^bb1
  ^bb2173:
    %1141 = pdl_interp.get_operand 1 of %1131
    %1142 = pdl_interp.get_defining_op of %1141 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %1142 : !pdl.operation -> ^bb2174, ^bb1
  ^bb2174:
    %1143 = pdl_interp.get_operand 1 of %1129
    %1144 = pdl_interp.get_defining_op of %1143 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %1144 : !pdl.operation -> ^bb2175, ^bb1
  ^bb2175:
    pdl_interp.is_not_null %1133 : !pdl.value -> ^bb2176, ^bb1
  ^bb2176:
    pdl_interp.check_operation_name of %1134 is "arith.subf" -> ^bb2177, ^bb1
  ^bb2177:
    pdl_interp.check_operand_count of %1134 is 2 -> ^bb2178, ^bb1
  ^bb2178:
    pdl_interp.check_result_count of %1134 is 1 -> ^bb2179, ^bb1
  ^bb2179:
    %1145 = pdl_interp.get_result 0 of %1134
    pdl_interp.is_not_null %1145 : !pdl.value -> ^bb2180, ^bb1
  ^bb2180:
    pdl_interp.are_equal %1145, %1133 : !pdl.value -> ^bb2181, ^bb1
  ^bb2181:
    pdl_interp.is_not_null %1143 : !pdl.value -> ^bb2182, ^bb1
  ^bb2182:
    %1146 = pdl_interp.get_value_type of %1138 : !pdl.type
    pdl_interp.are_equal %1146, %1126 : !pdl.type -> ^bb2183, ^bb1
  ^bb2183:
    %1147 = pdl_interp.get_value_type of %1145 : !pdl.type
    pdl_interp.are_equal %1147, %1126 : !pdl.type -> ^bb2184, ^bb1
  ^bb2184:
    %1148 = pdl_interp.get_value_type of %1140 : !pdl.type
    pdl_interp.are_equal %1148, %1126 : !pdl.type -> ^bb2185, ^bb1
  ^bb2185:
    pdl_interp.is_not_null %1136 : !pdl.value -> ^bb2186, ^bb1
  ^bb2186:
    pdl_interp.check_operation_name of %1137 is "arith.addf" -> ^bb2187, ^bb1
  ^bb2187:
    pdl_interp.check_operand_count of %1137 is 2 -> ^bb2188, ^bb1
  ^bb2188:
    pdl_interp.check_result_count of %1137 is 1 -> ^bb2189, ^bb1
  ^bb2189:
    %1149 = pdl_interp.get_result 0 of %1137
    pdl_interp.is_not_null %1149 : !pdl.value -> ^bb2190, ^bb1
  ^bb2190:
    pdl_interp.are_equal %1149, %1136 : !pdl.value -> ^bb2191, ^bb1
  ^bb2191:
    pdl_interp.is_not_null %1141 : !pdl.value -> ^bb2192, ^bb1
  ^bb2192:
    pdl_interp.check_operation_name of %1142 is "arith.constant" -> ^bb2193, ^bb1
  ^bb2193:
    pdl_interp.check_operation_name of %1144 is "arith.constant" -> ^bb2194, ^bb1
  ^bb2194:
    pdl_interp.check_operand_count of %1142 is 0 -> ^bb2195, ^bb1
  ^bb2195:
    pdl_interp.check_operand_count of %1144 is 0 -> ^bb2196, ^bb1
  ^bb2196:
    pdl_interp.check_result_count of %1142 is 1 -> ^bb2197, ^bb1
  ^bb2197:
    pdl_interp.check_result_count of %1144 is 1 -> ^bb2198, ^bb1
  ^bb2198:
    %1150 = pdl_interp.get_operand 0 of %1137
    pdl_interp.is_not_null %1150 : !pdl.value -> ^bb2199, ^bb1
  ^bb2199:
    %1151 = pdl_interp.get_operand 1 of %1137
    pdl_interp.is_not_null %1151 : !pdl.value -> ^bb2200, ^bb1
  ^bb2200:
    %1152 = pdl_interp.get_operand 0 of %1134
    pdl_interp.are_equal %1150, %1152 : !pdl.value -> ^bb2201, ^bb1
  ^bb2201:
    %1153 = pdl_interp.get_operand 1 of %1134
    pdl_interp.are_equal %1151, %1153 : !pdl.value -> ^bb2202, ^bb1
  ^bb2202:
    %1154 = pdl_interp.get_attribute "value" of %1142
    pdl_interp.is_not_null %1154 : !pdl.attribute -> ^bb2203, ^bb1
  ^bb2203:
    %1155 = pdl_interp.get_attribute "value" of %1144
    pdl_interp.is_not_null %1155 : !pdl.attribute -> ^bb2204, ^bb1
  ^bb2204:
    pdl_interp.check_attribute %1154 is 2.000000e+00 : f32 -> ^bb2205, ^bb1
  ^bb2205:
    pdl_interp.check_attribute %1155 is 2.000000e+00 : f32 -> ^bb2206, ^bb1
  ^bb2206:
    %1156 = pdl_interp.get_result 0 of %1142
    pdl_interp.is_not_null %1156 : !pdl.value -> ^bb2207, ^bb1
  ^bb2207:
    %1157 = pdl_interp.get_result 0 of %1144
    pdl_interp.is_not_null %1157 : !pdl.value -> ^bb2208, ^bb1
  ^bb2208:
    pdl_interp.are_equal %1156, %1141 : !pdl.value -> ^bb2209, ^bb1
  ^bb2209:
    pdl_interp.are_equal %1157, %1143 : !pdl.value -> ^bb2210, ^bb1
  ^bb2210:
    %1158 = pdl_interp.get_value_type of %1150 : !pdl.type
    pdl_interp.are_equal %1158, %1126 : !pdl.type -> ^bb2211, ^bb1
  ^bb2211:
    %1159 = pdl_interp.get_value_type of %1151 : !pdl.type
    pdl_interp.are_equal %1159, %1126 : !pdl.type -> ^bb2212, ^bb1
  ^bb2212:
    %1160 = pdl_interp.get_value_type of %1149 : !pdl.type
    pdl_interp.are_equal %1160, %1126 : !pdl.type -> ^bb2213, ^bb1
  ^bb2213:
    %1161 = pdl_interp.get_value_type of %1156 : !pdl.type
    pdl_interp.are_equal %1161, %1126 : !pdl.type -> ^bb2214, ^bb1
  ^bb2214:
    %1162 = pdl_interp.get_value_type of %1157 : !pdl.type
    pdl_interp.are_equal %1162, %1126 : !pdl.type -> ^bb2215, ^bb1
  ^bb2215:
    pdl_interp.record_match @rewriters::@sum_cos_rev(%1150, %1151, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1
  ^bb1950:
    pdl_interp.check_operand_count of %1017 is 1 -> ^bb2216, ^bb1
  ^bb2216:
    pdl_interp.check_result_count of %1017 is 1 -> ^bb2217, ^bb1
  ^bb2217:
    %1163 = pdl_interp.get_result 0 of %1017
    pdl_interp.is_not_null %1163 : !pdl.value -> ^bb2218, ^bb1
  ^bb2218:
    pdl_interp.are_equal %1163, %1016 : !pdl.value -> ^bb2219, ^bb1
  ^bb2219:
    pdl_interp.switch_operation_name of %1019 to ["math.sinh", "math.cosh"](^bb2220, ^bb2221) -> ^bb1
  ^bb2220:
    pdl_interp.check_operand_count of %1019 is 1 -> ^bb2222, ^bb1
  ^bb2222:
    pdl_interp.check_result_count of %1019 is 1 -> ^bb2223, ^bb1
  ^bb2223:
    %1164 = pdl_interp.get_result 0 of %1019
    pdl_interp.is_not_null %1164 : !pdl.value -> ^bb2224, ^bb1
  ^bb2224:
    pdl_interp.are_equal %1164, %1018 : !pdl.value -> ^bb2225, ^bb1
  ^bb2225:
    %1165 = pdl_interp.get_attribute "value" of %3
    pdl_interp.is_not_null %1165 : !pdl.attribute -> ^bb2226, ^bb1
  ^bb2226:
    pdl_interp.check_attribute %1165 is 2.000000e+00 : f32 -> ^bb2227, ^bb1
  ^bb2227:
    %1166 = pdl_interp.get_value_type of %1014 : !pdl.type
    %1167 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1166, %1167 : !pdl.type -> ^bb2228, ^bb1
  ^bb2228:
    pdl_interp.check_type %1166 is f32 -> ^bb2229, ^bb1
  ^bb2229:
    %1168 = pdl_interp.get_operand 0 of %1019
    %1169 = pdl_interp.get_defining_op of %1168 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1169 : !pdl.operation -> ^bb2230, ^bb1
  ^bb2230:
    %1170 = pdl_interp.get_operand 0 of %1017
    %1171 = pdl_interp.get_defining_op of %1170 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1171 : !pdl.operation -> ^bb2231, ^bb1
  ^bb2231:
    %1172 = pdl_interp.get_value_type of %1015 : !pdl.type
    pdl_interp.are_equal %1166, %1172 : !pdl.type -> ^bb2232, ^bb1
  ^bb2232:
    pdl_interp.is_not_null %1170 : !pdl.value -> ^bb2233, ^bb1
  ^bb2233:
    %1173 = pdl_interp.get_operand 0 of %1169
    %1174 = pdl_interp.get_defining_op of %1173 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1174 : !pdl.operation -> ^bb2234, ^bb1
  ^bb2234:
    %1175 = pdl_interp.get_value_type of %1163 : !pdl.type
    pdl_interp.are_equal %1175, %1166 : !pdl.type -> ^bb2235, ^bb1
  ^bb2235:
    %1176 = pdl_interp.get_operand 0 of %1171
    %1177 = pdl_interp.get_defining_op of %1176 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1177 : !pdl.operation -> ^bb2236, ^bb1
  ^bb2236:
    pdl_interp.is_not_null %1168 : !pdl.value -> ^bb2237, ^bb1
  ^bb2237:
    pdl_interp.check_operation_name of %1169 is "arith.divf" -> ^bb2238, ^bb1
  ^bb2238:
    pdl_interp.check_operand_count of %1169 is 2 -> ^bb2239, ^bb1
  ^bb2239:
    pdl_interp.check_result_count of %1169 is 1 -> ^bb2240, ^bb1
  ^bb2240:
    %1178 = pdl_interp.get_result 0 of %1169
    pdl_interp.is_not_null %1178 : !pdl.value -> ^bb2241, ^bb1
  ^bb2241:
    pdl_interp.are_equal %1178, %1168 : !pdl.value -> ^bb2242, ^bb1
  ^bb2242:
    %1179 = pdl_interp.get_value_type of %1164 : !pdl.type
    pdl_interp.are_equal %1179, %1166 : !pdl.type -> ^bb2243, ^bb1
  ^bb2243:
    pdl_interp.check_operation_name of %1171 is "arith.divf" -> ^bb2244, ^bb1
  ^bb2244:
    pdl_interp.check_operand_count of %1171 is 2 -> ^bb2245, ^bb1
  ^bb2245:
    pdl_interp.check_result_count of %1171 is 1 -> ^bb2246, ^bb1
  ^bb2246:
    %1180 = pdl_interp.get_result 0 of %1171
    pdl_interp.is_not_null %1180 : !pdl.value -> ^bb2247, ^bb1
  ^bb2247:
    pdl_interp.are_equal %1180, %1170 : !pdl.value -> ^bb2248, ^bb1
  ^bb2248:
    %1181 = pdl_interp.get_operand 1 of %1171
    %1182 = pdl_interp.get_defining_op of %1181 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %1182 : !pdl.operation -> ^bb2249, ^bb1
  ^bb2249:
    %1183 = pdl_interp.get_operand 1 of %1169
    %1184 = pdl_interp.get_defining_op of %1183 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %1184 : !pdl.operation -> ^bb2250, ^bb1
  ^bb2250:
    pdl_interp.is_not_null %1173 : !pdl.value -> ^bb2251, ^bb1
  ^bb2251:
    pdl_interp.check_operation_name of %1174 is "arith.subf" -> ^bb2252, ^bb1
  ^bb2252:
    pdl_interp.check_operand_count of %1174 is 2 -> ^bb2253, ^bb1
  ^bb2253:
    pdl_interp.check_result_count of %1174 is 1 -> ^bb2254, ^bb1
  ^bb2254:
    %1185 = pdl_interp.get_result 0 of %1174
    pdl_interp.is_not_null %1185 : !pdl.value -> ^bb2255, ^bb1
  ^bb2255:
    pdl_interp.are_equal %1185, %1173 : !pdl.value -> ^bb2256, ^bb1
  ^bb2256:
    pdl_interp.is_not_null %1183 : !pdl.value -> ^bb2257, ^bb1
  ^bb2257:
    %1186 = pdl_interp.get_value_type of %1178 : !pdl.type
    pdl_interp.are_equal %1186, %1166 : !pdl.type -> ^bb2258, ^bb1
  ^bb2258:
    %1187 = pdl_interp.get_value_type of %1185 : !pdl.type
    pdl_interp.are_equal %1187, %1166 : !pdl.type -> ^bb2259, ^bb1
  ^bb2259:
    %1188 = pdl_interp.get_value_type of %1180 : !pdl.type
    pdl_interp.are_equal %1188, %1166 : !pdl.type -> ^bb2260, ^bb1
  ^bb2260:
    pdl_interp.is_not_null %1176 : !pdl.value -> ^bb2261, ^bb1
  ^bb2261:
    pdl_interp.check_operation_name of %1177 is "arith.addf" -> ^bb2262, ^bb1
  ^bb2262:
    pdl_interp.check_operand_count of %1177 is 2 -> ^bb2263, ^bb1
  ^bb2263:
    pdl_interp.check_result_count of %1177 is 1 -> ^bb2264, ^bb1
  ^bb2264:
    %1189 = pdl_interp.get_result 0 of %1177
    pdl_interp.is_not_null %1189 : !pdl.value -> ^bb2265, ^bb1
  ^bb2265:
    pdl_interp.are_equal %1189, %1176 : !pdl.value -> ^bb2266, ^bb1
  ^bb2266:
    pdl_interp.is_not_null %1181 : !pdl.value -> ^bb2267, ^bb1
  ^bb2267:
    pdl_interp.check_operation_name of %1182 is "arith.constant" -> ^bb2268, ^bb1
  ^bb2268:
    pdl_interp.check_operation_name of %1184 is "arith.constant" -> ^bb2269, ^bb1
  ^bb2269:
    pdl_interp.check_operand_count of %1182 is 0 -> ^bb2270, ^bb1
  ^bb2270:
    pdl_interp.check_operand_count of %1184 is 0 -> ^bb2271, ^bb1
  ^bb2271:
    pdl_interp.check_result_count of %1182 is 1 -> ^bb2272, ^bb1
  ^bb2272:
    pdl_interp.check_result_count of %1184 is 1 -> ^bb2273, ^bb1
  ^bb2273:
    %1190 = pdl_interp.get_operand 0 of %1177
    pdl_interp.is_not_null %1190 : !pdl.value -> ^bb2274, ^bb1
  ^bb2274:
    %1191 = pdl_interp.get_operand 1 of %1177
    pdl_interp.is_not_null %1191 : !pdl.value -> ^bb2275, ^bb1
  ^bb2275:
    %1192 = pdl_interp.get_operand 0 of %1174
    pdl_interp.are_equal %1190, %1192 : !pdl.value -> ^bb2276, ^bb1
  ^bb2276:
    %1193 = pdl_interp.get_operand 1 of %1174
    pdl_interp.are_equal %1191, %1193 : !pdl.value -> ^bb2277, ^bb1
  ^bb2277:
    %1194 = pdl_interp.get_attribute "value" of %1182
    pdl_interp.is_not_null %1194 : !pdl.attribute -> ^bb2278, ^bb1
  ^bb2278:
    %1195 = pdl_interp.get_attribute "value" of %1184
    pdl_interp.is_not_null %1195 : !pdl.attribute -> ^bb2279, ^bb1
  ^bb2279:
    pdl_interp.check_attribute %1194 is 2.000000e+00 : f32 -> ^bb2280, ^bb1
  ^bb2280:
    pdl_interp.check_attribute %1195 is 2.000000e+00 : f32 -> ^bb2281, ^bb1
  ^bb2281:
    %1196 = pdl_interp.get_result 0 of %1182
    pdl_interp.is_not_null %1196 : !pdl.value -> ^bb2282, ^bb1
  ^bb2282:
    %1197 = pdl_interp.get_result 0 of %1184
    pdl_interp.is_not_null %1197 : !pdl.value -> ^bb2283, ^bb1
  ^bb2283:
    pdl_interp.are_equal %1196, %1181 : !pdl.value -> ^bb2284, ^bb1
  ^bb2284:
    pdl_interp.are_equal %1197, %1183 : !pdl.value -> ^bb2285, ^bb1
  ^bb2285:
    %1198 = pdl_interp.get_value_type of %1190 : !pdl.type
    pdl_interp.are_equal %1198, %1166 : !pdl.type -> ^bb2286, ^bb1
  ^bb2286:
    %1199 = pdl_interp.get_value_type of %1191 : !pdl.type
    pdl_interp.are_equal %1199, %1166 : !pdl.type -> ^bb2287, ^bb1
  ^bb2287:
    %1200 = pdl_interp.get_value_type of %1189 : !pdl.type
    pdl_interp.are_equal %1200, %1166 : !pdl.type -> ^bb2288, ^bb1
  ^bb2288:
    %1201 = pdl_interp.get_value_type of %1196 : !pdl.type
    pdl_interp.are_equal %1201, %1166 : !pdl.type -> ^bb2289, ^bb1
  ^bb2289:
    %1202 = pdl_interp.get_value_type of %1197 : !pdl.type
    pdl_interp.are_equal %1202, %1166 : !pdl.type -> ^bb2290, ^bb1
  ^bb2290:
    pdl_interp.record_match @rewriters::@diff_cosh_rev(%1190, %1191, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1
  ^bb2221:
    pdl_interp.check_operand_count of %1019 is 1 -> ^bb2291, ^bb1
  ^bb2291:
    pdl_interp.check_result_count of %1019 is 1 -> ^bb2292, ^bb1
  ^bb2292:
    %1203 = pdl_interp.get_result 0 of %1019
    pdl_interp.is_not_null %1203 : !pdl.value -> ^bb2293, ^bb1
  ^bb2293:
    pdl_interp.are_equal %1203, %1018 : !pdl.value -> ^bb2294, ^bb1
  ^bb2294:
    %1204 = pdl_interp.get_attribute "value" of %3
    pdl_interp.is_not_null %1204 : !pdl.attribute -> ^bb2295, ^bb1
  ^bb2295:
    pdl_interp.check_attribute %1204 is 2.000000e+00 : f32 -> ^bb2296, ^bb1
  ^bb2296:
    %1205 = pdl_interp.get_value_type of %1014 : !pdl.type
    %1206 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1205, %1206 : !pdl.type -> ^bb2297, ^bb1
  ^bb2297:
    pdl_interp.check_type %1205 is f32 -> ^bb2298, ^bb1
  ^bb2298:
    %1207 = pdl_interp.get_value_type of %1015 : !pdl.type
    pdl_interp.are_equal %1205, %1207 : !pdl.type -> ^bb2299, ^bb2300
  ^bb2300:
    %1208 = pdl_interp.get_operand 0 of %1019
    %1209 = pdl_interp.get_defining_op of %1208 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1209 : !pdl.operation -> ^bb2301, ^bb1
  ^bb2301:
    %1210 = pdl_interp.get_operand 0 of %1017
    %1211 = pdl_interp.get_defining_op of %1210 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1211 : !pdl.operation -> ^bb2302, ^bb1
  ^bb2302:
    %1212 = pdl_interp.get_value_type of %1015 : !pdl.type
    pdl_interp.are_equal %1205, %1212 : !pdl.type -> ^bb2303, ^bb1
  ^bb2303:
    pdl_interp.is_not_null %1210 : !pdl.value -> ^bb2304, ^bb1
  ^bb2304:
    %1213 = pdl_interp.get_operand 0 of %1209
    %1214 = pdl_interp.get_defining_op of %1213 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1214 : !pdl.operation -> ^bb2305, ^bb1
  ^bb2305:
    %1215 = pdl_interp.get_value_type of %1163 : !pdl.type
    pdl_interp.are_equal %1215, %1205 : !pdl.type -> ^bb2306, ^bb1
  ^bb2306:
    %1216 = pdl_interp.get_operand 0 of %1211
    %1217 = pdl_interp.get_defining_op of %1216 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1217 : !pdl.operation -> ^bb2307, ^bb1
  ^bb2307:
    pdl_interp.is_not_null %1208 : !pdl.value -> ^bb2308, ^bb1
  ^bb2308:
    pdl_interp.check_operation_name of %1209 is "arith.divf" -> ^bb2309, ^bb1
  ^bb2309:
    pdl_interp.check_operand_count of %1209 is 2 -> ^bb2310, ^bb1
  ^bb2310:
    pdl_interp.check_result_count of %1209 is 1 -> ^bb2311, ^bb1
  ^bb2311:
    %1218 = pdl_interp.get_result 0 of %1209
    pdl_interp.is_not_null %1218 : !pdl.value -> ^bb2312, ^bb1
  ^bb2312:
    pdl_interp.are_equal %1218, %1208 : !pdl.value -> ^bb2313, ^bb1
  ^bb2313:
    %1219 = pdl_interp.get_value_type of %1203 : !pdl.type
    pdl_interp.are_equal %1219, %1205 : !pdl.type -> ^bb2314, ^bb1
  ^bb2314:
    pdl_interp.check_operation_name of %1211 is "arith.divf" -> ^bb2315, ^bb1
  ^bb2315:
    pdl_interp.check_operand_count of %1211 is 2 -> ^bb2316, ^bb1
  ^bb2316:
    pdl_interp.check_result_count of %1211 is 1 -> ^bb2317, ^bb1
  ^bb2317:
    %1220 = pdl_interp.get_result 0 of %1211
    pdl_interp.is_not_null %1220 : !pdl.value -> ^bb2318, ^bb1
  ^bb2318:
    pdl_interp.are_equal %1220, %1210 : !pdl.value -> ^bb2319, ^bb1
  ^bb2319:
    %1221 = pdl_interp.get_operand 1 of %1211
    %1222 = pdl_interp.get_defining_op of %1221 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %1222 : !pdl.operation -> ^bb2320, ^bb1
  ^bb2320:
    %1223 = pdl_interp.get_operand 1 of %1209
    %1224 = pdl_interp.get_defining_op of %1223 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %1224 : !pdl.operation -> ^bb2321, ^bb1
  ^bb2321:
    pdl_interp.is_not_null %1213 : !pdl.value -> ^bb2322, ^bb1
  ^bb2322:
    pdl_interp.check_operation_name of %1214 is "arith.subf" -> ^bb2323, ^bb1
  ^bb2323:
    pdl_interp.check_operand_count of %1214 is 2 -> ^bb2324, ^bb1
  ^bb2324:
    pdl_interp.check_result_count of %1214 is 1 -> ^bb2325, ^bb1
  ^bb2325:
    %1225 = pdl_interp.get_result 0 of %1214
    pdl_interp.is_not_null %1225 : !pdl.value -> ^bb2326, ^bb1
  ^bb2326:
    pdl_interp.are_equal %1225, %1213 : !pdl.value -> ^bb2327, ^bb1
  ^bb2327:
    pdl_interp.is_not_null %1223 : !pdl.value -> ^bb2328, ^bb1
  ^bb2328:
    %1226 = pdl_interp.get_value_type of %1218 : !pdl.type
    pdl_interp.are_equal %1226, %1205 : !pdl.type -> ^bb2329, ^bb1
  ^bb2329:
    %1227 = pdl_interp.get_value_type of %1225 : !pdl.type
    pdl_interp.are_equal %1227, %1205 : !pdl.type -> ^bb2330, ^bb1
  ^bb2330:
    %1228 = pdl_interp.get_value_type of %1220 : !pdl.type
    pdl_interp.are_equal %1228, %1205 : !pdl.type -> ^bb2331, ^bb1
  ^bb2331:
    pdl_interp.is_not_null %1216 : !pdl.value -> ^bb2332, ^bb1
  ^bb2332:
    pdl_interp.check_operation_name of %1217 is "arith.addf" -> ^bb2333, ^bb1
  ^bb2333:
    pdl_interp.check_operand_count of %1217 is 2 -> ^bb2334, ^bb1
  ^bb2334:
    pdl_interp.check_result_count of %1217 is 1 -> ^bb2335, ^bb1
  ^bb2335:
    %1229 = pdl_interp.get_result 0 of %1217
    pdl_interp.is_not_null %1229 : !pdl.value -> ^bb2336, ^bb1
  ^bb2336:
    pdl_interp.are_equal %1229, %1216 : !pdl.value -> ^bb2337, ^bb1
  ^bb2337:
    pdl_interp.is_not_null %1221 : !pdl.value -> ^bb2338, ^bb1
  ^bb2338:
    pdl_interp.check_operation_name of %1222 is "arith.constant" -> ^bb2339, ^bb1
  ^bb2339:
    pdl_interp.check_operation_name of %1224 is "arith.constant" -> ^bb2340, ^bb1
  ^bb2340:
    pdl_interp.check_operand_count of %1222 is 0 -> ^bb2341, ^bb1
  ^bb2341:
    pdl_interp.check_operand_count of %1224 is 0 -> ^bb2342, ^bb1
  ^bb2342:
    pdl_interp.check_result_count of %1222 is 1 -> ^bb2343, ^bb1
  ^bb2343:
    pdl_interp.check_result_count of %1224 is 1 -> ^bb2344, ^bb1
  ^bb2344:
    %1230 = pdl_interp.get_operand 0 of %1217
    pdl_interp.is_not_null %1230 : !pdl.value -> ^bb2345, ^bb1
  ^bb2345:
    %1231 = pdl_interp.get_operand 1 of %1217
    pdl_interp.is_not_null %1231 : !pdl.value -> ^bb2346, ^bb1
  ^bb2346:
    %1232 = pdl_interp.get_operand 0 of %1214
    pdl_interp.are_equal %1230, %1232 : !pdl.value -> ^bb2347, ^bb1
  ^bb2347:
    %1233 = pdl_interp.get_operand 1 of %1214
    pdl_interp.are_equal %1231, %1233 : !pdl.value -> ^bb2348, ^bb1
  ^bb2348:
    %1234 = pdl_interp.get_attribute "value" of %1222
    pdl_interp.is_not_null %1234 : !pdl.attribute -> ^bb2349, ^bb1
  ^bb2349:
    %1235 = pdl_interp.get_attribute "value" of %1224
    pdl_interp.is_not_null %1235 : !pdl.attribute -> ^bb2350, ^bb1
  ^bb2350:
    pdl_interp.check_attribute %1234 is 2.000000e+00 : f32 -> ^bb2351, ^bb1
  ^bb2351:
    pdl_interp.check_attribute %1235 is 2.000000e+00 : f32 -> ^bb2352, ^bb1
  ^bb2352:
    %1236 = pdl_interp.get_result 0 of %1222
    pdl_interp.is_not_null %1236 : !pdl.value -> ^bb2353, ^bb1
  ^bb2353:
    %1237 = pdl_interp.get_result 0 of %1224
    pdl_interp.is_not_null %1237 : !pdl.value -> ^bb2354, ^bb1
  ^bb2354:
    pdl_interp.are_equal %1236, %1221 : !pdl.value -> ^bb2355, ^bb1
  ^bb2355:
    pdl_interp.are_equal %1237, %1223 : !pdl.value -> ^bb2356, ^bb1
  ^bb2356:
    %1238 = pdl_interp.get_value_type of %1230 : !pdl.type
    pdl_interp.are_equal %1238, %1205 : !pdl.type -> ^bb2357, ^bb1
  ^bb2357:
    %1239 = pdl_interp.get_value_type of %1231 : !pdl.type
    pdl_interp.are_equal %1239, %1205 : !pdl.type -> ^bb2358, ^bb1
  ^bb2358:
    %1240 = pdl_interp.get_value_type of %1229 : !pdl.type
    pdl_interp.are_equal %1240, %1205 : !pdl.type -> ^bb2359, ^bb1
  ^bb2359:
    %1241 = pdl_interp.get_value_type of %1236 : !pdl.type
    pdl_interp.are_equal %1241, %1205 : !pdl.type -> ^bb2360, ^bb1
  ^bb2360:
    %1242 = pdl_interp.get_value_type of %1237 : !pdl.type
    pdl_interp.are_equal %1242, %1205 : !pdl.type -> ^bb2361, ^bb1
  ^bb2361:
    pdl_interp.record_match @rewriters::@sum_sinh_rev(%1230, %1231, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1
  ^bb2299:
    %1243 = pdl_interp.get_operand 0 of %1017
    pdl_interp.is_not_null %1243 : !pdl.value -> ^bb2362, ^bb2300
  ^bb2362:
    %1244 = pdl_interp.get_value_type of %1163 : !pdl.type
    pdl_interp.are_equal %1244, %1205 : !pdl.type -> ^bb2363, ^bb2300
  ^bb2363:
    %1245 = pdl_interp.get_value_type of %1203 : !pdl.type
    pdl_interp.are_equal %1245, %1205 : !pdl.type -> ^bb2364, ^bb2300
  ^bb2364:
    %1246 = pdl_interp.get_operand 0 of %1019
    pdl_interp.are_equal %1243, %1246 : !pdl.value -> ^bb2365, ^bb2300
  ^bb2365:
    %1247 = pdl_interp.get_value_type of %1243 : !pdl.type
    pdl_interp.are_equal %1247, %1205 : !pdl.type -> ^bb2366, ^bb2300
  ^bb2366:
    pdl_interp.record_match @rewriters::@sinh_2_rev(%1243, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb2300
  ^bb1951:
    pdl_interp.check_operand_count of %1017 is 1 -> ^bb2367, ^bb1
  ^bb2367:
    pdl_interp.check_result_count of %1017 is 1 -> ^bb2368, ^bb1
  ^bb2368:
    %1248 = pdl_interp.get_result 0 of %1017
    pdl_interp.is_not_null %1248 : !pdl.value -> ^bb2369, ^bb1
  ^bb2369:
    pdl_interp.are_equal %1248, %1016 : !pdl.value -> ^bb2370, ^bb1
  ^bb2370:
    pdl_interp.switch_operation_name of %1019 to ["math.sinh", "math.cosh"](^bb2371, ^bb2372) -> ^bb1
  ^bb2371:
    pdl_interp.check_operand_count of %1019 is 1 -> ^bb2373, ^bb1
  ^bb2373:
    pdl_interp.check_result_count of %1019 is 1 -> ^bb2374, ^bb1
  ^bb2374:
    %1249 = pdl_interp.get_result 0 of %1019
    pdl_interp.is_not_null %1249 : !pdl.value -> ^bb2375, ^bb1
  ^bb2375:
    pdl_interp.are_equal %1249, %1018 : !pdl.value -> ^bb2376, ^bb1
  ^bb2376:
    %1250 = pdl_interp.get_attribute "value" of %3
    pdl_interp.is_not_null %1250 : !pdl.attribute -> ^bb2377, ^bb1
  ^bb2377:
    pdl_interp.check_attribute %1250 is 2.000000e+00 : f32 -> ^bb2378, ^bb1
  ^bb2378:
    %1251 = pdl_interp.get_value_type of %1014 : !pdl.type
    %1252 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1251, %1252 : !pdl.type -> ^bb2379, ^bb1
  ^bb2379:
    pdl_interp.check_type %1251 is f32 -> ^bb2380, ^bb1
  ^bb2380:
    %1253 = pdl_interp.get_operand 0 of %1019
    %1254 = pdl_interp.get_defining_op of %1253 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1254 : !pdl.operation -> ^bb2381, ^bb1
  ^bb2381:
    %1255 = pdl_interp.get_operand 0 of %1017
    %1256 = pdl_interp.get_defining_op of %1255 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1256 : !pdl.operation -> ^bb2382, ^bb1
  ^bb2382:
    %1257 = pdl_interp.get_value_type of %1015 : !pdl.type
    pdl_interp.are_equal %1251, %1257 : !pdl.type -> ^bb2383, ^bb1
  ^bb2383:
    pdl_interp.is_not_null %1255 : !pdl.value -> ^bb2384, ^bb1
  ^bb2384:
    %1258 = pdl_interp.get_operand 0 of %1254
    %1259 = pdl_interp.get_defining_op of %1258 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1259 : !pdl.operation -> ^bb2385, ^bb1
  ^bb2385:
    %1260 = pdl_interp.get_value_type of %1248 : !pdl.type
    pdl_interp.are_equal %1260, %1251 : !pdl.type -> ^bb2386, ^bb1
  ^bb2386:
    %1261 = pdl_interp.get_operand 0 of %1256
    %1262 = pdl_interp.get_defining_op of %1261 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1262 : !pdl.operation -> ^bb2387, ^bb1
  ^bb2387:
    pdl_interp.is_not_null %1253 : !pdl.value -> ^bb2388, ^bb1
  ^bb2388:
    pdl_interp.check_operation_name of %1254 is "arith.divf" -> ^bb2389, ^bb1
  ^bb2389:
    pdl_interp.check_operand_count of %1254 is 2 -> ^bb2390, ^bb1
  ^bb2390:
    pdl_interp.check_result_count of %1254 is 1 -> ^bb2391, ^bb1
  ^bb2391:
    %1263 = pdl_interp.get_result 0 of %1254
    pdl_interp.is_not_null %1263 : !pdl.value -> ^bb2392, ^bb1
  ^bb2392:
    pdl_interp.are_equal %1263, %1253 : !pdl.value -> ^bb2393, ^bb1
  ^bb2393:
    %1264 = pdl_interp.get_value_type of %1249 : !pdl.type
    pdl_interp.are_equal %1264, %1251 : !pdl.type -> ^bb2394, ^bb1
  ^bb2394:
    pdl_interp.check_operation_name of %1256 is "arith.divf" -> ^bb2395, ^bb1
  ^bb2395:
    pdl_interp.check_operand_count of %1256 is 2 -> ^bb2396, ^bb1
  ^bb2396:
    pdl_interp.check_result_count of %1256 is 1 -> ^bb2397, ^bb1
  ^bb2397:
    %1265 = pdl_interp.get_result 0 of %1256
    pdl_interp.is_not_null %1265 : !pdl.value -> ^bb2398, ^bb1
  ^bb2398:
    pdl_interp.are_equal %1265, %1255 : !pdl.value -> ^bb2399, ^bb1
  ^bb2399:
    %1266 = pdl_interp.get_operand 1 of %1256
    %1267 = pdl_interp.get_defining_op of %1266 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %1267 : !pdl.operation -> ^bb2400, ^bb1
  ^bb2400:
    %1268 = pdl_interp.get_operand 1 of %1254
    %1269 = pdl_interp.get_defining_op of %1268 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %1269 : !pdl.operation -> ^bb2401, ^bb1
  ^bb2401:
    pdl_interp.is_not_null %1258 : !pdl.value -> ^bb2402, ^bb1
  ^bb2402:
    pdl_interp.check_operation_name of %1259 is "arith.subf" -> ^bb2403, ^bb1
  ^bb2403:
    pdl_interp.check_operand_count of %1259 is 2 -> ^bb2404, ^bb1
  ^bb2404:
    pdl_interp.check_result_count of %1259 is 1 -> ^bb2405, ^bb1
  ^bb2405:
    %1270 = pdl_interp.get_result 0 of %1259
    pdl_interp.is_not_null %1270 : !pdl.value -> ^bb2406, ^bb1
  ^bb2406:
    pdl_interp.are_equal %1270, %1258 : !pdl.value -> ^bb2407, ^bb1
  ^bb2407:
    pdl_interp.is_not_null %1268 : !pdl.value -> ^bb2408, ^bb1
  ^bb2408:
    %1271 = pdl_interp.get_value_type of %1263 : !pdl.type
    pdl_interp.are_equal %1271, %1251 : !pdl.type -> ^bb2409, ^bb1
  ^bb2409:
    %1272 = pdl_interp.get_value_type of %1270 : !pdl.type
    pdl_interp.are_equal %1272, %1251 : !pdl.type -> ^bb2410, ^bb1
  ^bb2410:
    %1273 = pdl_interp.get_value_type of %1265 : !pdl.type
    pdl_interp.are_equal %1273, %1251 : !pdl.type -> ^bb2411, ^bb1
  ^bb2411:
    pdl_interp.is_not_null %1261 : !pdl.value -> ^bb2412, ^bb1
  ^bb2412:
    pdl_interp.check_operation_name of %1262 is "arith.addf" -> ^bb2413, ^bb1
  ^bb2413:
    pdl_interp.check_operand_count of %1262 is 2 -> ^bb2414, ^bb1
  ^bb2414:
    pdl_interp.check_result_count of %1262 is 1 -> ^bb2415, ^bb1
  ^bb2415:
    %1274 = pdl_interp.get_result 0 of %1262
    pdl_interp.is_not_null %1274 : !pdl.value -> ^bb2416, ^bb1
  ^bb2416:
    pdl_interp.are_equal %1274, %1261 : !pdl.value -> ^bb2417, ^bb1
  ^bb2417:
    pdl_interp.is_not_null %1266 : !pdl.value -> ^bb2418, ^bb1
  ^bb2418:
    pdl_interp.check_operation_name of %1267 is "arith.constant" -> ^bb2419, ^bb1
  ^bb2419:
    pdl_interp.check_operation_name of %1269 is "arith.constant" -> ^bb2420, ^bb1
  ^bb2420:
    pdl_interp.check_operand_count of %1267 is 0 -> ^bb2421, ^bb1
  ^bb2421:
    pdl_interp.check_operand_count of %1269 is 0 -> ^bb2422, ^bb1
  ^bb2422:
    pdl_interp.check_result_count of %1267 is 1 -> ^bb2423, ^bb1
  ^bb2423:
    pdl_interp.check_result_count of %1269 is 1 -> ^bb2424, ^bb1
  ^bb2424:
    %1275 = pdl_interp.get_operand 0 of %1262
    pdl_interp.is_not_null %1275 : !pdl.value -> ^bb2425, ^bb1
  ^bb2425:
    %1276 = pdl_interp.get_operand 1 of %1262
    pdl_interp.is_not_null %1276 : !pdl.value -> ^bb2426, ^bb1
  ^bb2426:
    %1277 = pdl_interp.get_operand 0 of %1259
    pdl_interp.are_equal %1275, %1277 : !pdl.value -> ^bb2427, ^bb1
  ^bb2427:
    %1278 = pdl_interp.get_operand 1 of %1259
    pdl_interp.are_equal %1276, %1278 : !pdl.value -> ^bb2428, ^bb1
  ^bb2428:
    %1279 = pdl_interp.get_attribute "value" of %1267
    pdl_interp.is_not_null %1279 : !pdl.attribute -> ^bb2429, ^bb1
  ^bb2429:
    %1280 = pdl_interp.get_attribute "value" of %1269
    pdl_interp.is_not_null %1280 : !pdl.attribute -> ^bb2430, ^bb1
  ^bb2430:
    pdl_interp.check_attribute %1279 is 2.000000e+00 : f32 -> ^bb2431, ^bb1
  ^bb2431:
    pdl_interp.check_attribute %1280 is 2.000000e+00 : f32 -> ^bb2432, ^bb1
  ^bb2432:
    %1281 = pdl_interp.get_result 0 of %1267
    pdl_interp.is_not_null %1281 : !pdl.value -> ^bb2433, ^bb1
  ^bb2433:
    %1282 = pdl_interp.get_result 0 of %1269
    pdl_interp.is_not_null %1282 : !pdl.value -> ^bb2434, ^bb1
  ^bb2434:
    pdl_interp.are_equal %1281, %1266 : !pdl.value -> ^bb2435, ^bb1
  ^bb2435:
    pdl_interp.are_equal %1282, %1268 : !pdl.value -> ^bb2436, ^bb1
  ^bb2436:
    %1283 = pdl_interp.get_value_type of %1275 : !pdl.type
    pdl_interp.are_equal %1283, %1251 : !pdl.type -> ^bb2437, ^bb1
  ^bb2437:
    %1284 = pdl_interp.get_value_type of %1276 : !pdl.type
    pdl_interp.are_equal %1284, %1251 : !pdl.type -> ^bb2438, ^bb1
  ^bb2438:
    %1285 = pdl_interp.get_value_type of %1274 : !pdl.type
    pdl_interp.are_equal %1285, %1251 : !pdl.type -> ^bb2439, ^bb1
  ^bb2439:
    %1286 = pdl_interp.get_value_type of %1281 : !pdl.type
    pdl_interp.are_equal %1286, %1251 : !pdl.type -> ^bb2440, ^bb1
  ^bb2440:
    %1287 = pdl_interp.get_value_type of %1282 : !pdl.type
    pdl_interp.are_equal %1287, %1251 : !pdl.type -> ^bb2441, ^bb1
  ^bb2441:
    pdl_interp.record_match @rewriters::@diff_sinh_rev(%1275, %1276, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1
  ^bb2372:
    pdl_interp.check_operand_count of %1019 is 1 -> ^bb2442, ^bb1
  ^bb2442:
    pdl_interp.check_result_count of %1019 is 1 -> ^bb2443, ^bb1
  ^bb2443:
    %1288 = pdl_interp.get_result 0 of %1019
    pdl_interp.is_not_null %1288 : !pdl.value -> ^bb2444, ^bb1
  ^bb2444:
    pdl_interp.are_equal %1288, %1018 : !pdl.value -> ^bb2445, ^bb1
  ^bb2445:
    %1289 = pdl_interp.get_attribute "value" of %3
    pdl_interp.is_not_null %1289 : !pdl.attribute -> ^bb2446, ^bb1
  ^bb2446:
    pdl_interp.check_attribute %1289 is 2.000000e+00 : f32 -> ^bb2447, ^bb1
  ^bb2447:
    %1290 = pdl_interp.get_value_type of %1014 : !pdl.type
    %1291 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1290, %1291 : !pdl.type -> ^bb2448, ^bb1
  ^bb2448:
    pdl_interp.check_type %1290 is f32 -> ^bb2449, ^bb1
  ^bb2449:
    %1292 = pdl_interp.get_operand 0 of %1019
    %1293 = pdl_interp.get_defining_op of %1292 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1293 : !pdl.operation -> ^bb2450, ^bb1
  ^bb2450:
    %1294 = pdl_interp.get_operand 0 of %1017
    %1295 = pdl_interp.get_defining_op of %1294 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1295 : !pdl.operation -> ^bb2451, ^bb1
  ^bb2451:
    %1296 = pdl_interp.get_value_type of %1015 : !pdl.type
    pdl_interp.are_equal %1290, %1296 : !pdl.type -> ^bb2452, ^bb1
  ^bb2452:
    pdl_interp.is_not_null %1294 : !pdl.value -> ^bb2453, ^bb1
  ^bb2453:
    %1297 = pdl_interp.get_operand 0 of %1293
    %1298 = pdl_interp.get_defining_op of %1297 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1298 : !pdl.operation -> ^bb2454, ^bb1
  ^bb2454:
    %1299 = pdl_interp.get_value_type of %1248 : !pdl.type
    pdl_interp.are_equal %1299, %1290 : !pdl.type -> ^bb2455, ^bb1
  ^bb2455:
    %1300 = pdl_interp.get_operand 0 of %1295
    %1301 = pdl_interp.get_defining_op of %1300 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1301 : !pdl.operation -> ^bb2456, ^bb1
  ^bb2456:
    pdl_interp.is_not_null %1292 : !pdl.value -> ^bb2457, ^bb1
  ^bb2457:
    pdl_interp.check_operation_name of %1293 is "arith.divf" -> ^bb2458, ^bb1
  ^bb2458:
    pdl_interp.check_operand_count of %1293 is 2 -> ^bb2459, ^bb1
  ^bb2459:
    pdl_interp.check_result_count of %1293 is 1 -> ^bb2460, ^bb1
  ^bb2460:
    %1302 = pdl_interp.get_result 0 of %1293
    pdl_interp.is_not_null %1302 : !pdl.value -> ^bb2461, ^bb1
  ^bb2461:
    pdl_interp.are_equal %1302, %1292 : !pdl.value -> ^bb2462, ^bb1
  ^bb2462:
    %1303 = pdl_interp.get_value_type of %1288 : !pdl.type
    pdl_interp.are_equal %1303, %1290 : !pdl.type -> ^bb2463, ^bb1
  ^bb2463:
    pdl_interp.check_operation_name of %1295 is "arith.divf" -> ^bb2464, ^bb1
  ^bb2464:
    pdl_interp.check_operand_count of %1295 is 2 -> ^bb2465, ^bb1
  ^bb2465:
    pdl_interp.check_result_count of %1295 is 1 -> ^bb2466, ^bb1
  ^bb2466:
    %1304 = pdl_interp.get_result 0 of %1295
    pdl_interp.is_not_null %1304 : !pdl.value -> ^bb2467, ^bb1
  ^bb2467:
    pdl_interp.are_equal %1304, %1294 : !pdl.value -> ^bb2468, ^bb1
  ^bb2468:
    %1305 = pdl_interp.get_operand 1 of %1295
    %1306 = pdl_interp.get_defining_op of %1305 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %1306 : !pdl.operation -> ^bb2469, ^bb1
  ^bb2469:
    %1307 = pdl_interp.get_operand 1 of %1293
    %1308 = pdl_interp.get_defining_op of %1307 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %1308 : !pdl.operation -> ^bb2470, ^bb1
  ^bb2470:
    pdl_interp.is_not_null %1297 : !pdl.value -> ^bb2471, ^bb1
  ^bb2471:
    pdl_interp.check_operation_name of %1298 is "arith.subf" -> ^bb2472, ^bb1
  ^bb2472:
    pdl_interp.check_operand_count of %1298 is 2 -> ^bb2473, ^bb1
  ^bb2473:
    pdl_interp.check_result_count of %1298 is 1 -> ^bb2474, ^bb1
  ^bb2474:
    %1309 = pdl_interp.get_result 0 of %1298
    pdl_interp.is_not_null %1309 : !pdl.value -> ^bb2475, ^bb1
  ^bb2475:
    pdl_interp.are_equal %1309, %1297 : !pdl.value -> ^bb2476, ^bb1
  ^bb2476:
    pdl_interp.is_not_null %1307 : !pdl.value -> ^bb2477, ^bb1
  ^bb2477:
    %1310 = pdl_interp.get_value_type of %1302 : !pdl.type
    pdl_interp.are_equal %1310, %1290 : !pdl.type -> ^bb2478, ^bb1
  ^bb2478:
    %1311 = pdl_interp.get_value_type of %1309 : !pdl.type
    pdl_interp.are_equal %1311, %1290 : !pdl.type -> ^bb2479, ^bb1
  ^bb2479:
    %1312 = pdl_interp.get_value_type of %1304 : !pdl.type
    pdl_interp.are_equal %1312, %1290 : !pdl.type -> ^bb2480, ^bb1
  ^bb2480:
    pdl_interp.is_not_null %1300 : !pdl.value -> ^bb2481, ^bb1
  ^bb2481:
    pdl_interp.check_operation_name of %1301 is "arith.addf" -> ^bb2482, ^bb1
  ^bb2482:
    pdl_interp.check_operand_count of %1301 is 2 -> ^bb2483, ^bb1
  ^bb2483:
    pdl_interp.check_result_count of %1301 is 1 -> ^bb2484, ^bb1
  ^bb2484:
    %1313 = pdl_interp.get_result 0 of %1301
    pdl_interp.is_not_null %1313 : !pdl.value -> ^bb2485, ^bb1
  ^bb2485:
    pdl_interp.are_equal %1313, %1300 : !pdl.value -> ^bb2486, ^bb1
  ^bb2486:
    pdl_interp.is_not_null %1305 : !pdl.value -> ^bb2487, ^bb1
  ^bb2487:
    pdl_interp.check_operation_name of %1306 is "arith.constant" -> ^bb2488, ^bb1
  ^bb2488:
    pdl_interp.check_operation_name of %1308 is "arith.constant" -> ^bb2489, ^bb1
  ^bb2489:
    pdl_interp.check_operand_count of %1306 is 0 -> ^bb2490, ^bb1
  ^bb2490:
    pdl_interp.check_operand_count of %1308 is 0 -> ^bb2491, ^bb1
  ^bb2491:
    pdl_interp.check_result_count of %1306 is 1 -> ^bb2492, ^bb1
  ^bb2492:
    pdl_interp.check_result_count of %1308 is 1 -> ^bb2493, ^bb1
  ^bb2493:
    %1314 = pdl_interp.get_operand 0 of %1301
    pdl_interp.is_not_null %1314 : !pdl.value -> ^bb2494, ^bb1
  ^bb2494:
    %1315 = pdl_interp.get_operand 1 of %1301
    pdl_interp.is_not_null %1315 : !pdl.value -> ^bb2495, ^bb1
  ^bb2495:
    %1316 = pdl_interp.get_operand 0 of %1298
    pdl_interp.are_equal %1314, %1316 : !pdl.value -> ^bb2496, ^bb1
  ^bb2496:
    %1317 = pdl_interp.get_operand 1 of %1298
    pdl_interp.are_equal %1315, %1317 : !pdl.value -> ^bb2497, ^bb1
  ^bb2497:
    %1318 = pdl_interp.get_attribute "value" of %1306
    pdl_interp.is_not_null %1318 : !pdl.attribute -> ^bb2498, ^bb1
  ^bb2498:
    %1319 = pdl_interp.get_attribute "value" of %1308
    pdl_interp.is_not_null %1319 : !pdl.attribute -> ^bb2499, ^bb1
  ^bb2499:
    pdl_interp.check_attribute %1318 is 2.000000e+00 : f32 -> ^bb2500, ^bb1
  ^bb2500:
    pdl_interp.check_attribute %1319 is 2.000000e+00 : f32 -> ^bb2501, ^bb1
  ^bb2501:
    %1320 = pdl_interp.get_result 0 of %1306
    pdl_interp.is_not_null %1320 : !pdl.value -> ^bb2502, ^bb1
  ^bb2502:
    %1321 = pdl_interp.get_result 0 of %1308
    pdl_interp.is_not_null %1321 : !pdl.value -> ^bb2503, ^bb1
  ^bb2503:
    pdl_interp.are_equal %1320, %1305 : !pdl.value -> ^bb2504, ^bb1
  ^bb2504:
    pdl_interp.are_equal %1321, %1307 : !pdl.value -> ^bb2505, ^bb1
  ^bb2505:
    %1322 = pdl_interp.get_value_type of %1314 : !pdl.type
    pdl_interp.are_equal %1322, %1290 : !pdl.type -> ^bb2506, ^bb1
  ^bb2506:
    %1323 = pdl_interp.get_value_type of %1315 : !pdl.type
    pdl_interp.are_equal %1323, %1290 : !pdl.type -> ^bb2507, ^bb1
  ^bb2507:
    %1324 = pdl_interp.get_value_type of %1313 : !pdl.type
    pdl_interp.are_equal %1324, %1290 : !pdl.type -> ^bb2508, ^bb1
  ^bb2508:
    %1325 = pdl_interp.get_value_type of %1320 : !pdl.type
    pdl_interp.are_equal %1325, %1290 : !pdl.type -> ^bb2509, ^bb1
  ^bb2509:
    %1326 = pdl_interp.get_value_type of %1321 : !pdl.type
    pdl_interp.are_equal %1326, %1290 : !pdl.type -> ^bb2510, ^bb1
  ^bb2510:
    pdl_interp.record_match @rewriters::@sum_cosh_rev(%1314, %1315, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1
  ^bb1937:
    pdl_interp.check_operand_count of %768 is 1 -> ^bb2511, ^bb1
  ^bb2511:
    pdl_interp.check_result_count of %768 is 1 -> ^bb2512, ^bb1
  ^bb2512:
    %1327 = pdl_interp.get_result 0 of %768
    pdl_interp.is_not_null %1327 : !pdl.value -> ^bb2513, ^bb1
  ^bb2513:
    pdl_interp.are_equal %1327, %767 : !pdl.value -> ^bb2514, ^bb1
  ^bb2514:
    %1328 = pdl_interp.get_operand 0 of %768
    pdl_interp.is_not_null %1328 : !pdl.value -> ^bb2515, ^bb1
  ^bb2515:
    %1329 = pdl_interp.get_attribute "value" of %3
    pdl_interp.is_not_null %1329 : !pdl.attribute -> ^bb2516, ^bb1
  ^bb2516:
    pdl_interp.check_attribute %1329 is 2.000000e+00 : f32 -> ^bb2517, ^bb1
  ^bb2517:
    %1330 = pdl_interp.get_value_type of %1014 : !pdl.type
    %1331 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1330, %1331 : !pdl.type -> ^bb2518, ^bb1
  ^bb2518:
    pdl_interp.check_type %1330 is f32 -> ^bb2519, ^bb1
  ^bb2519:
    %1332 = pdl_interp.get_value_type of %1327 : !pdl.type
    pdl_interp.are_equal %1330, %1332 : !pdl.type -> ^bb2520, ^bb1
  ^bb2520:
    %1333 = pdl_interp.get_value_type of %1328 : !pdl.type
    pdl_interp.are_equal %1330, %1333 : !pdl.type -> ^bb2521, ^bb1
  ^bb2521:
    pdl_interp.record_match @rewriters::@sinh_undef_rev(%1328, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1
  ^bb1938:
    pdl_interp.check_operand_count of %768 is 1 -> ^bb2522, ^bb1
  ^bb2522:
    pdl_interp.check_result_count of %768 is 1 -> ^bb2523, ^bb1
  ^bb2523:
    %1334 = pdl_interp.get_result 0 of %768
    pdl_interp.is_not_null %1334 : !pdl.value -> ^bb2524, ^bb1
  ^bb2524:
    pdl_interp.are_equal %1334, %767 : !pdl.value -> ^bb2525, ^bb1
  ^bb2525:
    %1335 = pdl_interp.get_operand 0 of %768
    pdl_interp.is_not_null %1335 : !pdl.value -> ^bb2526, ^bb1
  ^bb2526:
    %1336 = pdl_interp.get_attribute "value" of %3
    pdl_interp.is_not_null %1336 : !pdl.attribute -> ^bb2527, ^bb1
  ^bb2527:
    pdl_interp.check_attribute %1336 is 2.000000e+00 : f32 -> ^bb2528, ^bb1
  ^bb2528:
    %1337 = pdl_interp.get_value_type of %1014 : !pdl.type
    %1338 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1337, %1338 : !pdl.type -> ^bb2529, ^bb1
  ^bb2529:
    pdl_interp.check_type %1337 is f32 -> ^bb2530, ^bb1
  ^bb2530:
    %1339 = pdl_interp.get_value_type of %1334 : !pdl.type
    pdl_interp.are_equal %1337, %1339 : !pdl.type -> ^bb2531, ^bb1
  ^bb2531:
    %1340 = pdl_interp.get_value_type of %1335 : !pdl.type
    pdl_interp.are_equal %1337, %1340 : !pdl.type -> ^bb2532, ^bb1
  ^bb2532:
    pdl_interp.record_match @rewriters::@cosh_undef_rev(%1335, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1
  ^bb1939:
    pdl_interp.check_operand_count of %768 is 1 -> ^bb2533, ^bb1
  ^bb2533:
    pdl_interp.check_result_count of %768 is 1 -> ^bb2534, ^bb1
  ^bb2534:
    %1341 = pdl_interp.get_result 0 of %768
    pdl_interp.is_not_null %1341 : !pdl.value -> ^bb2535, ^bb1
  ^bb2535:
    pdl_interp.are_equal %1341, %767 : !pdl.value -> ^bb2536, ^bb1
  ^bb2536:
    %1342 = pdl_interp.get_operand 0 of %768
    pdl_interp.is_not_null %1342 : !pdl.value -> ^bb2537, ^bb1
  ^bb2537:
    %1343 = pdl_interp.get_attribute "value" of %3
    pdl_interp.is_not_null %1343 : !pdl.attribute -> ^bb2538, ^bb1
  ^bb2538:
    pdl_interp.check_attribute %1343 is 2.000000e+00 : f32 -> ^bb2539, ^bb1
  ^bb2539:
    %1344 = pdl_interp.get_value_type of %1014 : !pdl.type
    %1345 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1344, %1345 : !pdl.type -> ^bb2540, ^bb1
  ^bb2540:
    pdl_interp.check_type %1344 is f32 -> ^bb2541, ^bb1
  ^bb2541:
    %1346 = pdl_interp.get_value_type of %1341 : !pdl.type
    pdl_interp.are_equal %1344, %1346 : !pdl.type -> ^bb2542, ^bb1
  ^bb2542:
    %1347 = pdl_interp.get_value_type of %1342 : !pdl.type
    pdl_interp.are_equal %1344, %1347 : !pdl.type -> ^bb2543, ^bb1
  ^bb2543:
    pdl_interp.record_match @rewriters::@acosh_2_rev(%1342, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1
  ^bb1478:
    pdl_interp.switch_operation_name of %3 to ["arith.mulf", "arith.divf", "arith.constant", "arith.negf", "arith.addf", "arith.subf", "math.powf"](^bb2544, ^bb2545, ^bb2546, ^bb2547, ^bb2548, ^bb2549, ^bb2550) -> ^bb1479
  ^bb2544:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb2551, ^bb1479
  ^bb2551:
    pdl_interp.check_result_count of %3 is 1 -> ^bb2552, ^bb1479
  ^bb2552:
    %1348 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %1348 : !pdl.value -> ^bb2553, ^bb1479
  ^bb2553:
    pdl_interp.are_equal %1348, %2 : !pdl.value -> ^bb2554, ^bb1479
  ^bb2554:
    %1349 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %1349 : !pdl.value -> ^bb2555, ^bb1479
  ^bb2555:
    %1350 = pdl_interp.get_operand 1 of %0
    pdl_interp.is_not_null %1350 : !pdl.value -> ^bb2556, ^bb2557
  ^bb2557:
    %1351 = pdl_interp.get_value_type of %1349 : !pdl.type
    %1352 = pdl_interp.get_value_type of %1348 : !pdl.type
    pdl_interp.are_equal %1351, %1352 : !pdl.type -> ^bb2558, ^bb1479
  ^bb2558:
    %1353 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1351, %1353 : !pdl.type -> ^bb2559, ^bb1479
  ^bb2559:
    pdl_interp.check_type %1351 is f32 -> ^bb2560, ^bb1479
  ^bb2560:
    %1354 = pdl_interp.get_operand 1 of %3
    pdl_interp.are_equal %1349, %1354 : !pdl.value -> ^bb2561, ^bb1479
  ^bb2561:
    %1355 = pdl_interp.get_operand 1 of %0
    pdl_interp.are_equal %1349, %1355 : !pdl.value -> ^bb2562, ^bb1479
  ^bb2562:
    pdl_interp.record_match @rewriters::@pow3(%1349, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1479
  ^bb2556:
    %1356 = pdl_interp.get_value_type of %1349 : !pdl.type
    %1357 = pdl_interp.get_value_type of %1348 : !pdl.type
    pdl_interp.are_equal %1356, %1357 : !pdl.type -> ^bb2563, ^bb2557
  ^bb2563:
    %1358 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1356, %1358 : !pdl.type -> ^bb2564, ^bb2557
  ^bb2564:
    pdl_interp.check_type %1356 is f32 -> ^bb2565, ^bb2557
  ^bb2565:
    %1359 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %1359 : !pdl.value -> ^bb2566, ^bb2557
  ^bb2566:
    %1360 = pdl_interp.get_value_type of %1359 : !pdl.type
    pdl_interp.are_equal %1356, %1360 : !pdl.type -> ^bb2567, ^bb2557
  ^bb2567:
    %1361 = pdl_interp.get_value_type of %1350 : !pdl.type
    pdl_interp.are_equal %1356, %1361 : !pdl.type -> ^bb2568, ^bb2557
  ^bb2568:
    pdl_interp.record_match @rewriters::@associate_mullmul(%1359, %1350, %1349, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb2557
  ^bb2545:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb2569, ^bb1479
  ^bb2569:
    pdl_interp.check_result_count of %3 is 1 -> ^bb2570, ^bb1479
  ^bb2570:
    %1362 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %1362 : !pdl.value -> ^bb2571, ^bb1479
  ^bb2571:
    pdl_interp.are_equal %1362, %2 : !pdl.value -> ^bb2572, ^bb1479
  ^bb2572:
    %1363 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %1363 : !pdl.value -> ^bb2573, ^bb1479
  ^bb2573:
    %1364 = pdl_interp.get_operand 1 of %0
    pdl_interp.is_not_null %1364 : !pdl.value -> ^bb2574, ^bb2575
  ^bb2575:
    %1365 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %1365 : !pdl.value -> ^bb2576, ^bb1479
  ^bb2576:
    %1366 = pdl_interp.get_defining_op of %1363 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1366 : !pdl.operation -> ^bb2577, ^bb1479
  ^bb2577:
    pdl_interp.check_operation_name of %1366 is "arith.constant" -> ^bb2578, ^bb1479
  ^bb2578:
    pdl_interp.check_operand_count of %1366 is 0 -> ^bb2579, ^bb1479
  ^bb2579:
    pdl_interp.check_result_count of %1366 is 1 -> ^bb2580, ^bb1479
  ^bb2580:
    %1367 = pdl_interp.get_result 0 of %1366
    pdl_interp.is_not_null %1367 : !pdl.value -> ^bb2581, ^bb1479
  ^bb2581:
    pdl_interp.are_equal %1367, %1363 : !pdl.value -> ^bb2582, ^bb1479
  ^bb2582:
    %1368 = pdl_interp.get_attribute "value" of %1366
    pdl_interp.is_not_null %1368 : !pdl.attribute -> ^bb2583, ^bb1479
  ^bb2583:
    pdl_interp.check_attribute %1368 is 1.000000e+00 : f32 -> ^bb2584, ^bb1479
  ^bb2584:
    %1369 = pdl_interp.get_value_type of %1367 : !pdl.type
    %1370 = pdl_interp.get_value_type of %1362 : !pdl.type
    pdl_interp.are_equal %1369, %1370 : !pdl.type -> ^bb2585, ^bb1479
  ^bb2585:
    %1371 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1369, %1371 : !pdl.type -> ^bb2586, ^bb1479
  ^bb2586:
    pdl_interp.check_type %1369 is f32 -> ^bb2587, ^bb1479
  ^bb2587:
    %1372 = pdl_interp.get_value_type of %1365 : !pdl.type
    pdl_interp.are_equal %1369, %1372 : !pdl.type -> ^bb2588, ^bb1479
  ^bb2588:
    %1373 = pdl_interp.get_operand 1 of %0
    pdl_interp.are_equal %1365, %1373 : !pdl.value -> ^bb2589, ^bb1479
  ^bb2589:
    pdl_interp.record_match @rewriters::@lft_mult_inverse(%0 : !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1479
  ^bb2574:
    %1374 = pdl_interp.get_value_type of %1363 : !pdl.type
    %1375 = pdl_interp.get_value_type of %1362 : !pdl.type
    pdl_interp.are_equal %1374, %1375 : !pdl.type -> ^bb2590, ^bb2575
  ^bb2590:
    %1376 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1374, %1376 : !pdl.type -> ^bb2591, ^bb2575
  ^bb2591:
    pdl_interp.check_type %1374 is f32 -> ^bb2592, ^bb2575
  ^bb2592:
    %1377 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %1377 : !pdl.value -> ^bb2593, ^bb2575
  ^bb2593:
    %1378 = pdl_interp.get_value_type of %1377 : !pdl.type
    pdl_interp.are_equal %1374, %1378 : !pdl.type -> ^bb2594, ^bb2575
  ^bb2594:
    %1379 = pdl_interp.get_value_type of %1364 : !pdl.type
    pdl_interp.are_equal %1374, %1379 : !pdl.type -> ^bb2595, ^bb2575
  ^bb2595:
    pdl_interp.record_match @rewriters::@associate_mulldiv(%1363, %1364, %1377, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb2575
  ^bb2546:
    pdl_interp.check_operand_count of %3 is 0 -> ^bb2596, ^bb1479
  ^bb2596:
    pdl_interp.check_result_count of %3 is 1 -> ^bb2597, ^bb1479
  ^bb2597:
    %1380 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %1380 : !pdl.value -> ^bb2598, ^bb1479
  ^bb2598:
    pdl_interp.are_equal %1380, %2 : !pdl.value -> ^bb2599, ^bb1479
  ^bb2599:
    %1381 = pdl_interp.get_operand 1 of %0
    pdl_interp.is_not_null %1381 : !pdl.value -> ^bb2600, ^bb1479
  ^bb2600:
    %1382 = pdl_interp.get_attribute "value" of %3
    pdl_interp.is_not_null %1382 : !pdl.attribute -> ^bb2601, ^bb1479
  ^bb2601:
    pdl_interp.switch_attribute %1382 to [0.000000e+00 : f32, 1.000000e+00 : f32, -1.000000e+00 : f32, 2.000000e+00 : f32](^bb2602, ^bb2603, ^bb2604, ^bb2605) -> ^bb1479
  ^bb2602:
    %1383 = pdl_interp.get_value_type of %1380 : !pdl.type
    %1384 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1383, %1384 : !pdl.type -> ^bb2606, ^bb1479
  ^bb2606:
    pdl_interp.check_type %1383 is f32 -> ^bb2607, ^bb1479
  ^bb2607:
    %1385 = pdl_interp.get_value_type of %1381 : !pdl.type
    pdl_interp.are_equal %1383, %1385 : !pdl.type -> ^bb2608, ^bb1479
  ^bb2608:
    pdl_interp.record_match @rewriters::@mul0_lft(%0 : !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1479
  ^bb2603:
    %1386 = pdl_interp.get_value_type of %1380 : !pdl.type
    %1387 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1386, %1387 : !pdl.type -> ^bb2609, ^bb1479
  ^bb2609:
    pdl_interp.check_type %1386 is f32 -> ^bb2610, ^bb1479
  ^bb2610:
    %1388 = pdl_interp.get_value_type of %1381 : !pdl.type
    pdl_interp.are_equal %1386, %1388 : !pdl.type -> ^bb2611, ^bb1479
  ^bb2611:
    pdl_interp.record_match @rewriters::@mul_lft_identity(%1381, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1479
  ^bb2604:
    %1389 = pdl_interp.get_value_type of %1380 : !pdl.type
    %1390 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1389, %1390 : !pdl.type -> ^bb2612, ^bb1479
  ^bb2612:
    pdl_interp.check_type %1389 is f32 -> ^bb2613, ^bb1479
  ^bb2613:
    %1391 = pdl_interp.get_value_type of %1381 : !pdl.type
    pdl_interp.are_equal %1389, %1391 : !pdl.type -> ^bb2614, ^bb1479
  ^bb2614:
    pdl_interp.record_match @rewriters::@mul_1_neg(%1381, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1479
  ^bb2605:
    %1392 = pdl_interp.get_value_type of %1380 : !pdl.type
    %1393 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1392, %1393 : !pdl.type -> ^bb2615, ^bb1479
  ^bb2615:
    pdl_interp.check_type %1392 is f32 -> ^bb2616, ^bb1479
  ^bb2616:
    %1394 = pdl_interp.get_value_type of %1381 : !pdl.type
    pdl_interp.are_equal %1392, %1394 : !pdl.type -> ^bb2617, ^bb1479
  ^bb2617:
    pdl_interp.record_match @rewriters::@count_2_rev(%1381, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1479
  ^bb2547:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb2618, ^bb1479
  ^bb2618:
    pdl_interp.check_result_count of %3 is 1 -> ^bb2619, ^bb1479
  ^bb2619:
    %1395 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %1395 : !pdl.value -> ^bb2620, ^bb1479
  ^bb2620:
    pdl_interp.are_equal %1395, %2 : !pdl.value -> ^bb2621, ^bb1479
  ^bb2621:
    %1396 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %1396 : !pdl.value -> ^bb2622, ^bb1479
  ^bb2622:
    %1397 = pdl_interp.get_operand 1 of %0
    pdl_interp.is_not_null %1397 : !pdl.value -> ^bb2623, ^bb1479
  ^bb2623:
    %1398 = pdl_interp.get_value_type of %1396 : !pdl.type
    %1399 = pdl_interp.get_value_type of %1395 : !pdl.type
    pdl_interp.are_equal %1398, %1399 : !pdl.type -> ^bb2624, ^bb1479
  ^bb2624:
    %1400 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1398, %1400 : !pdl.type -> ^bb2625, ^bb1479
  ^bb2625:
    pdl_interp.check_type %1398 is f32 -> ^bb2626, ^bb1479
  ^bb2626:
    %1401 = pdl_interp.get_value_type of %1397 : !pdl.type
    pdl_interp.are_equal %1398, %1401 : !pdl.type -> ^bb2627, ^bb1479
  ^bb2627:
    pdl_interp.record_match @rewriters::@distribute_lft_neg_out(%1396, %1397, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1479
  ^bb2548:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb2628, ^bb1479
  ^bb2628:
    pdl_interp.check_result_count of %3 is 1 -> ^bb2629, ^bb1479
  ^bb2629:
    %1402 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %1402 : !pdl.value -> ^bb2630, ^bb1479
  ^bb2630:
    pdl_interp.are_equal %1402, %2 : !pdl.value -> ^bb2631, ^bb1479
  ^bb2631:
    %1403 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %1403 : !pdl.value -> ^bb2632, ^bb1479
  ^bb2632:
    %1404 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %1404 : !pdl.value -> ^bb2633, ^bb1479
  ^bb2633:
    %1405 = pdl_interp.get_defining_op of %1404 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %1405 : !pdl.operation -> ^bb2634, ^bb1479
  ^bb2634:
    %1406 = pdl_interp.get_defining_op of %1403 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1406 : !pdl.operation -> ^bb2635, ^bb1479
  ^bb2635:
    pdl_interp.check_operation_name of %1405 is "arith.divf" -> ^bb2636, ^bb1479
  ^bb2636:
    pdl_interp.check_operand_count of %1405 is 2 -> ^bb2637, ^bb1479
  ^bb2637:
    pdl_interp.check_result_count of %1405 is 1 -> ^bb2638, ^bb1479
  ^bb2638:
    %1407 = pdl_interp.get_result 0 of %1405
    pdl_interp.is_not_null %1407 : !pdl.value -> ^bb2639, ^bb1479
  ^bb2639:
    pdl_interp.are_equal %1407, %1404 : !pdl.value -> ^bb2640, ^bb1479
  ^bb2640:
    pdl_interp.check_operation_name of %1406 is "arith.constant" -> ^bb2641, ^bb1479
  ^bb2641:
    pdl_interp.check_operand_count of %1406 is 0 -> ^bb2642, ^bb1479
  ^bb2642:
    pdl_interp.check_result_count of %1406 is 1 -> ^bb2643, ^bb1479
  ^bb2643:
    %1408 = pdl_interp.get_result 0 of %1406
    pdl_interp.is_not_null %1408 : !pdl.value -> ^bb2644, ^bb1479
  ^bb2644:
    pdl_interp.are_equal %1408, %1403 : !pdl.value -> ^bb2645, ^bb1479
  ^bb2645:
    %1409 = pdl_interp.get_operand 0 of %1405
    pdl_interp.is_not_null %1409 : !pdl.value -> ^bb2646, ^bb1479
  ^bb2646:
    %1410 = pdl_interp.get_attribute "value" of %1406
    pdl_interp.is_not_null %1410 : !pdl.attribute -> ^bb2647, ^bb1479
  ^bb2647:
    pdl_interp.check_attribute %1410 is 1.000000e+00 : f32 -> ^bb2648, ^bb1479
  ^bb2648:
    %1411 = pdl_interp.get_value_type of %1408 : !pdl.type
    %1412 = pdl_interp.get_value_type of %1402 : !pdl.type
    pdl_interp.are_equal %1411, %1412 : !pdl.type -> ^bb2649, ^bb1479
  ^bb2649:
    %1413 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1411, %1413 : !pdl.type -> ^bb2650, ^bb1479
  ^bb2650:
    pdl_interp.check_type %1411 is f32 -> ^bb2651, ^bb1479
  ^bb2651:
    %1414 = pdl_interp.get_value_type of %1407 : !pdl.type
    pdl_interp.are_equal %1411, %1414 : !pdl.type -> ^bb2652, ^bb1479
  ^bb2652:
    %1415 = pdl_interp.get_value_type of %1409 : !pdl.type
    pdl_interp.are_equal %1411, %1415 : !pdl.type -> ^bb2653, ^bb1479
  ^bb2653:
    %1416 = pdl_interp.get_operand 1 of %1405
    pdl_interp.is_not_null %1416 : !pdl.value -> ^bb2654, ^bb1479
  ^bb2654:
    %1417 = pdl_interp.get_operand 1 of %0
    pdl_interp.are_equal %1416, %1417 : !pdl.value -> ^bb2655, ^bb1479
  ^bb2655:
    %1418 = pdl_interp.get_value_type of %1416 : !pdl.type
    pdl_interp.are_equal %1411, %1418 : !pdl.type -> ^bb2656, ^bb1479
  ^bb2656:
    pdl_interp.record_match @rewriters::@sum_to_mult_rev(%1416, %1409, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1479
  ^bb2549:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb2657, ^bb1479
  ^bb2657:
    pdl_interp.check_result_count of %3 is 1 -> ^bb2658, ^bb1479
  ^bb2658:
    %1419 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %1419 : !pdl.value -> ^bb2659, ^bb1479
  ^bb2659:
    pdl_interp.are_equal %1419, %2 : !pdl.value -> ^bb2660, ^bb1479
  ^bb2660:
    %1420 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %1420 : !pdl.value -> ^bb2661, ^bb1479
  ^bb2661:
    %1421 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %1421 : !pdl.value -> ^bb2662, ^bb1479
  ^bb2662:
    %1422 = pdl_interp.get_defining_op of %1421 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %1422 : !pdl.operation -> ^bb2663, ^bb1479
  ^bb2663:
    %1423 = pdl_interp.get_defining_op of %1420 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1423 : !pdl.operation -> ^bb2664, ^bb1479
  ^bb2664:
    pdl_interp.check_operation_name of %1422 is "arith.divf" -> ^bb2665, ^bb1479
  ^bb2665:
    pdl_interp.check_operand_count of %1422 is 2 -> ^bb2666, ^bb1479
  ^bb2666:
    pdl_interp.check_result_count of %1422 is 1 -> ^bb2667, ^bb1479
  ^bb2667:
    %1424 = pdl_interp.get_result 0 of %1422
    pdl_interp.is_not_null %1424 : !pdl.value -> ^bb2668, ^bb1479
  ^bb2668:
    pdl_interp.are_equal %1424, %1421 : !pdl.value -> ^bb2669, ^bb1479
  ^bb2669:
    pdl_interp.check_operation_name of %1423 is "arith.constant" -> ^bb2670, ^bb1479
  ^bb2670:
    pdl_interp.check_operand_count of %1423 is 0 -> ^bb2671, ^bb1479
  ^bb2671:
    pdl_interp.check_result_count of %1423 is 1 -> ^bb2672, ^bb1479
  ^bb2672:
    %1425 = pdl_interp.get_result 0 of %1423
    pdl_interp.is_not_null %1425 : !pdl.value -> ^bb2673, ^bb1479
  ^bb2673:
    pdl_interp.are_equal %1425, %1420 : !pdl.value -> ^bb2674, ^bb1479
  ^bb2674:
    %1426 = pdl_interp.get_operand 0 of %1422
    pdl_interp.is_not_null %1426 : !pdl.value -> ^bb2675, ^bb1479
  ^bb2675:
    %1427 = pdl_interp.get_attribute "value" of %1423
    pdl_interp.is_not_null %1427 : !pdl.attribute -> ^bb2676, ^bb1479
  ^bb2676:
    pdl_interp.check_attribute %1427 is 1.000000e+00 : f32 -> ^bb2677, ^bb1479
  ^bb2677:
    %1428 = pdl_interp.get_value_type of %1425 : !pdl.type
    %1429 = pdl_interp.get_value_type of %1419 : !pdl.type
    pdl_interp.are_equal %1428, %1429 : !pdl.type -> ^bb2678, ^bb1479
  ^bb2678:
    %1430 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1428, %1430 : !pdl.type -> ^bb2679, ^bb1479
  ^bb2679:
    pdl_interp.check_type %1428 is f32 -> ^bb2680, ^bb1479
  ^bb2680:
    %1431 = pdl_interp.get_value_type of %1424 : !pdl.type
    pdl_interp.are_equal %1428, %1431 : !pdl.type -> ^bb2681, ^bb1479
  ^bb2681:
    %1432 = pdl_interp.get_value_type of %1426 : !pdl.type
    pdl_interp.are_equal %1428, %1432 : !pdl.type -> ^bb2682, ^bb1479
  ^bb2682:
    %1433 = pdl_interp.get_operand 1 of %1422
    pdl_interp.is_not_null %1433 : !pdl.value -> ^bb2683, ^bb1479
  ^bb2683:
    %1434 = pdl_interp.get_operand 1 of %0
    pdl_interp.are_equal %1433, %1434 : !pdl.value -> ^bb2684, ^bb1479
  ^bb2684:
    %1435 = pdl_interp.get_value_type of %1433 : !pdl.type
    pdl_interp.are_equal %1428, %1435 : !pdl.type -> ^bb2685, ^bb1479
  ^bb2685:
    pdl_interp.record_match @rewriters::@sub_to_mult_rev(%1433, %1426, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1479
  ^bb2550:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb2686, ^bb1479
  ^bb2686:
    pdl_interp.check_result_count of %3 is 1 -> ^bb2687, ^bb1479
  ^bb2687:
    %1436 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %1436 : !pdl.value -> ^bb2688, ^bb1479
  ^bb2688:
    pdl_interp.are_equal %1436, %2 : !pdl.value -> ^bb2689, ^bb1479
  ^bb2689:
    %1437 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %1437 : !pdl.value -> ^bb2690, ^bb1479
  ^bb2690:
    %1438 = pdl_interp.get_value_type of %1437 : !pdl.type
    %1439 = pdl_interp.get_value_type of %1436 : !pdl.type
    pdl_interp.are_equal %1438, %1439 : !pdl.type -> ^bb2691, ^bb1479
  ^bb2691:
    %1440 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1438, %1440 : !pdl.type -> ^bb2692, ^bb1479
  ^bb2692:
    pdl_interp.check_type %1438 is f32 -> ^bb2693, ^bb1479
  ^bb2693:
    %1441 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %1441 : !pdl.value -> ^bb2694, ^bb1479
  ^bb2694:
    %1442 = pdl_interp.get_value_type of %1441 : !pdl.type
    pdl_interp.are_equal %1438, %1442 : !pdl.type -> ^bb2695, ^bb1479
  ^bb2695:
    %1443 = pdl_interp.get_operand 1 of %0
    pdl_interp.are_equal %1437, %1443 : !pdl.value -> ^bb2696, ^bb1479
  ^bb2696:
    pdl_interp.record_match @rewriters::@pow_plus(%1441, %1437, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb1479
  ^bb28:
    pdl_interp.check_operand_count of %0 is 2 -> ^bb2697, ^bb1
  ^bb2697:
    pdl_interp.check_result_count of %0 is 1 -> ^bb2698, ^bb1
  ^bb2698:
    pdl_interp.is_not_null %2 : !pdl.value -> ^bb2699, ^bb2700
  ^bb2700:
    %1444 = pdl_interp.get_operand 1 of %0
    %1445 = pdl_interp.get_defining_op of %1444 : !pdl.value {position = "root.operand[1].defining_op"}
    pdl_interp.is_not_null %1445 : !pdl.operation -> ^bb2701, ^bb1
  ^bb2701:
    pdl_interp.is_not_null %2 : !pdl.value -> ^bb2702, ^bb1
  ^bb2702:
    pdl_interp.switch_operation_name of %3 to ["arith.constant", "arith.mulf", "arith.negf", "math.absf", "math.sqrt", "math.powf", "math.cbrt", "math.exp", "math.sin", "arith.subf", "arith.addf", "math.sinh", "math.log"](^bb2703, ^bb2704, ^bb2705, ^bb2706, ^bb2707, ^bb2708, ^bb2709, ^bb2710, ^bb2711, ^bb2712, ^bb2713, ^bb2714, ^bb2715) -> ^bb1
  ^bb2703:
    pdl_interp.check_operand_count of %3 is 0 -> ^bb2716, ^bb1
  ^bb2716:
    pdl_interp.check_result_count of %3 is 1 -> ^bb2717, ^bb1
  ^bb2717:
    %1446 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %1446 : !pdl.value -> ^bb2718, ^bb1
  ^bb2718:
    pdl_interp.are_equal %1446, %2 : !pdl.value -> ^bb2719, ^bb1
  ^bb2719:
    pdl_interp.is_not_null %1444 : !pdl.value -> ^bb2720, ^bb1
  ^bb2720:
    pdl_interp.switch_operation_name of %1445 to ["arith.divf", "math.exp", "math.powf", "math.sqrt"](^bb2721, ^bb2722, ^bb2723, ^bb2724) -> ^bb1
  ^bb2721:
    pdl_interp.check_operand_count of %1445 is 2 -> ^bb2725, ^bb1
  ^bb2725:
    pdl_interp.check_result_count of %1445 is 1 -> ^bb2726, ^bb1
  ^bb2726:
    %1447 = pdl_interp.get_result 0 of %1445
    pdl_interp.is_not_null %1447 : !pdl.value -> ^bb2727, ^bb1
  ^bb2727:
    pdl_interp.are_equal %1447, %1444 : !pdl.value -> ^bb2728, ^bb1
  ^bb2728:
    %1448 = pdl_interp.get_operand 0 of %1445
    pdl_interp.is_not_null %1448 : !pdl.value -> ^bb2729, ^bb1
  ^bb2729:
    %1449 = pdl_interp.get_defining_op of %1448 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1449 : !pdl.operation -> ^bb2730, ^bb2731
  ^bb2731:
    %1450 = pdl_interp.get_operand 1 of %1445
    pdl_interp.is_not_null %1450 : !pdl.value -> ^bb2732, ^bb1
  ^bb2732:
    %1451 = pdl_interp.get_attribute "value" of %3
    pdl_interp.is_not_null %1451 : !pdl.attribute -> ^bb2733, ^bb1
  ^bb2733:
    pdl_interp.check_attribute %1451 is 1.000000e+00 : f32 -> ^bb2734, ^bb1
  ^bb2734:
    %1452 = pdl_interp.get_value_type of %1446 : !pdl.type
    %1453 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1452, %1453 : !pdl.type -> ^bb2735, ^bb1
  ^bb2735:
    pdl_interp.check_type %1452 is f32 -> ^bb2736, ^bb1
  ^bb2736:
    %1454 = pdl_interp.get_value_type of %1447 : !pdl.type
    pdl_interp.are_equal %1452, %1454 : !pdl.type -> ^bb2737, ^bb1
  ^bb2737:
    %1455 = pdl_interp.get_value_type of %1448 : !pdl.type
    pdl_interp.are_equal %1452, %1455 : !pdl.type -> ^bb2738, ^bb1
  ^bb2738:
    %1456 = pdl_interp.get_value_type of %1450 : !pdl.type
    pdl_interp.are_equal %1452, %1456 : !pdl.type -> ^bb2739, ^bb1
  ^bb2739:
    pdl_interp.record_match @rewriters::@div_flip_rev(%1450, %1448, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb1
  ^bb2730:
    %1457 = pdl_interp.get_operand 1 of %1445
    pdl_interp.is_not_null %1457 : !pdl.value -> ^bb2740, ^bb2731
  ^bb2740:
    pdl_interp.check_operation_name of %1449 is "arith.constant" -> ^bb2741, ^bb2731
  ^bb2741:
    pdl_interp.check_operand_count of %1449 is 0 -> ^bb2742, ^bb2731
  ^bb2742:
    pdl_interp.check_result_count of %1449 is 1 -> ^bb2743, ^bb2731
  ^bb2743:
    %1458 = pdl_interp.get_result 0 of %1449
    pdl_interp.is_not_null %1458 : !pdl.value -> ^bb2744, ^bb2731
  ^bb2744:
    pdl_interp.are_equal %1458, %1448 : !pdl.value -> ^bb2745, ^bb2731
  ^bb2745:
    %1459 = pdl_interp.get_attribute "value" of %3
    pdl_interp.is_not_null %1459 : !pdl.attribute -> ^bb2746, ^bb2731
  ^bb2746:
    pdl_interp.check_attribute %1459 is 1.000000e+00 : f32 -> ^bb2747, ^bb2731
  ^bb2747:
    %1460 = pdl_interp.get_value_type of %1446 : !pdl.type
    %1461 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1460, %1461 : !pdl.type -> ^bb2748, ^bb2731
  ^bb2748:
    pdl_interp.check_type %1460 is f32 -> ^bb2749, ^bb2731
  ^bb2749:
    %1462 = pdl_interp.get_value_type of %1447 : !pdl.type
    pdl_interp.are_equal %1460, %1462 : !pdl.type -> ^bb2750, ^bb2731
  ^bb2750:
    %1463 = pdl_interp.get_value_type of %1458 : !pdl.type
    pdl_interp.are_equal %1463, %1460 : !pdl.type -> ^bb2751, ^bb2731
  ^bb2751:
    %1464 = pdl_interp.get_attribute "value" of %1449
    pdl_interp.is_not_null %1464 : !pdl.attribute -> ^bb2752, ^bb2731
  ^bb2752:
    pdl_interp.check_attribute %1464 is 1.000000e+00 : f32 -> ^bb2753, ^bb2731
  ^bb2753:
    %1465 = pdl_interp.get_value_type of %1457 : !pdl.type
    pdl_interp.are_equal %1460, %1465 : !pdl.type -> ^bb2754, ^bb2731
  ^bb2754:
    pdl_interp.record_match @rewriters::@remove_double_div(%1457, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb2731
  ^bb2722:
    pdl_interp.check_operand_count of %1445 is 1 -> ^bb2755, ^bb1
  ^bb2755:
    pdl_interp.check_result_count of %1445 is 1 -> ^bb2756, ^bb1
  ^bb2756:
    %1466 = pdl_interp.get_result 0 of %1445
    pdl_interp.is_not_null %1466 : !pdl.value -> ^bb2757, ^bb1
  ^bb2757:
    pdl_interp.are_equal %1466, %1444 : !pdl.value -> ^bb2758, ^bb1
  ^bb2758:
    %1467 = pdl_interp.get_operand 0 of %1445
    pdl_interp.is_not_null %1467 : !pdl.value -> ^bb2759, ^bb1
  ^bb2759:
    %1468 = pdl_interp.get_attribute "value" of %3
    pdl_interp.is_not_null %1468 : !pdl.attribute -> ^bb2760, ^bb1
  ^bb2760:
    pdl_interp.check_attribute %1468 is 1.000000e+00 : f32 -> ^bb2761, ^bb1
  ^bb2761:
    %1469 = pdl_interp.get_value_type of %1446 : !pdl.type
    %1470 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1469, %1470 : !pdl.type -> ^bb2762, ^bb1
  ^bb2762:
    pdl_interp.check_type %1469 is f32 -> ^bb2763, ^bb1
  ^bb2763:
    %1471 = pdl_interp.get_value_type of %1466 : !pdl.type
    pdl_interp.are_equal %1469, %1471 : !pdl.type -> ^bb2764, ^bb1
  ^bb2764:
    %1472 = pdl_interp.get_value_type of %1467 : !pdl.type
    pdl_interp.are_equal %1469, %1472 : !pdl.type -> ^bb2765, ^bb1
  ^bb2765:
    pdl_interp.record_match @rewriters::@rec_exp(%1467, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb1
  ^bb2723:
    pdl_interp.check_operand_count of %1445 is 2 -> ^bb2766, ^bb1
  ^bb2766:
    pdl_interp.check_result_count of %1445 is 1 -> ^bb2767, ^bb1
  ^bb2767:
    %1473 = pdl_interp.get_result 0 of %1445
    pdl_interp.is_not_null %1473 : !pdl.value -> ^bb2768, ^bb1
  ^bb2768:
    pdl_interp.are_equal %1473, %1444 : !pdl.value -> ^bb2769, ^bb1
  ^bb2769:
    %1474 = pdl_interp.get_operand 0 of %1445
    pdl_interp.is_not_null %1474 : !pdl.value -> ^bb2770, ^bb1
  ^bb2770:
    %1475 = pdl_interp.get_operand 1 of %1445
    pdl_interp.is_not_null %1475 : !pdl.value -> ^bb2771, ^bb1
  ^bb2771:
    %1476 = pdl_interp.get_attribute "value" of %3
    pdl_interp.is_not_null %1476 : !pdl.attribute -> ^bb2772, ^bb1
  ^bb2772:
    pdl_interp.check_attribute %1476 is 1.000000e+00 : f32 -> ^bb2773, ^bb1
  ^bb2773:
    %1477 = pdl_interp.get_value_type of %1446 : !pdl.type
    %1478 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1477, %1478 : !pdl.type -> ^bb2774, ^bb1
  ^bb2774:
    pdl_interp.check_type %1477 is f32 -> ^bb2775, ^bb1
  ^bb2775:
    %1479 = pdl_interp.get_value_type of %1473 : !pdl.type
    pdl_interp.are_equal %1477, %1479 : !pdl.type -> ^bb2776, ^bb1
  ^bb2776:
    %1480 = pdl_interp.get_value_type of %1474 : !pdl.type
    pdl_interp.are_equal %1477, %1480 : !pdl.type -> ^bb2777, ^bb1
  ^bb2777:
    %1481 = pdl_interp.get_value_type of %1475 : !pdl.type
    pdl_interp.are_equal %1477, %1481 : !pdl.type -> ^bb2778, ^bb1
  ^bb2778:
    pdl_interp.record_match @rewriters::@pow_flip(%1475, %1474, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb1
  ^bb2724:
    pdl_interp.check_operand_count of %1445 is 1 -> ^bb2779, ^bb1
  ^bb2779:
    pdl_interp.check_result_count of %1445 is 1 -> ^bb2780, ^bb1
  ^bb2780:
    %1482 = pdl_interp.get_result 0 of %1445
    pdl_interp.is_not_null %1482 : !pdl.value -> ^bb2781, ^bb1
  ^bb2781:
    pdl_interp.are_equal %1482, %1444 : !pdl.value -> ^bb2782, ^bb1
  ^bb2782:
    %1483 = pdl_interp.get_operand 0 of %1445
    pdl_interp.is_not_null %1483 : !pdl.value -> ^bb2783, ^bb1
  ^bb2783:
    %1484 = pdl_interp.get_defining_op of %1483 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1484 : !pdl.operation -> ^bb2784, ^bb1
  ^bb2784:
    pdl_interp.switch_operation_name of %1484 to ["arith.addf", "arith.subf"](^bb2785, ^bb2786) -> ^bb1
  ^bb2785:
    pdl_interp.check_operand_count of %1484 is 2 -> ^bb2787, ^bb1
  ^bb2787:
    pdl_interp.check_result_count of %1484 is 1 -> ^bb2788, ^bb1
  ^bb2788:
    %1485 = pdl_interp.get_result 0 of %1484
    pdl_interp.is_not_null %1485 : !pdl.value -> ^bb2789, ^bb1
  ^bb2789:
    pdl_interp.are_equal %1485, %1483 : !pdl.value -> ^bb2790, ^bb1
  ^bb2790:
    %1486 = pdl_interp.get_attribute "value" of %3
    pdl_interp.is_not_null %1486 : !pdl.attribute -> ^bb2791, ^bb1
  ^bb2791:
    pdl_interp.check_attribute %1486 is 1.000000e+00 : f32 -> ^bb2792, ^bb1
  ^bb2792:
    %1487 = pdl_interp.get_value_type of %1446 : !pdl.type
    %1488 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1487, %1488 : !pdl.type -> ^bb2793, ^bb1
  ^bb2793:
    pdl_interp.check_type %1487 is f32 -> ^bb2794, ^bb1
  ^bb2794:
    %1489 = pdl_interp.get_operand 0 of %1484
    %1490 = pdl_interp.get_defining_op of %1489 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1490 : !pdl.operation -> ^bb2795, ^bb1
  ^bb2795:
    %1491 = pdl_interp.get_value_type of %1482 : !pdl.type
    pdl_interp.are_equal %1487, %1491 : !pdl.type -> ^bb2796, ^bb1
  ^bb2796:
    pdl_interp.is_not_null %1489 : !pdl.value -> ^bb2797, ^bb1
  ^bb2797:
    %1492 = pdl_interp.get_value_type of %1485 : !pdl.type
    pdl_interp.are_equal %1492, %1487 : !pdl.type -> ^bb2798, ^bb1
  ^bb2798:
    pdl_interp.check_operation_name of %1490 is "arith.constant" -> ^bb2799, ^bb1
  ^bb2799:
    pdl_interp.check_operand_count of %1490 is 0 -> ^bb2800, ^bb1
  ^bb2800:
    pdl_interp.check_result_count of %1490 is 1 -> ^bb2801, ^bb1
  ^bb2801:
    %1493 = pdl_interp.get_result 0 of %1490
    pdl_interp.is_not_null %1493 : !pdl.value -> ^bb2802, ^bb1
  ^bb2802:
    pdl_interp.are_equal %1493, %1489 : !pdl.value -> ^bb2803, ^bb1
  ^bb2803:
    %1494 = pdl_interp.get_operand 1 of %1484
    %1495 = pdl_interp.get_defining_op of %1494 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %1495 : !pdl.operation -> ^bb2804, ^bb1
  ^bb2804:
    %1496 = pdl_interp.get_value_type of %1493 : !pdl.type
    pdl_interp.are_equal %1496, %1487 : !pdl.type -> ^bb2805, ^bb1
  ^bb2805:
    pdl_interp.is_not_null %1494 : !pdl.value -> ^bb2806, ^bb1
  ^bb2806:
    pdl_interp.check_operation_name of %1495 is "arith.mulf" -> ^bb2807, ^bb1
  ^bb2807:
    pdl_interp.check_operand_count of %1495 is 2 -> ^bb2808, ^bb1
  ^bb2808:
    pdl_interp.check_result_count of %1495 is 1 -> ^bb2809, ^bb1
  ^bb2809:
    %1497 = pdl_interp.get_attribute "value" of %1490
    pdl_interp.is_not_null %1497 : !pdl.attribute -> ^bb2810, ^bb1
  ^bb2810:
    pdl_interp.check_attribute %1497 is 1.000000e+00 : f32 -> ^bb2811, ^bb1
  ^bb2811:
    %1498 = pdl_interp.get_result 0 of %1495
    pdl_interp.is_not_null %1498 : !pdl.value -> ^bb2812, ^bb1
  ^bb2812:
    pdl_interp.are_equal %1498, %1494 : !pdl.value -> ^bb2813, ^bb1
  ^bb2813:
    %1499 = pdl_interp.get_operand 0 of %1495
    pdl_interp.is_not_null %1499 : !pdl.value -> ^bb2814, ^bb1
  ^bb2814:
    %1500 = pdl_interp.get_operand 1 of %1495
    pdl_interp.are_equal %1499, %1500 : !pdl.value -> ^bb2815, ^bb1
  ^bb2815:
    %1501 = pdl_interp.get_value_type of %1499 : !pdl.type
    pdl_interp.are_equal %1501, %1487 : !pdl.type -> ^bb2816, ^bb1
  ^bb2816:
    %1502 = pdl_interp.get_value_type of %1498 : !pdl.type
    pdl_interp.are_equal %1502, %1487 : !pdl.type -> ^bb2817, ^bb1
  ^bb2817:
    pdl_interp.record_match @rewriters::@cos_atan_rev(%1499, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb1
  ^bb2786:
    pdl_interp.check_operand_count of %1484 is 2 -> ^bb2818, ^bb1
  ^bb2818:
    pdl_interp.check_result_count of %1484 is 1 -> ^bb2819, ^bb1
  ^bb2819:
    %1503 = pdl_interp.get_result 0 of %1484
    pdl_interp.is_not_null %1503 : !pdl.value -> ^bb2820, ^bb1
  ^bb2820:
    pdl_interp.are_equal %1503, %1483 : !pdl.value -> ^bb2821, ^bb1
  ^bb2821:
    %1504 = pdl_interp.get_attribute "value" of %3
    pdl_interp.is_not_null %1504 : !pdl.attribute -> ^bb2822, ^bb1
  ^bb2822:
    pdl_interp.check_attribute %1504 is 1.000000e+00 : f32 -> ^bb2823, ^bb1
  ^bb2823:
    %1505 = pdl_interp.get_value_type of %1446 : !pdl.type
    %1506 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1505, %1506 : !pdl.type -> ^bb2824, ^bb1
  ^bb2824:
    pdl_interp.check_type %1505 is f32 -> ^bb2825, ^bb1
  ^bb2825:
    %1507 = pdl_interp.get_operand 0 of %1484
    %1508 = pdl_interp.get_defining_op of %1507 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1508 : !pdl.operation -> ^bb2826, ^bb1
  ^bb2826:
    %1509 = pdl_interp.get_value_type of %1482 : !pdl.type
    pdl_interp.are_equal %1505, %1509 : !pdl.type -> ^bb2827, ^bb1
  ^bb2827:
    pdl_interp.is_not_null %1507 : !pdl.value -> ^bb2828, ^bb1
  ^bb2828:
    %1510 = pdl_interp.get_value_type of %1503 : !pdl.type
    pdl_interp.are_equal %1510, %1505 : !pdl.type -> ^bb2829, ^bb1
  ^bb2829:
    pdl_interp.check_operation_name of %1508 is "arith.constant" -> ^bb2830, ^bb1
  ^bb2830:
    pdl_interp.check_operand_count of %1508 is 0 -> ^bb2831, ^bb1
  ^bb2831:
    pdl_interp.check_result_count of %1508 is 1 -> ^bb2832, ^bb1
  ^bb2832:
    %1511 = pdl_interp.get_result 0 of %1508
    pdl_interp.is_not_null %1511 : !pdl.value -> ^bb2833, ^bb1
  ^bb2833:
    pdl_interp.are_equal %1511, %1507 : !pdl.value -> ^bb2834, ^bb1
  ^bb2834:
    %1512 = pdl_interp.get_operand 1 of %1484
    %1513 = pdl_interp.get_defining_op of %1512 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %1513 : !pdl.operation -> ^bb2835, ^bb1
  ^bb2835:
    %1514 = pdl_interp.get_value_type of %1511 : !pdl.type
    pdl_interp.are_equal %1514, %1505 : !pdl.type -> ^bb2836, ^bb1
  ^bb2836:
    pdl_interp.is_not_null %1512 : !pdl.value -> ^bb2837, ^bb1
  ^bb2837:
    pdl_interp.check_operation_name of %1513 is "arith.mulf" -> ^bb2838, ^bb1
  ^bb2838:
    pdl_interp.check_operand_count of %1513 is 2 -> ^bb2839, ^bb1
  ^bb2839:
    pdl_interp.check_result_count of %1513 is 1 -> ^bb2840, ^bb1
  ^bb2840:
    %1515 = pdl_interp.get_attribute "value" of %1508
    pdl_interp.is_not_null %1515 : !pdl.attribute -> ^bb2841, ^bb1
  ^bb2841:
    pdl_interp.check_attribute %1515 is 1.000000e+00 : f32 -> ^bb2842, ^bb1
  ^bb2842:
    %1516 = pdl_interp.get_result 0 of %1513
    pdl_interp.is_not_null %1516 : !pdl.value -> ^bb2843, ^bb1
  ^bb2843:
    pdl_interp.are_equal %1516, %1512 : !pdl.value -> ^bb2844, ^bb1
  ^bb2844:
    %1517 = pdl_interp.get_operand 0 of %1513
    pdl_interp.is_not_null %1517 : !pdl.value -> ^bb2845, ^bb1
  ^bb2845:
    %1518 = pdl_interp.get_operand 1 of %1513
    pdl_interp.are_equal %1517, %1518 : !pdl.value -> ^bb2846, ^bb1
  ^bb2846:
    %1519 = pdl_interp.get_value_type of %1517 : !pdl.type
    pdl_interp.are_equal %1519, %1505 : !pdl.type -> ^bb2847, ^bb1
  ^bb2847:
    %1520 = pdl_interp.get_value_type of %1516 : !pdl.type
    pdl_interp.are_equal %1520, %1505 : !pdl.type -> ^bb2848, ^bb1
  ^bb2848:
    pdl_interp.record_match @rewriters::@cosh_atanh_rev(%1517, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb1
  ^bb2704:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb2849, ^bb1
  ^bb2849:
    pdl_interp.check_result_count of %3 is 1 -> ^bb2850, ^bb1
  ^bb2850:
    %1521 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %1521 : !pdl.value -> ^bb2851, ^bb1
  ^bb2851:
    pdl_interp.are_equal %1521, %2 : !pdl.value -> ^bb2852, ^bb1
  ^bb2852:
    %1522 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %1522 : !pdl.value -> ^bb2853, ^bb1
  ^bb2853:
    pdl_interp.is_not_null %1444 : !pdl.value -> ^bb2854, ^bb1
  ^bb2854:
    %1523 = pdl_interp.get_value_type of %1522 : !pdl.type
    %1524 = pdl_interp.get_value_type of %1521 : !pdl.type
    pdl_interp.are_equal %1523, %1524 : !pdl.type -> ^bb2855, ^bb2856
  ^bb2856:
    pdl_interp.switch_operation_name of %1445 to ["arith.subf", "arith.addf"](^bb2857, ^bb2858) -> ^bb1
  ^bb2857:
    pdl_interp.check_operand_count of %1445 is 2 -> ^bb2859, ^bb1
  ^bb2859:
    pdl_interp.check_result_count of %1445 is 1 -> ^bb2860, ^bb1
  ^bb2860:
    %1525 = pdl_interp.get_result 0 of %1445
    pdl_interp.is_not_null %1525 : !pdl.value -> ^bb2861, ^bb1
  ^bb2861:
    pdl_interp.are_equal %1525, %1444 : !pdl.value -> ^bb2862, ^bb1
  ^bb2862:
    %1526 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %1526 : !pdl.value -> ^bb2863, ^bb1
  ^bb2863:
    %1527 = pdl_interp.get_defining_op of %1526 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %1527 : !pdl.operation -> ^bb2864, ^bb1
  ^bb2864:
    %1528 = pdl_interp.get_defining_op of %1522 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1528 : !pdl.operation -> ^bb2865, ^bb1
  ^bb2865:
    %1529 = pdl_interp.get_operand 0 of %1445
    pdl_interp.is_not_null %1529 : !pdl.value -> ^bb2866, ^bb1
  ^bb2866:
    %1530 = pdl_interp.get_defining_op of %1529 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1530 : !pdl.operation -> ^bb2867, ^bb1
  ^bb2867:
    %1531 = pdl_interp.get_operand 1 of %1445
    %1532 = pdl_interp.get_defining_op of %1531 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %1532 : !pdl.operation -> ^bb2868, ^bb1
  ^bb2868:
    pdl_interp.is_not_null %1531 : !pdl.value -> ^bb2869, ^bb1
  ^bb2869:
    pdl_interp.check_operation_name of %1527 is "math.tan" -> ^bb2870, ^bb1
  ^bb2870:
    pdl_interp.check_operand_count of %1527 is 1 -> ^bb2871, ^bb1
  ^bb2871:
    pdl_interp.check_result_count of %1527 is 1 -> ^bb2872, ^bb1
  ^bb2872:
    %1533 = pdl_interp.get_result 0 of %1527
    pdl_interp.is_not_null %1533 : !pdl.value -> ^bb2873, ^bb1
  ^bb2873:
    pdl_interp.are_equal %1533, %1526 : !pdl.value -> ^bb2874, ^bb1
  ^bb2874:
    pdl_interp.check_operation_name of %1528 is "arith.constant" -> ^bb2875, ^bb1
  ^bb2875:
    pdl_interp.check_operand_count of %1528 is 0 -> ^bb2876, ^bb1
  ^bb2876:
    pdl_interp.check_result_count of %1528 is 1 -> ^bb2877, ^bb1
  ^bb2877:
    %1534 = pdl_interp.get_result 0 of %1528
    pdl_interp.is_not_null %1534 : !pdl.value -> ^bb2878, ^bb1
  ^bb2878:
    pdl_interp.are_equal %1534, %1522 : !pdl.value -> ^bb2879, ^bb1
  ^bb2879:
    pdl_interp.check_operation_name of %1530 is "arith.constant" -> ^bb2880, ^bb1
  ^bb2880:
    pdl_interp.check_operand_count of %1530 is 0 -> ^bb2881, ^bb1
  ^bb2881:
    pdl_interp.check_result_count of %1530 is 1 -> ^bb2882, ^bb1
  ^bb2882:
    %1535 = pdl_interp.get_result 0 of %1530
    pdl_interp.is_not_null %1535 : !pdl.value -> ^bb2883, ^bb1
  ^bb2883:
    pdl_interp.are_equal %1535, %1529 : !pdl.value -> ^bb2884, ^bb1
  ^bb2884:
    pdl_interp.check_operation_name of %1532 is "arith.mulf" -> ^bb2885, ^bb1
  ^bb2885:
    pdl_interp.check_operand_count of %1532 is 2 -> ^bb2886, ^bb1
  ^bb2886:
    pdl_interp.check_result_count of %1532 is 1 -> ^bb2887, ^bb1
  ^bb2887:
    %1536 = pdl_interp.get_result 0 of %1532
    pdl_interp.is_not_null %1536 : !pdl.value -> ^bb2888, ^bb1
  ^bb2888:
    pdl_interp.are_equal %1536, %1531 : !pdl.value -> ^bb2889, ^bb1
  ^bb2889:
    %1537 = pdl_interp.get_operand 0 of %1527
    pdl_interp.is_not_null %1537 : !pdl.value -> ^bb2890, ^bb1
  ^bb2890:
    %1538 = pdl_interp.get_operand 0 of %1532
    %1539 = pdl_interp.get_defining_op of %1538 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1539 : !pdl.operation -> ^bb2891, ^bb1
  ^bb2891:
    %1540 = pdl_interp.get_attribute "value" of %1528
    pdl_interp.is_not_null %1540 : !pdl.attribute -> ^bb2892, ^bb1
  ^bb2892:
    pdl_interp.check_attribute %1540 is 2.000000e+00 : f32 -> ^bb2893, ^bb1
  ^bb2893:
    %1541 = pdl_interp.get_value_type of %1534 : !pdl.type
    %1542 = pdl_interp.get_value_type of %1521 : !pdl.type
    pdl_interp.are_equal %1541, %1542 : !pdl.type -> ^bb2894, ^bb1
  ^bb2894:
    %1543 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1541, %1543 : !pdl.type -> ^bb2895, ^bb1
  ^bb2895:
    pdl_interp.check_type %1541 is f32 -> ^bb2896, ^bb1
  ^bb2896:
    pdl_interp.is_not_null %1538 : !pdl.value -> ^bb2897, ^bb1
  ^bb2897:
    pdl_interp.check_operation_name of %1539 is "math.tan" -> ^bb2898, ^bb1
  ^bb2898:
    pdl_interp.check_operand_count of %1539 is 1 -> ^bb2899, ^bb1
  ^bb2899:
    pdl_interp.check_result_count of %1539 is 1 -> ^bb2900, ^bb1
  ^bb2900:
    %1544 = pdl_interp.get_result 0 of %1539
    pdl_interp.is_not_null %1544 : !pdl.value -> ^bb2901, ^bb1
  ^bb2901:
    pdl_interp.are_equal %1544, %1538 : !pdl.value -> ^bb2902, ^bb1
  ^bb2902:
    %1545 = pdl_interp.get_attribute "value" of %1530
    pdl_interp.is_not_null %1545 : !pdl.attribute -> ^bb2903, ^bb1
  ^bb2903:
    pdl_interp.check_attribute %1545 is 1.000000e+00 : f32 -> ^bb2904, ^bb1
  ^bb2904:
    %1546 = pdl_interp.get_value_type of %1533 : !pdl.type
    pdl_interp.are_equal %1541, %1546 : !pdl.type -> ^bb2905, ^bb1
  ^bb2905:
    %1547 = pdl_interp.get_operand 1 of %1532
    %1548 = pdl_interp.get_defining_op of %1547 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %1548 : !pdl.operation -> ^bb2906, ^bb1
  ^bb2906:
    %1549 = pdl_interp.get_value_type of %1537 : !pdl.type
    pdl_interp.are_equal %1541, %1549 : !pdl.type -> ^bb2907, ^bb1
  ^bb2907:
    %1550 = pdl_interp.get_value_type of %1525 : !pdl.type
    pdl_interp.are_equal %1541, %1550 : !pdl.type -> ^bb2908, ^bb1
  ^bb2908:
    %1551 = pdl_interp.get_value_type of %1535 : !pdl.type
    pdl_interp.are_equal %1541, %1551 : !pdl.type -> ^bb2909, ^bb1
  ^bb2909:
    pdl_interp.is_not_null %1547 : !pdl.value -> ^bb2910, ^bb1
  ^bb2910:
    pdl_interp.check_operation_name of %1548 is "math.tan" -> ^bb2911, ^bb1
  ^bb2911:
    pdl_interp.check_operand_count of %1548 is 1 -> ^bb2912, ^bb1
  ^bb2912:
    pdl_interp.check_result_count of %1548 is 1 -> ^bb2913, ^bb1
  ^bb2913:
    %1552 = pdl_interp.get_result 0 of %1548
    pdl_interp.is_not_null %1552 : !pdl.value -> ^bb2914, ^bb1
  ^bb2914:
    pdl_interp.are_equal %1552, %1547 : !pdl.value -> ^bb2915, ^bb1
  ^bb2915:
    %1553 = pdl_interp.get_value_type of %1536 : !pdl.type
    pdl_interp.are_equal %1541, %1553 : !pdl.type -> ^bb2916, ^bb1
  ^bb2916:
    %1554 = pdl_interp.get_operand 0 of %1548
    pdl_interp.are_equal %1554, %1537 : !pdl.value -> ^bb2917, ^bb1
  ^bb2917:
    %1555 = pdl_interp.get_value_type of %1544 : !pdl.type
    pdl_interp.are_equal %1555, %1541 : !pdl.type -> ^bb2918, ^bb1
  ^bb2918:
    %1556 = pdl_interp.get_operand 0 of %1539
    pdl_interp.are_equal %1556, %1537 : !pdl.value -> ^bb2919, ^bb1
  ^bb2919:
    %1557 = pdl_interp.get_value_type of %1552 : !pdl.type
    pdl_interp.are_equal %1557, %1541 : !pdl.type -> ^bb2920, ^bb1
  ^bb2920:
    pdl_interp.record_match @rewriters::@_2_tan(%1537, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb1
  ^bb2858:
    pdl_interp.check_operand_count of %1445 is 2 -> ^bb2921, ^bb1
  ^bb2921:
    pdl_interp.check_result_count of %1445 is 1 -> ^bb2922, ^bb1
  ^bb2922:
    %1558 = pdl_interp.get_result 0 of %1445
    pdl_interp.is_not_null %1558 : !pdl.value -> ^bb2923, ^bb1
  ^bb2923:
    pdl_interp.are_equal %1558, %1444 : !pdl.value -> ^bb2924, ^bb1
  ^bb2924:
    %1559 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %1559 : !pdl.value -> ^bb2925, ^bb1
  ^bb2925:
    %1560 = pdl_interp.get_defining_op of %1559 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %1560 : !pdl.operation -> ^bb2926, ^bb1
  ^bb2926:
    %1561 = pdl_interp.get_defining_op of %1522 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1561 : !pdl.operation -> ^bb2927, ^bb1
  ^bb2927:
    %1562 = pdl_interp.get_operand 0 of %1445
    pdl_interp.is_not_null %1562 : !pdl.value -> ^bb2928, ^bb1
  ^bb2928:
    %1563 = pdl_interp.get_defining_op of %1562 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1563 : !pdl.operation -> ^bb2929, ^bb1
  ^bb2929:
    %1564 = pdl_interp.get_operand 1 of %1445
    %1565 = pdl_interp.get_defining_op of %1564 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %1565 : !pdl.operation -> ^bb2930, ^bb1
  ^bb2930:
    pdl_interp.is_not_null %1564 : !pdl.value -> ^bb2931, ^bb1
  ^bb2931:
    pdl_interp.check_operation_name of %1560 is "math.tanh" -> ^bb2932, ^bb1
  ^bb2932:
    pdl_interp.check_operand_count of %1560 is 1 -> ^bb2933, ^bb1
  ^bb2933:
    pdl_interp.check_result_count of %1560 is 1 -> ^bb2934, ^bb1
  ^bb2934:
    %1566 = pdl_interp.get_result 0 of %1560
    pdl_interp.is_not_null %1566 : !pdl.value -> ^bb2935, ^bb1
  ^bb2935:
    pdl_interp.are_equal %1566, %1559 : !pdl.value -> ^bb2936, ^bb1
  ^bb2936:
    pdl_interp.check_operation_name of %1561 is "arith.constant" -> ^bb2937, ^bb1
  ^bb2937:
    pdl_interp.check_operand_count of %1561 is 0 -> ^bb2938, ^bb1
  ^bb2938:
    pdl_interp.check_result_count of %1561 is 1 -> ^bb2939, ^bb1
  ^bb2939:
    %1567 = pdl_interp.get_result 0 of %1561
    pdl_interp.is_not_null %1567 : !pdl.value -> ^bb2940, ^bb1
  ^bb2940:
    pdl_interp.are_equal %1567, %1522 : !pdl.value -> ^bb2941, ^bb1
  ^bb2941:
    pdl_interp.check_operation_name of %1563 is "arith.constant" -> ^bb2942, ^bb1
  ^bb2942:
    pdl_interp.check_operand_count of %1563 is 0 -> ^bb2943, ^bb1
  ^bb2943:
    pdl_interp.check_result_count of %1563 is 1 -> ^bb2944, ^bb1
  ^bb2944:
    %1568 = pdl_interp.get_result 0 of %1563
    pdl_interp.is_not_null %1568 : !pdl.value -> ^bb2945, ^bb1
  ^bb2945:
    pdl_interp.are_equal %1568, %1562 : !pdl.value -> ^bb2946, ^bb1
  ^bb2946:
    pdl_interp.check_operation_name of %1565 is "arith.mulf" -> ^bb2947, ^bb1
  ^bb2947:
    pdl_interp.check_operand_count of %1565 is 2 -> ^bb2948, ^bb1
  ^bb2948:
    pdl_interp.check_result_count of %1565 is 1 -> ^bb2949, ^bb1
  ^bb2949:
    %1569 = pdl_interp.get_result 0 of %1565
    pdl_interp.is_not_null %1569 : !pdl.value -> ^bb2950, ^bb1
  ^bb2950:
    pdl_interp.are_equal %1569, %1564 : !pdl.value -> ^bb2951, ^bb1
  ^bb2951:
    %1570 = pdl_interp.get_operand 0 of %1560
    pdl_interp.is_not_null %1570 : !pdl.value -> ^bb2952, ^bb1
  ^bb2952:
    %1571 = pdl_interp.get_operand 0 of %1565
    %1572 = pdl_interp.get_defining_op of %1571 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1572 : !pdl.operation -> ^bb2953, ^bb1
  ^bb2953:
    %1573 = pdl_interp.get_attribute "value" of %1561
    pdl_interp.is_not_null %1573 : !pdl.attribute -> ^bb2954, ^bb1
  ^bb2954:
    pdl_interp.check_attribute %1573 is 2.000000e+00 : f32 -> ^bb2955, ^bb1
  ^bb2955:
    %1574 = pdl_interp.get_value_type of %1567 : !pdl.type
    %1575 = pdl_interp.get_value_type of %1521 : !pdl.type
    pdl_interp.are_equal %1574, %1575 : !pdl.type -> ^bb2956, ^bb1
  ^bb2956:
    %1576 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1574, %1576 : !pdl.type -> ^bb2957, ^bb1
  ^bb2957:
    pdl_interp.check_type %1574 is f32 -> ^bb2958, ^bb1
  ^bb2958:
    pdl_interp.is_not_null %1571 : !pdl.value -> ^bb2959, ^bb1
  ^bb2959:
    pdl_interp.check_operation_name of %1572 is "math.tanh" -> ^bb2960, ^bb1
  ^bb2960:
    pdl_interp.check_operand_count of %1572 is 1 -> ^bb2961, ^bb1
  ^bb2961:
    pdl_interp.check_result_count of %1572 is 1 -> ^bb2962, ^bb1
  ^bb2962:
    %1577 = pdl_interp.get_result 0 of %1572
    pdl_interp.is_not_null %1577 : !pdl.value -> ^bb2963, ^bb1
  ^bb2963:
    pdl_interp.are_equal %1577, %1571 : !pdl.value -> ^bb2964, ^bb1
  ^bb2964:
    %1578 = pdl_interp.get_attribute "value" of %1563
    pdl_interp.is_not_null %1578 : !pdl.attribute -> ^bb2965, ^bb1
  ^bb2965:
    pdl_interp.check_attribute %1578 is 1.000000e+00 : f32 -> ^bb2966, ^bb1
  ^bb2966:
    %1579 = pdl_interp.get_value_type of %1566 : !pdl.type
    pdl_interp.are_equal %1574, %1579 : !pdl.type -> ^bb2967, ^bb1
  ^bb2967:
    %1580 = pdl_interp.get_operand 1 of %1565
    %1581 = pdl_interp.get_defining_op of %1580 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %1581 : !pdl.operation -> ^bb2968, ^bb1
  ^bb2968:
    %1582 = pdl_interp.get_value_type of %1570 : !pdl.type
    pdl_interp.are_equal %1574, %1582 : !pdl.type -> ^bb2969, ^bb1
  ^bb2969:
    %1583 = pdl_interp.get_value_type of %1558 : !pdl.type
    pdl_interp.are_equal %1574, %1583 : !pdl.type -> ^bb2970, ^bb1
  ^bb2970:
    %1584 = pdl_interp.get_value_type of %1568 : !pdl.type
    pdl_interp.are_equal %1574, %1584 : !pdl.type -> ^bb2971, ^bb1
  ^bb2971:
    pdl_interp.is_not_null %1580 : !pdl.value -> ^bb2972, ^bb1
  ^bb2972:
    pdl_interp.check_operation_name of %1581 is "math.tanh" -> ^bb2973, ^bb1
  ^bb2973:
    pdl_interp.check_operand_count of %1581 is 1 -> ^bb2974, ^bb1
  ^bb2974:
    pdl_interp.check_result_count of %1581 is 1 -> ^bb2975, ^bb1
  ^bb2975:
    %1585 = pdl_interp.get_result 0 of %1581
    pdl_interp.is_not_null %1585 : !pdl.value -> ^bb2976, ^bb1
  ^bb2976:
    pdl_interp.are_equal %1585, %1580 : !pdl.value -> ^bb2977, ^bb1
  ^bb2977:
    %1586 = pdl_interp.get_value_type of %1569 : !pdl.type
    pdl_interp.are_equal %1574, %1586 : !pdl.type -> ^bb2978, ^bb1
  ^bb2978:
    %1587 = pdl_interp.get_operand 0 of %1581
    pdl_interp.are_equal %1587, %1570 : !pdl.value -> ^bb2979, ^bb1
  ^bb2979:
    %1588 = pdl_interp.get_value_type of %1577 : !pdl.type
    pdl_interp.are_equal %1588, %1574 : !pdl.type -> ^bb2980, ^bb1
  ^bb2980:
    %1589 = pdl_interp.get_operand 0 of %1572
    pdl_interp.are_equal %1589, %1570 : !pdl.value -> ^bb2981, ^bb1
  ^bb2981:
    %1590 = pdl_interp.get_value_type of %1585 : !pdl.type
    pdl_interp.are_equal %1590, %1574 : !pdl.type -> ^bb2982, ^bb1
  ^bb2982:
    pdl_interp.record_match @rewriters::@tanh_2_rev(%1570, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb1
  ^bb2855:
    %1591 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1523, %1591 : !pdl.type -> ^bb2983, ^bb2856
  ^bb2983:
    pdl_interp.check_type %1523 is f32 -> ^bb2984, ^bb2856
  ^bb2984:
    pdl_interp.check_operation_name of %1445 is "arith.mulf" -> ^bb2985, ^bb2856
  ^bb2985:
    pdl_interp.check_operand_count of %1445 is 2 -> ^bb2986, ^bb2856
  ^bb2986:
    pdl_interp.check_result_count of %1445 is 1 -> ^bb2987, ^bb2856
  ^bb2987:
    %1592 = pdl_interp.get_result 0 of %1445
    pdl_interp.is_not_null %1592 : !pdl.value -> ^bb2988, ^bb2856
  ^bb2988:
    pdl_interp.are_equal %1592, %1444 : !pdl.value -> ^bb2989, ^bb2856
  ^bb2989:
    %1593 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %1593 : !pdl.value -> ^bb2990, ^bb2856
  ^bb2990:
    %1594 = pdl_interp.get_operand 0 of %1445
    pdl_interp.is_not_null %1594 : !pdl.value -> ^bb2991, ^bb2856
  ^bb2991:
    %1595 = pdl_interp.get_operand 1 of %1445
    pdl_interp.is_not_null %1595 : !pdl.value -> ^bb2992, ^bb2856
  ^bb2992:
    %1596 = pdl_interp.get_value_type of %1592 : !pdl.type
    pdl_interp.are_equal %1523, %1596 : !pdl.type -> ^bb2993, ^bb2856
  ^bb2993:
    %1597 = pdl_interp.get_value_type of %1593 : !pdl.type
    pdl_interp.are_equal %1523, %1597 : !pdl.type -> ^bb2994, ^bb2856
  ^bb2994:
    %1598 = pdl_interp.get_value_type of %1594 : !pdl.type
    pdl_interp.are_equal %1523, %1598 : !pdl.type -> ^bb2995, ^bb2856
  ^bb2995:
    %1599 = pdl_interp.get_value_type of %1595 : !pdl.type
    pdl_interp.are_equal %1523, %1599 : !pdl.type -> ^bb2996, ^bb2856
  ^bb2996:
    pdl_interp.record_match @rewriters::@times_frac(%1522, %1594, %1593, %1595, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb2856
  ^bb2705:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb2997, ^bb1
  ^bb2997:
    pdl_interp.check_result_count of %3 is 1 -> ^bb2998, ^bb1
  ^bb2998:
    %1600 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %1600 : !pdl.value -> ^bb2999, ^bb1
  ^bb2999:
    pdl_interp.are_equal %1600, %2 : !pdl.value -> ^bb3000, ^bb1
  ^bb3000:
    %1601 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %1601 : !pdl.value -> ^bb3001, ^bb1
  ^bb3001:
    pdl_interp.is_not_null %1444 : !pdl.value -> ^bb3002, ^bb1
  ^bb3002:
    %1602 = pdl_interp.get_value_type of %1601 : !pdl.type
    %1603 = pdl_interp.get_value_type of %1600 : !pdl.type
    pdl_interp.are_equal %1602, %1603 : !pdl.type -> ^bb3003, ^bb3004
  ^bb3004:
    pdl_interp.check_operation_name of %1445 is "arith.addf" -> ^bb3005, ^bb1
  ^bb3005:
    pdl_interp.check_operand_count of %1445 is 2 -> ^bb3006, ^bb1
  ^bb3006:
    pdl_interp.check_result_count of %1445 is 1 -> ^bb3007, ^bb1
  ^bb3007:
    %1604 = pdl_interp.get_result 0 of %1445
    pdl_interp.is_not_null %1604 : !pdl.value -> ^bb3008, ^bb1
  ^bb3008:
    pdl_interp.are_equal %1604, %1444 : !pdl.value -> ^bb3009, ^bb1
  ^bb3009:
    %1605 = pdl_interp.get_defining_op of %1601 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1605 : !pdl.operation -> ^bb3010, ^bb1
  ^bb3010:
    %1606 = pdl_interp.get_operand 0 of %1445
    pdl_interp.is_not_null %1606 : !pdl.value -> ^bb3011, ^bb1
  ^bb3011:
    %1607 = pdl_interp.get_defining_op of %1606 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1607 : !pdl.operation -> ^bb3012, ^bb1
  ^bb3012:
    %1608 = pdl_interp.get_operand 1 of %1445
    %1609 = pdl_interp.get_defining_op of %1608 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %1609 : !pdl.operation -> ^bb3013, ^bb1
  ^bb3013:
    pdl_interp.is_not_null %1608 : !pdl.value -> ^bb3014, ^bb1
  ^bb3014:
    pdl_interp.check_operation_name of %1605 is "math.sin" -> ^bb3015, ^bb1
  ^bb3015:
    pdl_interp.check_operand_count of %1605 is 1 -> ^bb3016, ^bb1
  ^bb3016:
    pdl_interp.check_result_count of %1605 is 1 -> ^bb3017, ^bb1
  ^bb3017:
    %1610 = pdl_interp.get_result 0 of %1605
    pdl_interp.is_not_null %1610 : !pdl.value -> ^bb3018, ^bb1
  ^bb3018:
    pdl_interp.are_equal %1610, %1601 : !pdl.value -> ^bb3019, ^bb1
  ^bb3019:
    pdl_interp.check_operation_name of %1607 is "arith.constant" -> ^bb3020, ^bb1
  ^bb3020:
    pdl_interp.check_operand_count of %1607 is 0 -> ^bb3021, ^bb1
  ^bb3021:
    pdl_interp.check_result_count of %1607 is 1 -> ^bb3022, ^bb1
  ^bb3022:
    %1611 = pdl_interp.get_result 0 of %1607
    pdl_interp.is_not_null %1611 : !pdl.value -> ^bb3023, ^bb1
  ^bb3023:
    pdl_interp.are_equal %1611, %1606 : !pdl.value -> ^bb3024, ^bb1
  ^bb3024:
    pdl_interp.check_operation_name of %1609 is "math.cos" -> ^bb3025, ^bb1
  ^bb3025:
    pdl_interp.check_operand_count of %1609 is 1 -> ^bb3026, ^bb1
  ^bb3026:
    pdl_interp.check_result_count of %1609 is 1 -> ^bb3027, ^bb1
  ^bb3027:
    %1612 = pdl_interp.get_result 0 of %1609
    pdl_interp.is_not_null %1612 : !pdl.value -> ^bb3028, ^bb1
  ^bb3028:
    pdl_interp.are_equal %1612, %1608 : !pdl.value -> ^bb3029, ^bb1
  ^bb3029:
    %1613 = pdl_interp.get_operand 0 of %1605
    pdl_interp.is_not_null %1613 : !pdl.value -> ^bb3030, ^bb1
  ^bb3030:
    %1614 = pdl_interp.get_value_type of %1613 : !pdl.type
    %1615 = pdl_interp.get_value_type of %1610 : !pdl.type
    pdl_interp.are_equal %1614, %1615 : !pdl.type -> ^bb3031, ^bb1
  ^bb3031:
    %1616 = pdl_interp.get_value_type of %1600 : !pdl.type
    pdl_interp.are_equal %1614, %1616 : !pdl.type -> ^bb3032, ^bb1
  ^bb3032:
    %1617 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1614, %1617 : !pdl.type -> ^bb3033, ^bb1
  ^bb3033:
    pdl_interp.check_type %1614 is f32 -> ^bb3034, ^bb1
  ^bb3034:
    %1618 = pdl_interp.get_value_type of %1604 : !pdl.type
    pdl_interp.are_equal %1614, %1618 : !pdl.type -> ^bb3035, ^bb1
  ^bb3035:
    %1619 = pdl_interp.get_value_type of %1612 : !pdl.type
    pdl_interp.are_equal %1614, %1619 : !pdl.type -> ^bb3036, ^bb1
  ^bb3036:
    %1620 = pdl_interp.get_value_type of %1611 : !pdl.type
    pdl_interp.are_equal %1614, %1620 : !pdl.type -> ^bb3037, ^bb1
  ^bb3037:
    %1621 = pdl_interp.get_attribute "value" of %1607
    pdl_interp.is_not_null %1621 : !pdl.attribute -> ^bb3038, ^bb1
  ^bb3038:
    pdl_interp.check_attribute %1621 is 1.000000e+00 : f32 -> ^bb3039, ^bb1
  ^bb3039:
    %1622 = pdl_interp.get_operand 0 of %1609
    pdl_interp.are_equal %1613, %1622 : !pdl.value -> ^bb3040, ^bb1
  ^bb3040:
    pdl_interp.record_match @rewriters::@hang_0m_tan(%1613, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb1
  ^bb3003:
    %1623 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1602, %1623 : !pdl.type -> ^bb3041, ^bb3004
  ^bb3041:
    pdl_interp.check_type %1602 is f32 -> ^bb3042, ^bb3004
  ^bb3042:
    pdl_interp.check_operation_name of %1445 is "arith.negf" -> ^bb3043, ^bb3004
  ^bb3043:
    pdl_interp.check_operand_count of %1445 is 1 -> ^bb3044, ^bb3004
  ^bb3044:
    pdl_interp.check_result_count of %1445 is 1 -> ^bb3045, ^bb3004
  ^bb3045:
    %1624 = pdl_interp.get_result 0 of %1445
    pdl_interp.is_not_null %1624 : !pdl.value -> ^bb3046, ^bb3004
  ^bb3046:
    pdl_interp.are_equal %1624, %1444 : !pdl.value -> ^bb3047, ^bb3004
  ^bb3047:
    %1625 = pdl_interp.get_operand 0 of %1445
    pdl_interp.is_not_null %1625 : !pdl.value -> ^bb3048, ^bb3004
  ^bb3048:
    %1626 = pdl_interp.get_value_type of %1624 : !pdl.type
    pdl_interp.are_equal %1602, %1626 : !pdl.type -> ^bb3049, ^bb3004
  ^bb3049:
    %1627 = pdl_interp.get_value_type of %1625 : !pdl.type
    pdl_interp.are_equal %1602, %1627 : !pdl.type -> ^bb3050, ^bb3004
  ^bb3050:
    pdl_interp.record_match @rewriters::@frac_2neg_rev(%1601, %1625, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb3004
  ^bb2706:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb3051, ^bb1
  ^bb3051:
    pdl_interp.check_result_count of %3 is 1 -> ^bb3052, ^bb1
  ^bb3052:
    %1628 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %1628 : !pdl.value -> ^bb3053, ^bb1
  ^bb3053:
    pdl_interp.are_equal %1628, %2 : !pdl.value -> ^bb3054, ^bb1
  ^bb3054:
    %1629 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %1629 : !pdl.value -> ^bb3055, ^bb1
  ^bb3055:
    pdl_interp.is_not_null %1444 : !pdl.value -> ^bb3056, ^bb1
  ^bb3056:
    %1630 = pdl_interp.get_value_type of %1629 : !pdl.type
    %1631 = pdl_interp.get_value_type of %1628 : !pdl.type
    pdl_interp.are_equal %1630, %1631 : !pdl.type -> ^bb3057, ^bb3058
  ^bb3058:
    pdl_interp.check_operation_name of %1445 is "math.cbrt" -> ^bb3059, ^bb1
  ^bb3059:
    pdl_interp.check_operand_count of %1445 is 1 -> ^bb3060, ^bb1
  ^bb3060:
    pdl_interp.check_result_count of %1445 is 1 -> ^bb3061, ^bb1
  ^bb3061:
    %1632 = pdl_interp.get_result 0 of %1445
    pdl_interp.is_not_null %1632 : !pdl.value -> ^bb3062, ^bb1
  ^bb3062:
    pdl_interp.are_equal %1632, %1444 : !pdl.value -> ^bb3063, ^bb1
  ^bb3063:
    %1633 = pdl_interp.get_defining_op of %1629 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1633 : !pdl.operation -> ^bb3064, ^bb1
  ^bb3064:
    pdl_interp.check_operation_name of %1633 is "math.cbrt" -> ^bb3065, ^bb1
  ^bb3065:
    pdl_interp.check_operand_count of %1633 is 1 -> ^bb3066, ^bb1
  ^bb3066:
    pdl_interp.check_result_count of %1633 is 1 -> ^bb3067, ^bb1
  ^bb3067:
    %1634 = pdl_interp.get_result 0 of %1633
    pdl_interp.is_not_null %1634 : !pdl.value -> ^bb3068, ^bb1
  ^bb3068:
    pdl_interp.are_equal %1634, %1629 : !pdl.value -> ^bb3069, ^bb1
  ^bb3069:
    %1635 = pdl_interp.get_operand 0 of %1633
    pdl_interp.is_not_null %1635 : !pdl.value -> ^bb3070, ^bb1
  ^bb3070:
    %1636 = pdl_interp.get_value_type of %1635 : !pdl.type
    %1637 = pdl_interp.get_value_type of %1634 : !pdl.type
    pdl_interp.are_equal %1636, %1637 : !pdl.type -> ^bb3071, ^bb1
  ^bb3071:
    %1638 = pdl_interp.get_value_type of %1628 : !pdl.type
    pdl_interp.are_equal %1636, %1638 : !pdl.type -> ^bb3072, ^bb1
  ^bb3072:
    %1639 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1636, %1639 : !pdl.type -> ^bb3073, ^bb1
  ^bb3073:
    pdl_interp.check_type %1636 is f32 -> ^bb3074, ^bb1
  ^bb3074:
    %1640 = pdl_interp.get_value_type of %1632 : !pdl.type
    pdl_interp.are_equal %1636, %1640 : !pdl.type -> ^bb3075, ^bb1
  ^bb3075:
    %1641 = pdl_interp.get_operand 0 of %1445
    pdl_interp.are_equal %1635, %1641 : !pdl.value -> ^bb3076, ^bb1
  ^bb3076:
    pdl_interp.record_match @rewriters::@cbrt_div_cbrt2(%1635, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb1
  ^bb3057:
    %1642 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1630, %1642 : !pdl.type -> ^bb3077, ^bb3058
  ^bb3077:
    pdl_interp.check_type %1630 is f32 -> ^bb3078, ^bb3058
  ^bb3078:
    pdl_interp.check_operation_name of %1445 is "math.absf" -> ^bb3079, ^bb3058
  ^bb3079:
    pdl_interp.check_operand_count of %1445 is 1 -> ^bb3080, ^bb3058
  ^bb3080:
    pdl_interp.check_result_count of %1445 is 1 -> ^bb3081, ^bb3058
  ^bb3081:
    %1643 = pdl_interp.get_result 0 of %1445
    pdl_interp.is_not_null %1643 : !pdl.value -> ^bb3082, ^bb3058
  ^bb3082:
    pdl_interp.are_equal %1643, %1444 : !pdl.value -> ^bb3083, ^bb3058
  ^bb3083:
    %1644 = pdl_interp.get_operand 0 of %1445
    pdl_interp.is_not_null %1644 : !pdl.value -> ^bb3084, ^bb3058
  ^bb3084:
    %1645 = pdl_interp.get_value_type of %1643 : !pdl.type
    pdl_interp.are_equal %1630, %1645 : !pdl.type -> ^bb3085, ^bb3058
  ^bb3085:
    %1646 = pdl_interp.get_value_type of %1644 : !pdl.type
    pdl_interp.are_equal %1630, %1646 : !pdl.type -> ^bb3086, ^bb3058
  ^bb3086:
    pdl_interp.record_match @rewriters::@div_fabs(%1629, %1644, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb3058
  ^bb2707:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb3087, ^bb1
  ^bb3087:
    pdl_interp.check_result_count of %3 is 1 -> ^bb3088, ^bb1
  ^bb3088:
    %1647 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %1647 : !pdl.value -> ^bb3089, ^bb1
  ^bb3089:
    pdl_interp.are_equal %1647, %2 : !pdl.value -> ^bb3090, ^bb1
  ^bb3090:
    %1648 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %1648 : !pdl.value -> ^bb3091, ^bb1
  ^bb3091:
    pdl_interp.is_not_null %1444 : !pdl.value -> ^bb3092, ^bb1
  ^bb3092:
    %1649 = pdl_interp.get_value_type of %1648 : !pdl.type
    %1650 = pdl_interp.get_value_type of %1647 : !pdl.type
    pdl_interp.are_equal %1649, %1650 : !pdl.type -> ^bb3093, ^bb1
  ^bb3093:
    %1651 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1649, %1651 : !pdl.type -> ^bb3094, ^bb1
  ^bb3094:
    pdl_interp.check_type %1649 is f32 -> ^bb3095, ^bb1
  ^bb3095:
    pdl_interp.check_operation_name of %1445 is "math.sqrt" -> ^bb3096, ^bb1
  ^bb3096:
    pdl_interp.check_operand_count of %1445 is 1 -> ^bb3097, ^bb1
  ^bb3097:
    pdl_interp.check_result_count of %1445 is 1 -> ^bb3098, ^bb1
  ^bb3098:
    %1652 = pdl_interp.get_result 0 of %1445
    pdl_interp.is_not_null %1652 : !pdl.value -> ^bb3099, ^bb1
  ^bb3099:
    pdl_interp.are_equal %1652, %1444 : !pdl.value -> ^bb3100, ^bb1
  ^bb3100:
    %1653 = pdl_interp.get_operand 0 of %1445
    pdl_interp.is_not_null %1653 : !pdl.value -> ^bb3101, ^bb1
  ^bb3101:
    %1654 = pdl_interp.get_value_type of %1652 : !pdl.type
    pdl_interp.are_equal %1649, %1654 : !pdl.type -> ^bb3102, ^bb1
  ^bb3102:
    %1655 = pdl_interp.get_value_type of %1653 : !pdl.type
    pdl_interp.are_equal %1649, %1655 : !pdl.type -> ^bb3103, ^bb1
  ^bb3103:
    pdl_interp.record_match @rewriters::@sqrt_undiv(%1648, %1653, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb1
  ^bb2708:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb3104, ^bb1
  ^bb3104:
    pdl_interp.check_result_count of %3 is 1 -> ^bb3105, ^bb1
  ^bb3105:
    %1656 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %1656 : !pdl.value -> ^bb3106, ^bb1
  ^bb3106:
    pdl_interp.are_equal %1656, %2 : !pdl.value -> ^bb3107, ^bb1
  ^bb3107:
    %1657 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %1657 : !pdl.value -> ^bb3108, ^bb1
  ^bb3108:
    pdl_interp.is_not_null %1444 : !pdl.value -> ^bb3109, ^bb1
  ^bb3109:
    %1658 = pdl_interp.get_value_type of %1657 : !pdl.type
    %1659 = pdl_interp.get_value_type of %1656 : !pdl.type
    pdl_interp.are_equal %1658, %1659 : !pdl.type -> ^bb3110, ^bb1
  ^bb3110:
    %1660 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1658, %1660 : !pdl.type -> ^bb3111, ^bb1
  ^bb3111:
    pdl_interp.check_type %1658 is f32 -> ^bb3112, ^bb1
  ^bb3112:
    pdl_interp.check_operation_name of %1445 is "math.powf" -> ^bb3113, ^bb1
  ^bb3113:
    pdl_interp.check_operand_count of %1445 is 2 -> ^bb3114, ^bb1
  ^bb3114:
    pdl_interp.check_result_count of %1445 is 1 -> ^bb3115, ^bb1
  ^bb3115:
    %1661 = pdl_interp.get_result 0 of %1445
    pdl_interp.is_not_null %1661 : !pdl.value -> ^bb3116, ^bb1
  ^bb3116:
    pdl_interp.are_equal %1661, %1444 : !pdl.value -> ^bb3117, ^bb1
  ^bb3117:
    %1662 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %1662 : !pdl.value -> ^bb3118, ^bb1
  ^bb3118:
    %1663 = pdl_interp.get_defining_op of %1662 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %1663 : !pdl.operation -> ^bb3119, ^bb3120
  ^bb3120:
    %1664 = pdl_interp.get_operand 1 of %1445
    pdl_interp.is_not_null %1664 : !pdl.value -> ^bb3121, ^bb1
  ^bb3121:
    %1665 = pdl_interp.get_value_type of %1661 : !pdl.type
    pdl_interp.are_equal %1658, %1665 : !pdl.type -> ^bb3122, ^bb1
  ^bb3122:
    %1666 = pdl_interp.get_value_type of %1662 : !pdl.type
    pdl_interp.are_equal %1658, %1666 : !pdl.type -> ^bb3123, ^bb1
  ^bb3123:
    %1667 = pdl_interp.get_operand 0 of %1445
    pdl_interp.are_equal %1657, %1667 : !pdl.value -> ^bb3124, ^bb1
  ^bb3124:
    %1668 = pdl_interp.get_value_type of %1664 : !pdl.type
    pdl_interp.are_equal %1658, %1668 : !pdl.type -> ^bb3125, ^bb1
  ^bb3125:
    pdl_interp.record_match @rewriters::@pow_div(%1662, %1664, %1657, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb1
  ^bb3119:
    %1669 = pdl_interp.get_operand 0 of %1445
    pdl_interp.is_not_null %1669 : !pdl.value -> ^bb3126, ^bb3120
  ^bb3126:
    %1670 = pdl_interp.get_operand 1 of %1445
    %1671 = pdl_interp.get_defining_op of %1670 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %1671 : !pdl.operation -> ^bb3127, ^bb3120
  ^bb3127:
    pdl_interp.is_not_null %1670 : !pdl.value -> ^bb3128, ^bb3120
  ^bb3128:
    %1672 = pdl_interp.get_value_type of %1661 : !pdl.type
    pdl_interp.are_equal %1658, %1672 : !pdl.type -> ^bb3129, ^bb3120
  ^bb3129:
    pdl_interp.check_operation_name of %1663 is "arith.constant" -> ^bb3130, ^bb3120
  ^bb3130:
    pdl_interp.check_operand_count of %1663 is 0 -> ^bb3131, ^bb3120
  ^bb3131:
    pdl_interp.check_result_count of %1663 is 1 -> ^bb3132, ^bb3120
  ^bb3132:
    %1673 = pdl_interp.get_result 0 of %1663
    pdl_interp.is_not_null %1673 : !pdl.value -> ^bb3133, ^bb3120
  ^bb3133:
    pdl_interp.are_equal %1673, %1662 : !pdl.value -> ^bb3134, ^bb3120
  ^bb3134:
    pdl_interp.check_operation_name of %1671 is "arith.constant" -> ^bb3135, ^bb3120
  ^bb3135:
    pdl_interp.check_operand_count of %1671 is 0 -> ^bb3136, ^bb3120
  ^bb3136:
    pdl_interp.check_result_count of %1671 is 1 -> ^bb3137, ^bb3120
  ^bb3137:
    %1674 = pdl_interp.get_result 0 of %1671
    pdl_interp.is_not_null %1674 : !pdl.value -> ^bb3138, ^bb3120
  ^bb3138:
    pdl_interp.are_equal %1674, %1670 : !pdl.value -> ^bb3139, ^bb3120
  ^bb3139:
    %1675 = pdl_interp.get_value_type of %1669 : !pdl.type
    pdl_interp.are_equal %1658, %1675 : !pdl.type -> ^bb3140, ^bb3120
  ^bb3140:
    %1676 = pdl_interp.get_attribute "value" of %1663
    pdl_interp.is_not_null %1676 : !pdl.attribute -> ^bb3141, ^bb3120
  ^bb3141:
    pdl_interp.check_attribute %1676 is 3.000000e+00 : f32 -> ^bb3142, ^bb3120
  ^bb3142:
    %1677 = pdl_interp.get_value_type of %1673 : !pdl.type
    pdl_interp.are_equal %1677, %1658 : !pdl.type -> ^bb3143, ^bb3120
  ^bb3143:
    %1678 = pdl_interp.get_value_type of %1674 : !pdl.type
    pdl_interp.are_equal %1678, %1658 : !pdl.type -> ^bb3144, ^bb3120
  ^bb3144:
    %1679 = pdl_interp.get_attribute "value" of %1671
    pdl_interp.is_not_null %1679 : !pdl.attribute -> ^bb3145, ^bb3120
  ^bb3145:
    pdl_interp.check_attribute %1679 is 3.000000e+00 : f32 -> ^bb3146, ^bb3120
  ^bb3146:
    pdl_interp.record_match @rewriters::@cube_div_rev(%1657, %1669, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb3120
  ^bb2709:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb3147, ^bb1
  ^bb3147:
    pdl_interp.check_result_count of %3 is 1 -> ^bb3148, ^bb1
  ^bb3148:
    %1680 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %1680 : !pdl.value -> ^bb3149, ^bb1
  ^bb3149:
    pdl_interp.are_equal %1680, %2 : !pdl.value -> ^bb3150, ^bb1
  ^bb3150:
    %1681 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %1681 : !pdl.value -> ^bb3151, ^bb1
  ^bb3151:
    pdl_interp.is_not_null %1444 : !pdl.value -> ^bb3152, ^bb1
  ^bb3152:
    %1682 = pdl_interp.get_value_type of %1681 : !pdl.type
    %1683 = pdl_interp.get_value_type of %1680 : !pdl.type
    pdl_interp.are_equal %1682, %1683 : !pdl.type -> ^bb3153, ^bb1
  ^bb3153:
    %1684 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1682, %1684 : !pdl.type -> ^bb3154, ^bb1
  ^bb3154:
    pdl_interp.check_type %1682 is f32 -> ^bb3155, ^bb1
  ^bb3155:
    pdl_interp.switch_operation_name of %1445 to ["math.cbrt", "math.absf"](^bb3156, ^bb3157) -> ^bb1
  ^bb3156:
    pdl_interp.check_operand_count of %1445 is 1 -> ^bb3158, ^bb1
  ^bb3158:
    pdl_interp.check_result_count of %1445 is 1 -> ^bb3159, ^bb1
  ^bb3159:
    %1685 = pdl_interp.get_result 0 of %1445
    pdl_interp.is_not_null %1685 : !pdl.value -> ^bb3160, ^bb1
  ^bb3160:
    pdl_interp.are_equal %1685, %1444 : !pdl.value -> ^bb3161, ^bb1
  ^bb3161:
    %1686 = pdl_interp.get_operand 0 of %1445
    pdl_interp.is_not_null %1686 : !pdl.value -> ^bb3162, ^bb1
  ^bb3162:
    %1687 = pdl_interp.get_value_type of %1685 : !pdl.type
    pdl_interp.are_equal %1682, %1687 : !pdl.type -> ^bb3163, ^bb1
  ^bb3163:
    %1688 = pdl_interp.get_value_type of %1686 : !pdl.type
    pdl_interp.are_equal %1682, %1688 : !pdl.type -> ^bb3164, ^bb1
  ^bb3164:
    pdl_interp.record_match @rewriters::@cbrt_undiv(%1681, %1686, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb1
  ^bb3157:
    pdl_interp.check_operand_count of %1445 is 1 -> ^bb3165, ^bb1
  ^bb3165:
    pdl_interp.check_result_count of %1445 is 1 -> ^bb3166, ^bb1
  ^bb3166:
    %1689 = pdl_interp.get_result 0 of %1445
    pdl_interp.is_not_null %1689 : !pdl.value -> ^bb3167, ^bb1
  ^bb3167:
    pdl_interp.are_equal %1689, %1444 : !pdl.value -> ^bb3168, ^bb1
  ^bb3168:
    %1690 = pdl_interp.get_operand 0 of %1445
    pdl_interp.is_not_null %1690 : !pdl.value -> ^bb3169, ^bb1
  ^bb3169:
    %1691 = pdl_interp.get_defining_op of %1690 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1691 : !pdl.operation -> ^bb3170, ^bb1
  ^bb3170:
    %1692 = pdl_interp.get_value_type of %1689 : !pdl.type
    pdl_interp.are_equal %1682, %1692 : !pdl.type -> ^bb3171, ^bb1
  ^bb3171:
    pdl_interp.check_operation_name of %1691 is "math.cbrt" -> ^bb3172, ^bb1
  ^bb3172:
    pdl_interp.check_operand_count of %1691 is 1 -> ^bb3173, ^bb1
  ^bb3173:
    pdl_interp.check_result_count of %1691 is 1 -> ^bb3174, ^bb1
  ^bb3174:
    %1693 = pdl_interp.get_result 0 of %1691
    pdl_interp.is_not_null %1693 : !pdl.value -> ^bb3175, ^bb1
  ^bb3175:
    pdl_interp.are_equal %1693, %1690 : !pdl.value -> ^bb3176, ^bb1
  ^bb3176:
    %1694 = pdl_interp.get_value_type of %1693 : !pdl.type
    pdl_interp.are_equal %1694, %1682 : !pdl.type -> ^bb3177, ^bb1
  ^bb3177:
    %1695 = pdl_interp.get_operand 0 of %1691
    pdl_interp.are_equal %1695, %1681 : !pdl.value -> ^bb3178, ^bb1
  ^bb3178:
    pdl_interp.record_match @rewriters::@cbrt_div_cbrt(%1681, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb1
  ^bb2710:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb3179, ^bb1
  ^bb3179:
    pdl_interp.check_result_count of %3 is 1 -> ^bb3180, ^bb1
  ^bb3180:
    %1696 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %1696 : !pdl.value -> ^bb3181, ^bb1
  ^bb3181:
    pdl_interp.are_equal %1696, %2 : !pdl.value -> ^bb3182, ^bb1
  ^bb3182:
    %1697 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %1697 : !pdl.value -> ^bb3183, ^bb1
  ^bb3183:
    pdl_interp.is_not_null %1444 : !pdl.value -> ^bb3184, ^bb1
  ^bb3184:
    %1698 = pdl_interp.get_value_type of %1697 : !pdl.type
    %1699 = pdl_interp.get_value_type of %1696 : !pdl.type
    pdl_interp.are_equal %1698, %1699 : !pdl.type -> ^bb3185, ^bb1
  ^bb3185:
    %1700 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1698, %1700 : !pdl.type -> ^bb3186, ^bb1
  ^bb3186:
    pdl_interp.check_type %1698 is f32 -> ^bb3187, ^bb1
  ^bb3187:
    pdl_interp.check_operation_name of %1445 is "math.exp" -> ^bb3188, ^bb1
  ^bb3188:
    pdl_interp.check_operand_count of %1445 is 1 -> ^bb3189, ^bb1
  ^bb3189:
    pdl_interp.check_result_count of %1445 is 1 -> ^bb3190, ^bb1
  ^bb3190:
    %1701 = pdl_interp.get_result 0 of %1445
    pdl_interp.is_not_null %1701 : !pdl.value -> ^bb3191, ^bb1
  ^bb3191:
    pdl_interp.are_equal %1701, %1444 : !pdl.value -> ^bb3192, ^bb1
  ^bb3192:
    %1702 = pdl_interp.get_operand 0 of %1445
    pdl_interp.is_not_null %1702 : !pdl.value -> ^bb3193, ^bb1
  ^bb3193:
    %1703 = pdl_interp.get_value_type of %1701 : !pdl.type
    pdl_interp.are_equal %1698, %1703 : !pdl.type -> ^bb3194, ^bb1
  ^bb3194:
    %1704 = pdl_interp.get_value_type of %1702 : !pdl.type
    pdl_interp.are_equal %1698, %1704 : !pdl.type -> ^bb3195, ^bb1
  ^bb3195:
    pdl_interp.record_match @rewriters::@div_exp(%1697, %1702, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb1
  ^bb2711:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb3196, ^bb1
  ^bb3196:
    pdl_interp.check_result_count of %3 is 1 -> ^bb3197, ^bb1
  ^bb3197:
    %1705 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %1705 : !pdl.value -> ^bb3198, ^bb1
  ^bb3198:
    pdl_interp.are_equal %1705, %2 : !pdl.value -> ^bb3199, ^bb1
  ^bb3199:
    %1706 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %1706 : !pdl.value -> ^bb3200, ^bb1
  ^bb3200:
    pdl_interp.is_not_null %1444 : !pdl.value -> ^bb3201, ^bb1
  ^bb3201:
    %1707 = pdl_interp.get_value_type of %1706 : !pdl.type
    %1708 = pdl_interp.get_value_type of %1705 : !pdl.type
    pdl_interp.are_equal %1707, %1708 : !pdl.type -> ^bb3202, ^bb1
  ^bb3202:
    %1709 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1707, %1709 : !pdl.type -> ^bb3203, ^bb1
  ^bb3203:
    pdl_interp.check_type %1707 is f32 -> ^bb3204, ^bb1
  ^bb3204:
    pdl_interp.switch_operation_name of %1445 to ["arith.addf", "math.cos"](^bb3205, ^bb3206) -> ^bb1
  ^bb3205:
    pdl_interp.check_operand_count of %1445 is 2 -> ^bb3207, ^bb1
  ^bb3207:
    pdl_interp.check_result_count of %1445 is 1 -> ^bb3208, ^bb1
  ^bb3208:
    %1710 = pdl_interp.get_result 0 of %1445
    pdl_interp.is_not_null %1710 : !pdl.value -> ^bb3209, ^bb1
  ^bb3209:
    pdl_interp.are_equal %1710, %1444 : !pdl.value -> ^bb3210, ^bb1
  ^bb3210:
    %1711 = pdl_interp.get_operand 0 of %1445
    pdl_interp.is_not_null %1711 : !pdl.value -> ^bb3211, ^bb1
  ^bb3211:
    %1712 = pdl_interp.get_defining_op of %1711 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1712 : !pdl.operation -> ^bb3212, ^bb1
  ^bb3212:
    %1713 = pdl_interp.get_operand 1 of %1445
    %1714 = pdl_interp.get_defining_op of %1713 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %1714 : !pdl.operation -> ^bb3213, ^bb1
  ^bb3213:
    pdl_interp.is_not_null %1713 : !pdl.value -> ^bb3214, ^bb1
  ^bb3214:
    %1715 = pdl_interp.get_value_type of %1710 : !pdl.type
    pdl_interp.are_equal %1707, %1715 : !pdl.type -> ^bb3215, ^bb1
  ^bb3215:
    pdl_interp.check_operation_name of %1712 is "arith.constant" -> ^bb3216, ^bb1
  ^bb3216:
    pdl_interp.check_operand_count of %1712 is 0 -> ^bb3217, ^bb1
  ^bb3217:
    pdl_interp.check_result_count of %1712 is 1 -> ^bb3218, ^bb1
  ^bb3218:
    %1716 = pdl_interp.get_result 0 of %1712
    pdl_interp.is_not_null %1716 : !pdl.value -> ^bb3219, ^bb1
  ^bb3219:
    pdl_interp.are_equal %1716, %1711 : !pdl.value -> ^bb3220, ^bb1
  ^bb3220:
    pdl_interp.check_operation_name of %1714 is "math.cos" -> ^bb3221, ^bb1
  ^bb3221:
    pdl_interp.check_operand_count of %1714 is 1 -> ^bb3222, ^bb1
  ^bb3222:
    pdl_interp.check_result_count of %1714 is 1 -> ^bb3223, ^bb1
  ^bb3223:
    %1717 = pdl_interp.get_result 0 of %1714
    pdl_interp.is_not_null %1717 : !pdl.value -> ^bb3224, ^bb1
  ^bb3224:
    pdl_interp.are_equal %1717, %1713 : !pdl.value -> ^bb3225, ^bb1
  ^bb3225:
    %1718 = pdl_interp.get_attribute "value" of %1712
    pdl_interp.is_not_null %1718 : !pdl.attribute -> ^bb3226, ^bb1
  ^bb3226:
    pdl_interp.check_attribute %1718 is 1.000000e+00 : f32 -> ^bb3227, ^bb1
  ^bb3227:
    %1719 = pdl_interp.get_value_type of %1717 : !pdl.type
    pdl_interp.are_equal %1719, %1707 : !pdl.type -> ^bb3228, ^bb1
  ^bb3228:
    %1720 = pdl_interp.get_value_type of %1716 : !pdl.type
    pdl_interp.are_equal %1720, %1707 : !pdl.type -> ^bb3229, ^bb1
  ^bb3229:
    %1721 = pdl_interp.get_operand 0 of %1714
    pdl_interp.are_equal %1721, %1706 : !pdl.value -> ^bb3230, ^bb1
  ^bb3230:
    pdl_interp.record_match @rewriters::@hang_0p_tan(%1706, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb1
  ^bb3206:
    pdl_interp.check_operand_count of %1445 is 1 -> ^bb3231, ^bb1
  ^bb3231:
    pdl_interp.check_result_count of %1445 is 1 -> ^bb3232, ^bb1
  ^bb3232:
    %1722 = pdl_interp.get_result 0 of %1445
    pdl_interp.is_not_null %1722 : !pdl.value -> ^bb3233, ^bb1
  ^bb3233:
    pdl_interp.are_equal %1722, %1444 : !pdl.value -> ^bb3234, ^bb1
  ^bb3234:
    %1723 = pdl_interp.get_value_type of %1722 : !pdl.type
    pdl_interp.are_equal %1707, %1723 : !pdl.type -> ^bb3235, ^bb1
  ^bb3235:
    %1724 = pdl_interp.get_operand 0 of %1445
    pdl_interp.are_equal %1706, %1724 : !pdl.value -> ^bb3236, ^bb1
  ^bb3236:
    pdl_interp.record_match @rewriters::@quot_tan(%1706, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb1
  ^bb2712:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb3237, ^bb1
  ^bb3237:
    pdl_interp.check_result_count of %3 is 1 -> ^bb3238, ^bb1
  ^bb3238:
    %1725 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %1725 : !pdl.value -> ^bb3239, ^bb1
  ^bb3239:
    pdl_interp.are_equal %1725, %2 : !pdl.value -> ^bb3240, ^bb1
  ^bb3240:
    %1726 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %1726 : !pdl.value -> ^bb3241, ^bb1
  ^bb3241:
    pdl_interp.is_not_null %1444 : !pdl.value -> ^bb3242, ^bb1
  ^bb3242:
    pdl_interp.switch_operation_name of %1445 to ["math.sin", "arith.negf", "arith.addf", "arith.constant", "math.sinh"](^bb3243, ^bb3244, ^bb3245, ^bb3246, ^bb3247) -> ^bb1
  ^bb3243:
    pdl_interp.check_operand_count of %1445 is 1 -> ^bb3248, ^bb1
  ^bb3248:
    pdl_interp.check_result_count of %1445 is 1 -> ^bb3249, ^bb1
  ^bb3249:
    %1727 = pdl_interp.get_result 0 of %1445
    pdl_interp.is_not_null %1727 : !pdl.value -> ^bb3250, ^bb1
  ^bb3250:
    pdl_interp.are_equal %1727, %1444 : !pdl.value -> ^bb3251, ^bb1
  ^bb3251:
    %1728 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %1728 : !pdl.value -> ^bb3252, ^bb1
  ^bb3252:
    %1729 = pdl_interp.get_defining_op of %1728 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %1729 : !pdl.operation -> ^bb3253, ^bb1
  ^bb3253:
    %1730 = pdl_interp.get_defining_op of %1726 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1730 : !pdl.operation -> ^bb3254, ^bb1
  ^bb3254:
    pdl_interp.check_operation_name of %1729 is "math.cos" -> ^bb3255, ^bb1
  ^bb3255:
    pdl_interp.check_operand_count of %1729 is 1 -> ^bb3256, ^bb1
  ^bb3256:
    pdl_interp.check_result_count of %1729 is 1 -> ^bb3257, ^bb1
  ^bb3257:
    %1731 = pdl_interp.get_result 0 of %1729
    pdl_interp.is_not_null %1731 : !pdl.value -> ^bb3258, ^bb1
  ^bb3258:
    pdl_interp.are_equal %1731, %1728 : !pdl.value -> ^bb3259, ^bb1
  ^bb3259:
    pdl_interp.check_operation_name of %1730 is "arith.constant" -> ^bb3260, ^bb1
  ^bb3260:
    pdl_interp.check_operand_count of %1730 is 0 -> ^bb3261, ^bb1
  ^bb3261:
    pdl_interp.check_result_count of %1730 is 1 -> ^bb3262, ^bb1
  ^bb3262:
    %1732 = pdl_interp.get_result 0 of %1730
    pdl_interp.is_not_null %1732 : !pdl.value -> ^bb3263, ^bb1
  ^bb3263:
    pdl_interp.are_equal %1732, %1726 : !pdl.value -> ^bb3264, ^bb1
  ^bb3264:
    %1733 = pdl_interp.get_operand 0 of %1729
    pdl_interp.is_not_null %1733 : !pdl.value -> ^bb3265, ^bb1
  ^bb3265:
    %1734 = pdl_interp.get_attribute "value" of %1730
    pdl_interp.is_not_null %1734 : !pdl.attribute -> ^bb3266, ^bb1
  ^bb3266:
    pdl_interp.check_attribute %1734 is 1.000000e+00 : f32 -> ^bb3267, ^bb1
  ^bb3267:
    %1735 = pdl_interp.get_value_type of %1732 : !pdl.type
    %1736 = pdl_interp.get_value_type of %1725 : !pdl.type
    pdl_interp.are_equal %1735, %1736 : !pdl.type -> ^bb3268, ^bb1
  ^bb3268:
    %1737 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1735, %1737 : !pdl.type -> ^bb3269, ^bb1
  ^bb3269:
    pdl_interp.check_type %1735 is f32 -> ^bb3270, ^bb1
  ^bb3270:
    %1738 = pdl_interp.get_value_type of %1731 : !pdl.type
    pdl_interp.are_equal %1735, %1738 : !pdl.type -> ^bb3271, ^bb1
  ^bb3271:
    %1739 = pdl_interp.get_value_type of %1733 : !pdl.type
    pdl_interp.are_equal %1735, %1739 : !pdl.type -> ^bb3272, ^bb1
  ^bb3272:
    %1740 = pdl_interp.get_value_type of %1727 : !pdl.type
    pdl_interp.are_equal %1735, %1740 : !pdl.type -> ^bb3273, ^bb1
  ^bb3273:
    %1741 = pdl_interp.get_operand 0 of %1445
    pdl_interp.are_equal %1733, %1741 : !pdl.value -> ^bb3274, ^bb1
  ^bb3274:
    pdl_interp.record_match @rewriters::@hang_p0_tan(%1733, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb1
  ^bb3244:
    pdl_interp.check_operand_count of %1445 is 1 -> ^bb3275, ^bb1
  ^bb3275:
    pdl_interp.check_result_count of %1445 is 1 -> ^bb3276, ^bb1
  ^bb3276:
    %1742 = pdl_interp.get_result 0 of %1445
    pdl_interp.is_not_null %1742 : !pdl.value -> ^bb3277, ^bb1
  ^bb3277:
    pdl_interp.are_equal %1742, %1444 : !pdl.value -> ^bb3278, ^bb1
  ^bb3278:
    %1743 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %1743 : !pdl.value -> ^bb3279, ^bb1
  ^bb3279:
    %1744 = pdl_interp.get_defining_op of %1743 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %1744 : !pdl.operation -> ^bb3280, ^bb1
  ^bb3280:
    %1745 = pdl_interp.get_defining_op of %1726 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1745 : !pdl.operation -> ^bb3281, ^bb1
  ^bb3281:
    %1746 = pdl_interp.get_operand 0 of %1445
    pdl_interp.is_not_null %1746 : !pdl.value -> ^bb3282, ^bb1
  ^bb3282:
    %1747 = pdl_interp.get_defining_op of %1746 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1747 : !pdl.operation -> ^bb3283, ^bb1
  ^bb3283:
    pdl_interp.check_operation_name of %1744 is "math.cos" -> ^bb3284, ^bb1
  ^bb3284:
    pdl_interp.check_operand_count of %1744 is 1 -> ^bb3285, ^bb1
  ^bb3285:
    pdl_interp.check_result_count of %1744 is 1 -> ^bb3286, ^bb1
  ^bb3286:
    %1748 = pdl_interp.get_result 0 of %1744
    pdl_interp.is_not_null %1748 : !pdl.value -> ^bb3287, ^bb1
  ^bb3287:
    pdl_interp.are_equal %1748, %1743 : !pdl.value -> ^bb3288, ^bb1
  ^bb3288:
    pdl_interp.check_operation_name of %1745 is "arith.constant" -> ^bb3289, ^bb1
  ^bb3289:
    pdl_interp.check_operand_count of %1745 is 0 -> ^bb3290, ^bb1
  ^bb3290:
    pdl_interp.check_result_count of %1745 is 1 -> ^bb3291, ^bb1
  ^bb3291:
    %1749 = pdl_interp.get_result 0 of %1745
    pdl_interp.is_not_null %1749 : !pdl.value -> ^bb3292, ^bb1
  ^bb3292:
    pdl_interp.are_equal %1749, %1726 : !pdl.value -> ^bb3293, ^bb1
  ^bb3293:
    pdl_interp.check_operation_name of %1747 is "math.sin" -> ^bb3294, ^bb1
  ^bb3294:
    pdl_interp.check_operand_count of %1747 is 1 -> ^bb3295, ^bb1
  ^bb3295:
    pdl_interp.check_result_count of %1747 is 1 -> ^bb3296, ^bb1
  ^bb3296:
    %1750 = pdl_interp.get_result 0 of %1747
    pdl_interp.is_not_null %1750 : !pdl.value -> ^bb3297, ^bb1
  ^bb3297:
    pdl_interp.are_equal %1750, %1746 : !pdl.value -> ^bb3298, ^bb1
  ^bb3298:
    %1751 = pdl_interp.get_operand 0 of %1744
    pdl_interp.is_not_null %1751 : !pdl.value -> ^bb3299, ^bb1
  ^bb3299:
    %1752 = pdl_interp.get_attribute "value" of %1745
    pdl_interp.is_not_null %1752 : !pdl.attribute -> ^bb3300, ^bb1
  ^bb3300:
    pdl_interp.check_attribute %1752 is 1.000000e+00 : f32 -> ^bb3301, ^bb1
  ^bb3301:
    %1753 = pdl_interp.get_value_type of %1749 : !pdl.type
    %1754 = pdl_interp.get_value_type of %1725 : !pdl.type
    pdl_interp.are_equal %1753, %1754 : !pdl.type -> ^bb3302, ^bb1
  ^bb3302:
    %1755 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1753, %1755 : !pdl.type -> ^bb3303, ^bb1
  ^bb3303:
    pdl_interp.check_type %1753 is f32 -> ^bb3304, ^bb1
  ^bb3304:
    %1756 = pdl_interp.get_value_type of %1748 : !pdl.type
    pdl_interp.are_equal %1753, %1756 : !pdl.type -> ^bb3305, ^bb1
  ^bb3305:
    %1757 = pdl_interp.get_value_type of %1751 : !pdl.type
    pdl_interp.are_equal %1753, %1757 : !pdl.type -> ^bb3306, ^bb1
  ^bb3306:
    %1758 = pdl_interp.get_value_type of %1742 : !pdl.type
    pdl_interp.are_equal %1753, %1758 : !pdl.type -> ^bb3307, ^bb1
  ^bb3307:
    %1759 = pdl_interp.get_value_type of %1750 : !pdl.type
    pdl_interp.are_equal %1753, %1759 : !pdl.type -> ^bb3308, ^bb1
  ^bb3308:
    %1760 = pdl_interp.get_operand 0 of %1747
    pdl_interp.are_equal %1751, %1760 : !pdl.value -> ^bb3309, ^bb1
  ^bb3309:
    pdl_interp.record_match @rewriters::@hang_m0_tan(%1751, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb1
  ^bb3245:
    pdl_interp.check_operand_count of %1445 is 2 -> ^bb3310, ^bb1
  ^bb3310:
    pdl_interp.check_result_count of %1445 is 1 -> ^bb3311, ^bb1
  ^bb3311:
    %1761 = pdl_interp.get_result 0 of %1445
    pdl_interp.is_not_null %1761 : !pdl.value -> ^bb3312, ^bb1
  ^bb3312:
    pdl_interp.are_equal %1761, %1444 : !pdl.value -> ^bb3313, ^bb1
  ^bb3313:
    %1762 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %1762 : !pdl.value -> ^bb3314, ^bb1
  ^bb3314:
    %1763 = pdl_interp.get_defining_op of %1762 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %1763 : !pdl.operation -> ^bb3315, ^bb1
  ^bb3315:
    %1764 = pdl_interp.get_defining_op of %1726 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1764 : !pdl.operation -> ^bb3316, ^bb1
  ^bb3316:
    %1765 = pdl_interp.get_operand 0 of %1445
    pdl_interp.is_not_null %1765 : !pdl.value -> ^bb3317, ^bb1
  ^bb3317:
    %1766 = pdl_interp.get_defining_op of %1765 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1766 : !pdl.operation -> ^bb3318, ^bb1
  ^bb3318:
    %1767 = pdl_interp.get_operand 1 of %1445
    %1768 = pdl_interp.get_defining_op of %1767 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %1768 : !pdl.operation -> ^bb3319, ^bb1
  ^bb3319:
    pdl_interp.is_not_null %1767 : !pdl.value -> ^bb3320, ^bb1
  ^bb3320:
    pdl_interp.switch_operation_name of %1763 to ["math.sin", "arith.constant", "math.exp"](^bb3321, ^bb3322, ^bb3323) -> ^bb1
  ^bb3321:
    pdl_interp.check_operand_count of %1763 is 1 -> ^bb3324, ^bb1
  ^bb3324:
    pdl_interp.check_result_count of %1763 is 1 -> ^bb3325, ^bb1
  ^bb3325:
    %1769 = pdl_interp.get_result 0 of %1763
    pdl_interp.is_not_null %1769 : !pdl.value -> ^bb3326, ^bb1
  ^bb3326:
    pdl_interp.are_equal %1769, %1762 : !pdl.value -> ^bb3327, ^bb1
  ^bb3327:
    pdl_interp.check_operation_name of %1764 is "math.sin" -> ^bb3328, ^bb1
  ^bb3328:
    pdl_interp.check_operand_count of %1764 is 1 -> ^bb3329, ^bb1
  ^bb3329:
    pdl_interp.check_result_count of %1764 is 1 -> ^bb3330, ^bb1
  ^bb3330:
    %1770 = pdl_interp.get_result 0 of %1764
    pdl_interp.is_not_null %1770 : !pdl.value -> ^bb3331, ^bb1
  ^bb3331:
    pdl_interp.are_equal %1770, %1726 : !pdl.value -> ^bb3332, ^bb1
  ^bb3332:
    pdl_interp.check_operation_name of %1766 is "math.cos" -> ^bb3333, ^bb1
  ^bb3333:
    pdl_interp.check_operand_count of %1766 is 1 -> ^bb3334, ^bb1
  ^bb3334:
    pdl_interp.check_result_count of %1766 is 1 -> ^bb3335, ^bb1
  ^bb3335:
    %1771 = pdl_interp.get_result 0 of %1766
    pdl_interp.is_not_null %1771 : !pdl.value -> ^bb3336, ^bb1
  ^bb3336:
    pdl_interp.are_equal %1771, %1765 : !pdl.value -> ^bb3337, ^bb1
  ^bb3337:
    pdl_interp.check_operation_name of %1768 is "math.cos" -> ^bb3338, ^bb1
  ^bb3338:
    pdl_interp.check_operand_count of %1768 is 1 -> ^bb3339, ^bb1
  ^bb3339:
    pdl_interp.check_result_count of %1768 is 1 -> ^bb3340, ^bb1
  ^bb3340:
    %1772 = pdl_interp.get_result 0 of %1768
    pdl_interp.is_not_null %1772 : !pdl.value -> ^bb3341, ^bb1
  ^bb3341:
    pdl_interp.are_equal %1772, %1767 : !pdl.value -> ^bb3342, ^bb1
  ^bb3342:
    %1773 = pdl_interp.get_operand 0 of %1764
    pdl_interp.is_not_null %1773 : !pdl.value -> ^bb3343, ^bb1
  ^bb3343:
    %1774 = pdl_interp.get_value_type of %1773 : !pdl.type
    %1775 = pdl_interp.get_value_type of %1770 : !pdl.type
    pdl_interp.are_equal %1774, %1775 : !pdl.type -> ^bb3344, ^bb1
  ^bb3344:
    %1776 = pdl_interp.get_value_type of %1725 : !pdl.type
    pdl_interp.are_equal %1774, %1776 : !pdl.type -> ^bb3345, ^bb1
  ^bb3345:
    %1777 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1774, %1777 : !pdl.type -> ^bb3346, ^bb1
  ^bb3346:
    pdl_interp.check_type %1774 is f32 -> ^bb3347, ^bb1
  ^bb3347:
    %1778 = pdl_interp.get_operand 0 of %1763
    pdl_interp.is_not_null %1778 : !pdl.value -> ^bb3348, ^bb1
  ^bb3348:
    %1779 = pdl_interp.get_value_type of %1769 : !pdl.type
    pdl_interp.are_equal %1774, %1779 : !pdl.type -> ^bb3349, ^bb1
  ^bb3349:
    %1780 = pdl_interp.get_value_type of %1761 : !pdl.type
    pdl_interp.are_equal %1774, %1780 : !pdl.type -> ^bb3350, ^bb1
  ^bb3350:
    %1781 = pdl_interp.get_value_type of %1772 : !pdl.type
    pdl_interp.are_equal %1774, %1781 : !pdl.type -> ^bb3351, ^bb1
  ^bb3351:
    %1782 = pdl_interp.get_value_type of %1771 : !pdl.type
    pdl_interp.are_equal %1774, %1782 : !pdl.type -> ^bb3352, ^bb1
  ^bb3352:
    %1783 = pdl_interp.get_operand 0 of %1766
    pdl_interp.are_equal %1773, %1783 : !pdl.value -> ^bb3353, ^bb1
  ^bb3353:
    %1784 = pdl_interp.get_value_type of %1778 : !pdl.type
    pdl_interp.are_equal %1774, %1784 : !pdl.type -> ^bb3354, ^bb1
  ^bb3354:
    %1785 = pdl_interp.get_operand 0 of %1768
    pdl_interp.are_equal %1778, %1785 : !pdl.value -> ^bb3355, ^bb1
  ^bb3355:
    pdl_interp.record_match @rewriters::@hang_m_tan(%1773, %1778, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb1
  ^bb3322:
    pdl_interp.check_operand_count of %1763 is 0 -> ^bb3356, ^bb1
  ^bb3356:
    pdl_interp.check_result_count of %1763 is 1 -> ^bb3357, ^bb1
  ^bb3357:
    %1786 = pdl_interp.get_result 0 of %1763
    pdl_interp.is_not_null %1786 : !pdl.value -> ^bb3358, ^bb1
  ^bb3358:
    pdl_interp.are_equal %1786, %1762 : !pdl.value -> ^bb3359, ^bb1
  ^bb3359:
    pdl_interp.check_operation_name of %1764 is "math.exp" -> ^bb3360, ^bb1
  ^bb3360:
    pdl_interp.check_operand_count of %1764 is 1 -> ^bb3361, ^bb1
  ^bb3361:
    pdl_interp.check_result_count of %1764 is 1 -> ^bb3362, ^bb1
  ^bb3362:
    %1787 = pdl_interp.get_result 0 of %1764
    pdl_interp.is_not_null %1787 : !pdl.value -> ^bb3363, ^bb1
  ^bb3363:
    pdl_interp.are_equal %1787, %1726 : !pdl.value -> ^bb3364, ^bb1
  ^bb3364:
    pdl_interp.check_operation_name of %1766 is "math.exp" -> ^bb3365, ^bb1
  ^bb3365:
    pdl_interp.check_operand_count of %1766 is 1 -> ^bb3366, ^bb1
  ^bb3366:
    pdl_interp.check_result_count of %1766 is 1 -> ^bb3367, ^bb1
  ^bb3367:
    %1788 = pdl_interp.get_result 0 of %1766
    pdl_interp.is_not_null %1788 : !pdl.value -> ^bb3368, ^bb1
  ^bb3368:
    pdl_interp.are_equal %1788, %1765 : !pdl.value -> ^bb3369, ^bb1
  ^bb3369:
    pdl_interp.check_operation_name of %1768 is "arith.constant" -> ^bb3370, ^bb1
  ^bb3370:
    pdl_interp.check_operand_count of %1768 is 0 -> ^bb3371, ^bb1
  ^bb3371:
    pdl_interp.check_result_count of %1768 is 1 -> ^bb3372, ^bb1
  ^bb3372:
    %1789 = pdl_interp.get_result 0 of %1768
    pdl_interp.is_not_null %1789 : !pdl.value -> ^bb3373, ^bb1
  ^bb3373:
    pdl_interp.are_equal %1789, %1767 : !pdl.value -> ^bb3374, ^bb1
  ^bb3374:
    %1790 = pdl_interp.get_operand 0 of %1764
    pdl_interp.is_not_null %1790 : !pdl.value -> ^bb3375, ^bb1
  ^bb3375:
    %1791 = pdl_interp.get_operand 0 of %1766
    %1792 = pdl_interp.get_defining_op of %1791 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1792 : !pdl.operation -> ^bb3376, ^bb1
  ^bb3376:
    pdl_interp.is_not_null %1791 : !pdl.value -> ^bb3377, ^bb1
  ^bb3377:
    %1793 = pdl_interp.get_attribute "value" of %1763
    pdl_interp.is_not_null %1793 : !pdl.attribute -> ^bb3378, ^bb1
  ^bb3378:
    pdl_interp.check_attribute %1793 is 1.000000e+00 : f32 -> ^bb3379, ^bb1
  ^bb3379:
    %1794 = pdl_interp.get_operand 0 of %1792
    %1795 = pdl_interp.get_defining_op of %1794 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1795 : !pdl.operation -> ^bb3380, ^bb1
  ^bb3380:
    pdl_interp.check_operation_name of %1792 is "arith.mulf" -> ^bb3381, ^bb1
  ^bb3381:
    pdl_interp.check_operand_count of %1792 is 2 -> ^bb3382, ^bb1
  ^bb3382:
    pdl_interp.check_result_count of %1792 is 1 -> ^bb3383, ^bb1
  ^bb3383:
    %1796 = pdl_interp.get_result 0 of %1792
    pdl_interp.is_not_null %1796 : !pdl.value -> ^bb3384, ^bb1
  ^bb3384:
    pdl_interp.are_equal %1796, %1791 : !pdl.value -> ^bb3385, ^bb1
  ^bb3385:
    %1797 = pdl_interp.get_defining_op of %1790 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1797 : !pdl.operation -> ^bb3386, ^bb1
  ^bb3386:
    %1798 = pdl_interp.get_attribute "value" of %1768
    pdl_interp.is_not_null %1798 : !pdl.attribute -> ^bb3387, ^bb1
  ^bb3387:
    pdl_interp.check_attribute %1798 is 1.000000e+00 : f32 -> ^bb3388, ^bb1
  ^bb3388:
    pdl_interp.is_not_null %1794 : !pdl.value -> ^bb3389, ^bb1
  ^bb3389:
    pdl_interp.check_operation_name of %1795 is "arith.constant" -> ^bb3390, ^bb1
  ^bb3390:
    pdl_interp.check_operand_count of %1795 is 0 -> ^bb3391, ^bb1
  ^bb3391:
    pdl_interp.check_result_count of %1795 is 1 -> ^bb3392, ^bb1
  ^bb3392:
    %1799 = pdl_interp.get_result 0 of %1795
    pdl_interp.is_not_null %1799 : !pdl.value -> ^bb3393, ^bb1
  ^bb3393:
    pdl_interp.are_equal %1799, %1794 : !pdl.value -> ^bb3394, ^bb1
  ^bb3394:
    pdl_interp.check_operation_name of %1797 is "arith.mulf" -> ^bb3395, ^bb1
  ^bb3395:
    pdl_interp.check_operand_count of %1797 is 2 -> ^bb3396, ^bb1
  ^bb3396:
    pdl_interp.check_result_count of %1797 is 1 -> ^bb3397, ^bb1
  ^bb3397:
    %1800 = pdl_interp.get_result 0 of %1797
    pdl_interp.is_not_null %1800 : !pdl.value -> ^bb3398, ^bb1
  ^bb3398:
    pdl_interp.are_equal %1800, %1790 : !pdl.value -> ^bb3399, ^bb1
  ^bb3399:
    %1801 = pdl_interp.get_operand 0 of %1797
    pdl_interp.is_not_null %1801 : !pdl.value -> ^bb3400, ^bb1
  ^bb3400:
    %1802 = pdl_interp.get_operand 1 of %1797
    pdl_interp.is_not_null %1802 : !pdl.value -> ^bb3401, ^bb1
  ^bb3401:
    %1803 = pdl_interp.get_defining_op of %1801 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1803 : !pdl.operation -> ^bb3402, ^bb1
  ^bb3402:
    pdl_interp.check_operation_name of %1803 is "arith.constant" -> ^bb3403, ^bb1
  ^bb3403:
    pdl_interp.check_operand_count of %1803 is 0 -> ^bb3404, ^bb1
  ^bb3404:
    pdl_interp.check_result_count of %1803 is 1 -> ^bb3405, ^bb1
  ^bb3405:
    %1804 = pdl_interp.get_attribute "value" of %1803
    pdl_interp.is_not_null %1804 : !pdl.attribute -> ^bb3406, ^bb1
  ^bb3406:
    pdl_interp.check_attribute %1804 is 2.000000e+00 : f32 -> ^bb3407, ^bb1
  ^bb3407:
    %1805 = pdl_interp.get_result 0 of %1803
    pdl_interp.is_not_null %1805 : !pdl.value -> ^bb3408, ^bb1
  ^bb3408:
    pdl_interp.are_equal %1805, %1801 : !pdl.value -> ^bb3409, ^bb1
  ^bb3409:
    %1806 = pdl_interp.get_value_type of %1805 : !pdl.type
    %1807 = pdl_interp.get_value_type of %1802 : !pdl.type
    pdl_interp.are_equal %1806, %1807 : !pdl.type -> ^bb3410, ^bb1
  ^bb3410:
    %1808 = pdl_interp.get_value_type of %1800 : !pdl.type
    pdl_interp.are_equal %1806, %1808 : !pdl.type -> ^bb3411, ^bb1
  ^bb3411:
    %1809 = pdl_interp.get_value_type of %1787 : !pdl.type
    pdl_interp.are_equal %1806, %1809 : !pdl.type -> ^bb3412, ^bb1
  ^bb3412:
    %1810 = pdl_interp.get_value_type of %1725 : !pdl.type
    pdl_interp.are_equal %1806, %1810 : !pdl.type -> ^bb3413, ^bb1
  ^bb3413:
    %1811 = pdl_interp.get_value_type of %1761 : !pdl.type
    pdl_interp.are_equal %1806, %1811 : !pdl.type -> ^bb3414, ^bb1
  ^bb3414:
    %1812 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1806, %1812 : !pdl.type -> ^bb3415, ^bb1
  ^bb3415:
    pdl_interp.check_type %1806 is f32 -> ^bb3416, ^bb1
  ^bb3416:
    %1813 = pdl_interp.get_operand 1 of %1792
    pdl_interp.are_equal %1802, %1813 : !pdl.value -> ^bb3417, ^bb1
  ^bb3417:
    %1814 = pdl_interp.get_attribute "value" of %1795
    pdl_interp.is_not_null %1814 : !pdl.attribute -> ^bb3418, ^bb1
  ^bb3418:
    pdl_interp.check_attribute %1814 is 2.000000e+00 : f32 -> ^bb3419, ^bb1
  ^bb3419:
    %1815 = pdl_interp.get_value_type of %1786 : !pdl.type
    pdl_interp.are_equal %1806, %1815 : !pdl.type -> ^bb3420, ^bb1
  ^bb3420:
    %1816 = pdl_interp.get_value_type of %1799 : !pdl.type
    pdl_interp.are_equal %1806, %1816 : !pdl.type -> ^bb3421, ^bb1
  ^bb3421:
    %1817 = pdl_interp.get_value_type of %1796 : !pdl.type
    pdl_interp.are_equal %1806, %1817 : !pdl.type -> ^bb3422, ^bb1
  ^bb3422:
    %1818 = pdl_interp.get_value_type of %1788 : !pdl.type
    pdl_interp.are_equal %1806, %1818 : !pdl.type -> ^bb3423, ^bb1
  ^bb3423:
    %1819 = pdl_interp.get_value_type of %1789 : !pdl.type
    pdl_interp.are_equal %1806, %1819 : !pdl.type -> ^bb3424, ^bb1
  ^bb3424:
    pdl_interp.record_match @rewriters::@tanh_def_b_rev(%1802, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb1
  ^bb3323:
    pdl_interp.check_operand_count of %1763 is 1 -> ^bb3425, ^bb1
  ^bb3425:
    pdl_interp.check_result_count of %1763 is 1 -> ^bb3426, ^bb1
  ^bb3426:
    %1820 = pdl_interp.get_result 0 of %1763
    pdl_interp.is_not_null %1820 : !pdl.value -> ^bb3427, ^bb1
  ^bb3427:
    pdl_interp.are_equal %1820, %1762 : !pdl.value -> ^bb3428, ^bb1
  ^bb3428:
    pdl_interp.switch_operation_name of %1764 to ["arith.constant", "math.exp"](^bb3429, ^bb3430) -> ^bb1
  ^bb3429:
    pdl_interp.check_operand_count of %1764 is 0 -> ^bb3431, ^bb1
  ^bb3431:
    pdl_interp.check_result_count of %1764 is 1 -> ^bb3432, ^bb1
  ^bb3432:
    %1821 = pdl_interp.get_result 0 of %1764
    pdl_interp.is_not_null %1821 : !pdl.value -> ^bb3433, ^bb1
  ^bb3433:
    pdl_interp.are_equal %1821, %1726 : !pdl.value -> ^bb3434, ^bb1
  ^bb3434:
    pdl_interp.check_operation_name of %1766 is "arith.constant" -> ^bb3435, ^bb1
  ^bb3435:
    pdl_interp.check_operand_count of %1766 is 0 -> ^bb3436, ^bb1
  ^bb3436:
    pdl_interp.check_result_count of %1766 is 1 -> ^bb3437, ^bb1
  ^bb3437:
    %1822 = pdl_interp.get_result 0 of %1766
    pdl_interp.is_not_null %1822 : !pdl.value -> ^bb3438, ^bb1
  ^bb3438:
    pdl_interp.are_equal %1822, %1765 : !pdl.value -> ^bb3439, ^bb1
  ^bb3439:
    pdl_interp.check_operation_name of %1768 is "math.exp" -> ^bb3440, ^bb1
  ^bb3440:
    pdl_interp.check_operand_count of %1768 is 1 -> ^bb3441, ^bb1
  ^bb3441:
    pdl_interp.check_result_count of %1768 is 1 -> ^bb3442, ^bb1
  ^bb3442:
    %1823 = pdl_interp.get_result 0 of %1768
    pdl_interp.is_not_null %1823 : !pdl.value -> ^bb3443, ^bb1
  ^bb3443:
    pdl_interp.are_equal %1823, %1767 : !pdl.value -> ^bb3444, ^bb1
  ^bb3444:
    %1824 = pdl_interp.get_operand 0 of %1763
    pdl_interp.is_not_null %1824 : !pdl.value -> ^bb3445, ^bb1
  ^bb3445:
    %1825 = pdl_interp.get_operand 0 of %1768
    %1826 = pdl_interp.get_defining_op of %1825 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1826 : !pdl.operation -> ^bb3446, ^bb1
  ^bb3446:
    %1827 = pdl_interp.get_defining_op of %1824 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1827 : !pdl.operation -> ^bb3447, ^bb1
  ^bb3447:
    %1828 = pdl_interp.get_operand 0 of %1826
    %1829 = pdl_interp.get_defining_op of %1828 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1829 : !pdl.operation -> ^bb3448, ^bb1
  ^bb3448:
    %1830 = pdl_interp.get_attribute "value" of %1764
    pdl_interp.is_not_null %1830 : !pdl.attribute -> ^bb3449, ^bb1
  ^bb3449:
    pdl_interp.check_attribute %1830 is 1.000000e+00 : f32 -> ^bb3450, ^bb1
  ^bb3450:
    %1831 = pdl_interp.get_value_type of %1821 : !pdl.type
    %1832 = pdl_interp.get_value_type of %1725 : !pdl.type
    pdl_interp.are_equal %1831, %1832 : !pdl.type -> ^bb3451, ^bb1
  ^bb3451:
    %1833 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1831, %1833 : !pdl.type -> ^bb3452, ^bb1
  ^bb3452:
    pdl_interp.check_type %1831 is f32 -> ^bb3453, ^bb1
  ^bb3453:
    pdl_interp.is_not_null %1825 : !pdl.value -> ^bb3454, ^bb1
  ^bb3454:
    pdl_interp.check_operation_name of %1826 is "arith.mulf" -> ^bb3455, ^bb1
  ^bb3455:
    pdl_interp.check_operand_count of %1826 is 2 -> ^bb3456, ^bb1
  ^bb3456:
    pdl_interp.check_result_count of %1826 is 1 -> ^bb3457, ^bb1
  ^bb3457:
    %1834 = pdl_interp.get_result 0 of %1826
    pdl_interp.is_not_null %1834 : !pdl.value -> ^bb3458, ^bb1
  ^bb3458:
    pdl_interp.are_equal %1834, %1825 : !pdl.value -> ^bb3459, ^bb1
  ^bb3459:
    %1835 = pdl_interp.get_attribute "value" of %1766
    pdl_interp.is_not_null %1835 : !pdl.attribute -> ^bb3460, ^bb1
  ^bb3460:
    pdl_interp.check_attribute %1835 is 1.000000e+00 : f32 -> ^bb3461, ^bb1
  ^bb3461:
    pdl_interp.check_operation_name of %1827 is "arith.mulf" -> ^bb3462, ^bb1
  ^bb3462:
    pdl_interp.check_operand_count of %1827 is 2 -> ^bb3463, ^bb1
  ^bb3463:
    pdl_interp.check_result_count of %1827 is 1 -> ^bb3464, ^bb1
  ^bb3464:
    %1836 = pdl_interp.get_result 0 of %1827
    pdl_interp.is_not_null %1836 : !pdl.value -> ^bb3465, ^bb1
  ^bb3465:
    pdl_interp.are_equal %1836, %1824 : !pdl.value -> ^bb3466, ^bb1
  ^bb3466:
    pdl_interp.is_not_null %1828 : !pdl.value -> ^bb3467, ^bb1
  ^bb3467:
    pdl_interp.check_operation_name of %1829 is "arith.constant" -> ^bb3468, ^bb1
  ^bb3468:
    pdl_interp.check_operand_count of %1829 is 0 -> ^bb3469, ^bb1
  ^bb3469:
    pdl_interp.check_result_count of %1829 is 1 -> ^bb3470, ^bb1
  ^bb3470:
    %1837 = pdl_interp.get_result 0 of %1829
    pdl_interp.is_not_null %1837 : !pdl.value -> ^bb3471, ^bb1
  ^bb3471:
    pdl_interp.are_equal %1837, %1828 : !pdl.value -> ^bb3472, ^bb1
  ^bb3472:
    %1838 = pdl_interp.get_value_type of %1820 : !pdl.type
    pdl_interp.are_equal %1831, %1838 : !pdl.type -> ^bb3473, ^bb1
  ^bb3473:
    %1839 = pdl_interp.get_value_type of %1761 : !pdl.type
    pdl_interp.are_equal %1831, %1839 : !pdl.type -> ^bb3474, ^bb1
  ^bb3474:
    %1840 = pdl_interp.get_value_type of %1822 : !pdl.type
    pdl_interp.are_equal %1831, %1840 : !pdl.type -> ^bb3475, ^bb1
  ^bb3475:
    %1841 = pdl_interp.get_operand 0 of %1827
    pdl_interp.is_not_null %1841 : !pdl.value -> ^bb3476, ^bb1
  ^bb3476:
    %1842 = pdl_interp.get_defining_op of %1841 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1842 : !pdl.operation -> ^bb3477, ^bb1
  ^bb3477:
    %1843 = pdl_interp.get_value_type of %1823 : !pdl.type
    pdl_interp.are_equal %1831, %1843 : !pdl.type -> ^bb3478, ^bb1
  ^bb3478:
    %1844 = pdl_interp.get_value_type of %1834 : !pdl.type
    pdl_interp.are_equal %1844, %1831 : !pdl.type -> ^bb3479, ^bb1
  ^bb3479:
    %1845 = pdl_interp.get_attribute "value" of %1829
    pdl_interp.is_not_null %1845 : !pdl.attribute -> ^bb3480, ^bb1
  ^bb3480:
    pdl_interp.check_attribute %1845 is -2.000000e+00 : f32 -> ^bb3481, ^bb1
  ^bb3481:
    %1846 = pdl_interp.get_operand 1 of %1827
    pdl_interp.is_not_null %1846 : !pdl.value -> ^bb3482, ^bb1
  ^bb3482:
    pdl_interp.check_operation_name of %1842 is "arith.constant" -> ^bb3483, ^bb1
  ^bb3483:
    pdl_interp.check_operand_count of %1842 is 0 -> ^bb3484, ^bb1
  ^bb3484:
    pdl_interp.check_result_count of %1842 is 1 -> ^bb3485, ^bb1
  ^bb3485:
    %1847 = pdl_interp.get_result 0 of %1842
    pdl_interp.is_not_null %1847 : !pdl.value -> ^bb3486, ^bb1
  ^bb3486:
    pdl_interp.are_equal %1847, %1841 : !pdl.value -> ^bb3487, ^bb1
  ^bb3487:
    %1848 = pdl_interp.get_value_type of %1836 : !pdl.type
    pdl_interp.are_equal %1848, %1831 : !pdl.type -> ^bb3488, ^bb1
  ^bb3488:
    %1849 = pdl_interp.get_operand 1 of %1826
    pdl_interp.are_equal %1846, %1849 : !pdl.value -> ^bb3489, ^bb1
  ^bb3489:
    %1850 = pdl_interp.get_value_type of %1846 : !pdl.type
    pdl_interp.are_equal %1850, %1831 : !pdl.type -> ^bb3490, ^bb1
  ^bb3490:
    %1851 = pdl_interp.get_attribute "value" of %1842
    pdl_interp.is_not_null %1851 : !pdl.attribute -> ^bb3491, ^bb1
  ^bb3491:
    pdl_interp.check_attribute %1851 is -2.000000e+00 : f32 -> ^bb3492, ^bb1
  ^bb3492:
    %1852 = pdl_interp.get_value_type of %1847 : !pdl.type
    pdl_interp.are_equal %1852, %1831 : !pdl.type -> ^bb3493, ^bb1
  ^bb3493:
    %1853 = pdl_interp.get_value_type of %1837 : !pdl.type
    pdl_interp.are_equal %1853, %1831 : !pdl.type -> ^bb3494, ^bb1
  ^bb3494:
    pdl_interp.record_match @rewriters::@tanh_def_c_rev(%1846, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb1
  ^bb3430:
    pdl_interp.check_operand_count of %1764 is 1 -> ^bb3495, ^bb1
  ^bb3495:
    pdl_interp.check_result_count of %1764 is 1 -> ^bb3496, ^bb1
  ^bb3496:
    %1854 = pdl_interp.get_result 0 of %1764
    pdl_interp.is_not_null %1854 : !pdl.value -> ^bb3497, ^bb1
  ^bb3497:
    pdl_interp.are_equal %1854, %1726 : !pdl.value -> ^bb3498, ^bb1
  ^bb3498:
    pdl_interp.check_operation_name of %1766 is "math.exp" -> ^bb3499, ^bb1
  ^bb3499:
    pdl_interp.check_operand_count of %1766 is 1 -> ^bb3500, ^bb1
  ^bb3500:
    pdl_interp.check_result_count of %1766 is 1 -> ^bb3501, ^bb1
  ^bb3501:
    %1855 = pdl_interp.get_result 0 of %1766
    pdl_interp.is_not_null %1855 : !pdl.value -> ^bb3502, ^bb1
  ^bb3502:
    pdl_interp.are_equal %1855, %1765 : !pdl.value -> ^bb3503, ^bb1
  ^bb3503:
    pdl_interp.check_operation_name of %1768 is "math.exp" -> ^bb3504, ^bb1
  ^bb3504:
    pdl_interp.check_operand_count of %1768 is 1 -> ^bb3505, ^bb1
  ^bb3505:
    pdl_interp.check_result_count of %1768 is 1 -> ^bb3506, ^bb1
  ^bb3506:
    %1856 = pdl_interp.get_result 0 of %1768
    pdl_interp.is_not_null %1856 : !pdl.value -> ^bb3507, ^bb1
  ^bb3507:
    pdl_interp.are_equal %1856, %1767 : !pdl.value -> ^bb3508, ^bb1
  ^bb3508:
    %1857 = pdl_interp.get_operand 0 of %1764
    pdl_interp.is_not_null %1857 : !pdl.value -> ^bb3509, ^bb1
  ^bb3509:
    %1858 = pdl_interp.get_value_type of %1857 : !pdl.type
    %1859 = pdl_interp.get_value_type of %1854 : !pdl.type
    pdl_interp.are_equal %1858, %1859 : !pdl.type -> ^bb3510, ^bb1
  ^bb3510:
    %1860 = pdl_interp.get_value_type of %1725 : !pdl.type
    pdl_interp.are_equal %1858, %1860 : !pdl.type -> ^bb3511, ^bb1
  ^bb3511:
    %1861 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1858, %1861 : !pdl.type -> ^bb3512, ^bb1
  ^bb3512:
    pdl_interp.check_type %1858 is f32 -> ^bb3513, ^bb1
  ^bb3513:
    %1862 = pdl_interp.get_operand 0 of %1763
    pdl_interp.is_not_null %1862 : !pdl.value -> ^bb3514, ^bb1
  ^bb3514:
    %1863 = pdl_interp.get_operand 0 of %1768
    %1864 = pdl_interp.get_defining_op of %1863 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1864 : !pdl.operation -> ^bb3515, ^bb1
  ^bb3515:
    %1865 = pdl_interp.get_value_type of %1820 : !pdl.type
    pdl_interp.are_equal %1858, %1865 : !pdl.type -> ^bb3516, ^bb1
  ^bb3516:
    %1866 = pdl_interp.get_value_type of %1761 : !pdl.type
    pdl_interp.are_equal %1858, %1866 : !pdl.type -> ^bb3517, ^bb1
  ^bb3517:
    %1867 = pdl_interp.get_defining_op of %1862 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1867 : !pdl.operation -> ^bb3518, ^bb1
  ^bb3518:
    %1868 = pdl_interp.get_value_type of %1856 : !pdl.type
    pdl_interp.are_equal %1858, %1868 : !pdl.type -> ^bb3519, ^bb1
  ^bb3519:
    %1869 = pdl_interp.get_value_type of %1855 : !pdl.type
    pdl_interp.are_equal %1858, %1869 : !pdl.type -> ^bb3520, ^bb1
  ^bb3520:
    pdl_interp.is_not_null %1863 : !pdl.value -> ^bb3521, ^bb1
  ^bb3521:
    pdl_interp.check_operation_name of %1864 is "arith.negf" -> ^bb3522, ^bb1
  ^bb3522:
    pdl_interp.check_operand_count of %1864 is 1 -> ^bb3523, ^bb1
  ^bb3523:
    pdl_interp.check_result_count of %1864 is 1 -> ^bb3524, ^bb1
  ^bb3524:
    %1870 = pdl_interp.get_result 0 of %1864
    pdl_interp.is_not_null %1870 : !pdl.value -> ^bb3525, ^bb1
  ^bb3525:
    pdl_interp.are_equal %1870, %1863 : !pdl.value -> ^bb3526, ^bb1
  ^bb3526:
    %1871 = pdl_interp.get_operand 0 of %1766
    pdl_interp.are_equal %1857, %1871 : !pdl.value -> ^bb3527, ^bb1
  ^bb3527:
    pdl_interp.check_operation_name of %1867 is "arith.negf" -> ^bb3528, ^bb1
  ^bb3528:
    pdl_interp.check_operand_count of %1867 is 1 -> ^bb3529, ^bb1
  ^bb3529:
    pdl_interp.check_result_count of %1867 is 1 -> ^bb3530, ^bb1
  ^bb3530:
    %1872 = pdl_interp.get_result 0 of %1867
    pdl_interp.is_not_null %1872 : !pdl.value -> ^bb3531, ^bb1
  ^bb3531:
    pdl_interp.are_equal %1872, %1862 : !pdl.value -> ^bb3532, ^bb1
  ^bb3532:
    %1873 = pdl_interp.get_value_type of %1872 : !pdl.type
    pdl_interp.are_equal %1873, %1858 : !pdl.type -> ^bb3533, ^bb1
  ^bb3533:
    %1874 = pdl_interp.get_operand 0 of %1864
    pdl_interp.are_equal %1874, %1857 : !pdl.value -> ^bb3534, ^bb1
  ^bb3534:
    %1875 = pdl_interp.get_value_type of %1870 : !pdl.type
    pdl_interp.are_equal %1875, %1858 : !pdl.type -> ^bb3535, ^bb1
  ^bb3535:
    %1876 = pdl_interp.get_operand 0 of %1867
    pdl_interp.are_equal %1876, %1857 : !pdl.value -> ^bb3536, ^bb1
  ^bb3536:
    pdl_interp.record_match @rewriters::@tanh_undef(%1857, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb1
  ^bb3246:
    pdl_interp.check_operand_count of %1445 is 0 -> ^bb3537, ^bb1
  ^bb3537:
    pdl_interp.check_result_count of %1445 is 1 -> ^bb3538, ^bb1
  ^bb3538:
    %1877 = pdl_interp.get_result 0 of %1445
    pdl_interp.is_not_null %1877 : !pdl.value -> ^bb3539, ^bb1
  ^bb3539:
    pdl_interp.are_equal %1877, %1444 : !pdl.value -> ^bb3540, ^bb1
  ^bb3540:
    %1878 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %1878 : !pdl.value -> ^bb3541, ^bb1
  ^bb3541:
    %1879 = pdl_interp.get_defining_op of %1878 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %1879 : !pdl.operation -> ^bb3542, ^bb1
  ^bb3542:
    %1880 = pdl_interp.get_defining_op of %1726 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1880 : !pdl.operation -> ^bb3543, ^bb1
  ^bb3543:
    pdl_interp.switch_operation_name of %1879 to ["math.cos", "math.exp"](^bb3544, ^bb3545) -> ^bb1
  ^bb3544:
    pdl_interp.check_operand_count of %1879 is 1 -> ^bb3546, ^bb1
  ^bb3546:
    pdl_interp.check_result_count of %1879 is 1 -> ^bb3547, ^bb1
  ^bb3547:
    %1881 = pdl_interp.get_result 0 of %1879
    pdl_interp.is_not_null %1881 : !pdl.value -> ^bb3548, ^bb1
  ^bb3548:
    pdl_interp.are_equal %1881, %1878 : !pdl.value -> ^bb3549, ^bb1
  ^bb3549:
    pdl_interp.check_operation_name of %1880 is "math.cos" -> ^bb3550, ^bb1
  ^bb3550:
    pdl_interp.check_operand_count of %1880 is 1 -> ^bb3551, ^bb1
  ^bb3551:
    pdl_interp.check_result_count of %1880 is 1 -> ^bb3552, ^bb1
  ^bb3552:
    %1882 = pdl_interp.get_result 0 of %1880
    pdl_interp.is_not_null %1882 : !pdl.value -> ^bb3553, ^bb1
  ^bb3553:
    pdl_interp.are_equal %1882, %1726 : !pdl.value -> ^bb3554, ^bb1
  ^bb3554:
    %1883 = pdl_interp.get_operand 0 of %1880
    pdl_interp.is_not_null %1883 : !pdl.value -> ^bb3555, ^bb1
  ^bb3555:
    %1884 = pdl_interp.get_operand 0 of %1879
    pdl_interp.is_not_null %1884 : !pdl.value -> ^bb3556, ^bb1
  ^bb3556:
    %1885 = pdl_interp.get_attribute "value" of %1445
    pdl_interp.is_not_null %1885 : !pdl.attribute -> ^bb3557, ^bb1
  ^bb3557:
    pdl_interp.check_attribute %1885 is 2.000000e+00 : f32 -> ^bb3558, ^bb1
  ^bb3558:
    %1886 = pdl_interp.get_defining_op of %1884 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1886 : !pdl.operation -> ^bb3559, ^bb1
  ^bb3559:
    %1887 = pdl_interp.get_defining_op of %1883 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1887 : !pdl.operation -> ^bb3560, ^bb1
  ^bb3560:
    pdl_interp.check_operation_name of %1886 is "arith.addf" -> ^bb3561, ^bb1
  ^bb3561:
    pdl_interp.check_operand_count of %1886 is 2 -> ^bb3562, ^bb1
  ^bb3562:
    pdl_interp.check_result_count of %1886 is 1 -> ^bb3563, ^bb1
  ^bb3563:
    %1888 = pdl_interp.get_result 0 of %1886
    pdl_interp.is_not_null %1888 : !pdl.value -> ^bb3564, ^bb1
  ^bb3564:
    pdl_interp.are_equal %1888, %1884 : !pdl.value -> ^bb3565, ^bb1
  ^bb3565:
    pdl_interp.check_operation_name of %1887 is "arith.subf" -> ^bb3566, ^bb1
  ^bb3566:
    pdl_interp.check_operand_count of %1887 is 2 -> ^bb3567, ^bb1
  ^bb3567:
    pdl_interp.check_result_count of %1887 is 1 -> ^bb3568, ^bb1
  ^bb3568:
    %1889 = pdl_interp.get_result 0 of %1887
    pdl_interp.is_not_null %1889 : !pdl.value -> ^bb3569, ^bb1
  ^bb3569:
    pdl_interp.are_equal %1889, %1883 : !pdl.value -> ^bb3570, ^bb1
  ^bb3570:
    %1890 = pdl_interp.get_operand 0 of %1887
    pdl_interp.is_not_null %1890 : !pdl.value -> ^bb3571, ^bb1
  ^bb3571:
    %1891 = pdl_interp.get_operand 1 of %1887
    pdl_interp.is_not_null %1891 : !pdl.value -> ^bb3572, ^bb1
  ^bb3572:
    %1892 = pdl_interp.get_value_type of %1890 : !pdl.type
    %1893 = pdl_interp.get_value_type of %1889 : !pdl.type
    pdl_interp.are_equal %1892, %1893 : !pdl.type -> ^bb3573, ^bb1
  ^bb3573:
    %1894 = pdl_interp.get_value_type of %1882 : !pdl.type
    pdl_interp.are_equal %1892, %1894 : !pdl.type -> ^bb3574, ^bb1
  ^bb3574:
    %1895 = pdl_interp.get_value_type of %1881 : !pdl.type
    pdl_interp.are_equal %1892, %1895 : !pdl.type -> ^bb3575, ^bb1
  ^bb3575:
    %1896 = pdl_interp.get_value_type of %1725 : !pdl.type
    pdl_interp.are_equal %1892, %1896 : !pdl.type -> ^bb3576, ^bb1
  ^bb3576:
    %1897 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1892, %1897 : !pdl.type -> ^bb3577, ^bb1
  ^bb3577:
    pdl_interp.check_type %1892 is f32 -> ^bb3578, ^bb1
  ^bb3578:
    %1898 = pdl_interp.get_operand 0 of %1886
    pdl_interp.are_equal %1890, %1898 : !pdl.value -> ^bb3579, ^bb1
  ^bb3579:
    %1899 = pdl_interp.get_operand 1 of %1886
    pdl_interp.are_equal %1891, %1899 : !pdl.value -> ^bb3580, ^bb1
  ^bb3580:
    %1900 = pdl_interp.get_value_type of %1891 : !pdl.type
    pdl_interp.are_equal %1892, %1900 : !pdl.type -> ^bb3581, ^bb1
  ^bb3581:
    %1901 = pdl_interp.get_value_type of %1888 : !pdl.type
    pdl_interp.are_equal %1892, %1901 : !pdl.type -> ^bb3582, ^bb1
  ^bb3582:
    %1902 = pdl_interp.get_value_type of %1877 : !pdl.type
    pdl_interp.are_equal %1892, %1902 : !pdl.type -> ^bb3583, ^bb1
  ^bb3583:
    pdl_interp.record_match @rewriters::@sin_mult_rev(%1890, %1891, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb1
  ^bb3545:
    pdl_interp.check_operand_count of %1879 is 1 -> ^bb3584, ^bb1
  ^bb3584:
    pdl_interp.check_result_count of %1879 is 1 -> ^bb3585, ^bb1
  ^bb3585:
    %1903 = pdl_interp.get_result 0 of %1879
    pdl_interp.is_not_null %1903 : !pdl.value -> ^bb3586, ^bb1
  ^bb3586:
    pdl_interp.are_equal %1903, %1878 : !pdl.value -> ^bb3587, ^bb1
  ^bb3587:
    pdl_interp.check_operation_name of %1880 is "math.exp" -> ^bb3588, ^bb1
  ^bb3588:
    pdl_interp.check_operand_count of %1880 is 1 -> ^bb3589, ^bb1
  ^bb3589:
    pdl_interp.check_result_count of %1880 is 1 -> ^bb3590, ^bb1
  ^bb3590:
    %1904 = pdl_interp.get_result 0 of %1880
    pdl_interp.is_not_null %1904 : !pdl.value -> ^bb3591, ^bb1
  ^bb3591:
    pdl_interp.are_equal %1904, %1726 : !pdl.value -> ^bb3592, ^bb1
  ^bb3592:
    %1905 = pdl_interp.get_operand 0 of %1880
    pdl_interp.is_not_null %1905 : !pdl.value -> ^bb3593, ^bb1
  ^bb3593:
    %1906 = pdl_interp.get_value_type of %1905 : !pdl.type
    %1907 = pdl_interp.get_value_type of %1904 : !pdl.type
    pdl_interp.are_equal %1906, %1907 : !pdl.type -> ^bb3594, ^bb1
  ^bb3594:
    %1908 = pdl_interp.get_value_type of %1725 : !pdl.type
    pdl_interp.are_equal %1906, %1908 : !pdl.type -> ^bb3595, ^bb1
  ^bb3595:
    %1909 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1906, %1909 : !pdl.type -> ^bb3596, ^bb1
  ^bb3596:
    pdl_interp.check_type %1906 is f32 -> ^bb3597, ^bb1
  ^bb3597:
    %1910 = pdl_interp.get_operand 0 of %1879
    pdl_interp.is_not_null %1910 : !pdl.value -> ^bb3598, ^bb1
  ^bb3598:
    %1911 = pdl_interp.get_value_type of %1903 : !pdl.type
    pdl_interp.are_equal %1906, %1911 : !pdl.type -> ^bb3599, ^bb1
  ^bb3599:
    %1912 = pdl_interp.get_attribute "value" of %1445
    pdl_interp.is_not_null %1912 : !pdl.attribute -> ^bb3600, ^bb1
  ^bb3600:
    pdl_interp.check_attribute %1912 is 2.000000e+00 : f32 -> ^bb3601, ^bb1
  ^bb3601:
    %1913 = pdl_interp.get_value_type of %1877 : !pdl.type
    pdl_interp.are_equal %1906, %1913 : !pdl.type -> ^bb3602, ^bb1
  ^bb3602:
    %1914 = pdl_interp.get_defining_op of %1910 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1914 : !pdl.operation -> ^bb3603, ^bb1
  ^bb3603:
    pdl_interp.check_operation_name of %1914 is "arith.negf" -> ^bb3604, ^bb1
  ^bb3604:
    pdl_interp.check_operand_count of %1914 is 1 -> ^bb3605, ^bb1
  ^bb3605:
    pdl_interp.check_result_count of %1914 is 1 -> ^bb3606, ^bb1
  ^bb3606:
    %1915 = pdl_interp.get_result 0 of %1914
    pdl_interp.is_not_null %1915 : !pdl.value -> ^bb3607, ^bb1
  ^bb3607:
    pdl_interp.are_equal %1915, %1910 : !pdl.value -> ^bb3608, ^bb1
  ^bb3608:
    %1916 = pdl_interp.get_value_type of %1915 : !pdl.type
    pdl_interp.are_equal %1916, %1906 : !pdl.type -> ^bb3609, ^bb1
  ^bb3609:
    %1917 = pdl_interp.get_operand 0 of %1914
    pdl_interp.are_equal %1917, %1905 : !pdl.value -> ^bb3610, ^bb1
  ^bb3610:
    pdl_interp.record_match @rewriters::@sinh_def_rev(%1905, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb1
  ^bb3247:
    pdl_interp.check_operand_count of %1445 is 1 -> ^bb3611, ^bb1
  ^bb3611:
    pdl_interp.check_result_count of %1445 is 1 -> ^bb3612, ^bb1
  ^bb3612:
    %1918 = pdl_interp.get_result 0 of %1445
    pdl_interp.is_not_null %1918 : !pdl.value -> ^bb3613, ^bb1
  ^bb3613:
    pdl_interp.are_equal %1918, %1444 : !pdl.value -> ^bb3614, ^bb1
  ^bb3614:
    %1919 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %1919 : !pdl.value -> ^bb3615, ^bb1
  ^bb3615:
    %1920 = pdl_interp.get_defining_op of %1919 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %1920 : !pdl.operation -> ^bb3616, ^bb1
  ^bb3616:
    %1921 = pdl_interp.get_defining_op of %1726 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1921 : !pdl.operation -> ^bb3617, ^bb1
  ^bb3617:
    pdl_interp.check_operation_name of %1920 is "arith.constant" -> ^bb3618, ^bb1
  ^bb3618:
    pdl_interp.check_operand_count of %1920 is 0 -> ^bb3619, ^bb1
  ^bb3619:
    pdl_interp.check_result_count of %1920 is 1 -> ^bb3620, ^bb1
  ^bb3620:
    %1922 = pdl_interp.get_result 0 of %1920
    pdl_interp.is_not_null %1922 : !pdl.value -> ^bb3621, ^bb1
  ^bb3621:
    pdl_interp.are_equal %1922, %1919 : !pdl.value -> ^bb3622, ^bb1
  ^bb3622:
    pdl_interp.check_operation_name of %1921 is "math.cosh" -> ^bb3623, ^bb1
  ^bb3623:
    pdl_interp.check_operand_count of %1921 is 1 -> ^bb3624, ^bb1
  ^bb3624:
    pdl_interp.check_result_count of %1921 is 1 -> ^bb3625, ^bb1
  ^bb3625:
    %1923 = pdl_interp.get_result 0 of %1921
    pdl_interp.is_not_null %1923 : !pdl.value -> ^bb3626, ^bb1
  ^bb3626:
    pdl_interp.are_equal %1923, %1726 : !pdl.value -> ^bb3627, ^bb1
  ^bb3627:
    %1924 = pdl_interp.get_operand 0 of %1921
    pdl_interp.is_not_null %1924 : !pdl.value -> ^bb3628, ^bb1
  ^bb3628:
    %1925 = pdl_interp.get_value_type of %1924 : !pdl.type
    %1926 = pdl_interp.get_value_type of %1923 : !pdl.type
    pdl_interp.are_equal %1925, %1926 : !pdl.type -> ^bb3629, ^bb1
  ^bb3629:
    %1927 = pdl_interp.get_value_type of %1725 : !pdl.type
    pdl_interp.are_equal %1925, %1927 : !pdl.type -> ^bb3630, ^bb1
  ^bb3630:
    %1928 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1925, %1928 : !pdl.type -> ^bb3631, ^bb1
  ^bb3631:
    pdl_interp.check_type %1925 is f32 -> ^bb3632, ^bb1
  ^bb3632:
    %1929 = pdl_interp.get_value_type of %1922 : !pdl.type
    pdl_interp.are_equal %1925, %1929 : !pdl.type -> ^bb3633, ^bb1
  ^bb3633:
    %1930 = pdl_interp.get_value_type of %1918 : !pdl.type
    pdl_interp.are_equal %1925, %1930 : !pdl.type -> ^bb3634, ^bb1
  ^bb3634:
    %1931 = pdl_interp.get_attribute "value" of %1920
    pdl_interp.is_not_null %1931 : !pdl.attribute -> ^bb3635, ^bb1
  ^bb3635:
    pdl_interp.check_attribute %1931 is 1.000000e+00 : f32 -> ^bb3636, ^bb1
  ^bb3636:
    %1932 = pdl_interp.get_operand 0 of %1445
    pdl_interp.are_equal %1924, %1932 : !pdl.value -> ^bb3637, ^bb1
  ^bb3637:
    pdl_interp.record_match @rewriters::@tanh_1div2mul_rev(%1924, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb1
  ^bb2713:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb3638, ^bb1
  ^bb3638:
    pdl_interp.check_result_count of %3 is 1 -> ^bb3639, ^bb1
  ^bb3639:
    %1933 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %1933 : !pdl.value -> ^bb3640, ^bb1
  ^bb3640:
    pdl_interp.are_equal %1933, %2 : !pdl.value -> ^bb3641, ^bb1
  ^bb3641:
    %1934 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %1934 : !pdl.value -> ^bb3642, ^bb1
  ^bb3642:
    pdl_interp.is_not_null %1444 : !pdl.value -> ^bb3643, ^bb1
  ^bb3643:
    pdl_interp.switch_operation_name of %1445 to ["arith.addf", "arith.subf", "arith.constant"](^bb3644, ^bb3645, ^bb3646) -> ^bb1
  ^bb3644:
    pdl_interp.check_operand_count of %1445 is 2 -> ^bb3647, ^bb1
  ^bb3647:
    pdl_interp.check_result_count of %1445 is 1 -> ^bb3648, ^bb1
  ^bb3648:
    %1935 = pdl_interp.get_result 0 of %1445
    pdl_interp.is_not_null %1935 : !pdl.value -> ^bb3649, ^bb1
  ^bb3649:
    pdl_interp.are_equal %1935, %1444 : !pdl.value -> ^bb3650, ^bb1
  ^bb3650:
    %1936 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %1936 : !pdl.value -> ^bb3651, ^bb1
  ^bb3651:
    %1937 = pdl_interp.get_defining_op of %1936 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %1937 : !pdl.operation -> ^bb3652, ^bb1
  ^bb3652:
    %1938 = pdl_interp.get_defining_op of %1934 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1938 : !pdl.operation -> ^bb3653, ^bb1
  ^bb3653:
    %1939 = pdl_interp.get_operand 0 of %1445
    pdl_interp.is_not_null %1939 : !pdl.value -> ^bb3654, ^bb1
  ^bb3654:
    %1940 = pdl_interp.get_defining_op of %1939 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1940 : !pdl.operation -> ^bb3655, ^bb1
  ^bb3655:
    %1941 = pdl_interp.get_operand 1 of %1445
    %1942 = pdl_interp.get_defining_op of %1941 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %1942 : !pdl.operation -> ^bb3656, ^bb1
  ^bb3656:
    pdl_interp.is_not_null %1941 : !pdl.value -> ^bb3657, ^bb1
  ^bb3657:
    pdl_interp.switch_operation_name of %1937 to ["math.sin", "math.tanh"](^bb3658, ^bb3659) -> ^bb1
  ^bb3658:
    pdl_interp.check_operand_count of %1937 is 1 -> ^bb3660, ^bb1
  ^bb3660:
    pdl_interp.check_result_count of %1937 is 1 -> ^bb3661, ^bb1
  ^bb3661:
    %1943 = pdl_interp.get_result 0 of %1937
    pdl_interp.is_not_null %1943 : !pdl.value -> ^bb3662, ^bb1
  ^bb3662:
    pdl_interp.are_equal %1943, %1936 : !pdl.value -> ^bb3663, ^bb1
  ^bb3663:
    pdl_interp.check_operation_name of %1938 is "math.sin" -> ^bb3664, ^bb1
  ^bb3664:
    pdl_interp.check_operand_count of %1938 is 1 -> ^bb3665, ^bb1
  ^bb3665:
    pdl_interp.check_result_count of %1938 is 1 -> ^bb3666, ^bb1
  ^bb3666:
    %1944 = pdl_interp.get_result 0 of %1938
    pdl_interp.is_not_null %1944 : !pdl.value -> ^bb3667, ^bb1
  ^bb3667:
    pdl_interp.are_equal %1944, %1934 : !pdl.value -> ^bb3668, ^bb1
  ^bb3668:
    pdl_interp.check_operation_name of %1940 is "math.cos" -> ^bb3669, ^bb1
  ^bb3669:
    pdl_interp.check_operand_count of %1940 is 1 -> ^bb3670, ^bb1
  ^bb3670:
    pdl_interp.check_result_count of %1940 is 1 -> ^bb3671, ^bb1
  ^bb3671:
    %1945 = pdl_interp.get_result 0 of %1940
    pdl_interp.is_not_null %1945 : !pdl.value -> ^bb3672, ^bb1
  ^bb3672:
    pdl_interp.are_equal %1945, %1939 : !pdl.value -> ^bb3673, ^bb1
  ^bb3673:
    pdl_interp.check_operation_name of %1942 is "math.cos" -> ^bb3674, ^bb1
  ^bb3674:
    pdl_interp.check_operand_count of %1942 is 1 -> ^bb3675, ^bb1
  ^bb3675:
    pdl_interp.check_result_count of %1942 is 1 -> ^bb3676, ^bb1
  ^bb3676:
    %1946 = pdl_interp.get_result 0 of %1942
    pdl_interp.is_not_null %1946 : !pdl.value -> ^bb3677, ^bb1
  ^bb3677:
    pdl_interp.are_equal %1946, %1941 : !pdl.value -> ^bb3678, ^bb1
  ^bb3678:
    %1947 = pdl_interp.get_operand 0 of %1938
    pdl_interp.is_not_null %1947 : !pdl.value -> ^bb3679, ^bb1
  ^bb3679:
    %1948 = pdl_interp.get_value_type of %1947 : !pdl.type
    %1949 = pdl_interp.get_value_type of %1944 : !pdl.type
    pdl_interp.are_equal %1948, %1949 : !pdl.type -> ^bb3680, ^bb1
  ^bb3680:
    %1950 = pdl_interp.get_value_type of %1933 : !pdl.type
    pdl_interp.are_equal %1948, %1950 : !pdl.type -> ^bb3681, ^bb1
  ^bb3681:
    %1951 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1948, %1951 : !pdl.type -> ^bb3682, ^bb1
  ^bb3682:
    pdl_interp.check_type %1948 is f32 -> ^bb3683, ^bb1
  ^bb3683:
    %1952 = pdl_interp.get_operand 0 of %1937
    pdl_interp.is_not_null %1952 : !pdl.value -> ^bb3684, ^bb1
  ^bb3684:
    %1953 = pdl_interp.get_value_type of %1943 : !pdl.type
    pdl_interp.are_equal %1948, %1953 : !pdl.type -> ^bb3685, ^bb1
  ^bb3685:
    %1954 = pdl_interp.get_value_type of %1935 : !pdl.type
    pdl_interp.are_equal %1948, %1954 : !pdl.type -> ^bb3686, ^bb1
  ^bb3686:
    %1955 = pdl_interp.get_value_type of %1946 : !pdl.type
    pdl_interp.are_equal %1948, %1955 : !pdl.type -> ^bb3687, ^bb1
  ^bb3687:
    %1956 = pdl_interp.get_value_type of %1945 : !pdl.type
    pdl_interp.are_equal %1948, %1956 : !pdl.type -> ^bb3688, ^bb1
  ^bb3688:
    %1957 = pdl_interp.get_operand 0 of %1940
    pdl_interp.are_equal %1947, %1957 : !pdl.value -> ^bb3689, ^bb1
  ^bb3689:
    %1958 = pdl_interp.get_value_type of %1952 : !pdl.type
    pdl_interp.are_equal %1948, %1958 : !pdl.type -> ^bb3690, ^bb1
  ^bb3690:
    %1959 = pdl_interp.get_operand 0 of %1942
    pdl_interp.are_equal %1952, %1959 : !pdl.value -> ^bb3691, ^bb1
  ^bb3691:
    pdl_interp.record_match @rewriters::@hang_p_tan(%1947, %1952, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb1
  ^bb3659:
    pdl_interp.check_operand_count of %1937 is 1 -> ^bb3692, ^bb1
  ^bb3692:
    pdl_interp.check_result_count of %1937 is 1 -> ^bb3693, ^bb1
  ^bb3693:
    %1960 = pdl_interp.get_result 0 of %1937
    pdl_interp.is_not_null %1960 : !pdl.value -> ^bb3694, ^bb1
  ^bb3694:
    pdl_interp.are_equal %1960, %1936 : !pdl.value -> ^bb3695, ^bb1
  ^bb3695:
    pdl_interp.check_operation_name of %1938 is "math.tanh" -> ^bb3696, ^bb1
  ^bb3696:
    pdl_interp.check_operand_count of %1938 is 1 -> ^bb3697, ^bb1
  ^bb3697:
    pdl_interp.check_result_count of %1938 is 1 -> ^bb3698, ^bb1
  ^bb3698:
    %1961 = pdl_interp.get_result 0 of %1938
    pdl_interp.is_not_null %1961 : !pdl.value -> ^bb3699, ^bb1
  ^bb3699:
    pdl_interp.are_equal %1961, %1934 : !pdl.value -> ^bb3700, ^bb1
  ^bb3700:
    pdl_interp.check_operation_name of %1940 is "arith.constant" -> ^bb3701, ^bb1
  ^bb3701:
    pdl_interp.check_operand_count of %1940 is 0 -> ^bb3702, ^bb1
  ^bb3702:
    pdl_interp.check_result_count of %1940 is 1 -> ^bb3703, ^bb1
  ^bb3703:
    %1962 = pdl_interp.get_result 0 of %1940
    pdl_interp.is_not_null %1962 : !pdl.value -> ^bb3704, ^bb1
  ^bb3704:
    pdl_interp.are_equal %1962, %1939 : !pdl.value -> ^bb3705, ^bb1
  ^bb3705:
    pdl_interp.check_operation_name of %1942 is "arith.mulf" -> ^bb3706, ^bb1
  ^bb3706:
    pdl_interp.check_operand_count of %1942 is 2 -> ^bb3707, ^bb1
  ^bb3707:
    pdl_interp.check_result_count of %1942 is 1 -> ^bb3708, ^bb1
  ^bb3708:
    %1963 = pdl_interp.get_result 0 of %1942
    pdl_interp.is_not_null %1963 : !pdl.value -> ^bb3709, ^bb1
  ^bb3709:
    pdl_interp.are_equal %1963, %1941 : !pdl.value -> ^bb3710, ^bb1
  ^bb3710:
    %1964 = pdl_interp.get_operand 0 of %1938
    pdl_interp.is_not_null %1964 : !pdl.value -> ^bb3711, ^bb1
  ^bb3711:
    %1965 = pdl_interp.get_value_type of %1964 : !pdl.type
    %1966 = pdl_interp.get_value_type of %1961 : !pdl.type
    pdl_interp.are_equal %1965, %1966 : !pdl.type -> ^bb3712, ^bb1
  ^bb3712:
    %1967 = pdl_interp.get_value_type of %1933 : !pdl.type
    pdl_interp.are_equal %1965, %1967 : !pdl.type -> ^bb3713, ^bb1
  ^bb3713:
    %1968 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1965, %1968 : !pdl.type -> ^bb3714, ^bb1
  ^bb3714:
    pdl_interp.check_type %1965 is f32 -> ^bb3715, ^bb1
  ^bb3715:
    %1969 = pdl_interp.get_operand 0 of %1937
    pdl_interp.is_not_null %1969 : !pdl.value -> ^bb3716, ^bb1
  ^bb3716:
    %1970 = pdl_interp.get_operand 0 of %1942
    %1971 = pdl_interp.get_defining_op of %1970 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1971 : !pdl.operation -> ^bb3717, ^bb1
  ^bb3717:
    %1972 = pdl_interp.get_value_type of %1960 : !pdl.type
    pdl_interp.are_equal %1965, %1972 : !pdl.type -> ^bb3718, ^bb1
  ^bb3718:
    %1973 = pdl_interp.get_value_type of %1935 : !pdl.type
    pdl_interp.are_equal %1965, %1973 : !pdl.type -> ^bb3719, ^bb1
  ^bb3719:
    %1974 = pdl_interp.get_value_type of %1963 : !pdl.type
    pdl_interp.are_equal %1965, %1974 : !pdl.type -> ^bb3720, ^bb1
  ^bb3720:
    %1975 = pdl_interp.get_value_type of %1962 : !pdl.type
    pdl_interp.are_equal %1965, %1975 : !pdl.type -> ^bb3721, ^bb1
  ^bb3721:
    pdl_interp.is_not_null %1970 : !pdl.value -> ^bb3722, ^bb1
  ^bb3722:
    pdl_interp.check_operation_name of %1971 is "math.tanh" -> ^bb3723, ^bb1
  ^bb3723:
    pdl_interp.check_operand_count of %1971 is 1 -> ^bb3724, ^bb1
  ^bb3724:
    pdl_interp.check_result_count of %1971 is 1 -> ^bb3725, ^bb1
  ^bb3725:
    %1976 = pdl_interp.get_result 0 of %1971
    pdl_interp.is_not_null %1976 : !pdl.value -> ^bb3726, ^bb1
  ^bb3726:
    pdl_interp.are_equal %1976, %1970 : !pdl.value -> ^bb3727, ^bb1
  ^bb3727:
    %1977 = pdl_interp.get_attribute "value" of %1940
    pdl_interp.is_not_null %1977 : !pdl.attribute -> ^bb3728, ^bb1
  ^bb3728:
    pdl_interp.check_attribute %1977 is 1.000000e+00 : f32 -> ^bb3729, ^bb1
  ^bb3729:
    %1978 = pdl_interp.get_value_type of %1969 : !pdl.type
    pdl_interp.are_equal %1965, %1978 : !pdl.type -> ^bb3730, ^bb1
  ^bb3730:
    %1979 = pdl_interp.get_operand 1 of %1942
    %1980 = pdl_interp.get_defining_op of %1979 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %1980 : !pdl.operation -> ^bb3731, ^bb1
  ^bb3731:
    pdl_interp.is_not_null %1979 : !pdl.value -> ^bb3732, ^bb1
  ^bb3732:
    pdl_interp.check_operation_name of %1980 is "math.tanh" -> ^bb3733, ^bb1
  ^bb3733:
    pdl_interp.check_operand_count of %1980 is 1 -> ^bb3734, ^bb1
  ^bb3734:
    pdl_interp.check_result_count of %1980 is 1 -> ^bb3735, ^bb1
  ^bb3735:
    %1981 = pdl_interp.get_result 0 of %1980
    pdl_interp.is_not_null %1981 : !pdl.value -> ^bb3736, ^bb1
  ^bb3736:
    pdl_interp.are_equal %1981, %1979 : !pdl.value -> ^bb3737, ^bb1
  ^bb3737:
    %1982 = pdl_interp.get_operand 0 of %1980
    pdl_interp.are_equal %1982, %1969 : !pdl.value -> ^bb3738, ^bb1
  ^bb3738:
    %1983 = pdl_interp.get_operand 0 of %1971
    pdl_interp.are_equal %1983, %1964 : !pdl.value -> ^bb3739, ^bb1
  ^bb3739:
    %1984 = pdl_interp.get_value_type of %1976 : !pdl.type
    pdl_interp.are_equal %1984, %1965 : !pdl.type -> ^bb3740, ^bb1
  ^bb3740:
    %1985 = pdl_interp.get_value_type of %1981 : !pdl.type
    pdl_interp.are_equal %1985, %1965 : !pdl.type -> ^bb3741, ^bb1
  ^bb3741:
    pdl_interp.record_match @rewriters::@tanh_sum_rev(%1964, %1969, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb1
  ^bb3645:
    pdl_interp.check_operand_count of %1445 is 2 -> ^bb3742, ^bb1
  ^bb3742:
    pdl_interp.check_result_count of %1445 is 1 -> ^bb3743, ^bb1
  ^bb3743:
    %1986 = pdl_interp.get_result 0 of %1445
    pdl_interp.is_not_null %1986 : !pdl.value -> ^bb3744, ^bb1
  ^bb3744:
    pdl_interp.are_equal %1986, %1444 : !pdl.value -> ^bb3745, ^bb1
  ^bb3745:
    %1987 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %1987 : !pdl.value -> ^bb3746, ^bb1
  ^bb3746:
    %1988 = pdl_interp.get_defining_op of %1987 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %1988 : !pdl.operation -> ^bb3747, ^bb1
  ^bb3747:
    %1989 = pdl_interp.get_defining_op of %1934 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1989 : !pdl.operation -> ^bb3748, ^bb1
  ^bb3748:
    %1990 = pdl_interp.get_operand 0 of %1445
    pdl_interp.is_not_null %1990 : !pdl.value -> ^bb3749, ^bb1
  ^bb3749:
    %1991 = pdl_interp.get_defining_op of %1990 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %1991 : !pdl.operation -> ^bb3750, ^bb1
  ^bb3750:
    %1992 = pdl_interp.get_operand 1 of %1445
    %1993 = pdl_interp.get_defining_op of %1992 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %1993 : !pdl.operation -> ^bb3751, ^bb1
  ^bb3751:
    pdl_interp.is_not_null %1992 : !pdl.value -> ^bb3752, ^bb1
  ^bb3752:
    pdl_interp.check_operation_name of %1988 is "math.tan" -> ^bb3753, ^bb1
  ^bb3753:
    pdl_interp.check_operand_count of %1988 is 1 -> ^bb3754, ^bb1
  ^bb3754:
    pdl_interp.check_result_count of %1988 is 1 -> ^bb3755, ^bb1
  ^bb3755:
    %1994 = pdl_interp.get_result 0 of %1988
    pdl_interp.is_not_null %1994 : !pdl.value -> ^bb3756, ^bb1
  ^bb3756:
    pdl_interp.are_equal %1994, %1987 : !pdl.value -> ^bb3757, ^bb1
  ^bb3757:
    pdl_interp.check_operation_name of %1989 is "math.tan" -> ^bb3758, ^bb1
  ^bb3758:
    pdl_interp.check_operand_count of %1989 is 1 -> ^bb3759, ^bb1
  ^bb3759:
    pdl_interp.check_result_count of %1989 is 1 -> ^bb3760, ^bb1
  ^bb3760:
    %1995 = pdl_interp.get_result 0 of %1989
    pdl_interp.is_not_null %1995 : !pdl.value -> ^bb3761, ^bb1
  ^bb3761:
    pdl_interp.are_equal %1995, %1934 : !pdl.value -> ^bb3762, ^bb1
  ^bb3762:
    pdl_interp.check_operation_name of %1991 is "arith.constant" -> ^bb3763, ^bb1
  ^bb3763:
    pdl_interp.check_operand_count of %1991 is 0 -> ^bb3764, ^bb1
  ^bb3764:
    pdl_interp.check_result_count of %1991 is 1 -> ^bb3765, ^bb1
  ^bb3765:
    %1996 = pdl_interp.get_result 0 of %1991
    pdl_interp.is_not_null %1996 : !pdl.value -> ^bb3766, ^bb1
  ^bb3766:
    pdl_interp.are_equal %1996, %1990 : !pdl.value -> ^bb3767, ^bb1
  ^bb3767:
    pdl_interp.check_operation_name of %1993 is "arith.mulf" -> ^bb3768, ^bb1
  ^bb3768:
    pdl_interp.check_operand_count of %1993 is 2 -> ^bb3769, ^bb1
  ^bb3769:
    pdl_interp.check_result_count of %1993 is 1 -> ^bb3770, ^bb1
  ^bb3770:
    %1997 = pdl_interp.get_result 0 of %1993
    pdl_interp.is_not_null %1997 : !pdl.value -> ^bb3771, ^bb1
  ^bb3771:
    pdl_interp.are_equal %1997, %1992 : !pdl.value -> ^bb3772, ^bb1
  ^bb3772:
    %1998 = pdl_interp.get_operand 0 of %1989
    pdl_interp.is_not_null %1998 : !pdl.value -> ^bb3773, ^bb1
  ^bb3773:
    %1999 = pdl_interp.get_value_type of %1998 : !pdl.type
    %2000 = pdl_interp.get_value_type of %1995 : !pdl.type
    pdl_interp.are_equal %1999, %2000 : !pdl.type -> ^bb3774, ^bb1
  ^bb3774:
    %2001 = pdl_interp.get_value_type of %1933 : !pdl.type
    pdl_interp.are_equal %1999, %2001 : !pdl.type -> ^bb3775, ^bb1
  ^bb3775:
    %2002 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %1999, %2002 : !pdl.type -> ^bb3776, ^bb1
  ^bb3776:
    pdl_interp.check_type %1999 is f32 -> ^bb3777, ^bb1
  ^bb3777:
    %2003 = pdl_interp.get_operand 0 of %1988
    pdl_interp.is_not_null %2003 : !pdl.value -> ^bb3778, ^bb1
  ^bb3778:
    %2004 = pdl_interp.get_operand 0 of %1993
    %2005 = pdl_interp.get_defining_op of %2004 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %2005 : !pdl.operation -> ^bb3779, ^bb1
  ^bb3779:
    %2006 = pdl_interp.get_value_type of %1994 : !pdl.type
    pdl_interp.are_equal %1999, %2006 : !pdl.type -> ^bb3780, ^bb1
  ^bb3780:
    %2007 = pdl_interp.get_value_type of %1986 : !pdl.type
    pdl_interp.are_equal %1999, %2007 : !pdl.type -> ^bb3781, ^bb1
  ^bb3781:
    %2008 = pdl_interp.get_value_type of %1997 : !pdl.type
    pdl_interp.are_equal %1999, %2008 : !pdl.type -> ^bb3782, ^bb1
  ^bb3782:
    %2009 = pdl_interp.get_value_type of %1996 : !pdl.type
    pdl_interp.are_equal %1999, %2009 : !pdl.type -> ^bb3783, ^bb1
  ^bb3783:
    pdl_interp.is_not_null %2004 : !pdl.value -> ^bb3784, ^bb1
  ^bb3784:
    pdl_interp.check_operation_name of %2005 is "math.tan" -> ^bb3785, ^bb1
  ^bb3785:
    pdl_interp.check_operand_count of %2005 is 1 -> ^bb3786, ^bb1
  ^bb3786:
    pdl_interp.check_result_count of %2005 is 1 -> ^bb3787, ^bb1
  ^bb3787:
    %2010 = pdl_interp.get_result 0 of %2005
    pdl_interp.is_not_null %2010 : !pdl.value -> ^bb3788, ^bb1
  ^bb3788:
    pdl_interp.are_equal %2010, %2004 : !pdl.value -> ^bb3789, ^bb1
  ^bb3789:
    %2011 = pdl_interp.get_attribute "value" of %1991
    pdl_interp.is_not_null %2011 : !pdl.attribute -> ^bb3790, ^bb1
  ^bb3790:
    pdl_interp.check_attribute %2011 is 1.000000e+00 : f32 -> ^bb3791, ^bb1
  ^bb3791:
    %2012 = pdl_interp.get_value_type of %2003 : !pdl.type
    pdl_interp.are_equal %1999, %2012 : !pdl.type -> ^bb3792, ^bb1
  ^bb3792:
    %2013 = pdl_interp.get_operand 1 of %1993
    %2014 = pdl_interp.get_defining_op of %2013 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %2014 : !pdl.operation -> ^bb3793, ^bb1
  ^bb3793:
    pdl_interp.is_not_null %2013 : !pdl.value -> ^bb3794, ^bb1
  ^bb3794:
    pdl_interp.check_operation_name of %2014 is "math.tan" -> ^bb3795, ^bb1
  ^bb3795:
    pdl_interp.check_operand_count of %2014 is 1 -> ^bb3796, ^bb1
  ^bb3796:
    pdl_interp.check_result_count of %2014 is 1 -> ^bb3797, ^bb1
  ^bb3797:
    %2015 = pdl_interp.get_result 0 of %2014
    pdl_interp.is_not_null %2015 : !pdl.value -> ^bb3798, ^bb1
  ^bb3798:
    pdl_interp.are_equal %2015, %2013 : !pdl.value -> ^bb3799, ^bb1
  ^bb3799:
    %2016 = pdl_interp.get_operand 0 of %2014
    pdl_interp.are_equal %2016, %2003 : !pdl.value -> ^bb3800, ^bb1
  ^bb3800:
    %2017 = pdl_interp.get_operand 0 of %2005
    pdl_interp.are_equal %2017, %1998 : !pdl.value -> ^bb3801, ^bb1
  ^bb3801:
    %2018 = pdl_interp.get_value_type of %2010 : !pdl.type
    pdl_interp.are_equal %2018, %1999 : !pdl.type -> ^bb3802, ^bb1
  ^bb3802:
    %2019 = pdl_interp.get_value_type of %2015 : !pdl.type
    pdl_interp.are_equal %2019, %1999 : !pdl.type -> ^bb3803, ^bb1
  ^bb3803:
    pdl_interp.record_match @rewriters::@tan_sum_rev(%1998, %2003, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb1
  ^bb3646:
    pdl_interp.check_operand_count of %1445 is 0 -> ^bb3804, ^bb1
  ^bb3804:
    pdl_interp.check_result_count of %1445 is 1 -> ^bb3805, ^bb1
  ^bb3805:
    %2020 = pdl_interp.get_result 0 of %1445
    pdl_interp.is_not_null %2020 : !pdl.value -> ^bb3806, ^bb1
  ^bb3806:
    pdl_interp.are_equal %2020, %1444 : !pdl.value -> ^bb3807, ^bb1
  ^bb3807:
    %2021 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2021 : !pdl.value -> ^bb3808, ^bb1
  ^bb3808:
    %2022 = pdl_interp.get_defining_op of %2021 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %2022 : !pdl.operation -> ^bb3809, ^bb1
  ^bb3809:
    %2023 = pdl_interp.get_defining_op of %1934 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %2023 : !pdl.operation -> ^bb3810, ^bb1
  ^bb3810:
    pdl_interp.switch_operation_name of %2022 to ["math.cos", "math.sin", "math.exp"](^bb3811, ^bb3812, ^bb3813) -> ^bb1
  ^bb3811:
    pdl_interp.check_operand_count of %2022 is 1 -> ^bb3814, ^bb1
  ^bb3814:
    pdl_interp.check_result_count of %2022 is 1 -> ^bb3815, ^bb1
  ^bb3815:
    %2024 = pdl_interp.get_result 0 of %2022
    pdl_interp.is_not_null %2024 : !pdl.value -> ^bb3816, ^bb1
  ^bb3816:
    pdl_interp.are_equal %2024, %2021 : !pdl.value -> ^bb3817, ^bb1
  ^bb3817:
    pdl_interp.check_operation_name of %2023 is "math.cos" -> ^bb3818, ^bb1
  ^bb3818:
    pdl_interp.check_operand_count of %2023 is 1 -> ^bb3819, ^bb1
  ^bb3819:
    pdl_interp.check_result_count of %2023 is 1 -> ^bb3820, ^bb1
  ^bb3820:
    %2025 = pdl_interp.get_result 0 of %2023
    pdl_interp.is_not_null %2025 : !pdl.value -> ^bb3821, ^bb1
  ^bb3821:
    pdl_interp.are_equal %2025, %1934 : !pdl.value -> ^bb3822, ^bb1
  ^bb3822:
    %2026 = pdl_interp.get_operand 0 of %2023
    pdl_interp.is_not_null %2026 : !pdl.value -> ^bb3823, ^bb1
  ^bb3823:
    %2027 = pdl_interp.get_operand 0 of %2022
    pdl_interp.is_not_null %2027 : !pdl.value -> ^bb3824, ^bb1
  ^bb3824:
    %2028 = pdl_interp.get_attribute "value" of %1445
    pdl_interp.is_not_null %2028 : !pdl.attribute -> ^bb3825, ^bb1
  ^bb3825:
    pdl_interp.check_attribute %2028 is 2.000000e+00 : f32 -> ^bb3826, ^bb1
  ^bb3826:
    %2029 = pdl_interp.get_defining_op of %2027 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %2029 : !pdl.operation -> ^bb3827, ^bb1
  ^bb3827:
    %2030 = pdl_interp.get_defining_op of %2026 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %2030 : !pdl.operation -> ^bb3828, ^bb1
  ^bb3828:
    pdl_interp.check_operation_name of %2029 is "arith.subf" -> ^bb3829, ^bb1
  ^bb3829:
    pdl_interp.check_operand_count of %2029 is 2 -> ^bb3830, ^bb1
  ^bb3830:
    pdl_interp.check_result_count of %2029 is 1 -> ^bb3831, ^bb1
  ^bb3831:
    %2031 = pdl_interp.get_result 0 of %2029
    pdl_interp.is_not_null %2031 : !pdl.value -> ^bb3832, ^bb1
  ^bb3832:
    pdl_interp.are_equal %2031, %2027 : !pdl.value -> ^bb3833, ^bb1
  ^bb3833:
    pdl_interp.check_operation_name of %2030 is "arith.addf" -> ^bb3834, ^bb1
  ^bb3834:
    pdl_interp.check_operand_count of %2030 is 2 -> ^bb3835, ^bb1
  ^bb3835:
    pdl_interp.check_result_count of %2030 is 1 -> ^bb3836, ^bb1
  ^bb3836:
    %2032 = pdl_interp.get_result 0 of %2030
    pdl_interp.is_not_null %2032 : !pdl.value -> ^bb3837, ^bb1
  ^bb3837:
    pdl_interp.are_equal %2032, %2026 : !pdl.value -> ^bb3838, ^bb1
  ^bb3838:
    %2033 = pdl_interp.get_operand 0 of %2030
    pdl_interp.is_not_null %2033 : !pdl.value -> ^bb3839, ^bb1
  ^bb3839:
    %2034 = pdl_interp.get_operand 1 of %2030
    pdl_interp.is_not_null %2034 : !pdl.value -> ^bb3840, ^bb1
  ^bb3840:
    %2035 = pdl_interp.get_value_type of %2033 : !pdl.type
    %2036 = pdl_interp.get_value_type of %2032 : !pdl.type
    pdl_interp.are_equal %2035, %2036 : !pdl.type -> ^bb3841, ^bb1
  ^bb3841:
    %2037 = pdl_interp.get_value_type of %2025 : !pdl.type
    pdl_interp.are_equal %2035, %2037 : !pdl.type -> ^bb3842, ^bb1
  ^bb3842:
    %2038 = pdl_interp.get_value_type of %2024 : !pdl.type
    pdl_interp.are_equal %2035, %2038 : !pdl.type -> ^bb3843, ^bb1
  ^bb3843:
    %2039 = pdl_interp.get_value_type of %1933 : !pdl.type
    pdl_interp.are_equal %2035, %2039 : !pdl.type -> ^bb3844, ^bb1
  ^bb3844:
    %2040 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2035, %2040 : !pdl.type -> ^bb3845, ^bb1
  ^bb3845:
    pdl_interp.check_type %2035 is f32 -> ^bb3846, ^bb1
  ^bb3846:
    %2041 = pdl_interp.get_operand 0 of %2029
    pdl_interp.are_equal %2033, %2041 : !pdl.value -> ^bb3847, ^bb1
  ^bb3847:
    %2042 = pdl_interp.get_operand 1 of %2029
    pdl_interp.are_equal %2034, %2042 : !pdl.value -> ^bb3848, ^bb1
  ^bb3848:
    %2043 = pdl_interp.get_value_type of %2034 : !pdl.type
    pdl_interp.are_equal %2035, %2043 : !pdl.type -> ^bb3849, ^bb1
  ^bb3849:
    %2044 = pdl_interp.get_value_type of %2031 : !pdl.type
    pdl_interp.are_equal %2035, %2044 : !pdl.type -> ^bb3850, ^bb1
  ^bb3850:
    %2045 = pdl_interp.get_value_type of %2020 : !pdl.type
    pdl_interp.are_equal %2035, %2045 : !pdl.type -> ^bb3851, ^bb1
  ^bb3851:
    pdl_interp.record_match @rewriters::@cos_mult_rev(%2033, %2034, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb1
  ^bb3812:
    pdl_interp.check_operand_count of %2022 is 1 -> ^bb3852, ^bb1
  ^bb3852:
    pdl_interp.check_result_count of %2022 is 1 -> ^bb3853, ^bb1
  ^bb3853:
    %2046 = pdl_interp.get_result 0 of %2022
    pdl_interp.is_not_null %2046 : !pdl.value -> ^bb3854, ^bb1
  ^bb3854:
    pdl_interp.are_equal %2046, %2021 : !pdl.value -> ^bb3855, ^bb1
  ^bb3855:
    pdl_interp.check_operation_name of %2023 is "math.sin" -> ^bb3856, ^bb1
  ^bb3856:
    pdl_interp.check_operand_count of %2023 is 1 -> ^bb3857, ^bb1
  ^bb3857:
    pdl_interp.check_result_count of %2023 is 1 -> ^bb3858, ^bb1
  ^bb3858:
    %2047 = pdl_interp.get_result 0 of %2023
    pdl_interp.is_not_null %2047 : !pdl.value -> ^bb3859, ^bb1
  ^bb3859:
    pdl_interp.are_equal %2047, %1934 : !pdl.value -> ^bb3860, ^bb1
  ^bb3860:
    %2048 = pdl_interp.get_operand 0 of %2023
    pdl_interp.is_not_null %2048 : !pdl.value -> ^bb3861, ^bb1
  ^bb3861:
    %2049 = pdl_interp.get_operand 0 of %2022
    pdl_interp.is_not_null %2049 : !pdl.value -> ^bb3862, ^bb1
  ^bb3862:
    %2050 = pdl_interp.get_attribute "value" of %1445
    pdl_interp.is_not_null %2050 : !pdl.attribute -> ^bb3863, ^bb1
  ^bb3863:
    pdl_interp.check_attribute %2050 is 2.000000e+00 : f32 -> ^bb3864, ^bb1
  ^bb3864:
    %2051 = pdl_interp.get_defining_op of %2049 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %2051 : !pdl.operation -> ^bb3865, ^bb1
  ^bb3865:
    %2052 = pdl_interp.get_defining_op of %2048 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %2052 : !pdl.operation -> ^bb3866, ^bb1
  ^bb3866:
    pdl_interp.check_operation_name of %2051 is "arith.addf" -> ^bb3867, ^bb1
  ^bb3867:
    pdl_interp.check_operand_count of %2051 is 2 -> ^bb3868, ^bb1
  ^bb3868:
    pdl_interp.check_result_count of %2051 is 1 -> ^bb3869, ^bb1
  ^bb3869:
    %2053 = pdl_interp.get_result 0 of %2051
    pdl_interp.is_not_null %2053 : !pdl.value -> ^bb3870, ^bb1
  ^bb3870:
    pdl_interp.are_equal %2053, %2049 : !pdl.value -> ^bb3871, ^bb1
  ^bb3871:
    pdl_interp.check_operation_name of %2052 is "arith.subf" -> ^bb3872, ^bb1
  ^bb3872:
    pdl_interp.check_operand_count of %2052 is 2 -> ^bb3873, ^bb1
  ^bb3873:
    pdl_interp.check_result_count of %2052 is 1 -> ^bb3874, ^bb1
  ^bb3874:
    %2054 = pdl_interp.get_result 0 of %2052
    pdl_interp.is_not_null %2054 : !pdl.value -> ^bb3875, ^bb1
  ^bb3875:
    pdl_interp.are_equal %2054, %2048 : !pdl.value -> ^bb3876, ^bb1
  ^bb3876:
    %2055 = pdl_interp.get_operand 0 of %2052
    pdl_interp.is_not_null %2055 : !pdl.value -> ^bb3877, ^bb1
  ^bb3877:
    %2056 = pdl_interp.get_operand 1 of %2052
    pdl_interp.is_not_null %2056 : !pdl.value -> ^bb3878, ^bb1
  ^bb3878:
    %2057 = pdl_interp.get_value_type of %2055 : !pdl.type
    %2058 = pdl_interp.get_value_type of %2054 : !pdl.type
    pdl_interp.are_equal %2057, %2058 : !pdl.type -> ^bb3879, ^bb1
  ^bb3879:
    %2059 = pdl_interp.get_value_type of %2047 : !pdl.type
    pdl_interp.are_equal %2057, %2059 : !pdl.type -> ^bb3880, ^bb1
  ^bb3880:
    %2060 = pdl_interp.get_value_type of %2046 : !pdl.type
    pdl_interp.are_equal %2057, %2060 : !pdl.type -> ^bb3881, ^bb1
  ^bb3881:
    %2061 = pdl_interp.get_value_type of %1933 : !pdl.type
    pdl_interp.are_equal %2057, %2061 : !pdl.type -> ^bb3882, ^bb1
  ^bb3882:
    %2062 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2057, %2062 : !pdl.type -> ^bb3883, ^bb1
  ^bb3883:
    pdl_interp.check_type %2057 is f32 -> ^bb3884, ^bb1
  ^bb3884:
    %2063 = pdl_interp.get_operand 0 of %2051
    pdl_interp.are_equal %2055, %2063 : !pdl.value -> ^bb3885, ^bb1
  ^bb3885:
    %2064 = pdl_interp.get_operand 1 of %2051
    pdl_interp.are_equal %2056, %2064 : !pdl.value -> ^bb3886, ^bb1
  ^bb3886:
    %2065 = pdl_interp.get_value_type of %2056 : !pdl.type
    pdl_interp.are_equal %2057, %2065 : !pdl.type -> ^bb3887, ^bb1
  ^bb3887:
    %2066 = pdl_interp.get_value_type of %2053 : !pdl.type
    pdl_interp.are_equal %2057, %2066 : !pdl.type -> ^bb3888, ^bb1
  ^bb3888:
    %2067 = pdl_interp.get_value_type of %2020 : !pdl.type
    pdl_interp.are_equal %2057, %2067 : !pdl.type -> ^bb3889, ^bb1
  ^bb3889:
    pdl_interp.record_match @rewriters::@sin_cos_mult_rev(%2055, %2056, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb1
  ^bb3813:
    pdl_interp.check_operand_count of %2022 is 1 -> ^bb3890, ^bb1
  ^bb3890:
    pdl_interp.check_result_count of %2022 is 1 -> ^bb3891, ^bb1
  ^bb3891:
    %2068 = pdl_interp.get_result 0 of %2022
    pdl_interp.is_not_null %2068 : !pdl.value -> ^bb3892, ^bb1
  ^bb3892:
    pdl_interp.are_equal %2068, %2021 : !pdl.value -> ^bb3893, ^bb1
  ^bb3893:
    pdl_interp.check_operation_name of %2023 is "math.exp" -> ^bb3894, ^bb1
  ^bb3894:
    pdl_interp.check_operand_count of %2023 is 1 -> ^bb3895, ^bb1
  ^bb3895:
    pdl_interp.check_result_count of %2023 is 1 -> ^bb3896, ^bb1
  ^bb3896:
    %2069 = pdl_interp.get_result 0 of %2023
    pdl_interp.is_not_null %2069 : !pdl.value -> ^bb3897, ^bb1
  ^bb3897:
    pdl_interp.are_equal %2069, %1934 : !pdl.value -> ^bb3898, ^bb1
  ^bb3898:
    %2070 = pdl_interp.get_operand 0 of %2023
    pdl_interp.is_not_null %2070 : !pdl.value -> ^bb3899, ^bb1
  ^bb3899:
    %2071 = pdl_interp.get_value_type of %2070 : !pdl.type
    %2072 = pdl_interp.get_value_type of %2069 : !pdl.type
    pdl_interp.are_equal %2071, %2072 : !pdl.type -> ^bb3900, ^bb1
  ^bb3900:
    %2073 = pdl_interp.get_value_type of %1933 : !pdl.type
    pdl_interp.are_equal %2071, %2073 : !pdl.type -> ^bb3901, ^bb1
  ^bb3901:
    %2074 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2071, %2074 : !pdl.type -> ^bb3902, ^bb1
  ^bb3902:
    pdl_interp.check_type %2071 is f32 -> ^bb3903, ^bb1
  ^bb3903:
    %2075 = pdl_interp.get_operand 0 of %2022
    pdl_interp.is_not_null %2075 : !pdl.value -> ^bb3904, ^bb1
  ^bb3904:
    %2076 = pdl_interp.get_value_type of %2068 : !pdl.type
    pdl_interp.are_equal %2071, %2076 : !pdl.type -> ^bb3905, ^bb1
  ^bb3905:
    %2077 = pdl_interp.get_attribute "value" of %1445
    pdl_interp.is_not_null %2077 : !pdl.attribute -> ^bb3906, ^bb1
  ^bb3906:
    pdl_interp.check_attribute %2077 is 2.000000e+00 : f32 -> ^bb3907, ^bb1
  ^bb3907:
    %2078 = pdl_interp.get_value_type of %2020 : !pdl.type
    pdl_interp.are_equal %2071, %2078 : !pdl.type -> ^bb3908, ^bb1
  ^bb3908:
    %2079 = pdl_interp.get_defining_op of %2075 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %2079 : !pdl.operation -> ^bb3909, ^bb1
  ^bb3909:
    pdl_interp.check_operation_name of %2079 is "arith.negf" -> ^bb3910, ^bb1
  ^bb3910:
    pdl_interp.check_operand_count of %2079 is 1 -> ^bb3911, ^bb1
  ^bb3911:
    pdl_interp.check_result_count of %2079 is 1 -> ^bb3912, ^bb1
  ^bb3912:
    %2080 = pdl_interp.get_result 0 of %2079
    pdl_interp.is_not_null %2080 : !pdl.value -> ^bb3913, ^bb1
  ^bb3913:
    pdl_interp.are_equal %2080, %2075 : !pdl.value -> ^bb3914, ^bb1
  ^bb3914:
    %2081 = pdl_interp.get_value_type of %2080 : !pdl.type
    pdl_interp.are_equal %2081, %2071 : !pdl.type -> ^bb3915, ^bb1
  ^bb3915:
    %2082 = pdl_interp.get_operand 0 of %2079
    pdl_interp.are_equal %2082, %2070 : !pdl.value -> ^bb3916, ^bb1
  ^bb3916:
    pdl_interp.record_match @rewriters::@cosh_def_rev(%2070, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb1
  ^bb2714:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb3917, ^bb1
  ^bb3917:
    pdl_interp.check_result_count of %3 is 1 -> ^bb3918, ^bb1
  ^bb3918:
    %2083 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2083 : !pdl.value -> ^bb3919, ^bb1
  ^bb3919:
    pdl_interp.are_equal %2083, %2 : !pdl.value -> ^bb3920, ^bb1
  ^bb3920:
    %2084 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2084 : !pdl.value -> ^bb3921, ^bb1
  ^bb3921:
    pdl_interp.is_not_null %1444 : !pdl.value -> ^bb3922, ^bb1
  ^bb3922:
    %2085 = pdl_interp.get_value_type of %2084 : !pdl.type
    %2086 = pdl_interp.get_value_type of %2083 : !pdl.type
    pdl_interp.are_equal %2085, %2086 : !pdl.type -> ^bb3923, ^bb1
  ^bb3923:
    %2087 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2085, %2087 : !pdl.type -> ^bb3924, ^bb1
  ^bb3924:
    pdl_interp.check_type %2085 is f32 -> ^bb3925, ^bb1
  ^bb3925:
    pdl_interp.switch_operation_name of %1445 to ["arith.addf", "math.sqrt"](^bb3926, ^bb3927) -> ^bb1
  ^bb3926:
    pdl_interp.check_operand_count of %1445 is 2 -> ^bb3928, ^bb1
  ^bb3928:
    pdl_interp.check_result_count of %1445 is 1 -> ^bb3929, ^bb1
  ^bb3929:
    %2088 = pdl_interp.get_result 0 of %1445
    pdl_interp.is_not_null %2088 : !pdl.value -> ^bb3930, ^bb1
  ^bb3930:
    pdl_interp.are_equal %2088, %1444 : !pdl.value -> ^bb3931, ^bb1
  ^bb3931:
    %2089 = pdl_interp.get_operand 0 of %1445
    pdl_interp.is_not_null %2089 : !pdl.value -> ^bb3932, ^bb1
  ^bb3932:
    %2090 = pdl_interp.get_defining_op of %2089 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %2090 : !pdl.operation -> ^bb3933, ^bb1
  ^bb3933:
    %2091 = pdl_interp.get_operand 1 of %1445
    %2092 = pdl_interp.get_defining_op of %2091 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %2092 : !pdl.operation -> ^bb3934, ^bb1
  ^bb3934:
    pdl_interp.is_not_null %2091 : !pdl.value -> ^bb3935, ^bb1
  ^bb3935:
    %2093 = pdl_interp.get_value_type of %2088 : !pdl.type
    pdl_interp.are_equal %2085, %2093 : !pdl.type -> ^bb3936, ^bb1
  ^bb3936:
    pdl_interp.check_operation_name of %2090 is "math.cosh" -> ^bb3937, ^bb1
  ^bb3937:
    pdl_interp.check_operand_count of %2090 is 1 -> ^bb3938, ^bb1
  ^bb3938:
    pdl_interp.check_result_count of %2090 is 1 -> ^bb3939, ^bb1
  ^bb3939:
    %2094 = pdl_interp.get_result 0 of %2090
    pdl_interp.is_not_null %2094 : !pdl.value -> ^bb3940, ^bb1
  ^bb3940:
    pdl_interp.are_equal %2094, %2089 : !pdl.value -> ^bb3941, ^bb1
  ^bb3941:
    pdl_interp.check_operation_name of %2092 is "arith.constant" -> ^bb3942, ^bb1
  ^bb3942:
    pdl_interp.check_operand_count of %2092 is 0 -> ^bb3943, ^bb1
  ^bb3943:
    pdl_interp.check_result_count of %2092 is 1 -> ^bb3944, ^bb1
  ^bb3944:
    %2095 = pdl_interp.get_result 0 of %2092
    pdl_interp.is_not_null %2095 : !pdl.value -> ^bb3945, ^bb1
  ^bb3945:
    pdl_interp.are_equal %2095, %2091 : !pdl.value -> ^bb3946, ^bb1
  ^bb3946:
    %2096 = pdl_interp.get_value_type of %2095 : !pdl.type
    pdl_interp.are_equal %2096, %2085 : !pdl.type -> ^bb3947, ^bb1
  ^bb3947:
    %2097 = pdl_interp.get_attribute "value" of %2092
    pdl_interp.is_not_null %2097 : !pdl.attribute -> ^bb3948, ^bb1
  ^bb3948:
    pdl_interp.check_attribute %2097 is 1.000000e+00 : f32 -> ^bb3949, ^bb1
  ^bb3949:
    %2098 = pdl_interp.get_value_type of %2094 : !pdl.type
    pdl_interp.are_equal %2098, %2085 : !pdl.type -> ^bb3950, ^bb1
  ^bb3950:
    %2099 = pdl_interp.get_operand 0 of %2090
    pdl_interp.are_equal %2099, %2084 : !pdl.value -> ^bb3951, ^bb1
  ^bb3951:
    pdl_interp.record_match @rewriters::@tanh_1div2_rev(%2084, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb1
  ^bb3927:
    pdl_interp.check_operand_count of %1445 is 1 -> ^bb3952, ^bb1
  ^bb3952:
    pdl_interp.check_result_count of %1445 is 1 -> ^bb3953, ^bb1
  ^bb3953:
    %2100 = pdl_interp.get_result 0 of %1445
    pdl_interp.is_not_null %2100 : !pdl.value -> ^bb3954, ^bb1
  ^bb3954:
    pdl_interp.are_equal %2100, %1444 : !pdl.value -> ^bb3955, ^bb1
  ^bb3955:
    %2101 = pdl_interp.get_operand 0 of %1445
    pdl_interp.is_not_null %2101 : !pdl.value -> ^bb3956, ^bb1
  ^bb3956:
    %2102 = pdl_interp.get_defining_op of %2101 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %2102 : !pdl.operation -> ^bb3957, ^bb1
  ^bb3957:
    %2103 = pdl_interp.get_value_type of %2100 : !pdl.type
    pdl_interp.are_equal %2085, %2103 : !pdl.type -> ^bb3958, ^bb1
  ^bb3958:
    pdl_interp.check_operation_name of %2102 is "arith.mulf" -> ^bb3959, ^bb1
  ^bb3959:
    pdl_interp.check_operand_count of %2102 is 2 -> ^bb3960, ^bb1
  ^bb3960:
    pdl_interp.check_result_count of %2102 is 1 -> ^bb3961, ^bb1
  ^bb3961:
    %2104 = pdl_interp.get_result 0 of %2102
    pdl_interp.is_not_null %2104 : !pdl.value -> ^bb3962, ^bb1
  ^bb3962:
    pdl_interp.are_equal %2104, %2101 : !pdl.value -> ^bb3963, ^bb1
  ^bb3963:
    %2105 = pdl_interp.get_operand 0 of %2102
    %2106 = pdl_interp.get_defining_op of %2105 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %2106 : !pdl.operation -> ^bb3964, ^bb1
  ^bb3964:
    pdl_interp.is_not_null %2105 : !pdl.value -> ^bb3965, ^bb1
  ^bb3965:
    pdl_interp.check_operation_name of %2106 is "arith.constant" -> ^bb3966, ^bb1
  ^bb3966:
    pdl_interp.check_operand_count of %2106 is 0 -> ^bb3967, ^bb1
  ^bb3967:
    pdl_interp.check_result_count of %2106 is 1 -> ^bb3968, ^bb1
  ^bb3968:
    %2107 = pdl_interp.get_result 0 of %2106
    pdl_interp.is_not_null %2107 : !pdl.value -> ^bb3969, ^bb1
  ^bb3969:
    pdl_interp.are_equal %2107, %2105 : !pdl.value -> ^bb3970, ^bb1
  ^bb3970:
    %2108 = pdl_interp.get_operand 1 of %2102
    %2109 = pdl_interp.get_defining_op of %2108 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %2109 : !pdl.operation -> ^bb3971, ^bb1
  ^bb3971:
    %2110 = pdl_interp.get_value_type of %2104 : !pdl.type
    pdl_interp.are_equal %2110, %2085 : !pdl.type -> ^bb3972, ^bb1
  ^bb3972:
    pdl_interp.is_not_null %2108 : !pdl.value -> ^bb3973, ^bb1
  ^bb3973:
    pdl_interp.check_operation_name of %2109 is "arith.addf" -> ^bb3974, ^bb1
  ^bb3974:
    pdl_interp.check_operand_count of %2109 is 2 -> ^bb3975, ^bb1
  ^bb3975:
    pdl_interp.check_result_count of %2109 is 1 -> ^bb3976, ^bb1
  ^bb3976:
    %2111 = pdl_interp.get_attribute "value" of %2106
    pdl_interp.is_not_null %2111 : !pdl.attribute -> ^bb3977, ^bb1
  ^bb3977:
    pdl_interp.check_attribute %2111 is 2.000000e+00 : f32 -> ^bb3978, ^bb1
  ^bb3978:
    %2112 = pdl_interp.get_result 0 of %2109
    pdl_interp.is_not_null %2112 : !pdl.value -> ^bb3979, ^bb1
  ^bb3979:
    pdl_interp.are_equal %2112, %2108 : !pdl.value -> ^bb3980, ^bb1
  ^bb3980:
    %2113 = pdl_interp.get_operand 0 of %2109
    pdl_interp.is_not_null %2113 : !pdl.value -> ^bb3981, ^bb1
  ^bb3981:
    %2114 = pdl_interp.get_defining_op of %2113 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %2114 : !pdl.operation -> ^bb3982, ^bb1
  ^bb3982:
    %2115 = pdl_interp.get_operand 1 of %2109
    %2116 = pdl_interp.get_defining_op of %2115 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op.operand[1].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %2116 : !pdl.operation -> ^bb3983, ^bb1
  ^bb3983:
    pdl_interp.is_not_null %2115 : !pdl.value -> ^bb3984, ^bb1
  ^bb3984:
    %2117 = pdl_interp.get_value_type of %2107 : !pdl.type
    pdl_interp.are_equal %2117, %2085 : !pdl.type -> ^bb3985, ^bb1
  ^bb3985:
    %2118 = pdl_interp.get_value_type of %2112 : !pdl.type
    pdl_interp.are_equal %2118, %2085 : !pdl.type -> ^bb3986, ^bb1
  ^bb3986:
    pdl_interp.check_operation_name of %2114 is "math.cosh" -> ^bb3987, ^bb1
  ^bb3987:
    pdl_interp.check_operation_name of %2116 is "arith.constant" -> ^bb3988, ^bb1
  ^bb3988:
    pdl_interp.check_operand_count of %2114 is 1 -> ^bb3989, ^bb1
  ^bb3989:
    pdl_interp.check_operand_count of %2116 is 0 -> ^bb3990, ^bb1
  ^bb3990:
    pdl_interp.check_result_count of %2114 is 1 -> ^bb3991, ^bb1
  ^bb3991:
    pdl_interp.check_result_count of %2116 is 1 -> ^bb3992, ^bb1
  ^bb3992:
    %2119 = pdl_interp.get_operand 0 of %2114
    pdl_interp.are_equal %2119, %2084 : !pdl.value -> ^bb3993, ^bb1
  ^bb3993:
    %2120 = pdl_interp.get_attribute "value" of %2116
    pdl_interp.is_not_null %2120 : !pdl.attribute -> ^bb3994, ^bb1
  ^bb3994:
    pdl_interp.check_attribute %2120 is 1.000000e+00 : f32 -> ^bb3995, ^bb1
  ^bb3995:
    %2121 = pdl_interp.get_result 0 of %2114
    pdl_interp.is_not_null %2121 : !pdl.value -> ^bb3996, ^bb1
  ^bb3996:
    %2122 = pdl_interp.get_result 0 of %2116
    pdl_interp.is_not_null %2122 : !pdl.value -> ^bb3997, ^bb1
  ^bb3997:
    pdl_interp.are_equal %2121, %2113 : !pdl.value -> ^bb3998, ^bb1
  ^bb3998:
    pdl_interp.are_equal %2122, %2115 : !pdl.value -> ^bb3999, ^bb1
  ^bb3999:
    %2123 = pdl_interp.get_value_type of %2121 : !pdl.type
    pdl_interp.are_equal %2123, %2085 : !pdl.type -> ^bb4000, ^bb1
  ^bb4000:
    %2124 = pdl_interp.get_value_type of %2122 : !pdl.type
    pdl_interp.are_equal %2124, %2085 : !pdl.type -> ^bb4001, ^bb1
  ^bb4001:
    pdl_interp.record_match @rewriters::@sinh_1div2_rev(%2084, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb1
  ^bb2715:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb4002, ^bb1
  ^bb4002:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4003, ^bb1
  ^bb4003:
    %2125 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2125 : !pdl.value -> ^bb4004, ^bb1
  ^bb4004:
    pdl_interp.are_equal %2125, %2 : !pdl.value -> ^bb4005, ^bb1
  ^bb4005:
    %2126 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2126 : !pdl.value -> ^bb4006, ^bb1
  ^bb4006:
    pdl_interp.is_not_null %1444 : !pdl.value -> ^bb4007, ^bb1
  ^bb4007:
    pdl_interp.check_operation_name of %1445 is "arith.constant" -> ^bb4008, ^bb1
  ^bb4008:
    pdl_interp.check_operand_count of %1445 is 0 -> ^bb4009, ^bb1
  ^bb4009:
    pdl_interp.check_result_count of %1445 is 1 -> ^bb4010, ^bb1
  ^bb4010:
    %2127 = pdl_interp.get_result 0 of %1445
    pdl_interp.is_not_null %2127 : !pdl.value -> ^bb4011, ^bb1
  ^bb4011:
    pdl_interp.are_equal %2127, %1444 : !pdl.value -> ^bb4012, ^bb1
  ^bb4012:
    %2128 = pdl_interp.get_defining_op of %2126 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %2128 : !pdl.operation -> ^bb4013, ^bb1
  ^bb4013:
    pdl_interp.check_operation_name of %2128 is "arith.divf" -> ^bb4014, ^bb1
  ^bb4014:
    pdl_interp.check_operand_count of %2128 is 2 -> ^bb4015, ^bb1
  ^bb4015:
    pdl_interp.check_result_count of %2128 is 1 -> ^bb4016, ^bb1
  ^bb4016:
    %2129 = pdl_interp.get_result 0 of %2128
    pdl_interp.is_not_null %2129 : !pdl.value -> ^bb4017, ^bb1
  ^bb4017:
    pdl_interp.are_equal %2129, %2126 : !pdl.value -> ^bb4018, ^bb1
  ^bb4018:
    %2130 = pdl_interp.get_operand 0 of %2128
    pdl_interp.is_not_null %2130 : !pdl.value -> ^bb4019, ^bb1
  ^bb4019:
    %2131 = pdl_interp.get_attribute "value" of %1445
    pdl_interp.is_not_null %2131 : !pdl.attribute -> ^bb4020, ^bb1
  ^bb4020:
    pdl_interp.check_attribute %2131 is 2.000000e+00 : f32 -> ^bb4021, ^bb1
  ^bb4021:
    %2132 = pdl_interp.get_defining_op of %2130 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %2132 : !pdl.operation -> ^bb4022, ^bb1
  ^bb4022:
    %2133 = pdl_interp.get_operand 1 of %2128
    %2134 = pdl_interp.get_defining_op of %2133 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %2134 : !pdl.operation -> ^bb4023, ^bb1
  ^bb4023:
    pdl_interp.check_operation_name of %2132 is "arith.addf" -> ^bb4024, ^bb1
  ^bb4024:
    pdl_interp.check_operand_count of %2132 is 2 -> ^bb4025, ^bb1
  ^bb4025:
    pdl_interp.check_result_count of %2132 is 1 -> ^bb4026, ^bb1
  ^bb4026:
    %2135 = pdl_interp.get_result 0 of %2132
    pdl_interp.is_not_null %2135 : !pdl.value -> ^bb4027, ^bb1
  ^bb4027:
    pdl_interp.are_equal %2135, %2130 : !pdl.value -> ^bb4028, ^bb1
  ^bb4028:
    pdl_interp.is_not_null %2133 : !pdl.value -> ^bb4029, ^bb1
  ^bb4029:
    %2136 = pdl_interp.get_operand 0 of %2132
    pdl_interp.is_not_null %2136 : !pdl.value -> ^bb4030, ^bb1
  ^bb4030:
    pdl_interp.check_operation_name of %2134 is "arith.subf" -> ^bb4031, ^bb1
  ^bb4031:
    pdl_interp.check_operand_count of %2134 is 2 -> ^bb4032, ^bb1
  ^bb4032:
    pdl_interp.check_result_count of %2134 is 1 -> ^bb4033, ^bb1
  ^bb4033:
    %2137 = pdl_interp.get_result 0 of %2134
    pdl_interp.is_not_null %2137 : !pdl.value -> ^bb4034, ^bb1
  ^bb4034:
    pdl_interp.are_equal %2137, %2133 : !pdl.value -> ^bb4035, ^bb1
  ^bb4035:
    %2138 = pdl_interp.get_operand 1 of %2132
    pdl_interp.is_not_null %2138 : !pdl.value -> ^bb4036, ^bb1
  ^bb4036:
    %2139 = pdl_interp.get_defining_op of %2136 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %2139 : !pdl.operation -> ^bb4037, ^bb1
  ^bb4037:
    %2140 = pdl_interp.get_operand 0 of %2134
    pdl_interp.is_not_null %2140 : !pdl.value -> ^bb4038, ^bb1
  ^bb4038:
    pdl_interp.check_operation_name of %2139 is "arith.constant" -> ^bb4039, ^bb1
  ^bb4039:
    pdl_interp.check_operand_count of %2139 is 0 -> ^bb4040, ^bb1
  ^bb4040:
    pdl_interp.check_result_count of %2139 is 1 -> ^bb4041, ^bb1
  ^bb4041:
    %2141 = pdl_interp.get_attribute "value" of %2139
    pdl_interp.is_not_null %2141 : !pdl.attribute -> ^bb4042, ^bb1
  ^bb4042:
    pdl_interp.check_attribute %2141 is 1.000000e+00 : f32 -> ^bb4043, ^bb1
  ^bb4043:
    %2142 = pdl_interp.get_result 0 of %2139
    pdl_interp.is_not_null %2142 : !pdl.value -> ^bb4044, ^bb1
  ^bb4044:
    pdl_interp.are_equal %2142, %2136 : !pdl.value -> ^bb4045, ^bb1
  ^bb4045:
    %2143 = pdl_interp.get_value_type of %2142 : !pdl.type
    %2144 = pdl_interp.get_value_type of %2138 : !pdl.type
    pdl_interp.are_equal %2143, %2144 : !pdl.type -> ^bb4046, ^bb1
  ^bb4046:
    %2145 = pdl_interp.get_value_type of %2135 : !pdl.type
    pdl_interp.are_equal %2143, %2145 : !pdl.type -> ^bb4047, ^bb1
  ^bb4047:
    %2146 = pdl_interp.get_value_type of %2129 : !pdl.type
    pdl_interp.are_equal %2143, %2146 : !pdl.type -> ^bb4048, ^bb1
  ^bb4048:
    %2147 = pdl_interp.get_value_type of %2125 : !pdl.type
    pdl_interp.are_equal %2143, %2147 : !pdl.type -> ^bb4049, ^bb1
  ^bb4049:
    %2148 = pdl_interp.get_value_type of %2127 : !pdl.type
    pdl_interp.are_equal %2143, %2148 : !pdl.type -> ^bb4050, ^bb1
  ^bb4050:
    %2149 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2143, %2149 : !pdl.type -> ^bb4051, ^bb1
  ^bb4051:
    pdl_interp.check_type %2143 is f32 -> ^bb4052, ^bb1
  ^bb4052:
    %2150 = pdl_interp.get_defining_op of %2140 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %2150 : !pdl.operation -> ^bb4053, ^bb1
  ^bb4053:
    %2151 = pdl_interp.get_operand 1 of %2134
    pdl_interp.are_equal %2138, %2151 : !pdl.value -> ^bb4054, ^bb1
  ^bb4054:
    pdl_interp.check_operation_name of %2150 is "arith.constant" -> ^bb4055, ^bb1
  ^bb4055:
    pdl_interp.check_operand_count of %2150 is 0 -> ^bb4056, ^bb1
  ^bb4056:
    pdl_interp.check_result_count of %2150 is 1 -> ^bb4057, ^bb1
  ^bb4057:
    %2152 = pdl_interp.get_attribute "value" of %2150
    pdl_interp.is_not_null %2152 : !pdl.attribute -> ^bb4058, ^bb1
  ^bb4058:
    pdl_interp.check_attribute %2152 is 1.000000e+00 : f32 -> ^bb4059, ^bb1
  ^bb4059:
    %2153 = pdl_interp.get_result 0 of %2150
    pdl_interp.is_not_null %2153 : !pdl.value -> ^bb4060, ^bb1
  ^bb4060:
    pdl_interp.are_equal %2153, %2140 : !pdl.value -> ^bb4061, ^bb1
  ^bb4061:
    %2154 = pdl_interp.get_value_type of %2153 : !pdl.type
    pdl_interp.are_equal %2143, %2154 : !pdl.type -> ^bb4062, ^bb1
  ^bb4062:
    %2155 = pdl_interp.get_value_type of %2137 : !pdl.type
    pdl_interp.are_equal %2143, %2155 : !pdl.type -> ^bb4063, ^bb1
  ^bb4063:
    pdl_interp.record_match @rewriters::@atanh_def_rev(%2138, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb1
  ^bb2699:
    pdl_interp.switch_operation_name of %3 to ["arith.divf", "arith.mulf", "arith.constant", "arith.negf", "arith.addf", "arith.subf", "math.absf", "math.cbrt", "math.sqrt"](^bb4064, ^bb4065, ^bb4066, ^bb4067, ^bb4068, ^bb4069, ^bb4070, ^bb4071, ^bb4072) -> ^bb2700
  ^bb4064:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb4073, ^bb2700
  ^bb4073:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4074, ^bb2700
  ^bb4074:
    %2156 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2156 : !pdl.value -> ^bb4075, ^bb2700
  ^bb4075:
    pdl_interp.are_equal %2156, %2 : !pdl.value -> ^bb4076, ^bb2700
  ^bb4076:
    %2157 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2157 : !pdl.value -> ^bb4077, ^bb2700
  ^bb4077:
    %2158 = pdl_interp.get_operand 1 of %0
    pdl_interp.is_not_null %2158 : !pdl.value -> ^bb4078, ^bb2700
  ^bb4078:
    %2159 = pdl_interp.get_value_type of %2157 : !pdl.type
    %2160 = pdl_interp.get_value_type of %2156 : !pdl.type
    pdl_interp.are_equal %2159, %2160 : !pdl.type -> ^bb4079, ^bb2700
  ^bb4079:
    %2161 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2159, %2161 : !pdl.type -> ^bb4080, ^bb2700
  ^bb4080:
    pdl_interp.check_type %2159 is f32 -> ^bb4081, ^bb2700
  ^bb4081:
    %2162 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2162 : !pdl.value -> ^bb4082, ^bb2700
  ^bb4082:
    %2163 = pdl_interp.get_value_type of %2162 : !pdl.type
    pdl_interp.are_equal %2159, %2163 : !pdl.type -> ^bb4083, ^bb2700
  ^bb4083:
    %2164 = pdl_interp.get_value_type of %2158 : !pdl.type
    pdl_interp.are_equal %2159, %2164 : !pdl.type -> ^bb4084, ^bb2700
  ^bb4084:
    pdl_interp.record_match @rewriters::@associate_divldiv(%2162, %2158, %2157, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb2700
  ^bb4065:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb4085, ^bb2700
  ^bb4085:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4086, ^bb2700
  ^bb4086:
    %2165 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2165 : !pdl.value -> ^bb4087, ^bb2700
  ^bb4087:
    pdl_interp.are_equal %2165, %2 : !pdl.value -> ^bb4088, ^bb2700
  ^bb4088:
    %2166 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2166 : !pdl.value -> ^bb4089, ^bb2700
  ^bb4089:
    %2167 = pdl_interp.get_operand 1 of %0
    pdl_interp.is_not_null %2167 : !pdl.value -> ^bb4090, ^bb2700
  ^bb4090:
    %2168 = pdl_interp.get_value_type of %2166 : !pdl.type
    %2169 = pdl_interp.get_value_type of %2165 : !pdl.type
    pdl_interp.are_equal %2168, %2169 : !pdl.type -> ^bb4091, ^bb2700
  ^bb4091:
    %2170 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2168, %2170 : !pdl.type -> ^bb4092, ^bb2700
  ^bb4092:
    pdl_interp.check_type %2168 is f32 -> ^bb4093, ^bb2700
  ^bb4093:
    %2171 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2171 : !pdl.value -> ^bb4094, ^bb2700
  ^bb4094:
    %2172 = pdl_interp.get_value_type of %2171 : !pdl.type
    pdl_interp.are_equal %2168, %2172 : !pdl.type -> ^bb4095, ^bb2700
  ^bb4095:
    %2173 = pdl_interp.get_value_type of %2167 : !pdl.type
    pdl_interp.are_equal %2168, %2173 : !pdl.type -> ^bb4096, ^bb2700
  ^bb4096:
    pdl_interp.record_match @rewriters::@associate_divlmul(%2171, %2167, %2166, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb2700
  ^bb4066:
    pdl_interp.check_operand_count of %3 is 0 -> ^bb4097, ^bb2700
  ^bb4097:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4098, ^bb2700
  ^bb4098:
    %2174 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2174 : !pdl.value -> ^bb4099, ^bb2700
  ^bb4099:
    pdl_interp.are_equal %2174, %2 : !pdl.value -> ^bb4100, ^bb2700
  ^bb4100:
    %2175 = pdl_interp.get_operand 1 of %0
    pdl_interp.is_not_null %2175 : !pdl.value -> ^bb4101, ^bb2700
  ^bb4101:
    %2176 = pdl_interp.get_attribute "value" of %3
    pdl_interp.is_not_null %2176 : !pdl.attribute -> ^bb4102, ^bb2700
  ^bb4102:
    pdl_interp.switch_attribute %2176 to [0.000000e+00 : f32, 1.000000e+00 : f32](^bb4103, ^bb4104) -> ^bb2700
  ^bb4103:
    %2177 = pdl_interp.get_value_type of %2174 : !pdl.type
    %2178 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2177, %2178 : !pdl.type -> ^bb4105, ^bb2700
  ^bb4105:
    pdl_interp.check_type %2177 is f32 -> ^bb4106, ^bb2700
  ^bb4106:
    %2179 = pdl_interp.get_value_type of %2175 : !pdl.type
    pdl_interp.are_equal %2177, %2179 : !pdl.type -> ^bb4107, ^bb2700
  ^bb4107:
    pdl_interp.record_match @rewriters::@div0(%0 : !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb2700
  ^bb4104:
    %2180 = pdl_interp.get_value_type of %2174 : !pdl.type
    %2181 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2180, %2181 : !pdl.type -> ^bb4108, ^bb2700
  ^bb4108:
    pdl_interp.check_type %2180 is f32 -> ^bb4109, ^bb2700
  ^bb4109:
    %2182 = pdl_interp.get_value_type of %2175 : !pdl.type
    pdl_interp.are_equal %2180, %2182 : !pdl.type -> ^bb4110, ^bb2700
  ^bb4110:
    pdl_interp.record_match @rewriters::@inv_pow(%2175, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb2700
  ^bb4067:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb4111, ^bb2700
  ^bb4111:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4112, ^bb2700
  ^bb4112:
    %2183 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2183 : !pdl.value -> ^bb4113, ^bb2700
  ^bb4113:
    pdl_interp.are_equal %2183, %2 : !pdl.value -> ^bb4114, ^bb2700
  ^bb4114:
    %2184 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2184 : !pdl.value -> ^bb4115, ^bb2700
  ^bb4115:
    %2185 = pdl_interp.get_operand 1 of %0
    pdl_interp.is_not_null %2185 : !pdl.value -> ^bb4116, ^bb2700
  ^bb4116:
    %2186 = pdl_interp.get_value_type of %2184 : !pdl.type
    %2187 = pdl_interp.get_value_type of %2183 : !pdl.type
    pdl_interp.are_equal %2186, %2187 : !pdl.type -> ^bb4117, ^bb2700
  ^bb4117:
    %2188 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2186, %2188 : !pdl.type -> ^bb4118, ^bb2700
  ^bb4118:
    pdl_interp.check_type %2186 is f32 -> ^bb4119, ^bb2700
  ^bb4119:
    %2189 = pdl_interp.get_value_type of %2185 : !pdl.type
    pdl_interp.are_equal %2186, %2189 : !pdl.type -> ^bb4120, ^bb2700
  ^bb4120:
    pdl_interp.record_match @rewriters::@distribute_frac_neg(%2184, %2185, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb2700
  ^bb4068:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb4121, ^bb2700
  ^bb4121:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4122, ^bb2700
  ^bb4122:
    %2190 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2190 : !pdl.value -> ^bb4123, ^bb2700
  ^bb4123:
    pdl_interp.are_equal %2190, %2 : !pdl.value -> ^bb4124, ^bb2700
  ^bb4124:
    %2191 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2191 : !pdl.value -> ^bb4125, ^bb2700
  ^bb4125:
    %2192 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2192 : !pdl.value -> ^bb4126, ^bb4127
  ^bb4127:
    %2193 = pdl_interp.get_operand 1 of %0
    pdl_interp.is_not_null %2193 : !pdl.value -> ^bb4128, ^bb2700
  ^bb4128:
    %2194 = pdl_interp.get_value_type of %2191 : !pdl.type
    %2195 = pdl_interp.get_value_type of %2190 : !pdl.type
    pdl_interp.are_equal %2194, %2195 : !pdl.type -> ^bb4129, ^bb2700
  ^bb4129:
    %2196 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2194, %2196 : !pdl.type -> ^bb4130, ^bb2700
  ^bb4130:
    pdl_interp.check_type %2194 is f32 -> ^bb4131, ^bb2700
  ^bb4131:
    %2197 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2197 : !pdl.value -> ^bb4132, ^bb2700
  ^bb4132:
    %2198 = pdl_interp.get_value_type of %2197 : !pdl.type
    pdl_interp.are_equal %2194, %2198 : !pdl.type -> ^bb4133, ^bb2700
  ^bb4133:
    %2199 = pdl_interp.get_value_type of %2193 : !pdl.type
    pdl_interp.are_equal %2194, %2199 : !pdl.type -> ^bb4134, ^bb2700
  ^bb4134:
    pdl_interp.record_match @rewriters::@div_add(%2191, %2193, %2197, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb2700
  ^bb4126:
    %2200 = pdl_interp.get_defining_op of %2191 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %2200 : !pdl.operation -> ^bb4135, ^bb4127
  ^bb4135:
    pdl_interp.check_operation_name of %2200 is "arith.mulf" -> ^bb4136, ^bb4127
  ^bb4136:
    pdl_interp.check_operand_count of %2200 is 2 -> ^bb4137, ^bb4127
  ^bb4137:
    pdl_interp.check_result_count of %2200 is 1 -> ^bb4138, ^bb4127
  ^bb4138:
    %2201 = pdl_interp.get_result 0 of %2200
    pdl_interp.is_not_null %2201 : !pdl.value -> ^bb4139, ^bb4127
  ^bb4139:
    pdl_interp.are_equal %2201, %2191 : !pdl.value -> ^bb4140, ^bb4127
  ^bb4140:
    %2202 = pdl_interp.get_operand 0 of %2200
    pdl_interp.is_not_null %2202 : !pdl.value -> ^bb4141, ^bb4127
  ^bb4141:
    %2203 = pdl_interp.get_value_type of %2202 : !pdl.type
    %2204 = pdl_interp.get_value_type of %2201 : !pdl.type
    pdl_interp.are_equal %2203, %2204 : !pdl.type -> ^bb4142, ^bb4127
  ^bb4142:
    %2205 = pdl_interp.get_value_type of %2190 : !pdl.type
    pdl_interp.are_equal %2203, %2205 : !pdl.type -> ^bb4143, ^bb4127
  ^bb4143:
    %2206 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2203, %2206 : !pdl.type -> ^bb4144, ^bb4127
  ^bb4144:
    pdl_interp.check_type %2203 is f32 -> ^bb4145, ^bb4127
  ^bb4145:
    %2207 = pdl_interp.get_operand 1 of %2200
    pdl_interp.is_not_null %2207 : !pdl.value -> ^bb4146, ^bb4127
  ^bb4146:
    %2208 = pdl_interp.get_value_type of %2192 : !pdl.type
    pdl_interp.are_equal %2203, %2208 : !pdl.type -> ^bb4147, ^bb4127
  ^bb4147:
    %2209 = pdl_interp.get_operand 1 of %0
    pdl_interp.are_equal %2207, %2209 : !pdl.value -> ^bb4148, ^bb4127
  ^bb4148:
    %2210 = pdl_interp.get_value_type of %2207 : !pdl.type
    pdl_interp.are_equal %2203, %2210 : !pdl.type -> ^bb4149, ^bb4127
  ^bb4149:
    pdl_interp.record_match @rewriters::@add_to_fraction_rev(%2192, %2207, %2202, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb4127
  ^bb4069:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb4150, ^bb2700
  ^bb4150:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4151, ^bb2700
  ^bb4151:
    %2211 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2211 : !pdl.value -> ^bb4152, ^bb2700
  ^bb4152:
    pdl_interp.are_equal %2211, %2 : !pdl.value -> ^bb4153, ^bb2700
  ^bb4153:
    %2212 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2212 : !pdl.value -> ^bb4154, ^bb2700
  ^bb4154:
    %2213 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2213 : !pdl.value -> ^bb4155, ^bb4156
  ^bb4156:
    %2214 = pdl_interp.get_operand 1 of %0
    pdl_interp.is_not_null %2214 : !pdl.value -> ^bb4157, ^bb2700
  ^bb4157:
    %2215 = pdl_interp.get_value_type of %2212 : !pdl.type
    %2216 = pdl_interp.get_value_type of %2211 : !pdl.type
    pdl_interp.are_equal %2215, %2216 : !pdl.type -> ^bb4158, ^bb2700
  ^bb4158:
    %2217 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2215, %2217 : !pdl.type -> ^bb4159, ^bb2700
  ^bb4159:
    pdl_interp.check_type %2215 is f32 -> ^bb4160, ^bb2700
  ^bb4160:
    %2218 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2218 : !pdl.value -> ^bb4161, ^bb2700
  ^bb4161:
    %2219 = pdl_interp.get_value_type of %2218 : !pdl.type
    pdl_interp.are_equal %2215, %2219 : !pdl.type -> ^bb4162, ^bb2700
  ^bb4162:
    %2220 = pdl_interp.get_value_type of %2214 : !pdl.type
    pdl_interp.are_equal %2215, %2220 : !pdl.type -> ^bb4163, ^bb2700
  ^bb4163:
    pdl_interp.record_match @rewriters::@div_sub(%2212, %2214, %2218, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb2700
  ^bb4155:
    %2221 = pdl_interp.get_defining_op of %2212 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %2221 : !pdl.operation -> ^bb4164, ^bb4156
  ^bb4164:
    pdl_interp.check_operation_name of %2221 is "arith.mulf" -> ^bb4165, ^bb4156
  ^bb4165:
    pdl_interp.check_operand_count of %2221 is 2 -> ^bb4166, ^bb4156
  ^bb4166:
    pdl_interp.check_result_count of %2221 is 1 -> ^bb4167, ^bb4156
  ^bb4167:
    %2222 = pdl_interp.get_result 0 of %2221
    pdl_interp.is_not_null %2222 : !pdl.value -> ^bb4168, ^bb4156
  ^bb4168:
    pdl_interp.are_equal %2222, %2212 : !pdl.value -> ^bb4169, ^bb4156
  ^bb4169:
    %2223 = pdl_interp.get_operand 0 of %2221
    pdl_interp.is_not_null %2223 : !pdl.value -> ^bb4170, ^bb4156
  ^bb4170:
    %2224 = pdl_interp.get_value_type of %2223 : !pdl.type
    %2225 = pdl_interp.get_value_type of %2222 : !pdl.type
    pdl_interp.are_equal %2224, %2225 : !pdl.type -> ^bb4171, ^bb4156
  ^bb4171:
    %2226 = pdl_interp.get_value_type of %2211 : !pdl.type
    pdl_interp.are_equal %2224, %2226 : !pdl.type -> ^bb4172, ^bb4156
  ^bb4172:
    %2227 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2224, %2227 : !pdl.type -> ^bb4173, ^bb4156
  ^bb4173:
    pdl_interp.check_type %2224 is f32 -> ^bb4174, ^bb4156
  ^bb4174:
    %2228 = pdl_interp.get_operand 1 of %2221
    pdl_interp.is_not_null %2228 : !pdl.value -> ^bb4175, ^bb4156
  ^bb4175:
    %2229 = pdl_interp.get_value_type of %2213 : !pdl.type
    pdl_interp.are_equal %2224, %2229 : !pdl.type -> ^bb4176, ^bb4156
  ^bb4176:
    %2230 = pdl_interp.get_operand 1 of %0
    pdl_interp.are_equal %2228, %2230 : !pdl.value -> ^bb4177, ^bb4156
  ^bb4177:
    %2231 = pdl_interp.get_value_type of %2228 : !pdl.type
    pdl_interp.are_equal %2224, %2231 : !pdl.type -> ^bb4178, ^bb4156
  ^bb4178:
    pdl_interp.record_match @rewriters::@sub_to_fraction_rev(%2213, %2228, %2223, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb4156
  ^bb4070:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb4179, ^bb2700
  ^bb4179:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4180, ^bb2700
  ^bb4180:
    %2232 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2232 : !pdl.value -> ^bb4181, ^bb2700
  ^bb4181:
    pdl_interp.are_equal %2232, %2 : !pdl.value -> ^bb4182, ^bb2700
  ^bb4182:
    %2233 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2233 : !pdl.value -> ^bb4183, ^bb2700
  ^bb4183:
    %2234 = pdl_interp.get_value_type of %2233 : !pdl.type
    %2235 = pdl_interp.get_value_type of %2232 : !pdl.type
    pdl_interp.are_equal %2234, %2235 : !pdl.type -> ^bb4184, ^bb2700
  ^bb4184:
    %2236 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2234, %2236 : !pdl.type -> ^bb4185, ^bb2700
  ^bb4185:
    pdl_interp.check_type %2234 is f32 -> ^bb4186, ^bb2700
  ^bb4186:
    %2237 = pdl_interp.get_operand 1 of %0
    pdl_interp.are_equal %2233, %2237 : !pdl.value -> ^bb4187, ^bb2700
  ^bb4187:
    pdl_interp.record_match @rewriters::@fabs_lhs_div(%2233, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb2700
  ^bb4071:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb4188, ^bb2700
  ^bb4188:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4189, ^bb2700
  ^bb4189:
    %2238 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2238 : !pdl.value -> ^bb4190, ^bb2700
  ^bb4190:
    pdl_interp.are_equal %2238, %2 : !pdl.value -> ^bb4191, ^bb2700
  ^bb4191:
    %2239 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2239 : !pdl.value -> ^bb4192, ^bb2700
  ^bb4192:
    %2240 = pdl_interp.get_value_type of %2239 : !pdl.type
    %2241 = pdl_interp.get_value_type of %2238 : !pdl.type
    pdl_interp.are_equal %2240, %2241 : !pdl.type -> ^bb4193, ^bb2700
  ^bb4193:
    %2242 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2240, %2242 : !pdl.type -> ^bb4194, ^bb2700
  ^bb4194:
    pdl_interp.check_type %2240 is f32 -> ^bb4195, ^bb2700
  ^bb4195:
    %2243 = pdl_interp.get_operand 1 of %0
    pdl_interp.are_equal %2239, %2243 : !pdl.value -> ^bb4196, ^bb2700
  ^bb4196:
    pdl_interp.record_match @rewriters::@fabs_cbrt_rev(%2239, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb2700
  ^bb4072:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb4197, ^bb2700
  ^bb4197:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4198, ^bb2700
  ^bb4198:
    %2244 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2244 : !pdl.value -> ^bb4199, ^bb2700
  ^bb4199:
    pdl_interp.are_equal %2244, %2 : !pdl.value -> ^bb4200, ^bb2700
  ^bb4200:
    %2245 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2245 : !pdl.value -> ^bb4201, ^bb2700
  ^bb4201:
    %2246 = pdl_interp.get_defining_op of %2245 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %2246 : !pdl.operation -> ^bb4202, ^bb2700
  ^bb4202:
    pdl_interp.check_operation_name of %2246 is "arith.subf" -> ^bb4203, ^bb2700
  ^bb4203:
    pdl_interp.check_operand_count of %2246 is 2 -> ^bb4204, ^bb2700
  ^bb4204:
    pdl_interp.check_result_count of %2246 is 1 -> ^bb4205, ^bb2700
  ^bb4205:
    %2247 = pdl_interp.get_result 0 of %2246
    pdl_interp.is_not_null %2247 : !pdl.value -> ^bb4206, ^bb2700
  ^bb4206:
    pdl_interp.are_equal %2247, %2245 : !pdl.value -> ^bb4207, ^bb2700
  ^bb4207:
    %2248 = pdl_interp.get_operand 0 of %2246
    pdl_interp.is_not_null %2248 : !pdl.value -> ^bb4208, ^bb2700
  ^bb4208:
    %2249 = pdl_interp.get_defining_op of %2248 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %2249 : !pdl.operation -> ^bb4209, ^bb2700
  ^bb4209:
    %2250 = pdl_interp.get_operand 1 of %2246
    %2251 = pdl_interp.get_defining_op of %2250 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %2251 : !pdl.operation -> ^bb4210, ^bb2700
  ^bb4210:
    pdl_interp.check_operation_name of %2249 is "arith.constant" -> ^bb4211, ^bb2700
  ^bb4211:
    pdl_interp.check_operand_count of %2249 is 0 -> ^bb4212, ^bb2700
  ^bb4212:
    pdl_interp.check_result_count of %2249 is 1 -> ^bb4213, ^bb2700
  ^bb4213:
    %2252 = pdl_interp.get_result 0 of %2249
    pdl_interp.is_not_null %2252 : !pdl.value -> ^bb4214, ^bb2700
  ^bb4214:
    pdl_interp.are_equal %2252, %2248 : !pdl.value -> ^bb4215, ^bb2700
  ^bb4215:
    pdl_interp.is_not_null %2250 : !pdl.value -> ^bb4216, ^bb2700
  ^bb4216:
    pdl_interp.check_operation_name of %2251 is "arith.mulf" -> ^bb4217, ^bb2700
  ^bb4217:
    pdl_interp.check_operand_count of %2251 is 2 -> ^bb4218, ^bb2700
  ^bb4218:
    pdl_interp.check_result_count of %2251 is 1 -> ^bb4219, ^bb2700
  ^bb4219:
    %2253 = pdl_interp.get_result 0 of %2251
    pdl_interp.is_not_null %2253 : !pdl.value -> ^bb4220, ^bb2700
  ^bb4220:
    pdl_interp.are_equal %2253, %2250 : !pdl.value -> ^bb4221, ^bb2700
  ^bb4221:
    %2254 = pdl_interp.get_operand 0 of %2251
    pdl_interp.is_not_null %2254 : !pdl.value -> ^bb4222, ^bb2700
  ^bb4222:
    %2255 = pdl_interp.get_operand 1 of %2251
    pdl_interp.are_equal %2254, %2255 : !pdl.value -> ^bb4223, ^bb2700
  ^bb4223:
    %2256 = pdl_interp.get_attribute "value" of %2249
    pdl_interp.is_not_null %2256 : !pdl.attribute -> ^bb4224, ^bb2700
  ^bb4224:
    pdl_interp.check_attribute %2256 is 1.000000e+00 : f32 -> ^bb4225, ^bb2700
  ^bb4225:
    %2257 = pdl_interp.get_value_type of %2252 : !pdl.type
    %2258 = pdl_interp.get_value_type of %2254 : !pdl.type
    pdl_interp.are_equal %2257, %2258 : !pdl.type -> ^bb4226, ^bb2700
  ^bb4226:
    %2259 = pdl_interp.get_value_type of %2253 : !pdl.type
    pdl_interp.are_equal %2257, %2259 : !pdl.type -> ^bb4227, ^bb2700
  ^bb4227:
    %2260 = pdl_interp.get_value_type of %2247 : !pdl.type
    pdl_interp.are_equal %2257, %2260 : !pdl.type -> ^bb4228, ^bb2700
  ^bb4228:
    %2261 = pdl_interp.get_value_type of %2244 : !pdl.type
    pdl_interp.are_equal %2257, %2261 : !pdl.type -> ^bb4229, ^bb2700
  ^bb4229:
    %2262 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2257, %2262 : !pdl.type -> ^bb4230, ^bb2700
  ^bb4230:
    pdl_interp.check_type %2257 is f32 -> ^bb4231, ^bb2700
  ^bb4231:
    %2263 = pdl_interp.get_operand 1 of %0
    pdl_interp.are_equal %2254, %2263 : !pdl.value -> ^bb4232, ^bb2700
  ^bb4232:
    pdl_interp.record_match @rewriters::@tan_acos_rev(%2254, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb2700
  ^bb29:
    pdl_interp.check_operand_count of %0 is 1 -> ^bb4233, ^bb1
  ^bb4233:
    pdl_interp.check_result_count of %0 is 1 -> ^bb4234, ^bb1
  ^bb4234:
    pdl_interp.is_not_null %2 : !pdl.value -> ^bb4235, ^bb1
  ^bb4235:
    pdl_interp.switch_operation_name of %3 to ["arith.negf", "arith.mulf", "arith.addf", "arith.divf", "arith.subf", "math.copysign", "math.powf", "math.cbrt", "math.log", "math.sin", "math.tan", "math.asin", "math.atan", "math.sinh"](^bb4236, ^bb4237, ^bb4238, ^bb4239, ^bb4240, ^bb4241, ^bb4242, ^bb4243, ^bb4244, ^bb4245, ^bb4246, ^bb4247, ^bb4248, ^bb4249) -> ^bb1
  ^bb4236:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb4250, ^bb1
  ^bb4250:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4251, ^bb1
  ^bb4251:
    %2264 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2264 : !pdl.value -> ^bb4252, ^bb1
  ^bb4252:
    pdl_interp.are_equal %2264, %2 : !pdl.value -> ^bb4253, ^bb1
  ^bb4253:
    %2265 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2265 : !pdl.value -> ^bb4254, ^bb1
  ^bb4254:
    %2266 = pdl_interp.get_value_type of %2265 : !pdl.type
    %2267 = pdl_interp.get_value_type of %2264 : !pdl.type
    pdl_interp.are_equal %2266, %2267 : !pdl.type -> ^bb4255, ^bb1
  ^bb4255:
    %2268 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2266, %2268 : !pdl.type -> ^bb4256, ^bb1
  ^bb4256:
    pdl_interp.check_type %2266 is f32 -> ^bb4257, ^bb1
  ^bb4257:
    pdl_interp.record_match @rewriters::@remove_double_neg(%2265, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.negf") -> ^bb1
  ^bb4237:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb4258, ^bb1
  ^bb4258:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4259, ^bb1
  ^bb4259:
    %2269 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2269 : !pdl.value -> ^bb4260, ^bb1
  ^bb4260:
    pdl_interp.are_equal %2269, %2 : !pdl.value -> ^bb4261, ^bb1
  ^bb4261:
    %2270 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2270 : !pdl.value -> ^bb4262, ^bb1
  ^bb4262:
    %2271 = pdl_interp.get_value_type of %2270 : !pdl.type
    %2272 = pdl_interp.get_value_type of %2269 : !pdl.type
    pdl_interp.are_equal %2271, %2272 : !pdl.type -> ^bb4263, ^bb1
  ^bb4263:
    %2273 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2271, %2273 : !pdl.type -> ^bb4264, ^bb1
  ^bb4264:
    pdl_interp.check_type %2271 is f32 -> ^bb4265, ^bb1
  ^bb4265:
    %2274 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2274 : !pdl.value -> ^bb4266, ^bb1
  ^bb4266:
    %2275 = pdl_interp.get_value_type of %2274 : !pdl.type
    pdl_interp.are_equal %2271, %2275 : !pdl.type -> ^bb4267, ^bb1
  ^bb4267:
    pdl_interp.record_match @rewriters::@distribute_rgt_neg_in(%2274, %2270, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.negf") -> ^bb4268
  ^bb4268:
    pdl_interp.record_match @rewriters::@distribute_lft_neg_in(%2270, %2274, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.negf") -> ^bb1
  ^bb4238:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb4269, ^bb1
  ^bb4269:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4270, ^bb1
  ^bb4270:
    %2276 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2276 : !pdl.value -> ^bb4271, ^bb1
  ^bb4271:
    pdl_interp.are_equal %2276, %2 : !pdl.value -> ^bb4272, ^bb1
  ^bb4272:
    %2277 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2277 : !pdl.value -> ^bb4273, ^bb1
  ^bb4273:
    %2278 = pdl_interp.get_value_type of %2277 : !pdl.type
    %2279 = pdl_interp.get_value_type of %2276 : !pdl.type
    pdl_interp.are_equal %2278, %2279 : !pdl.type -> ^bb4274, ^bb1
  ^bb4274:
    %2280 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2278, %2280 : !pdl.type -> ^bb4275, ^bb1
  ^bb4275:
    pdl_interp.check_type %2278 is f32 -> ^bb4276, ^bb1
  ^bb4276:
    %2281 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2281 : !pdl.value -> ^bb4277, ^bb1
  ^bb4277:
    %2282 = pdl_interp.get_value_type of %2281 : !pdl.type
    pdl_interp.are_equal %2278, %2282 : !pdl.type -> ^bb4278, ^bb1
  ^bb4278:
    pdl_interp.record_match @rewriters::@distribute_neg_in(%2277, %2281, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.negf") -> ^bb1
  ^bb4239:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb4279, ^bb1
  ^bb4279:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4280, ^bb1
  ^bb4280:
    %2283 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2283 : !pdl.value -> ^bb4281, ^bb1
  ^bb4281:
    pdl_interp.are_equal %2283, %2 : !pdl.value -> ^bb4282, ^bb1
  ^bb4282:
    %2284 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2284 : !pdl.value -> ^bb4283, ^bb1
  ^bb4283:
    %2285 = pdl_interp.get_value_type of %2284 : !pdl.type
    %2286 = pdl_interp.get_value_type of %2283 : !pdl.type
    pdl_interp.are_equal %2285, %2286 : !pdl.type -> ^bb4284, ^bb1
  ^bb4284:
    %2287 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2285, %2287 : !pdl.type -> ^bb4285, ^bb1
  ^bb4285:
    pdl_interp.check_type %2285 is f32 -> ^bb4286, ^bb1
  ^bb4286:
    %2288 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2288 : !pdl.value -> ^bb4287, ^bb1
  ^bb4287:
    %2289 = pdl_interp.get_value_type of %2288 : !pdl.type
    pdl_interp.are_equal %2285, %2289 : !pdl.type -> ^bb4288, ^bb1
  ^bb4288:
    pdl_interp.record_match @rewriters::@distribute_neg_frac2(%2288, %2284, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.negf") -> ^bb4289
  ^bb4289:
    pdl_interp.record_match @rewriters::@distribute_neg_frac(%2284, %2288, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.negf") -> ^bb1
  ^bb4240:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb4290, ^bb1
  ^bb4290:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4291, ^bb1
  ^bb4291:
    %2290 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2290 : !pdl.value -> ^bb4292, ^bb1
  ^bb4292:
    pdl_interp.are_equal %2290, %2 : !pdl.value -> ^bb4293, ^bb1
  ^bb4293:
    %2291 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2291 : !pdl.value -> ^bb4294, ^bb1
  ^bb4294:
    %2292 = pdl_interp.get_value_type of %2291 : !pdl.type
    %2293 = pdl_interp.get_value_type of %2290 : !pdl.type
    pdl_interp.are_equal %2292, %2293 : !pdl.type -> ^bb4295, ^bb1
  ^bb4295:
    %2294 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2292, %2294 : !pdl.type -> ^bb4296, ^bb1
  ^bb4296:
    pdl_interp.check_type %2292 is f32 -> ^bb4297, ^bb1
  ^bb4297:
    %2295 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2295 : !pdl.value -> ^bb4298, ^bb1
  ^bb4298:
    %2296 = pdl_interp.get_value_type of %2295 : !pdl.type
    pdl_interp.are_equal %2292, %2296 : !pdl.type -> ^bb4299, ^bb1
  ^bb4299:
    pdl_interp.record_match @rewriters::@sub_negate(%2295, %2291, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.negf") -> ^bb1
  ^bb4241:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb4300, ^bb1
  ^bb4300:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4301, ^bb1
  ^bb4301:
    %2297 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2297 : !pdl.value -> ^bb4302, ^bb1
  ^bb4302:
    pdl_interp.are_equal %2297, %2 : !pdl.value -> ^bb4303, ^bb1
  ^bb4303:
    %2298 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2298 : !pdl.value -> ^bb4304, ^bb1
  ^bb4304:
    %2299 = pdl_interp.get_value_type of %2298 : !pdl.type
    %2300 = pdl_interp.get_value_type of %2297 : !pdl.type
    pdl_interp.are_equal %2299, %2300 : !pdl.type -> ^bb4305, ^bb1
  ^bb4305:
    %2301 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2299, %2301 : !pdl.type -> ^bb4306, ^bb1
  ^bb4306:
    pdl_interp.check_type %2299 is f32 -> ^bb4307, ^bb1
  ^bb4307:
    %2302 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2302 : !pdl.value -> ^bb4308, ^bb1
  ^bb4308:
    %2303 = pdl_interp.get_value_type of %2302 : !pdl.type
    pdl_interp.are_equal %2299, %2303 : !pdl.type -> ^bb4309, ^bb1
  ^bb4309:
    pdl_interp.record_match @rewriters::@neg_copysign(%2302, %2298, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.negf") -> ^bb1
  ^bb4242:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb4310, ^bb1
  ^bb4310:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4311, ^bb1
  ^bb4311:
    %2304 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2304 : !pdl.value -> ^bb4312, ^bb1
  ^bb4312:
    pdl_interp.are_equal %2304, %2 : !pdl.value -> ^bb4313, ^bb1
  ^bb4313:
    %2305 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2305 : !pdl.value -> ^bb4314, ^bb1
  ^bb4314:
    %2306 = pdl_interp.get_value_type of %2305 : !pdl.type
    %2307 = pdl_interp.get_value_type of %2304 : !pdl.type
    pdl_interp.are_equal %2306, %2307 : !pdl.type -> ^bb4315, ^bb1
  ^bb4315:
    %2308 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2306, %2308 : !pdl.type -> ^bb4316, ^bb1
  ^bb4316:
    pdl_interp.check_type %2306 is f32 -> ^bb4317, ^bb1
  ^bb4317:
    %2309 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2309 : !pdl.value -> ^bb4318, ^bb1
  ^bb4318:
    %2310 = pdl_interp.get_defining_op of %2309 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %2310 : !pdl.operation -> ^bb4319, ^bb1
  ^bb4319:
    pdl_interp.check_operation_name of %2310 is "arith.constant" -> ^bb4320, ^bb1
  ^bb4320:
    pdl_interp.check_operand_count of %2310 is 0 -> ^bb4321, ^bb1
  ^bb4321:
    pdl_interp.check_result_count of %2310 is 1 -> ^bb4322, ^bb1
  ^bb4322:
    %2311 = pdl_interp.get_result 0 of %2310
    pdl_interp.is_not_null %2311 : !pdl.value -> ^bb4323, ^bb1
  ^bb4323:
    pdl_interp.are_equal %2311, %2309 : !pdl.value -> ^bb4324, ^bb1
  ^bb4324:
    %2312 = pdl_interp.get_attribute "value" of %2310
    pdl_interp.is_not_null %2312 : !pdl.attribute -> ^bb4325, ^bb1
  ^bb4325:
    pdl_interp.check_attribute %2312 is 3.000000e+00 : f32 -> ^bb4326, ^bb1
  ^bb4326:
    %2313 = pdl_interp.get_value_type of %2311 : !pdl.type
    pdl_interp.are_equal %2313, %2306 : !pdl.type -> ^bb4327, ^bb1
  ^bb4327:
    pdl_interp.record_match @rewriters::@cube_neg_rev(%2305, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.negf") -> ^bb1
  ^bb4243:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb4328, ^bb1
  ^bb4328:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4329, ^bb1
  ^bb4329:
    %2314 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2314 : !pdl.value -> ^bb4330, ^bb1
  ^bb4330:
    pdl_interp.are_equal %2314, %2 : !pdl.value -> ^bb4331, ^bb1
  ^bb4331:
    %2315 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2315 : !pdl.value -> ^bb4332, ^bb1
  ^bb4332:
    %2316 = pdl_interp.get_value_type of %2315 : !pdl.type
    %2317 = pdl_interp.get_value_type of %2314 : !pdl.type
    pdl_interp.are_equal %2316, %2317 : !pdl.type -> ^bb4333, ^bb1
  ^bb4333:
    %2318 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2316, %2318 : !pdl.type -> ^bb4334, ^bb1
  ^bb4334:
    pdl_interp.check_type %2316 is f32 -> ^bb4335, ^bb1
  ^bb4335:
    pdl_interp.record_match @rewriters::@cbrt_neg_rev(%2315, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.negf") -> ^bb1
  ^bb4244:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb4336, ^bb1
  ^bb4336:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4337, ^bb1
  ^bb4337:
    %2319 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2319 : !pdl.value -> ^bb4338, ^bb1
  ^bb4338:
    pdl_interp.are_equal %2319, %2 : !pdl.value -> ^bb4339, ^bb1
  ^bb4339:
    %2320 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2320 : !pdl.value -> ^bb4340, ^bb1
  ^bb4340:
    %2321 = pdl_interp.get_value_type of %2320 : !pdl.type
    %2322 = pdl_interp.get_value_type of %2319 : !pdl.type
    pdl_interp.are_equal %2321, %2322 : !pdl.type -> ^bb4341, ^bb1
  ^bb4341:
    %2323 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2321, %2323 : !pdl.type -> ^bb4342, ^bb1
  ^bb4342:
    pdl_interp.check_type %2321 is f32 -> ^bb4343, ^bb1
  ^bb4343:
    pdl_interp.record_match @rewriters::@neg_log(%2320, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.negf") -> ^bb1
  ^bb4245:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb4344, ^bb1
  ^bb4344:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4345, ^bb1
  ^bb4345:
    %2324 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2324 : !pdl.value -> ^bb4346, ^bb1
  ^bb4346:
    pdl_interp.are_equal %2324, %2 : !pdl.value -> ^bb4347, ^bb1
  ^bb4347:
    %2325 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2325 : !pdl.value -> ^bb4348, ^bb1
  ^bb4348:
    %2326 = pdl_interp.get_value_type of %2325 : !pdl.type
    %2327 = pdl_interp.get_value_type of %2324 : !pdl.type
    pdl_interp.are_equal %2326, %2327 : !pdl.type -> ^bb4349, ^bb1
  ^bb4349:
    %2328 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2326, %2328 : !pdl.type -> ^bb4350, ^bb1
  ^bb4350:
    pdl_interp.check_type %2326 is f32 -> ^bb4351, ^bb1
  ^bb4351:
    pdl_interp.record_match @rewriters::@sin_neg_rev(%2325, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.negf") -> ^bb1
  ^bb4246:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb4352, ^bb1
  ^bb4352:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4353, ^bb1
  ^bb4353:
    %2329 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2329 : !pdl.value -> ^bb4354, ^bb1
  ^bb4354:
    pdl_interp.are_equal %2329, %2 : !pdl.value -> ^bb4355, ^bb1
  ^bb4355:
    %2330 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2330 : !pdl.value -> ^bb4356, ^bb1
  ^bb4356:
    %2331 = pdl_interp.get_value_type of %2330 : !pdl.type
    %2332 = pdl_interp.get_value_type of %2329 : !pdl.type
    pdl_interp.are_equal %2331, %2332 : !pdl.type -> ^bb4357, ^bb1
  ^bb4357:
    %2333 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2331, %2333 : !pdl.type -> ^bb4358, ^bb1
  ^bb4358:
    pdl_interp.check_type %2331 is f32 -> ^bb4359, ^bb1
  ^bb4359:
    pdl_interp.record_match @rewriters::@tan_neg_rev(%2330, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.negf") -> ^bb1
  ^bb4247:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb4360, ^bb1
  ^bb4360:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4361, ^bb1
  ^bb4361:
    %2334 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2334 : !pdl.value -> ^bb4362, ^bb1
  ^bb4362:
    pdl_interp.are_equal %2334, %2 : !pdl.value -> ^bb4363, ^bb1
  ^bb4363:
    %2335 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2335 : !pdl.value -> ^bb4364, ^bb1
  ^bb4364:
    %2336 = pdl_interp.get_value_type of %2335 : !pdl.type
    %2337 = pdl_interp.get_value_type of %2334 : !pdl.type
    pdl_interp.are_equal %2336, %2337 : !pdl.type -> ^bb4365, ^bb1
  ^bb4365:
    %2338 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2336, %2338 : !pdl.type -> ^bb4366, ^bb1
  ^bb4366:
    pdl_interp.check_type %2336 is f32 -> ^bb4367, ^bb1
  ^bb4367:
    pdl_interp.record_match @rewriters::@asin_neg_rev(%2335, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.negf") -> ^bb1
  ^bb4248:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb4368, ^bb1
  ^bb4368:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4369, ^bb1
  ^bb4369:
    %2339 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2339 : !pdl.value -> ^bb4370, ^bb1
  ^bb4370:
    pdl_interp.are_equal %2339, %2 : !pdl.value -> ^bb4371, ^bb1
  ^bb4371:
    %2340 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2340 : !pdl.value -> ^bb4372, ^bb1
  ^bb4372:
    %2341 = pdl_interp.get_value_type of %2340 : !pdl.type
    %2342 = pdl_interp.get_value_type of %2339 : !pdl.type
    pdl_interp.are_equal %2341, %2342 : !pdl.type -> ^bb4373, ^bb1
  ^bb4373:
    %2343 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2341, %2343 : !pdl.type -> ^bb4374, ^bb1
  ^bb4374:
    pdl_interp.check_type %2341 is f32 -> ^bb4375, ^bb1
  ^bb4375:
    pdl_interp.record_match @rewriters::@atan_neg_rev(%2340, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.negf") -> ^bb1
  ^bb4249:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb4376, ^bb1
  ^bb4376:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4377, ^bb1
  ^bb4377:
    %2344 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2344 : !pdl.value -> ^bb4378, ^bb1
  ^bb4378:
    pdl_interp.are_equal %2344, %2 : !pdl.value -> ^bb4379, ^bb1
  ^bb4379:
    %2345 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2345 : !pdl.value -> ^bb4380, ^bb1
  ^bb4380:
    %2346 = pdl_interp.get_value_type of %2345 : !pdl.type
    %2347 = pdl_interp.get_value_type of %2344 : !pdl.type
    pdl_interp.are_equal %2346, %2347 : !pdl.type -> ^bb4381, ^bb1
  ^bb4381:
    %2348 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2346, %2348 : !pdl.type -> ^bb4382, ^bb1
  ^bb4382:
    pdl_interp.check_type %2346 is f32 -> ^bb4383, ^bb1
  ^bb4383:
    pdl_interp.record_match @rewriters::@sinh_neg_rev(%2345, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.negf") -> ^bb1
  ^bb30:
    pdl_interp.check_operand_count of %0 is 2 -> ^bb4384, ^bb1
  ^bb4384:
    pdl_interp.check_result_count of %0 is 1 -> ^bb4385, ^bb1
  ^bb4385:
    %2349 = pdl_interp.get_operand 1 of %0
    %2350 = pdl_interp.get_defining_op of %2349 : !pdl.value {position = "root.operand[1].defining_op"}
    pdl_interp.is_not_null %2350 : !pdl.operation -> ^bb4386, ^bb4387
  ^bb4387:
    pdl_interp.is_not_null %2 : !pdl.value -> ^bb4388, ^bb1
  ^bb4388:
    pdl_interp.switch_operation_name of %3 to ["math.sqrt", "math.cbrt", "arith.constant", "math.exp"](^bb4389, ^bb4390, ^bb4391, ^bb4392) -> ^bb1
  ^bb4389:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb4393, ^bb1
  ^bb4393:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4394, ^bb1
  ^bb4394:
    %2351 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2351 : !pdl.value -> ^bb4395, ^bb1
  ^bb4395:
    pdl_interp.are_equal %2351, %2 : !pdl.value -> ^bb4396, ^bb1
  ^bb4396:
    %2352 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2352 : !pdl.value -> ^bb4397, ^bb1
  ^bb4397:
    %2353 = pdl_interp.get_operand 1 of %0
    pdl_interp.is_not_null %2353 : !pdl.value -> ^bb4398, ^bb1
  ^bb4398:
    %2354 = pdl_interp.get_value_type of %2352 : !pdl.type
    %2355 = pdl_interp.get_value_type of %2351 : !pdl.type
    pdl_interp.are_equal %2354, %2355 : !pdl.type -> ^bb4399, ^bb1
  ^bb4399:
    %2356 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2354, %2356 : !pdl.type -> ^bb4400, ^bb1
  ^bb4400:
    pdl_interp.check_type %2354 is f32 -> ^bb4401, ^bb1
  ^bb4401:
    %2357 = pdl_interp.get_value_type of %2353 : !pdl.type
    pdl_interp.are_equal %2354, %2357 : !pdl.type -> ^bb4402, ^bb1
  ^bb4402:
    pdl_interp.record_match @rewriters::@sqrt_pow2(%2353, %2352, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.powf") -> ^bb1
  ^bb4390:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb4403, ^bb1
  ^bb4403:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4404, ^bb1
  ^bb4404:
    %2358 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2358 : !pdl.value -> ^bb4405, ^bb1
  ^bb4405:
    pdl_interp.are_equal %2358, %2 : !pdl.value -> ^bb4406, ^bb1
  ^bb4406:
    %2359 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2359 : !pdl.value -> ^bb4407, ^bb1
  ^bb4407:
    %2360 = pdl_interp.get_operand 1 of %0
    pdl_interp.is_not_null %2360 : !pdl.value -> ^bb4408, ^bb1
  ^bb4408:
    %2361 = pdl_interp.get_value_type of %2359 : !pdl.type
    %2362 = pdl_interp.get_value_type of %2358 : !pdl.type
    pdl_interp.are_equal %2361, %2362 : !pdl.type -> ^bb4409, ^bb1
  ^bb4409:
    %2363 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2361, %2363 : !pdl.type -> ^bb4410, ^bb1
  ^bb4410:
    pdl_interp.check_type %2361 is f32 -> ^bb4411, ^bb1
  ^bb4411:
    %2364 = pdl_interp.get_value_type of %2360 : !pdl.type
    pdl_interp.are_equal %2361, %2364 : !pdl.type -> ^bb4412, ^bb1
  ^bb4412:
    pdl_interp.record_match @rewriters::@pow_cbrt(%2360, %2359, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.powf") -> ^bb1
  ^bb4391:
    pdl_interp.check_operand_count of %3 is 0 -> ^bb4413, ^bb1
  ^bb4413:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4414, ^bb1
  ^bb4414:
    %2365 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2365 : !pdl.value -> ^bb4415, ^bb1
  ^bb4415:
    pdl_interp.are_equal %2365, %2 : !pdl.value -> ^bb4416, ^bb1
  ^bb4416:
    %2366 = pdl_interp.get_operand 1 of %0
    pdl_interp.is_not_null %2366 : !pdl.value -> ^bb4417, ^bb1
  ^bb4417:
    %2367 = pdl_interp.get_attribute "value" of %3
    pdl_interp.is_not_null %2367 : !pdl.attribute -> ^bb4418, ^bb1
  ^bb4418:
    pdl_interp.switch_attribute %2367 to [1.000000e+00 : f32, 0.000000e+00 : f32](^bb4419, ^bb4420) -> ^bb1
  ^bb4419:
    %2368 = pdl_interp.get_value_type of %2365 : !pdl.type
    %2369 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2368, %2369 : !pdl.type -> ^bb4421, ^bb1
  ^bb4421:
    pdl_interp.check_type %2368 is f32 -> ^bb4422, ^bb1
  ^bb4422:
    %2370 = pdl_interp.get_value_type of %2366 : !pdl.type
    pdl_interp.are_equal %2368, %2370 : !pdl.type -> ^bb4423, ^bb1
  ^bb4423:
    pdl_interp.record_match @rewriters::@pow_base_1(%0 : !pdl.operation) : benefit(1), loc([]), root("math.powf") -> ^bb1
  ^bb4420:
    %2371 = pdl_interp.get_value_type of %2365 : !pdl.type
    %2372 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2371, %2372 : !pdl.type -> ^bb4424, ^bb1
  ^bb4424:
    pdl_interp.check_type %2371 is f32 -> ^bb4425, ^bb1
  ^bb4425:
    %2373 = pdl_interp.get_value_type of %2366 : !pdl.type
    pdl_interp.are_equal %2371, %2373 : !pdl.type -> ^bb4426, ^bb1
  ^bb4426:
    pdl_interp.record_match @rewriters::@pow_base_0(%0 : !pdl.operation) : benefit(1), loc([]), root("math.powf") -> ^bb1
  ^bb4392:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb4427, ^bb1
  ^bb4427:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4428, ^bb1
  ^bb4428:
    %2374 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2374 : !pdl.value -> ^bb4429, ^bb1
  ^bb4429:
    pdl_interp.are_equal %2374, %2 : !pdl.value -> ^bb4430, ^bb1
  ^bb4430:
    %2375 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2375 : !pdl.value -> ^bb4431, ^bb1
  ^bb4431:
    %2376 = pdl_interp.get_operand 1 of %0
    pdl_interp.is_not_null %2376 : !pdl.value -> ^bb4432, ^bb1
  ^bb4432:
    %2377 = pdl_interp.get_value_type of %2375 : !pdl.type
    %2378 = pdl_interp.get_value_type of %2374 : !pdl.type
    pdl_interp.are_equal %2377, %2378 : !pdl.type -> ^bb4433, ^bb1
  ^bb4433:
    %2379 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2377, %2379 : !pdl.type -> ^bb4434, ^bb1
  ^bb4434:
    pdl_interp.check_type %2377 is f32 -> ^bb4435, ^bb1
  ^bb4435:
    %2380 = pdl_interp.get_value_type of %2376 : !pdl.type
    pdl_interp.are_equal %2377, %2380 : !pdl.type -> ^bb4436, ^bb1
  ^bb4436:
    pdl_interp.record_match @rewriters::@pow_exp(%2375, %2376, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.powf") -> ^bb1
  ^bb4386:
    pdl_interp.is_not_null %2 : !pdl.value -> ^bb4437, ^bb4387
  ^bb4437:
    pdl_interp.switch_operation_name of %3 to ["arith.addf", "arith.subf", "math.cbrt", "arith.negf", "arith.mulf", "arith.divf", "math.exp"](^bb4438, ^bb4439, ^bb4440, ^bb4441, ^bb4442, ^bb4443, ^bb4444) -> ^bb4387
  ^bb4438:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb4445, ^bb4387
  ^bb4445:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4446, ^bb4387
  ^bb4446:
    %2381 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2381 : !pdl.value -> ^bb4447, ^bb4387
  ^bb4447:
    pdl_interp.are_equal %2381, %2 : !pdl.value -> ^bb4448, ^bb4387
  ^bb4448:
    %2382 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2382 : !pdl.value -> ^bb4449, ^bb4387
  ^bb4449:
    pdl_interp.is_not_null %2349 : !pdl.value -> ^bb4450, ^bb4387
  ^bb4450:
    %2383 = pdl_interp.get_value_type of %2382 : !pdl.type
    %2384 = pdl_interp.get_value_type of %2381 : !pdl.type
    pdl_interp.are_equal %2383, %2384 : !pdl.type -> ^bb4451, ^bb4387
  ^bb4451:
    %2385 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2383, %2385 : !pdl.type -> ^bb4452, ^bb4387
  ^bb4452:
    pdl_interp.check_type %2383 is f32 -> ^bb4453, ^bb4387
  ^bb4453:
    pdl_interp.check_operation_name of %2350 is "arith.constant" -> ^bb4454, ^bb4387
  ^bb4454:
    pdl_interp.check_operand_count of %2350 is 0 -> ^bb4455, ^bb4387
  ^bb4455:
    pdl_interp.check_result_count of %2350 is 1 -> ^bb4456, ^bb4387
  ^bb4456:
    %2386 = pdl_interp.get_result 0 of %2350
    pdl_interp.is_not_null %2386 : !pdl.value -> ^bb4457, ^bb4387
  ^bb4457:
    pdl_interp.are_equal %2386, %2349 : !pdl.value -> ^bb4458, ^bb4387
  ^bb4458:
    %2387 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2387 : !pdl.value -> ^bb4459, ^bb4387
  ^bb4459:
    %2388 = pdl_interp.get_value_type of %2386 : !pdl.type
    pdl_interp.are_equal %2383, %2388 : !pdl.type -> ^bb4460, ^bb4387
  ^bb4460:
    %2389 = pdl_interp.get_value_type of %2387 : !pdl.type
    pdl_interp.are_equal %2383, %2389 : !pdl.type -> ^bb4461, ^bb4387
  ^bb4461:
    %2390 = pdl_interp.get_attribute "value" of %2350
    pdl_interp.is_not_null %2390 : !pdl.attribute -> ^bb4462, ^bb4387
  ^bb4462:
    pdl_interp.check_attribute %2390 is 2.000000e+00 : f32 -> ^bb4463, ^bb4387
  ^bb4463:
    pdl_interp.record_match @rewriters::@sum_square_pow(%2382, %2387, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.powf") -> ^bb4387
  ^bb4439:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb4464, ^bb4387
  ^bb4464:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4465, ^bb4387
  ^bb4465:
    %2391 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2391 : !pdl.value -> ^bb4466, ^bb4387
  ^bb4466:
    pdl_interp.are_equal %2391, %2 : !pdl.value -> ^bb4467, ^bb4387
  ^bb4467:
    %2392 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2392 : !pdl.value -> ^bb4468, ^bb4387
  ^bb4468:
    pdl_interp.is_not_null %2349 : !pdl.value -> ^bb4469, ^bb4387
  ^bb4469:
    %2393 = pdl_interp.get_value_type of %2392 : !pdl.type
    %2394 = pdl_interp.get_value_type of %2391 : !pdl.type
    pdl_interp.are_equal %2393, %2394 : !pdl.type -> ^bb4470, ^bb4387
  ^bb4470:
    %2395 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2393, %2395 : !pdl.type -> ^bb4471, ^bb4387
  ^bb4471:
    pdl_interp.check_type %2393 is f32 -> ^bb4472, ^bb4387
  ^bb4472:
    pdl_interp.check_operation_name of %2350 is "arith.constant" -> ^bb4473, ^bb4387
  ^bb4473:
    pdl_interp.check_operand_count of %2350 is 0 -> ^bb4474, ^bb4387
  ^bb4474:
    pdl_interp.check_result_count of %2350 is 1 -> ^bb4475, ^bb4387
  ^bb4475:
    %2396 = pdl_interp.get_result 0 of %2350
    pdl_interp.is_not_null %2396 : !pdl.value -> ^bb4476, ^bb4387
  ^bb4476:
    pdl_interp.are_equal %2396, %2349 : !pdl.value -> ^bb4477, ^bb4387
  ^bb4477:
    %2397 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2397 : !pdl.value -> ^bb4478, ^bb4387
  ^bb4478:
    %2398 = pdl_interp.get_value_type of %2396 : !pdl.type
    pdl_interp.are_equal %2393, %2398 : !pdl.type -> ^bb4479, ^bb4387
  ^bb4479:
    %2399 = pdl_interp.get_value_type of %2397 : !pdl.type
    pdl_interp.are_equal %2393, %2399 : !pdl.type -> ^bb4480, ^bb4387
  ^bb4480:
    %2400 = pdl_interp.get_attribute "value" of %2350
    pdl_interp.is_not_null %2400 : !pdl.attribute -> ^bb4481, ^bb4387
  ^bb4481:
    pdl_interp.check_attribute %2400 is 2.000000e+00 : f32 -> ^bb4482, ^bb4387
  ^bb4482:
    pdl_interp.record_match @rewriters::@sub_square_pow(%2392, %2397, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.powf") -> ^bb4387
  ^bb4440:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb4483, ^bb4387
  ^bb4483:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4484, ^bb4387
  ^bb4484:
    %2401 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2401 : !pdl.value -> ^bb4485, ^bb4387
  ^bb4485:
    pdl_interp.are_equal %2401, %2 : !pdl.value -> ^bb4486, ^bb4387
  ^bb4486:
    %2402 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2402 : !pdl.value -> ^bb4487, ^bb4387
  ^bb4487:
    pdl_interp.is_not_null %2349 : !pdl.value -> ^bb4488, ^bb4387
  ^bb4488:
    %2403 = pdl_interp.get_value_type of %2402 : !pdl.type
    %2404 = pdl_interp.get_value_type of %2401 : !pdl.type
    pdl_interp.are_equal %2403, %2404 : !pdl.type -> ^bb4489, ^bb4387
  ^bb4489:
    %2405 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2403, %2405 : !pdl.type -> ^bb4490, ^bb4387
  ^bb4490:
    pdl_interp.check_type %2403 is f32 -> ^bb4491, ^bb4387
  ^bb4491:
    pdl_interp.check_operation_name of %2350 is "arith.constant" -> ^bb4492, ^bb4387
  ^bb4492:
    pdl_interp.check_operand_count of %2350 is 0 -> ^bb4493, ^bb4387
  ^bb4493:
    pdl_interp.check_result_count of %2350 is 1 -> ^bb4494, ^bb4387
  ^bb4494:
    %2406 = pdl_interp.get_result 0 of %2350
    pdl_interp.is_not_null %2406 : !pdl.value -> ^bb4495, ^bb4387
  ^bb4495:
    pdl_interp.are_equal %2406, %2349 : !pdl.value -> ^bb4496, ^bb4387
  ^bb4496:
    %2407 = pdl_interp.get_value_type of %2406 : !pdl.type
    pdl_interp.are_equal %2403, %2407 : !pdl.type -> ^bb4497, ^bb4387
  ^bb4497:
    %2408 = pdl_interp.get_attribute "value" of %2350
    pdl_interp.is_not_null %2408 : !pdl.attribute -> ^bb4498, ^bb4387
  ^bb4498:
    pdl_interp.check_attribute %2408 is 3.000000e+00 : f32 -> ^bb4499, ^bb4387
  ^bb4499:
    pdl_interp.record_match @rewriters::@rem_cube_cbrt(%2402, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.powf") -> ^bb4387
  ^bb4441:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb4500, ^bb4387
  ^bb4500:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4501, ^bb4387
  ^bb4501:
    %2409 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2409 : !pdl.value -> ^bb4502, ^bb4387
  ^bb4502:
    pdl_interp.are_equal %2409, %2 : !pdl.value -> ^bb4503, ^bb4387
  ^bb4503:
    %2410 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2410 : !pdl.value -> ^bb4504, ^bb4387
  ^bb4504:
    pdl_interp.is_not_null %2349 : !pdl.value -> ^bb4505, ^bb4387
  ^bb4505:
    %2411 = pdl_interp.get_value_type of %2410 : !pdl.type
    %2412 = pdl_interp.get_value_type of %2409 : !pdl.type
    pdl_interp.are_equal %2411, %2412 : !pdl.type -> ^bb4506, ^bb4387
  ^bb4506:
    %2413 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2411, %2413 : !pdl.type -> ^bb4507, ^bb4387
  ^bb4507:
    pdl_interp.check_type %2411 is f32 -> ^bb4508, ^bb4387
  ^bb4508:
    pdl_interp.check_operation_name of %2350 is "arith.constant" -> ^bb4509, ^bb4387
  ^bb4509:
    pdl_interp.check_operand_count of %2350 is 0 -> ^bb4510, ^bb4387
  ^bb4510:
    pdl_interp.check_result_count of %2350 is 1 -> ^bb4511, ^bb4387
  ^bb4511:
    %2414 = pdl_interp.get_result 0 of %2350
    pdl_interp.is_not_null %2414 : !pdl.value -> ^bb4512, ^bb4387
  ^bb4512:
    pdl_interp.are_equal %2414, %2349 : !pdl.value -> ^bb4513, ^bb4387
  ^bb4513:
    %2415 = pdl_interp.get_value_type of %2414 : !pdl.type
    pdl_interp.are_equal %2411, %2415 : !pdl.type -> ^bb4514, ^bb4387
  ^bb4514:
    %2416 = pdl_interp.get_attribute "value" of %2350
    pdl_interp.is_not_null %2416 : !pdl.attribute -> ^bb4515, ^bb4387
  ^bb4515:
    pdl_interp.check_attribute %2416 is 3.000000e+00 : f32 -> ^bb4516, ^bb4387
  ^bb4516:
    pdl_interp.record_match @rewriters::@cube_neg(%2410, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.powf") -> ^bb4387
  ^bb4442:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb4517, ^bb4387
  ^bb4517:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4518, ^bb4387
  ^bb4518:
    %2417 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2417 : !pdl.value -> ^bb4519, ^bb4387
  ^bb4519:
    pdl_interp.are_equal %2417, %2 : !pdl.value -> ^bb4520, ^bb4387
  ^bb4520:
    %2418 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2418 : !pdl.value -> ^bb4521, ^bb4387
  ^bb4521:
    pdl_interp.is_not_null %2349 : !pdl.value -> ^bb4522, ^bb4387
  ^bb4522:
    %2419 = pdl_interp.get_value_type of %2418 : !pdl.type
    %2420 = pdl_interp.get_value_type of %2417 : !pdl.type
    pdl_interp.are_equal %2419, %2420 : !pdl.type -> ^bb4523, ^bb4387
  ^bb4523:
    %2421 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2419, %2421 : !pdl.type -> ^bb4524, ^bb4387
  ^bb4524:
    pdl_interp.check_type %2419 is f32 -> ^bb4525, ^bb4387
  ^bb4525:
    pdl_interp.check_operation_name of %2350 is "arith.constant" -> ^bb4526, ^bb4387
  ^bb4526:
    pdl_interp.check_operand_count of %2350 is 0 -> ^bb4527, ^bb4387
  ^bb4527:
    pdl_interp.check_result_count of %2350 is 1 -> ^bb4528, ^bb4387
  ^bb4528:
    %2422 = pdl_interp.get_result 0 of %2350
    pdl_interp.is_not_null %2422 : !pdl.value -> ^bb4529, ^bb4387
  ^bb4529:
    pdl_interp.are_equal %2422, %2349 : !pdl.value -> ^bb4530, ^bb4387
  ^bb4530:
    %2423 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2423 : !pdl.value -> ^bb4531, ^bb4387
  ^bb4531:
    %2424 = pdl_interp.get_value_type of %2422 : !pdl.type
    pdl_interp.are_equal %2419, %2424 : !pdl.type -> ^bb4532, ^bb4387
  ^bb4532:
    %2425 = pdl_interp.get_value_type of %2423 : !pdl.type
    pdl_interp.are_equal %2419, %2425 : !pdl.type -> ^bb4533, ^bb4387
  ^bb4533:
    %2426 = pdl_interp.get_attribute "value" of %2350
    pdl_interp.is_not_null %2426 : !pdl.attribute -> ^bb4534, ^bb4387
  ^bb4534:
    pdl_interp.check_attribute %2426 is 3.000000e+00 : f32 -> ^bb4535, ^bb4387
  ^bb4535:
    pdl_interp.record_match @rewriters::@cube_prod(%2418, %2423, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.powf") -> ^bb4387
  ^bb4443:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb4536, ^bb4387
  ^bb4536:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4537, ^bb4387
  ^bb4537:
    %2427 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2427 : !pdl.value -> ^bb4538, ^bb4387
  ^bb4538:
    pdl_interp.are_equal %2427, %2 : !pdl.value -> ^bb4539, ^bb4387
  ^bb4539:
    %2428 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2428 : !pdl.value -> ^bb4540, ^bb4387
  ^bb4540:
    pdl_interp.is_not_null %2349 : !pdl.value -> ^bb4541, ^bb4387
  ^bb4541:
    %2429 = pdl_interp.get_value_type of %2428 : !pdl.type
    %2430 = pdl_interp.get_value_type of %2427 : !pdl.type
    pdl_interp.are_equal %2429, %2430 : !pdl.type -> ^bb4542, ^bb4387
  ^bb4542:
    %2431 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2429, %2431 : !pdl.type -> ^bb4543, ^bb4387
  ^bb4543:
    pdl_interp.check_type %2429 is f32 -> ^bb4544, ^bb4387
  ^bb4544:
    pdl_interp.check_operation_name of %2350 is "arith.constant" -> ^bb4545, ^bb4387
  ^bb4545:
    pdl_interp.check_operand_count of %2350 is 0 -> ^bb4546, ^bb4387
  ^bb4546:
    pdl_interp.check_result_count of %2350 is 1 -> ^bb4547, ^bb4387
  ^bb4547:
    %2432 = pdl_interp.get_result 0 of %2350
    pdl_interp.is_not_null %2432 : !pdl.value -> ^bb4548, ^bb4387
  ^bb4548:
    pdl_interp.are_equal %2432, %2349 : !pdl.value -> ^bb4549, ^bb4387
  ^bb4549:
    %2433 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2433 : !pdl.value -> ^bb4550, ^bb4387
  ^bb4550:
    %2434 = pdl_interp.get_value_type of %2432 : !pdl.type
    pdl_interp.are_equal %2429, %2434 : !pdl.type -> ^bb4551, ^bb4387
  ^bb4551:
    %2435 = pdl_interp.get_value_type of %2433 : !pdl.type
    pdl_interp.are_equal %2429, %2435 : !pdl.type -> ^bb4552, ^bb4387
  ^bb4552:
    %2436 = pdl_interp.get_attribute "value" of %2350
    pdl_interp.is_not_null %2436 : !pdl.attribute -> ^bb4553, ^bb4387
  ^bb4553:
    pdl_interp.check_attribute %2436 is 3.000000e+00 : f32 -> ^bb4554, ^bb4387
  ^bb4554:
    pdl_interp.record_match @rewriters::@cube_div(%2428, %2433, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.powf") -> ^bb4387
  ^bb4444:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb4555, ^bb4387
  ^bb4555:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4556, ^bb4387
  ^bb4556:
    %2437 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2437 : !pdl.value -> ^bb4557, ^bb4387
  ^bb4557:
    pdl_interp.are_equal %2437, %2 : !pdl.value -> ^bb4558, ^bb4387
  ^bb4558:
    %2438 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2438 : !pdl.value -> ^bb4559, ^bb4387
  ^bb4559:
    pdl_interp.is_not_null %2349 : !pdl.value -> ^bb4560, ^bb4387
  ^bb4560:
    %2439 = pdl_interp.get_value_type of %2438 : !pdl.type
    %2440 = pdl_interp.get_value_type of %2437 : !pdl.type
    pdl_interp.are_equal %2439, %2440 : !pdl.type -> ^bb4561, ^bb4387
  ^bb4561:
    %2441 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2439, %2441 : !pdl.type -> ^bb4562, ^bb4387
  ^bb4562:
    pdl_interp.check_type %2439 is f32 -> ^bb4563, ^bb4387
  ^bb4563:
    pdl_interp.check_operation_name of %2350 is "arith.constant" -> ^bb4564, ^bb4387
  ^bb4564:
    pdl_interp.check_operand_count of %2350 is 0 -> ^bb4565, ^bb4387
  ^bb4565:
    pdl_interp.check_result_count of %2350 is 1 -> ^bb4566, ^bb4387
  ^bb4566:
    %2442 = pdl_interp.get_result 0 of %2350
    pdl_interp.is_not_null %2442 : !pdl.value -> ^bb4567, ^bb4387
  ^bb4567:
    pdl_interp.are_equal %2442, %2349 : !pdl.value -> ^bb4568, ^bb4387
  ^bb4568:
    %2443 = pdl_interp.get_value_type of %2442 : !pdl.type
    pdl_interp.are_equal %2439, %2443 : !pdl.type -> ^bb4569, ^bb4387
  ^bb4569:
    %2444 = pdl_interp.get_attribute "value" of %2350
    pdl_interp.is_not_null %2444 : !pdl.attribute -> ^bb4570, ^bb4387
  ^bb4570:
    pdl_interp.check_attribute %2444 is 3.000000e+00 : f32 -> ^bb4571, ^bb4387
  ^bb4571:
    pdl_interp.record_match @rewriters::@exp_lft_cube_rev(%2438, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.powf") -> ^bb4387
  ^bb31:
    pdl_interp.check_operand_count of %0 is 1 -> ^bb4572, ^bb1
  ^bb4572:
    pdl_interp.check_result_count of %0 is 1 -> ^bb4573, ^bb1
  ^bb4573:
    pdl_interp.is_not_null %2 : !pdl.value -> ^bb4574, ^bb1
  ^bb4574:
    pdl_interp.switch_operation_name of %3 to ["arith.mulf", "math.cbrt", "arith.divf", "math.exp", "arith.subf", "arith.addf"](^bb4575, ^bb4576, ^bb4577, ^bb4578, ^bb4579, ^bb4580) -> ^bb1
  ^bb4575:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb4581, ^bb1
  ^bb4581:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4582, ^bb1
  ^bb4582:
    %2445 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2445 : !pdl.value -> ^bb4583, ^bb1
  ^bb4583:
    pdl_interp.are_equal %2445, %2 : !pdl.value -> ^bb4584, ^bb1
  ^bb4584:
    %2446 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2446 : !pdl.value -> ^bb4585, ^bb1
  ^bb4585:
    %2447 = pdl_interp.get_value_type of %2446 : !pdl.type
    %2448 = pdl_interp.get_value_type of %2445 : !pdl.type
    pdl_interp.are_equal %2447, %2448 : !pdl.type -> ^bb4586, ^bb1
  ^bb4586:
    %2449 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2447, %2449 : !pdl.type -> ^bb4587, ^bb1
  ^bb4587:
    pdl_interp.check_type %2447 is f32 -> ^bb4588, ^bb1
  ^bb4588:
    %2450 = pdl_interp.get_operand 1 of %3
    pdl_interp.are_equal %2446, %2450 : !pdl.value -> ^bb4589, ^bb4590
  ^bb4590:
    %2451 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2451 : !pdl.value -> ^bb4591, ^bb1
  ^bb4591:
    %2452 = pdl_interp.get_value_type of %2451 : !pdl.type
    pdl_interp.are_equal %2447, %2452 : !pdl.type -> ^bb4592, ^bb1
  ^bb4592:
    pdl_interp.record_match @rewriters::@sqrt_prod(%2446, %2451, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.sqrt") -> ^bb1
  ^bb4589:
    pdl_interp.record_match @rewriters::@rem_sqrt_square(%2446, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.sqrt") -> ^bb4590
  ^bb4576:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb4593, ^bb1
  ^bb4593:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4594, ^bb1
  ^bb4594:
    %2453 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2453 : !pdl.value -> ^bb4595, ^bb1
  ^bb4595:
    pdl_interp.are_equal %2453, %2 : !pdl.value -> ^bb4596, ^bb1
  ^bb4596:
    %2454 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2454 : !pdl.value -> ^bb4597, ^bb1
  ^bb4597:
    %2455 = pdl_interp.get_value_type of %2454 : !pdl.type
    %2456 = pdl_interp.get_value_type of %2453 : !pdl.type
    pdl_interp.are_equal %2455, %2456 : !pdl.type -> ^bb4598, ^bb1
  ^bb4598:
    %2457 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2455, %2457 : !pdl.type -> ^bb4599, ^bb1
  ^bb4599:
    pdl_interp.check_type %2455 is f32 -> ^bb4600, ^bb1
  ^bb4600:
    pdl_interp.record_match @rewriters::@sqrt_cbrt(%2454, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.sqrt") -> ^bb1
  ^bb4577:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb4601, ^bb1
  ^bb4601:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4602, ^bb1
  ^bb4602:
    %2458 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2458 : !pdl.value -> ^bb4603, ^bb1
  ^bb4603:
    pdl_interp.are_equal %2458, %2 : !pdl.value -> ^bb4604, ^bb1
  ^bb4604:
    %2459 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2459 : !pdl.value -> ^bb4605, ^bb1
  ^bb4605:
    %2460 = pdl_interp.get_value_type of %2459 : !pdl.type
    %2461 = pdl_interp.get_value_type of %2458 : !pdl.type
    pdl_interp.are_equal %2460, %2461 : !pdl.type -> ^bb4606, ^bb4607
  ^bb4607:
    %2462 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2462 : !pdl.value -> ^bb4608, ^bb1
  ^bb4608:
    %2463 = pdl_interp.get_defining_op of %2462 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %2463 : !pdl.operation -> ^bb4609, ^bb1
  ^bb4609:
    %2464 = pdl_interp.get_defining_op of %2459 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %2464 : !pdl.operation -> ^bb4610, ^bb1
  ^bb4610:
    pdl_interp.check_operation_name of %2463 is "arith.constant" -> ^bb4611, ^bb1
  ^bb4611:
    pdl_interp.check_operand_count of %2463 is 0 -> ^bb4612, ^bb1
  ^bb4612:
    pdl_interp.check_result_count of %2463 is 1 -> ^bb4613, ^bb1
  ^bb4613:
    %2465 = pdl_interp.get_result 0 of %2463
    pdl_interp.is_not_null %2465 : !pdl.value -> ^bb4614, ^bb1
  ^bb4614:
    pdl_interp.are_equal %2465, %2462 : !pdl.value -> ^bb4615, ^bb1
  ^bb4615:
    pdl_interp.check_operation_name of %2464 is "arith.addf" -> ^bb4616, ^bb1
  ^bb4616:
    pdl_interp.check_operand_count of %2464 is 2 -> ^bb4617, ^bb1
  ^bb4617:
    pdl_interp.check_result_count of %2464 is 1 -> ^bb4618, ^bb1
  ^bb4618:
    %2466 = pdl_interp.get_result 0 of %2464
    pdl_interp.is_not_null %2466 : !pdl.value -> ^bb4619, ^bb1
  ^bb4619:
    pdl_interp.are_equal %2466, %2459 : !pdl.value -> ^bb4620, ^bb1
  ^bb4620:
    %2467 = pdl_interp.get_operand 0 of %2464
    pdl_interp.is_not_null %2467 : !pdl.value -> ^bb4621, ^bb1
  ^bb4621:
    %2468 = pdl_interp.get_attribute "value" of %2463
    pdl_interp.is_not_null %2468 : !pdl.attribute -> ^bb4622, ^bb1
  ^bb4622:
    pdl_interp.check_attribute %2468 is 2.000000e+00 : f32 -> ^bb4623, ^bb1
  ^bb4623:
    %2469 = pdl_interp.get_defining_op of %2467 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %2469 : !pdl.operation -> ^bb4624, ^bb1
  ^bb4624:
    %2470 = pdl_interp.get_operand 1 of %2464
    %2471 = pdl_interp.get_defining_op of %2470 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %2471 : !pdl.operation -> ^bb4625, ^bb1
  ^bb4625:
    pdl_interp.check_operation_name of %2469 is "math.cosh" -> ^bb4626, ^bb1
  ^bb4626:
    pdl_interp.check_operand_count of %2469 is 1 -> ^bb4627, ^bb1
  ^bb4627:
    pdl_interp.check_result_count of %2469 is 1 -> ^bb4628, ^bb1
  ^bb4628:
    %2472 = pdl_interp.get_result 0 of %2469
    pdl_interp.is_not_null %2472 : !pdl.value -> ^bb4629, ^bb1
  ^bb4629:
    pdl_interp.are_equal %2472, %2467 : !pdl.value -> ^bb4630, ^bb1
  ^bb4630:
    pdl_interp.is_not_null %2470 : !pdl.value -> ^bb4631, ^bb1
  ^bb4631:
    %2473 = pdl_interp.get_operand 0 of %2469
    pdl_interp.is_not_null %2473 : !pdl.value -> ^bb4632, ^bb1
  ^bb4632:
    pdl_interp.check_operation_name of %2471 is "arith.constant" -> ^bb4633, ^bb1
  ^bb4633:
    pdl_interp.check_operand_count of %2471 is 0 -> ^bb4634, ^bb1
  ^bb4634:
    pdl_interp.check_result_count of %2471 is 1 -> ^bb4635, ^bb1
  ^bb4635:
    %2474 = pdl_interp.get_result 0 of %2471
    pdl_interp.is_not_null %2474 : !pdl.value -> ^bb4636, ^bb1
  ^bb4636:
    pdl_interp.are_equal %2474, %2470 : !pdl.value -> ^bb4637, ^bb1
  ^bb4637:
    %2475 = pdl_interp.get_value_type of %2473 : !pdl.type
    %2476 = pdl_interp.get_value_type of %2472 : !pdl.type
    pdl_interp.are_equal %2475, %2476 : !pdl.type -> ^bb4638, ^bb1
  ^bb4638:
    %2477 = pdl_interp.get_value_type of %2466 : !pdl.type
    pdl_interp.are_equal %2475, %2477 : !pdl.type -> ^bb4639, ^bb1
  ^bb4639:
    %2478 = pdl_interp.get_value_type of %2465 : !pdl.type
    pdl_interp.are_equal %2475, %2478 : !pdl.type -> ^bb4640, ^bb1
  ^bb4640:
    %2479 = pdl_interp.get_value_type of %2458 : !pdl.type
    pdl_interp.are_equal %2475, %2479 : !pdl.type -> ^bb4641, ^bb1
  ^bb4641:
    %2480 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2475, %2480 : !pdl.type -> ^bb4642, ^bb1
  ^bb4642:
    pdl_interp.check_type %2475 is f32 -> ^bb4643, ^bb1
  ^bb4643:
    %2481 = pdl_interp.get_attribute "value" of %2471
    pdl_interp.is_not_null %2481 : !pdl.attribute -> ^bb4644, ^bb1
  ^bb4644:
    pdl_interp.check_attribute %2481 is 1.000000e+00 : f32 -> ^bb4645, ^bb1
  ^bb4645:
    %2482 = pdl_interp.get_value_type of %2474 : !pdl.type
    pdl_interp.are_equal %2475, %2482 : !pdl.type -> ^bb4646, ^bb1
  ^bb4646:
    pdl_interp.record_match @rewriters::@cosh_1div2_rev(%2473, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.sqrt") -> ^bb1
  ^bb4606:
    %2483 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2460, %2483 : !pdl.type -> ^bb4647, ^bb4607
  ^bb4647:
    pdl_interp.check_type %2460 is f32 -> ^bb4648, ^bb4607
  ^bb4648:
    %2484 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2484 : !pdl.value -> ^bb4649, ^bb4607
  ^bb4649:
    %2485 = pdl_interp.get_value_type of %2484 : !pdl.type
    pdl_interp.are_equal %2460, %2485 : !pdl.type -> ^bb4650, ^bb4607
  ^bb4650:
    pdl_interp.record_match @rewriters::@sqrt_div(%2459, %2484, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.sqrt") -> ^bb4607
  ^bb4578:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb4651, ^bb1
  ^bb4651:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4652, ^bb1
  ^bb4652:
    %2486 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2486 : !pdl.value -> ^bb4653, ^bb1
  ^bb4653:
    pdl_interp.are_equal %2486, %2 : !pdl.value -> ^bb4654, ^bb1
  ^bb4654:
    %2487 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2487 : !pdl.value -> ^bb4655, ^bb1
  ^bb4655:
    %2488 = pdl_interp.get_value_type of %2487 : !pdl.type
    %2489 = pdl_interp.get_value_type of %2486 : !pdl.type
    pdl_interp.are_equal %2488, %2489 : !pdl.type -> ^bb4656, ^bb1
  ^bb4656:
    %2490 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2488, %2490 : !pdl.type -> ^bb4657, ^bb1
  ^bb4657:
    pdl_interp.check_type %2488 is f32 -> ^bb4658, ^bb1
  ^bb4658:
    pdl_interp.record_match @rewriters::@exp_sqrt_rev(%2487, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.sqrt") -> ^bb1
  ^bb4579:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb4659, ^bb1
  ^bb4659:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4660, ^bb1
  ^bb4660:
    %2491 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2491 : !pdl.value -> ^bb4661, ^bb1
  ^bb4661:
    pdl_interp.are_equal %2491, %2 : !pdl.value -> ^bb4662, ^bb1
  ^bb4662:
    %2492 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2492 : !pdl.value -> ^bb4663, ^bb1
  ^bb4663:
    %2493 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2493 : !pdl.value -> ^bb4664, ^bb1
  ^bb4664:
    %2494 = pdl_interp.get_defining_op of %2493 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %2494 : !pdl.operation -> ^bb4665, ^bb1
  ^bb4665:
    %2495 = pdl_interp.get_defining_op of %2492 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %2495 : !pdl.operation -> ^bb4666, ^bb1
  ^bb4666:
    pdl_interp.check_operation_name of %2494 is "arith.mulf" -> ^bb4667, ^bb1
  ^bb4667:
    pdl_interp.check_operand_count of %2494 is 2 -> ^bb4668, ^bb1
  ^bb4668:
    pdl_interp.check_result_count of %2494 is 1 -> ^bb4669, ^bb1
  ^bb4669:
    %2496 = pdl_interp.get_result 0 of %2494
    pdl_interp.is_not_null %2496 : !pdl.value -> ^bb4670, ^bb1
  ^bb4670:
    pdl_interp.are_equal %2496, %2493 : !pdl.value -> ^bb4671, ^bb1
  ^bb4671:
    pdl_interp.check_operation_name of %2495 is "arith.constant" -> ^bb4672, ^bb1
  ^bb4672:
    pdl_interp.check_operand_count of %2495 is 0 -> ^bb4673, ^bb1
  ^bb4673:
    pdl_interp.check_result_count of %2495 is 1 -> ^bb4674, ^bb1
  ^bb4674:
    %2497 = pdl_interp.get_result 0 of %2495
    pdl_interp.is_not_null %2497 : !pdl.value -> ^bb4675, ^bb1
  ^bb4675:
    pdl_interp.are_equal %2497, %2492 : !pdl.value -> ^bb4676, ^bb1
  ^bb4676:
    %2498 = pdl_interp.get_operand 0 of %2494
    pdl_interp.is_not_null %2498 : !pdl.value -> ^bb4677, ^bb1
  ^bb4677:
    %2499 = pdl_interp.get_attribute "value" of %2495
    pdl_interp.is_not_null %2499 : !pdl.attribute -> ^bb4678, ^bb1
  ^bb4678:
    pdl_interp.check_attribute %2499 is 1.000000e+00 : f32 -> ^bb4679, ^bb1
  ^bb4679:
    %2500 = pdl_interp.get_value_type of %2497 : !pdl.type
    %2501 = pdl_interp.get_value_type of %2491 : !pdl.type
    pdl_interp.are_equal %2500, %2501 : !pdl.type -> ^bb4680, ^bb1
  ^bb4680:
    %2502 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2500, %2502 : !pdl.type -> ^bb4681, ^bb1
  ^bb4681:
    pdl_interp.check_type %2500 is f32 -> ^bb4682, ^bb1
  ^bb4682:
    %2503 = pdl_interp.get_value_type of %2496 : !pdl.type
    pdl_interp.are_equal %2500, %2503 : !pdl.type -> ^bb4683, ^bb1
  ^bb4683:
    %2504 = pdl_interp.get_value_type of %2498 : !pdl.type
    pdl_interp.are_equal %2500, %2504 : !pdl.type -> ^bb4684, ^bb1
  ^bb4684:
    %2505 = pdl_interp.get_operand 1 of %2494
    pdl_interp.are_equal %2498, %2505 : !pdl.value -> ^bb4685, ^bb1
  ^bb4685:
    pdl_interp.record_match @rewriters::@sin_acos_rev(%2498, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.sqrt") -> ^bb4686
  ^bb4686:
    pdl_interp.record_match @rewriters::@cos_asin_rev(%2498, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.sqrt") -> ^bb1
  ^bb4580:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb4687, ^bb1
  ^bb4687:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4688, ^bb1
  ^bb4688:
    %2506 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2506 : !pdl.value -> ^bb4689, ^bb1
  ^bb4689:
    pdl_interp.are_equal %2506, %2 : !pdl.value -> ^bb4690, ^bb1
  ^bb4690:
    %2507 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2507 : !pdl.value -> ^bb4691, ^bb1
  ^bb4691:
    %2508 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2508 : !pdl.value -> ^bb4692, ^bb1
  ^bb4692:
    %2509 = pdl_interp.get_defining_op of %2508 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %2509 : !pdl.operation -> ^bb4693, ^bb1
  ^bb4693:
    %2510 = pdl_interp.get_defining_op of %2507 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %2510 : !pdl.operation -> ^bb4694, ^bb1
  ^bb4694:
    pdl_interp.check_operation_name of %2509 is "arith.constant" -> ^bb4695, ^bb1
  ^bb4695:
    pdl_interp.check_operand_count of %2509 is 0 -> ^bb4696, ^bb1
  ^bb4696:
    pdl_interp.check_result_count of %2509 is 1 -> ^bb4697, ^bb1
  ^bb4697:
    %2511 = pdl_interp.get_result 0 of %2509
    pdl_interp.is_not_null %2511 : !pdl.value -> ^bb4698, ^bb1
  ^bb4698:
    pdl_interp.are_equal %2511, %2508 : !pdl.value -> ^bb4699, ^bb1
  ^bb4699:
    pdl_interp.check_operation_name of %2510 is "arith.mulf" -> ^bb4700, ^bb1
  ^bb4700:
    pdl_interp.check_operand_count of %2510 is 2 -> ^bb4701, ^bb1
  ^bb4701:
    pdl_interp.check_result_count of %2510 is 1 -> ^bb4702, ^bb1
  ^bb4702:
    %2512 = pdl_interp.get_result 0 of %2510
    pdl_interp.is_not_null %2512 : !pdl.value -> ^bb4703, ^bb1
  ^bb4703:
    pdl_interp.are_equal %2512, %2507 : !pdl.value -> ^bb4704, ^bb1
  ^bb4704:
    %2513 = pdl_interp.get_operand 0 of %2510
    pdl_interp.is_not_null %2513 : !pdl.value -> ^bb4705, ^bb1
  ^bb4705:
    %2514 = pdl_interp.get_value_type of %2513 : !pdl.type
    %2515 = pdl_interp.get_value_type of %2512 : !pdl.type
    pdl_interp.are_equal %2514, %2515 : !pdl.type -> ^bb4706, ^bb1
  ^bb4706:
    %2516 = pdl_interp.get_value_type of %2506 : !pdl.type
    pdl_interp.are_equal %2514, %2516 : !pdl.type -> ^bb4707, ^bb1
  ^bb4707:
    %2517 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2514, %2517 : !pdl.type -> ^bb4708, ^bb1
  ^bb4708:
    pdl_interp.check_type %2514 is f32 -> ^bb4709, ^bb1
  ^bb4709:
    %2518 = pdl_interp.get_value_type of %2511 : !pdl.type
    pdl_interp.are_equal %2514, %2518 : !pdl.type -> ^bb4710, ^bb1
  ^bb4710:
    %2519 = pdl_interp.get_attribute "value" of %2509
    pdl_interp.is_not_null %2519 : !pdl.attribute -> ^bb4711, ^bb1
  ^bb4711:
    pdl_interp.check_attribute %2519 is 1.000000e+00 : f32 -> ^bb4712, ^bb1
  ^bb4712:
    %2520 = pdl_interp.get_operand 1 of %2510
    pdl_interp.are_equal %2513, %2520 : !pdl.value -> ^bb4713, ^bb1
  ^bb4713:
    pdl_interp.record_match @rewriters::@cosh_asinh_rev(%2513, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.sqrt") -> ^bb1
  ^bb32:
    pdl_interp.check_operand_count of %0 is 1 -> ^bb4714, ^bb1
  ^bb4714:
    pdl_interp.check_result_count of %0 is 1 -> ^bb4715, ^bb1
  ^bb4715:
    pdl_interp.is_not_null %2 : !pdl.value -> ^bb4716, ^bb1
  ^bb4716:
    pdl_interp.switch_operation_name of %3 to ["math.sqrt", "math.powf", "arith.mulf", "arith.divf", "arith.negf", "math.absf", "math.exp"](^bb4717, ^bb4718, ^bb4719, ^bb4720, ^bb4721, ^bb4722, ^bb4723) -> ^bb1
  ^bb4717:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb4724, ^bb1
  ^bb4724:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4725, ^bb1
  ^bb4725:
    %2521 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2521 : !pdl.value -> ^bb4726, ^bb1
  ^bb4726:
    pdl_interp.are_equal %2521, %2 : !pdl.value -> ^bb4727, ^bb1
  ^bb4727:
    %2522 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2522 : !pdl.value -> ^bb4728, ^bb1
  ^bb4728:
    %2523 = pdl_interp.get_value_type of %2522 : !pdl.type
    %2524 = pdl_interp.get_value_type of %2521 : !pdl.type
    pdl_interp.are_equal %2523, %2524 : !pdl.type -> ^bb4729, ^bb1
  ^bb4729:
    %2525 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2523, %2525 : !pdl.type -> ^bb4730, ^bb1
  ^bb4730:
    pdl_interp.check_type %2523 is f32 -> ^bb4731, ^bb1
  ^bb4731:
    pdl_interp.record_match @rewriters::@cbrt_sqrt(%2522, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.cbrt") -> ^bb1
  ^bb4718:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb4732, ^bb1
  ^bb4732:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4733, ^bb1
  ^bb4733:
    %2526 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2526 : !pdl.value -> ^bb4734, ^bb1
  ^bb4734:
    pdl_interp.are_equal %2526, %2 : !pdl.value -> ^bb4735, ^bb1
  ^bb4735:
    %2527 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2527 : !pdl.value -> ^bb4736, ^bb1
  ^bb4736:
    %2528 = pdl_interp.get_value_type of %2527 : !pdl.type
    %2529 = pdl_interp.get_value_type of %2526 : !pdl.type
    pdl_interp.are_equal %2528, %2529 : !pdl.type -> ^bb4737, ^bb1
  ^bb4737:
    %2530 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2528, %2530 : !pdl.type -> ^bb4738, ^bb1
  ^bb4738:
    pdl_interp.check_type %2528 is f32 -> ^bb4739, ^bb1
  ^bb4739:
    %2531 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2531 : !pdl.value -> ^bb4740, ^bb1
  ^bb4740:
    %2532 = pdl_interp.get_defining_op of %2531 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %2532 : !pdl.operation -> ^bb4741, ^bb4742
  ^bb4742:
    %2533 = pdl_interp.get_value_type of %2531 : !pdl.type
    pdl_interp.are_equal %2528, %2533 : !pdl.type -> ^bb4743, ^bb1
  ^bb4743:
    pdl_interp.record_match @rewriters::@cbrt_pow(%2531, %2527, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.cbrt") -> ^bb1
  ^bb4741:
    pdl_interp.check_operation_name of %2532 is "arith.constant" -> ^bb4744, ^bb4742
  ^bb4744:
    pdl_interp.check_operand_count of %2532 is 0 -> ^bb4745, ^bb4742
  ^bb4745:
    pdl_interp.check_result_count of %2532 is 1 -> ^bb4746, ^bb4742
  ^bb4746:
    %2534 = pdl_interp.get_result 0 of %2532
    pdl_interp.is_not_null %2534 : !pdl.value -> ^bb4747, ^bb4742
  ^bb4747:
    pdl_interp.are_equal %2534, %2531 : !pdl.value -> ^bb4748, ^bb4742
  ^bb4748:
    %2535 = pdl_interp.get_attribute "value" of %2532
    pdl_interp.is_not_null %2535 : !pdl.attribute -> ^bb4749, ^bb4742
  ^bb4749:
    pdl_interp.check_attribute %2535 is 3.000000e+00 : f32 -> ^bb4750, ^bb4742
  ^bb4750:
    %2536 = pdl_interp.get_value_type of %2534 : !pdl.type
    pdl_interp.are_equal %2536, %2528 : !pdl.type -> ^bb4751, ^bb4742
  ^bb4751:
    pdl_interp.record_match @rewriters::@rem_cbrt_cube(%2527, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.cbrt") -> ^bb4742
  ^bb4719:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb4752, ^bb1
  ^bb4752:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4753, ^bb1
  ^bb4753:
    %2537 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2537 : !pdl.value -> ^bb4754, ^bb1
  ^bb4754:
    pdl_interp.are_equal %2537, %2 : !pdl.value -> ^bb4755, ^bb1
  ^bb4755:
    %2538 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2538 : !pdl.value -> ^bb4756, ^bb1
  ^bb4756:
    %2539 = pdl_interp.get_value_type of %2538 : !pdl.type
    %2540 = pdl_interp.get_value_type of %2537 : !pdl.type
    pdl_interp.are_equal %2539, %2540 : !pdl.type -> ^bb4757, ^bb1
  ^bb4757:
    %2541 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2539, %2541 : !pdl.type -> ^bb4758, ^bb1
  ^bb4758:
    pdl_interp.check_type %2539 is f32 -> ^bb4759, ^bb1
  ^bb4759:
    %2542 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2542 : !pdl.value -> ^bb4760, ^bb1
  ^bb4760:
    %2543 = pdl_interp.get_value_type of %2542 : !pdl.type
    pdl_interp.are_equal %2539, %2543 : !pdl.type -> ^bb4761, ^bb1
  ^bb4761:
    pdl_interp.record_match @rewriters::@cbrt_prod(%2538, %2542, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.cbrt") -> ^bb1
  ^bb4720:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb4762, ^bb1
  ^bb4762:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4763, ^bb1
  ^bb4763:
    %2544 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2544 : !pdl.value -> ^bb4764, ^bb1
  ^bb4764:
    pdl_interp.are_equal %2544, %2 : !pdl.value -> ^bb4765, ^bb1
  ^bb4765:
    %2545 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2545 : !pdl.value -> ^bb4766, ^bb1
  ^bb4766:
    %2546 = pdl_interp.get_value_type of %2545 : !pdl.type
    %2547 = pdl_interp.get_value_type of %2544 : !pdl.type
    pdl_interp.are_equal %2546, %2547 : !pdl.type -> ^bb4767, ^bb1
  ^bb4767:
    %2548 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2546, %2548 : !pdl.type -> ^bb4768, ^bb1
  ^bb4768:
    pdl_interp.check_type %2546 is f32 -> ^bb4769, ^bb1
  ^bb4769:
    %2549 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2549 : !pdl.value -> ^bb4770, ^bb1
  ^bb4770:
    %2550 = pdl_interp.get_value_type of %2549 : !pdl.type
    pdl_interp.are_equal %2546, %2550 : !pdl.type -> ^bb4771, ^bb1
  ^bb4771:
    pdl_interp.record_match @rewriters::@cbrt_div(%2545, %2549, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.cbrt") -> ^bb1
  ^bb4721:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb4772, ^bb1
  ^bb4772:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4773, ^bb1
  ^bb4773:
    %2551 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2551 : !pdl.value -> ^bb4774, ^bb1
  ^bb4774:
    pdl_interp.are_equal %2551, %2 : !pdl.value -> ^bb4775, ^bb1
  ^bb4775:
    %2552 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2552 : !pdl.value -> ^bb4776, ^bb1
  ^bb4776:
    %2553 = pdl_interp.get_value_type of %2552 : !pdl.type
    %2554 = pdl_interp.get_value_type of %2551 : !pdl.type
    pdl_interp.are_equal %2553, %2554 : !pdl.type -> ^bb4777, ^bb1
  ^bb4777:
    %2555 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2553, %2555 : !pdl.type -> ^bb4778, ^bb1
  ^bb4778:
    pdl_interp.check_type %2553 is f32 -> ^bb4779, ^bb1
  ^bb4779:
    pdl_interp.record_match @rewriters::@cbrt_neg(%2552, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.cbrt") -> ^bb1
  ^bb4722:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb4780, ^bb1
  ^bb4780:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4781, ^bb1
  ^bb4781:
    %2556 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2556 : !pdl.value -> ^bb4782, ^bb1
  ^bb4782:
    pdl_interp.are_equal %2556, %2 : !pdl.value -> ^bb4783, ^bb1
  ^bb4783:
    %2557 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2557 : !pdl.value -> ^bb4784, ^bb1
  ^bb4784:
    %2558 = pdl_interp.get_value_type of %2557 : !pdl.type
    %2559 = pdl_interp.get_value_type of %2556 : !pdl.type
    pdl_interp.are_equal %2558, %2559 : !pdl.type -> ^bb4785, ^bb1
  ^bb4785:
    %2560 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2558, %2560 : !pdl.type -> ^bb4786, ^bb1
  ^bb4786:
    pdl_interp.check_type %2558 is f32 -> ^bb4787, ^bb1
  ^bb4787:
    pdl_interp.record_match @rewriters::@cbrt_fabs(%2557, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.cbrt") -> ^bb1
  ^bb4723:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb4788, ^bb1
  ^bb4788:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4789, ^bb1
  ^bb4789:
    %2561 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2561 : !pdl.value -> ^bb4790, ^bb1
  ^bb4790:
    pdl_interp.are_equal %2561, %2 : !pdl.value -> ^bb4791, ^bb1
  ^bb4791:
    %2562 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2562 : !pdl.value -> ^bb4792, ^bb1
  ^bb4792:
    %2563 = pdl_interp.get_value_type of %2562 : !pdl.type
    %2564 = pdl_interp.get_value_type of %2561 : !pdl.type
    pdl_interp.are_equal %2563, %2564 : !pdl.type -> ^bb4793, ^bb1
  ^bb4793:
    %2565 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2563, %2565 : !pdl.type -> ^bb4794, ^bb1
  ^bb4794:
    pdl_interp.check_type %2563 is f32 -> ^bb4795, ^bb1
  ^bb4795:
    pdl_interp.record_match @rewriters::@exp_cbrt_rev(%2562, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.cbrt") -> ^bb1
  ^bb33:
    pdl_interp.check_operand_count of %0 is 1 -> ^bb4796, ^bb1
  ^bb4796:
    pdl_interp.check_result_count of %0 is 1 -> ^bb4797, ^bb1
  ^bb4797:
    pdl_interp.is_not_null %2 : !pdl.value -> ^bb4798, ^bb1
  ^bb4798:
    pdl_interp.switch_operation_name of %3 to ["math.absf", "arith.subf", "arith.addf", "arith.negf", "arith.mulf", "arith.divf", "math.sqrt", "math.copysign", "math.cbrt", "math.exp"](^bb4799, ^bb4800, ^bb4801, ^bb4802, ^bb4803, ^bb4804, ^bb4805, ^bb4806, ^bb4807, ^bb4808) -> ^bb1
  ^bb4799:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb4809, ^bb1
  ^bb4809:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4810, ^bb1
  ^bb4810:
    %2566 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2566 : !pdl.value -> ^bb4811, ^bb1
  ^bb4811:
    pdl_interp.are_equal %2566, %2 : !pdl.value -> ^bb4812, ^bb1
  ^bb4812:
    %2567 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2567 : !pdl.value -> ^bb4813, ^bb1
  ^bb4813:
    %2568 = pdl_interp.get_value_type of %2567 : !pdl.type
    %2569 = pdl_interp.get_value_type of %2566 : !pdl.type
    pdl_interp.are_equal %2568, %2569 : !pdl.type -> ^bb4814, ^bb1
  ^bb4814:
    %2570 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2568, %2570 : !pdl.type -> ^bb4815, ^bb1
  ^bb4815:
    pdl_interp.check_type %2568 is f32 -> ^bb4816, ^bb1
  ^bb4816:
    pdl_interp.record_match @rewriters::@fabs_fabs(%2567, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.absf") -> ^bb1
  ^bb4800:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb4817, ^bb1
  ^bb4817:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4818, ^bb1
  ^bb4818:
    %2571 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2571 : !pdl.value -> ^bb4819, ^bb1
  ^bb4819:
    pdl_interp.are_equal %2571, %2 : !pdl.value -> ^bb4820, ^bb1
  ^bb4820:
    %2572 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2572 : !pdl.value -> ^bb4821, ^bb1
  ^bb4821:
    %2573 = pdl_interp.get_value_type of %2572 : !pdl.type
    %2574 = pdl_interp.get_value_type of %2571 : !pdl.type
    pdl_interp.are_equal %2573, %2574 : !pdl.type -> ^bb4822, ^bb1
  ^bb4822:
    %2575 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2573, %2575 : !pdl.type -> ^bb4823, ^bb1
  ^bb4823:
    pdl_interp.check_type %2573 is f32 -> ^bb4824, ^bb1
  ^bb4824:
    %2576 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2576 : !pdl.value -> ^bb4825, ^bb1
  ^bb4825:
    %2577 = pdl_interp.get_value_type of %2576 : !pdl.type
    pdl_interp.are_equal %2573, %2577 : !pdl.type -> ^bb4826, ^bb1
  ^bb4826:
    pdl_interp.record_match @rewriters::@fabs_sub(%2576, %2572, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.absf") -> ^bb1
  ^bb4801:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb4827, ^bb1
  ^bb4827:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4828, ^bb1
  ^bb4828:
    %2578 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2578 : !pdl.value -> ^bb4829, ^bb1
  ^bb4829:
    pdl_interp.are_equal %2578, %2 : !pdl.value -> ^bb4830, ^bb1
  ^bb4830:
    %2579 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2579 : !pdl.value -> ^bb4831, ^bb1
  ^bb4831:
    %2580 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2580 : !pdl.value -> ^bb4832, ^bb1
  ^bb4832:
    %2581 = pdl_interp.get_defining_op of %2580 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %2581 : !pdl.operation -> ^bb4833, ^bb1
  ^bb4833:
    %2582 = pdl_interp.get_defining_op of %2579 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %2582 : !pdl.operation -> ^bb4834, ^bb1
  ^bb4834:
    pdl_interp.check_operation_name of %2581 is "math.absf" -> ^bb4835, ^bb1
  ^bb4835:
    pdl_interp.check_operand_count of %2581 is 1 -> ^bb4836, ^bb1
  ^bb4836:
    pdl_interp.check_result_count of %2581 is 1 -> ^bb4837, ^bb1
  ^bb4837:
    %2583 = pdl_interp.get_result 0 of %2581
    pdl_interp.is_not_null %2583 : !pdl.value -> ^bb4838, ^bb1
  ^bb4838:
    pdl_interp.are_equal %2583, %2580 : !pdl.value -> ^bb4839, ^bb1
  ^bb4839:
    pdl_interp.check_operation_name of %2582 is "math.absf" -> ^bb4840, ^bb1
  ^bb4840:
    pdl_interp.check_operand_count of %2582 is 1 -> ^bb4841, ^bb1
  ^bb4841:
    pdl_interp.check_result_count of %2582 is 1 -> ^bb4842, ^bb1
  ^bb4842:
    %2584 = pdl_interp.get_result 0 of %2582
    pdl_interp.is_not_null %2584 : !pdl.value -> ^bb4843, ^bb1
  ^bb4843:
    pdl_interp.are_equal %2584, %2579 : !pdl.value -> ^bb4844, ^bb1
  ^bb4844:
    %2585 = pdl_interp.get_operand 0 of %2582
    pdl_interp.is_not_null %2585 : !pdl.value -> ^bb4845, ^bb1
  ^bb4845:
    %2586 = pdl_interp.get_value_type of %2585 : !pdl.type
    %2587 = pdl_interp.get_value_type of %2584 : !pdl.type
    pdl_interp.are_equal %2586, %2587 : !pdl.type -> ^bb4846, ^bb1
  ^bb4846:
    %2588 = pdl_interp.get_value_type of %2578 : !pdl.type
    pdl_interp.are_equal %2586, %2588 : !pdl.type -> ^bb4847, ^bb1
  ^bb4847:
    %2589 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2586, %2589 : !pdl.type -> ^bb4848, ^bb1
  ^bb4848:
    pdl_interp.check_type %2586 is f32 -> ^bb4849, ^bb1
  ^bb4849:
    %2590 = pdl_interp.get_operand 0 of %2581
    pdl_interp.is_not_null %2590 : !pdl.value -> ^bb4850, ^bb1
  ^bb4850:
    %2591 = pdl_interp.get_value_type of %2583 : !pdl.type
    pdl_interp.are_equal %2586, %2591 : !pdl.type -> ^bb4851, ^bb1
  ^bb4851:
    %2592 = pdl_interp.get_value_type of %2590 : !pdl.type
    pdl_interp.are_equal %2586, %2592 : !pdl.type -> ^bb4852, ^bb1
  ^bb4852:
    pdl_interp.record_match @rewriters::@fabs_add(%2585, %2590, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.absf") -> ^bb1
  ^bb4802:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb4853, ^bb1
  ^bb4853:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4854, ^bb1
  ^bb4854:
    %2593 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2593 : !pdl.value -> ^bb4855, ^bb1
  ^bb4855:
    pdl_interp.are_equal %2593, %2 : !pdl.value -> ^bb4856, ^bb1
  ^bb4856:
    %2594 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2594 : !pdl.value -> ^bb4857, ^bb1
  ^bb4857:
    %2595 = pdl_interp.get_value_type of %2594 : !pdl.type
    %2596 = pdl_interp.get_value_type of %2593 : !pdl.type
    pdl_interp.are_equal %2595, %2596 : !pdl.type -> ^bb4858, ^bb1
  ^bb4858:
    %2597 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2595, %2597 : !pdl.type -> ^bb4859, ^bb1
  ^bb4859:
    pdl_interp.check_type %2595 is f32 -> ^bb4860, ^bb1
  ^bb4860:
    pdl_interp.record_match @rewriters::@fabs_neg(%2594, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.absf") -> ^bb1
  ^bb4803:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb4861, ^bb1
  ^bb4861:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4862, ^bb1
  ^bb4862:
    %2598 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2598 : !pdl.value -> ^bb4863, ^bb1
  ^bb4863:
    pdl_interp.are_equal %2598, %2 : !pdl.value -> ^bb4864, ^bb1
  ^bb4864:
    %2599 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2599 : !pdl.value -> ^bb4865, ^bb1
  ^bb4865:
    %2600 = pdl_interp.get_value_type of %2599 : !pdl.type
    %2601 = pdl_interp.get_value_type of %2598 : !pdl.type
    pdl_interp.are_equal %2600, %2601 : !pdl.type -> ^bb4866, ^bb1
  ^bb4866:
    %2602 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2600, %2602 : !pdl.type -> ^bb4867, ^bb1
  ^bb4867:
    pdl_interp.check_type %2600 is f32 -> ^bb4868, ^bb1
  ^bb4868:
    %2603 = pdl_interp.get_operand 1 of %3
    pdl_interp.are_equal %2599, %2603 : !pdl.value -> ^bb4869, ^bb4870
  ^bb4870:
    %2604 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2604 : !pdl.value -> ^bb4871, ^bb1
  ^bb4871:
    %2605 = pdl_interp.get_value_type of %2604 : !pdl.type
    pdl_interp.are_equal %2600, %2605 : !pdl.type -> ^bb4872, ^bb1
  ^bb4872:
    pdl_interp.record_match @rewriters::@fabs_mul(%2599, %2604, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.absf") -> ^bb1
  ^bb4869:
    pdl_interp.record_match @rewriters::@fabs_sqr(%2599, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.absf") -> ^bb4870
  ^bb4804:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb4873, ^bb1
  ^bb4873:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4874, ^bb1
  ^bb4874:
    %2606 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2606 : !pdl.value -> ^bb4875, ^bb1
  ^bb4875:
    pdl_interp.are_equal %2606, %2 : !pdl.value -> ^bb4876, ^bb1
  ^bb4876:
    %2607 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2607 : !pdl.value -> ^bb4877, ^bb1
  ^bb4877:
    %2608 = pdl_interp.get_value_type of %2607 : !pdl.type
    %2609 = pdl_interp.get_value_type of %2606 : !pdl.type
    pdl_interp.are_equal %2608, %2609 : !pdl.type -> ^bb4878, ^bb4879
  ^bb4879:
    %2610 = pdl_interp.get_defining_op of %2607 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %2610 : !pdl.operation -> ^bb4880, ^bb1
  ^bb4880:
    pdl_interp.check_operation_name of %2610 is "math.cbrt" -> ^bb4881, ^bb1
  ^bb4881:
    pdl_interp.check_operand_count of %2610 is 1 -> ^bb4882, ^bb1
  ^bb4882:
    pdl_interp.check_result_count of %2610 is 1 -> ^bb4883, ^bb1
  ^bb4883:
    %2611 = pdl_interp.get_result 0 of %2610
    pdl_interp.is_not_null %2611 : !pdl.value -> ^bb4884, ^bb1
  ^bb4884:
    pdl_interp.are_equal %2611, %2607 : !pdl.value -> ^bb4885, ^bb1
  ^bb4885:
    %2612 = pdl_interp.get_operand 0 of %2610
    pdl_interp.is_not_null %2612 : !pdl.value -> ^bb4886, ^bb1
  ^bb4886:
    %2613 = pdl_interp.get_value_type of %2612 : !pdl.type
    %2614 = pdl_interp.get_value_type of %2611 : !pdl.type
    pdl_interp.are_equal %2613, %2614 : !pdl.type -> ^bb4887, ^bb1
  ^bb4887:
    %2615 = pdl_interp.get_value_type of %2606 : !pdl.type
    pdl_interp.are_equal %2613, %2615 : !pdl.type -> ^bb4888, ^bb1
  ^bb4888:
    %2616 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2613, %2616 : !pdl.type -> ^bb4889, ^bb1
  ^bb4889:
    pdl_interp.check_type %2613 is f32 -> ^bb4890, ^bb1
  ^bb4890:
    %2617 = pdl_interp.get_operand 1 of %3
    pdl_interp.are_equal %2612, %2617 : !pdl.value -> ^bb4891, ^bb1
  ^bb4891:
    pdl_interp.record_match @rewriters::@fabs_cbrt(%2612, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.absf") -> ^bb1
  ^bb4878:
    %2618 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2608, %2618 : !pdl.type -> ^bb4892, ^bb4879
  ^bb4892:
    pdl_interp.check_type %2608 is f32 -> ^bb4893, ^bb4879
  ^bb4893:
    %2619 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2619 : !pdl.value -> ^bb4894, ^bb4879
  ^bb4894:
    %2620 = pdl_interp.get_value_type of %2619 : !pdl.type
    pdl_interp.are_equal %2608, %2620 : !pdl.type -> ^bb4895, ^bb4879
  ^bb4895:
    pdl_interp.record_match @rewriters::@fabs_div(%2607, %2619, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.absf") -> ^bb4879
  ^bb4805:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb4896, ^bb1
  ^bb4896:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4897, ^bb1
  ^bb4897:
    %2621 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2621 : !pdl.value -> ^bb4898, ^bb1
  ^bb4898:
    pdl_interp.are_equal %2621, %2 : !pdl.value -> ^bb4899, ^bb1
  ^bb4899:
    %2622 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2622 : !pdl.value -> ^bb4900, ^bb1
  ^bb4900:
    %2623 = pdl_interp.get_value_type of %2622 : !pdl.type
    %2624 = pdl_interp.get_value_type of %2621 : !pdl.type
    pdl_interp.are_equal %2623, %2624 : !pdl.type -> ^bb4901, ^bb1
  ^bb4901:
    %2625 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2623, %2625 : !pdl.type -> ^bb4902, ^bb1
  ^bb4902:
    pdl_interp.check_type %2623 is f32 -> ^bb4903, ^bb1
  ^bb4903:
    pdl_interp.record_match @rewriters::@sqrt_fabs(%2622, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.absf") -> ^bb1
  ^bb4806:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb4904, ^bb1
  ^bb4904:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4905, ^bb1
  ^bb4905:
    %2626 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2626 : !pdl.value -> ^bb4906, ^bb1
  ^bb4906:
    pdl_interp.are_equal %2626, %2 : !pdl.value -> ^bb4907, ^bb1
  ^bb4907:
    %2627 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2627 : !pdl.value -> ^bb4908, ^bb1
  ^bb4908:
    %2628 = pdl_interp.get_value_type of %2627 : !pdl.type
    %2629 = pdl_interp.get_value_type of %2626 : !pdl.type
    pdl_interp.are_equal %2628, %2629 : !pdl.type -> ^bb4909, ^bb1
  ^bb4909:
    %2630 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2628, %2630 : !pdl.type -> ^bb4910, ^bb1
  ^bb4910:
    pdl_interp.check_type %2628 is f32 -> ^bb4911, ^bb1
  ^bb4911:
    %2631 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2631 : !pdl.value -> ^bb4912, ^bb1
  ^bb4912:
    %2632 = pdl_interp.get_value_type of %2631 : !pdl.type
    pdl_interp.are_equal %2628, %2632 : !pdl.type -> ^bb4913, ^bb1
  ^bb4913:
    pdl_interp.record_match @rewriters::@fabs_copysign(%2627, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.absf") -> ^bb1
  ^bb4807:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb4914, ^bb1
  ^bb4914:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4915, ^bb1
  ^bb4915:
    %2633 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2633 : !pdl.value -> ^bb4916, ^bb1
  ^bb4916:
    pdl_interp.are_equal %2633, %2 : !pdl.value -> ^bb4917, ^bb1
  ^bb4917:
    %2634 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2634 : !pdl.value -> ^bb4918, ^bb1
  ^bb4918:
    %2635 = pdl_interp.get_value_type of %2634 : !pdl.type
    %2636 = pdl_interp.get_value_type of %2633 : !pdl.type
    pdl_interp.are_equal %2635, %2636 : !pdl.type -> ^bb4919, ^bb1
  ^bb4919:
    %2637 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2635, %2637 : !pdl.type -> ^bb4920, ^bb1
  ^bb4920:
    pdl_interp.check_type %2635 is f32 -> ^bb4921, ^bb1
  ^bb4921:
    pdl_interp.record_match @rewriters::@cbrt_fabs_rev(%2634, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.absf") -> ^bb1
  ^bb4808:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb4922, ^bb1
  ^bb4922:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4923, ^bb1
  ^bb4923:
    %2638 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2638 : !pdl.value -> ^bb4924, ^bb1
  ^bb4924:
    pdl_interp.are_equal %2638, %2 : !pdl.value -> ^bb4925, ^bb1
  ^bb4925:
    %2639 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2639 : !pdl.value -> ^bb4926, ^bb1
  ^bb4926:
    %2640 = pdl_interp.get_value_type of %2639 : !pdl.type
    %2641 = pdl_interp.get_value_type of %2638 : !pdl.type
    pdl_interp.are_equal %2640, %2641 : !pdl.type -> ^bb4927, ^bb1
  ^bb4927:
    %2642 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2640, %2642 : !pdl.type -> ^bb4928, ^bb1
  ^bb4928:
    pdl_interp.check_type %2640 is f32 -> ^bb4929, ^bb1
  ^bb4929:
    pdl_interp.record_match @rewriters::@fabs_exp(%2639, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.absf") -> ^bb1
  ^bb34:
    pdl_interp.check_operand_count of %0 is 2 -> ^bb4930, ^bb1
  ^bb4930:
    pdl_interp.check_result_count of %0 is 1 -> ^bb4931, ^bb1
  ^bb4931:
    pdl_interp.is_not_null %2 : !pdl.value -> ^bb4932, ^bb1
  ^bb4932:
    pdl_interp.switch_operation_name of %3 to ["arith.negf", "math.absf"](^bb4933, ^bb4934) -> ^bb1
  ^bb4933:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb4935, ^bb1
  ^bb4935:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4936, ^bb1
  ^bb4936:
    %2643 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2643 : !pdl.value -> ^bb4937, ^bb1
  ^bb4937:
    pdl_interp.are_equal %2643, %2 : !pdl.value -> ^bb4938, ^bb1
  ^bb4938:
    %2644 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2644 : !pdl.value -> ^bb4939, ^bb1
  ^bb4939:
    %2645 = pdl_interp.get_operand 1 of %0
    pdl_interp.is_not_null %2645 : !pdl.value -> ^bb4940, ^bb1
  ^bb4940:
    %2646 = pdl_interp.get_value_type of %2644 : !pdl.type
    %2647 = pdl_interp.get_value_type of %2643 : !pdl.type
    pdl_interp.are_equal %2646, %2647 : !pdl.type -> ^bb4941, ^bb1
  ^bb4941:
    %2648 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2646, %2648 : !pdl.type -> ^bb4942, ^bb1
  ^bb4942:
    pdl_interp.check_type %2646 is f32 -> ^bb4943, ^bb1
  ^bb4943:
    %2649 = pdl_interp.get_value_type of %2645 : !pdl.type
    pdl_interp.are_equal %2646, %2649 : !pdl.type -> ^bb4944, ^bb1
  ^bb4944:
    pdl_interp.record_match @rewriters::@copysign_other_neg(%2644, %2645, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.copysign") -> ^bb1
  ^bb4934:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb4945, ^bb1
  ^bb4945:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4946, ^bb1
  ^bb4946:
    %2650 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2650 : !pdl.value -> ^bb4947, ^bb1
  ^bb4947:
    pdl_interp.are_equal %2650, %2 : !pdl.value -> ^bb4948, ^bb1
  ^bb4948:
    %2651 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2651 : !pdl.value -> ^bb4949, ^bb1
  ^bb4949:
    %2652 = pdl_interp.get_operand 1 of %0
    pdl_interp.is_not_null %2652 : !pdl.value -> ^bb4950, ^bb1
  ^bb4950:
    %2653 = pdl_interp.get_value_type of %2651 : !pdl.type
    %2654 = pdl_interp.get_value_type of %2650 : !pdl.type
    pdl_interp.are_equal %2653, %2654 : !pdl.type -> ^bb4951, ^bb1
  ^bb4951:
    %2655 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2653, %2655 : !pdl.type -> ^bb4952, ^bb1
  ^bb4952:
    pdl_interp.check_type %2653 is f32 -> ^bb4953, ^bb1
  ^bb4953:
    %2656 = pdl_interp.get_value_type of %2652 : !pdl.type
    pdl_interp.are_equal %2653, %2656 : !pdl.type -> ^bb4954, ^bb1
  ^bb4954:
    pdl_interp.record_match @rewriters::@copysign_other_fabs(%2651, %2652, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.copysign") -> ^bb1
  ^bb35:
    pdl_interp.check_operand_count of %0 is 1 -> ^bb4955, ^bb1
  ^bb4955:
    pdl_interp.check_result_count of %0 is 1 -> ^bb4956, ^bb1
  ^bb4956:
    pdl_interp.is_not_null %2 : !pdl.value -> ^bb4957, ^bb1
  ^bb4957:
    pdl_interp.switch_operation_name of %3 to ["math.log", "arith.constant", "arith.addf", "arith.negf", "arith.subf", "arith.mulf", "arith.divf"](^bb4958, ^bb4959, ^bb4960, ^bb4961, ^bb4962, ^bb4963, ^bb4964) -> ^bb1
  ^bb4958:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb4965, ^bb1
  ^bb4965:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4966, ^bb1
  ^bb4966:
    %2657 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2657 : !pdl.value -> ^bb4967, ^bb1
  ^bb4967:
    pdl_interp.are_equal %2657, %2 : !pdl.value -> ^bb4968, ^bb1
  ^bb4968:
    %2658 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2658 : !pdl.value -> ^bb4969, ^bb1
  ^bb4969:
    %2659 = pdl_interp.get_value_type of %2658 : !pdl.type
    %2660 = pdl_interp.get_value_type of %2657 : !pdl.type
    pdl_interp.are_equal %2659, %2660 : !pdl.type -> ^bb4970, ^bb1
  ^bb4970:
    %2661 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2659, %2661 : !pdl.type -> ^bb4971, ^bb1
  ^bb4971:
    pdl_interp.check_type %2659 is f32 -> ^bb4972, ^bb1
  ^bb4972:
    pdl_interp.record_match @rewriters::@rem_exp_log(%2658, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.exp") -> ^bb1
  ^bb4959:
    pdl_interp.check_operand_count of %3 is 0 -> ^bb4973, ^bb1
  ^bb4973:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4974, ^bb1
  ^bb4974:
    %2662 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2662 : !pdl.value -> ^bb4975, ^bb1
  ^bb4975:
    pdl_interp.are_equal %2662, %2 : !pdl.value -> ^bb4976, ^bb1
  ^bb4976:
    %2663 = pdl_interp.get_attribute "value" of %3
    pdl_interp.is_not_null %2663 : !pdl.attribute -> ^bb4977, ^bb1
  ^bb4977:
    pdl_interp.check_attribute %2663 is 0.000000e+00 : f32 -> ^bb4978, ^bb1
  ^bb4978:
    %2664 = pdl_interp.get_value_type of %2662 : !pdl.type
    %2665 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2664, %2665 : !pdl.type -> ^bb4979, ^bb1
  ^bb4979:
    pdl_interp.check_type %2664 is f32 -> ^bb4980, ^bb1
  ^bb4980:
    pdl_interp.record_match @rewriters::@exp_0(%0 : !pdl.operation) : benefit(1), loc([]), root("math.exp") -> ^bb1
  ^bb4960:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb4981, ^bb1
  ^bb4981:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4982, ^bb1
  ^bb4982:
    %2666 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2666 : !pdl.value -> ^bb4983, ^bb1
  ^bb4983:
    pdl_interp.are_equal %2666, %2 : !pdl.value -> ^bb4984, ^bb1
  ^bb4984:
    %2667 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2667 : !pdl.value -> ^bb4985, ^bb1
  ^bb4985:
    %2668 = pdl_interp.get_value_type of %2667 : !pdl.type
    %2669 = pdl_interp.get_value_type of %2666 : !pdl.type
    pdl_interp.are_equal %2668, %2669 : !pdl.type -> ^bb4986, ^bb1
  ^bb4986:
    %2670 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2668, %2670 : !pdl.type -> ^bb4987, ^bb1
  ^bb4987:
    pdl_interp.check_type %2668 is f32 -> ^bb4988, ^bb1
  ^bb4988:
    %2671 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2671 : !pdl.value -> ^bb4989, ^bb1
  ^bb4989:
    %2672 = pdl_interp.get_value_type of %2671 : !pdl.type
    pdl_interp.are_equal %2668, %2672 : !pdl.type -> ^bb4990, ^bb1
  ^bb4990:
    pdl_interp.record_match @rewriters::@exp_sum(%2667, %2671, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.exp") -> ^bb1
  ^bb4961:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb4991, ^bb1
  ^bb4991:
    pdl_interp.check_result_count of %3 is 1 -> ^bb4992, ^bb1
  ^bb4992:
    %2673 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2673 : !pdl.value -> ^bb4993, ^bb1
  ^bb4993:
    pdl_interp.are_equal %2673, %2 : !pdl.value -> ^bb4994, ^bb1
  ^bb4994:
    %2674 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2674 : !pdl.value -> ^bb4995, ^bb1
  ^bb4995:
    %2675 = pdl_interp.get_value_type of %2674 : !pdl.type
    %2676 = pdl_interp.get_value_type of %2673 : !pdl.type
    pdl_interp.are_equal %2675, %2676 : !pdl.type -> ^bb4996, ^bb1
  ^bb4996:
    %2677 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2675, %2677 : !pdl.type -> ^bb4997, ^bb1
  ^bb4997:
    pdl_interp.check_type %2675 is f32 -> ^bb4998, ^bb1
  ^bb4998:
    pdl_interp.record_match @rewriters::@sinhsub__cosh_rev(%2674, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.exp") -> ^bb4999
  ^bb4999:
    pdl_interp.record_match @rewriters::@exp_neg(%2674, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.exp") -> ^bb1
  ^bb4962:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb5000, ^bb1
  ^bb5000:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5001, ^bb1
  ^bb5001:
    %2678 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2678 : !pdl.value -> ^bb5002, ^bb1
  ^bb5002:
    pdl_interp.are_equal %2678, %2 : !pdl.value -> ^bb5003, ^bb1
  ^bb5003:
    %2679 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2679 : !pdl.value -> ^bb5004, ^bb1
  ^bb5004:
    %2680 = pdl_interp.get_value_type of %2679 : !pdl.type
    %2681 = pdl_interp.get_value_type of %2678 : !pdl.type
    pdl_interp.are_equal %2680, %2681 : !pdl.type -> ^bb5005, ^bb1
  ^bb5005:
    %2682 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2680, %2682 : !pdl.type -> ^bb5006, ^bb1
  ^bb5006:
    pdl_interp.check_type %2680 is f32 -> ^bb5007, ^bb1
  ^bb5007:
    %2683 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2683 : !pdl.value -> ^bb5008, ^bb1
  ^bb5008:
    %2684 = pdl_interp.get_value_type of %2683 : !pdl.type
    pdl_interp.are_equal %2680, %2684 : !pdl.type -> ^bb5009, ^bb1
  ^bb5009:
    pdl_interp.record_match @rewriters::@exp_diff(%2679, %2683, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.exp") -> ^bb1
  ^bb4963:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb5010, ^bb1
  ^bb5010:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5011, ^bb1
  ^bb5011:
    %2685 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2685 : !pdl.value -> ^bb5012, ^bb1
  ^bb5012:
    pdl_interp.are_equal %2685, %2 : !pdl.value -> ^bb5013, ^bb1
  ^bb5013:
    %2686 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2686 : !pdl.value -> ^bb5014, ^bb1
  ^bb5014:
    %2687 = pdl_interp.get_value_type of %2686 : !pdl.type
    %2688 = pdl_interp.get_value_type of %2685 : !pdl.type
    pdl_interp.are_equal %2687, %2688 : !pdl.type -> ^bb5015, ^bb5016
  ^bb5016:
    %2689 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2689 : !pdl.value -> ^bb5017, ^bb1
  ^bb5017:
    %2690 = pdl_interp.get_defining_op of %2686 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %2690 : !pdl.operation -> ^bb5018, ^bb1
  ^bb5018:
    pdl_interp.check_operation_name of %2690 is "math.log" -> ^bb5019, ^bb1
  ^bb5019:
    pdl_interp.check_operand_count of %2690 is 1 -> ^bb5020, ^bb1
  ^bb5020:
    pdl_interp.check_result_count of %2690 is 1 -> ^bb5021, ^bb1
  ^bb5021:
    %2691 = pdl_interp.get_result 0 of %2690
    pdl_interp.is_not_null %2691 : !pdl.value -> ^bb5022, ^bb1
  ^bb5022:
    pdl_interp.are_equal %2691, %2686 : !pdl.value -> ^bb5023, ^bb1
  ^bb5023:
    %2692 = pdl_interp.get_operand 0 of %2690
    pdl_interp.is_not_null %2692 : !pdl.value -> ^bb5024, ^bb1
  ^bb5024:
    %2693 = pdl_interp.get_value_type of %2692 : !pdl.type
    %2694 = pdl_interp.get_value_type of %2691 : !pdl.type
    pdl_interp.are_equal %2693, %2694 : !pdl.type -> ^bb5025, ^bb1
  ^bb5025:
    %2695 = pdl_interp.get_value_type of %2685 : !pdl.type
    pdl_interp.are_equal %2693, %2695 : !pdl.type -> ^bb5026, ^bb1
  ^bb5026:
    %2696 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2693, %2696 : !pdl.type -> ^bb5027, ^bb1
  ^bb5027:
    pdl_interp.check_type %2693 is f32 -> ^bb5028, ^bb1
  ^bb5028:
    %2697 = pdl_interp.get_value_type of %2689 : !pdl.type
    pdl_interp.are_equal %2693, %2697 : !pdl.type -> ^bb5029, ^bb1
  ^bb5029:
    pdl_interp.record_match @rewriters::@exp_to_pow(%2692, %2689, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.exp") -> ^bb1
  ^bb5015:
    %2698 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2687, %2698 : !pdl.type -> ^bb5030, ^bb5016
  ^bb5030:
    pdl_interp.check_type %2687 is f32 -> ^bb5031, ^bb5016
  ^bb5031:
    %2699 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2699 : !pdl.value -> ^bb5032, ^bb5016
  ^bb5032:
    %2700 = pdl_interp.get_value_type of %2699 : !pdl.type
    pdl_interp.are_equal %2687, %2700 : !pdl.type -> ^bb5033, ^bb5034
  ^bb5034:
    %2701 = pdl_interp.get_defining_op of %2699 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %2701 : !pdl.operation -> ^bb5035, ^bb5016
  ^bb5035:
    pdl_interp.check_operation_name of %2701 is "arith.constant" -> ^bb5036, ^bb5016
  ^bb5036:
    pdl_interp.check_operand_count of %2701 is 0 -> ^bb5037, ^bb5016
  ^bb5037:
    pdl_interp.check_result_count of %2701 is 1 -> ^bb5038, ^bb5016
  ^bb5038:
    %2702 = pdl_interp.get_result 0 of %2701
    pdl_interp.is_not_null %2702 : !pdl.value -> ^bb5039, ^bb5016
  ^bb5039:
    pdl_interp.are_equal %2702, %2699 : !pdl.value -> ^bb5040, ^bb5016
  ^bb5040:
    %2703 = pdl_interp.get_attribute "value" of %2701
    pdl_interp.is_not_null %2703 : !pdl.attribute -> ^bb5041, ^bb5016
  ^bb5041:
    pdl_interp.switch_attribute %2703 to [2.000000e+00 : f32, 3.000000e+00 : f32](^bb5042, ^bb5043) -> ^bb5016
  ^bb5042:
    %2704 = pdl_interp.get_value_type of %2702 : !pdl.type
    pdl_interp.are_equal %2704, %2687 : !pdl.type -> ^bb5044, ^bb5016
  ^bb5044:
    pdl_interp.record_match @rewriters::@exp_lft_sqr(%2686, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.exp") -> ^bb5016
  ^bb5043:
    %2705 = pdl_interp.get_value_type of %2702 : !pdl.type
    pdl_interp.are_equal %2705, %2687 : !pdl.type -> ^bb5045, ^bb5016
  ^bb5045:
    pdl_interp.record_match @rewriters::@exp_lft_cube(%2686, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.exp") -> ^bb5016
  ^bb5033:
    pdl_interp.record_match @rewriters::@exp_prod(%2686, %2699, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.exp") -> ^bb5034
  ^bb4964:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb5046, ^bb1
  ^bb5046:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5047, ^bb1
  ^bb5047:
    %2706 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2706 : !pdl.value -> ^bb5048, ^bb1
  ^bb5048:
    pdl_interp.are_equal %2706, %2 : !pdl.value -> ^bb5049, ^bb1
  ^bb5049:
    %2707 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2707 : !pdl.value -> ^bb5050, ^bb1
  ^bb5050:
    %2708 = pdl_interp.get_value_type of %2707 : !pdl.type
    %2709 = pdl_interp.get_value_type of %2706 : !pdl.type
    pdl_interp.are_equal %2708, %2709 : !pdl.type -> ^bb5051, ^bb1
  ^bb5051:
    %2710 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2708, %2710 : !pdl.type -> ^bb5052, ^bb1
  ^bb5052:
    pdl_interp.check_type %2708 is f32 -> ^bb5053, ^bb1
  ^bb5053:
    %2711 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2711 : !pdl.value -> ^bb5054, ^bb1
  ^bb5054:
    %2712 = pdl_interp.get_defining_op of %2711 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %2712 : !pdl.operation -> ^bb5055, ^bb1
  ^bb5055:
    pdl_interp.check_operation_name of %2712 is "arith.constant" -> ^bb5056, ^bb1
  ^bb5056:
    pdl_interp.check_operand_count of %2712 is 0 -> ^bb5057, ^bb1
  ^bb5057:
    pdl_interp.check_result_count of %2712 is 1 -> ^bb5058, ^bb1
  ^bb5058:
    %2713 = pdl_interp.get_result 0 of %2712
    pdl_interp.is_not_null %2713 : !pdl.value -> ^bb5059, ^bb1
  ^bb5059:
    pdl_interp.are_equal %2713, %2711 : !pdl.value -> ^bb5060, ^bb1
  ^bb5060:
    %2714 = pdl_interp.get_attribute "value" of %2712
    pdl_interp.is_not_null %2714 : !pdl.attribute -> ^bb5061, ^bb1
  ^bb5061:
    pdl_interp.switch_attribute %2714 to [2.000000e+00 : f32, 3.000000e+00 : f32](^bb5062, ^bb5063) -> ^bb1
  ^bb5062:
    %2715 = pdl_interp.get_value_type of %2713 : !pdl.type
    pdl_interp.are_equal %2715, %2708 : !pdl.type -> ^bb5064, ^bb1
  ^bb5064:
    pdl_interp.record_match @rewriters::@exp_sqrt(%2707, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.exp") -> ^bb1
  ^bb5063:
    %2716 = pdl_interp.get_value_type of %2713 : !pdl.type
    pdl_interp.are_equal %2716, %2708 : !pdl.type -> ^bb5065, ^bb1
  ^bb5065:
    pdl_interp.record_match @rewriters::@exp_cbrt(%2707, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.exp") -> ^bb1
  ^bb36:
    pdl_interp.check_operand_count of %0 is 1 -> ^bb5066, ^bb1
  ^bb5066:
    pdl_interp.check_result_count of %0 is 1 -> ^bb5067, ^bb1
  ^bb5067:
    pdl_interp.is_not_null %2 : !pdl.value -> ^bb5068, ^bb1
  ^bb5068:
    pdl_interp.switch_operation_name of %3 to ["math.exp", "arith.divf", "arith.mulf", "arith.addf"](^bb5069, ^bb5070, ^bb5071, ^bb5072) -> ^bb1
  ^bb5069:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb5073, ^bb1
  ^bb5073:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5074, ^bb1
  ^bb5074:
    %2717 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2717 : !pdl.value -> ^bb5075, ^bb1
  ^bb5075:
    pdl_interp.are_equal %2717, %2 : !pdl.value -> ^bb5076, ^bb1
  ^bb5076:
    %2718 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2718 : !pdl.value -> ^bb5077, ^bb1
  ^bb5077:
    %2719 = pdl_interp.get_value_type of %2718 : !pdl.type
    %2720 = pdl_interp.get_value_type of %2717 : !pdl.type
    pdl_interp.are_equal %2719, %2720 : !pdl.type -> ^bb5078, ^bb1
  ^bb5078:
    %2721 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2719, %2721 : !pdl.type -> ^bb5079, ^bb1
  ^bb5079:
    pdl_interp.check_type %2719 is f32 -> ^bb5080, ^bb1
  ^bb5080:
    pdl_interp.record_match @rewriters::@rem_log_exp(%2718, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.log") -> ^bb1
  ^bb5070:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb5081, ^bb1
  ^bb5081:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5082, ^bb1
  ^bb5082:
    %2722 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2722 : !pdl.value -> ^bb5083, ^bb1
  ^bb5083:
    pdl_interp.are_equal %2722, %2 : !pdl.value -> ^bb5084, ^bb1
  ^bb5084:
    %2723 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2723 : !pdl.value -> ^bb5085, ^bb1
  ^bb5085:
    %2724 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2724 : !pdl.value -> ^bb5086, ^bb5087
  ^bb5087:
    %2725 = pdl_interp.get_value_type of %2723 : !pdl.type
    %2726 = pdl_interp.get_value_type of %2722 : !pdl.type
    pdl_interp.are_equal %2725, %2726 : !pdl.type -> ^bb5088, ^bb1
  ^bb5088:
    %2727 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2725, %2727 : !pdl.type -> ^bb5089, ^bb1
  ^bb5089:
    pdl_interp.check_type %2725 is f32 -> ^bb5090, ^bb1
  ^bb5090:
    %2728 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2728 : !pdl.value -> ^bb5091, ^bb1
  ^bb5091:
    %2729 = pdl_interp.get_value_type of %2728 : !pdl.type
    pdl_interp.are_equal %2725, %2729 : !pdl.type -> ^bb5092, ^bb1
  ^bb5092:
    pdl_interp.record_match @rewriters::@log_div(%2723, %2728, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.log") -> ^bb1
  ^bb5086:
    %2730 = pdl_interp.get_defining_op of %2723 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %2730 : !pdl.operation -> ^bb5093, ^bb5087
  ^bb5093:
    pdl_interp.check_operation_name of %2730 is "arith.constant" -> ^bb5094, ^bb5087
  ^bb5094:
    pdl_interp.check_operand_count of %2730 is 0 -> ^bb5095, ^bb5087
  ^bb5095:
    pdl_interp.check_result_count of %2730 is 1 -> ^bb5096, ^bb5087
  ^bb5096:
    %2731 = pdl_interp.get_result 0 of %2730
    pdl_interp.is_not_null %2731 : !pdl.value -> ^bb5097, ^bb5087
  ^bb5097:
    pdl_interp.are_equal %2731, %2723 : !pdl.value -> ^bb5098, ^bb5087
  ^bb5098:
    %2732 = pdl_interp.get_attribute "value" of %2730
    pdl_interp.is_not_null %2732 : !pdl.attribute -> ^bb5099, ^bb5087
  ^bb5099:
    pdl_interp.check_attribute %2732 is 1.000000e+00 : f32 -> ^bb5100, ^bb5087
  ^bb5100:
    %2733 = pdl_interp.get_value_type of %2731 : !pdl.type
    %2734 = pdl_interp.get_value_type of %2722 : !pdl.type
    pdl_interp.are_equal %2733, %2734 : !pdl.type -> ^bb5101, ^bb5087
  ^bb5101:
    %2735 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2733, %2735 : !pdl.type -> ^bb5102, ^bb5087
  ^bb5102:
    pdl_interp.check_type %2733 is f32 -> ^bb5103, ^bb5087
  ^bb5103:
    %2736 = pdl_interp.get_value_type of %2724 : !pdl.type
    pdl_interp.are_equal %2733, %2736 : !pdl.type -> ^bb5104, ^bb5087
  ^bb5104:
    pdl_interp.record_match @rewriters::@log_rec(%2724, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.log") -> ^bb5087
  ^bb5071:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb5105, ^bb1
  ^bb5105:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5106, ^bb1
  ^bb5106:
    %2737 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2737 : !pdl.value -> ^bb5107, ^bb1
  ^bb5107:
    pdl_interp.are_equal %2737, %2 : !pdl.value -> ^bb5108, ^bb1
  ^bb5108:
    %2738 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2738 : !pdl.value -> ^bb5109, ^bb1
  ^bb5109:
    %2739 = pdl_interp.get_value_type of %2738 : !pdl.type
    %2740 = pdl_interp.get_value_type of %2737 : !pdl.type
    pdl_interp.are_equal %2739, %2740 : !pdl.type -> ^bb5110, ^bb1
  ^bb5110:
    %2741 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2739, %2741 : !pdl.type -> ^bb5111, ^bb1
  ^bb5111:
    pdl_interp.check_type %2739 is f32 -> ^bb5112, ^bb1
  ^bb5112:
    %2742 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2742 : !pdl.value -> ^bb5113, ^bb1
  ^bb5113:
    %2743 = pdl_interp.get_value_type of %2742 : !pdl.type
    pdl_interp.are_equal %2739, %2743 : !pdl.type -> ^bb5114, ^bb1
  ^bb5114:
    pdl_interp.record_match @rewriters::@log_prod(%2738, %2742, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.log") -> ^bb1
  ^bb5072:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb5115, ^bb1
  ^bb5115:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5116, ^bb1
  ^bb5116:
    %2744 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2744 : !pdl.value -> ^bb5117, ^bb1
  ^bb5117:
    pdl_interp.are_equal %2744, %2 : !pdl.value -> ^bb5118, ^bb1
  ^bb5118:
    %2745 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2745 : !pdl.value -> ^bb5119, ^bb1
  ^bb5119:
    %2746 = pdl_interp.get_value_type of %2745 : !pdl.type
    %2747 = pdl_interp.get_value_type of %2744 : !pdl.type
    pdl_interp.are_equal %2746, %2747 : !pdl.type -> ^bb5120, ^bb1
  ^bb5120:
    %2748 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2746, %2748 : !pdl.type -> ^bb5121, ^bb1
  ^bb5121:
    pdl_interp.check_type %2746 is f32 -> ^bb5122, ^bb1
  ^bb5122:
    %2749 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2749 : !pdl.value -> ^bb5123, ^bb1
  ^bb5123:
    %2750 = pdl_interp.get_defining_op of %2749 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %2750 : !pdl.operation -> ^bb5124, ^bb1
  ^bb5124:
    pdl_interp.check_operation_name of %2750 is "math.sqrt" -> ^bb5125, ^bb1
  ^bb5125:
    pdl_interp.check_operand_count of %2750 is 1 -> ^bb5126, ^bb1
  ^bb5126:
    pdl_interp.check_result_count of %2750 is 1 -> ^bb5127, ^bb1
  ^bb5127:
    %2751 = pdl_interp.get_result 0 of %2750
    pdl_interp.is_not_null %2751 : !pdl.value -> ^bb5128, ^bb1
  ^bb5128:
    pdl_interp.are_equal %2751, %2749 : !pdl.value -> ^bb5129, ^bb1
  ^bb5129:
    %2752 = pdl_interp.get_operand 0 of %2750
    pdl_interp.is_not_null %2752 : !pdl.value -> ^bb5130, ^bb1
  ^bb5130:
    %2753 = pdl_interp.get_defining_op of %2752 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %2753 : !pdl.operation -> ^bb5131, ^bb1
  ^bb5131:
    %2754 = pdl_interp.get_value_type of %2751 : !pdl.type
    pdl_interp.are_equal %2754, %2746 : !pdl.type -> ^bb5132, ^bb1
  ^bb5132:
    pdl_interp.switch_operation_name of %2753 to ["arith.addf", "arith.subf"](^bb5133, ^bb5134) -> ^bb1
  ^bb5133:
    pdl_interp.check_operand_count of %2753 is 2 -> ^bb5135, ^bb1
  ^bb5135:
    pdl_interp.check_result_count of %2753 is 1 -> ^bb5136, ^bb1
  ^bb5136:
    %2755 = pdl_interp.get_result 0 of %2753
    pdl_interp.is_not_null %2755 : !pdl.value -> ^bb5137, ^bb1
  ^bb5137:
    pdl_interp.are_equal %2755, %2752 : !pdl.value -> ^bb5138, ^bb1
  ^bb5138:
    %2756 = pdl_interp.get_operand 0 of %2753
    pdl_interp.is_not_null %2756 : !pdl.value -> ^bb5139, ^bb1
  ^bb5139:
    %2757 = pdl_interp.get_defining_op of %2756 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %2757 : !pdl.operation -> ^bb5140, ^bb1
  ^bb5140:
    %2758 = pdl_interp.get_operand 1 of %2753
    %2759 = pdl_interp.get_defining_op of %2758 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %2759 : !pdl.operation -> ^bb5141, ^bb1
  ^bb5141:
    pdl_interp.is_not_null %2758 : !pdl.value -> ^bb5142, ^bb1
  ^bb5142:
    pdl_interp.check_operation_name of %2757 is "arith.mulf" -> ^bb5143, ^bb1
  ^bb5143:
    pdl_interp.check_operand_count of %2757 is 2 -> ^bb5144, ^bb1
  ^bb5144:
    pdl_interp.check_result_count of %2757 is 1 -> ^bb5145, ^bb1
  ^bb5145:
    %2760 = pdl_interp.get_result 0 of %2757
    pdl_interp.is_not_null %2760 : !pdl.value -> ^bb5146, ^bb1
  ^bb5146:
    pdl_interp.are_equal %2760, %2756 : !pdl.value -> ^bb5147, ^bb1
  ^bb5147:
    %2761 = pdl_interp.get_value_type of %2755 : !pdl.type
    pdl_interp.are_equal %2761, %2746 : !pdl.type -> ^bb5148, ^bb1
  ^bb5148:
    pdl_interp.check_operation_name of %2759 is "arith.constant" -> ^bb5149, ^bb1
  ^bb5149:
    pdl_interp.check_operand_count of %2759 is 0 -> ^bb5150, ^bb1
  ^bb5150:
    pdl_interp.check_result_count of %2759 is 1 -> ^bb5151, ^bb1
  ^bb5151:
    %2762 = pdl_interp.get_operand 0 of %2757
    pdl_interp.are_equal %2762, %2745 : !pdl.value -> ^bb5152, ^bb1
  ^bb5152:
    %2763 = pdl_interp.get_operand 1 of %2757
    pdl_interp.are_equal %2763, %2745 : !pdl.value -> ^bb5153, ^bb1
  ^bb5153:
    %2764 = pdl_interp.get_attribute "value" of %2759
    pdl_interp.is_not_null %2764 : !pdl.attribute -> ^bb5154, ^bb1
  ^bb5154:
    pdl_interp.check_attribute %2764 is 1.000000e+00 : f32 -> ^bb5155, ^bb1
  ^bb5155:
    %2765 = pdl_interp.get_result 0 of %2759
    pdl_interp.is_not_null %2765 : !pdl.value -> ^bb5156, ^bb1
  ^bb5156:
    pdl_interp.are_equal %2765, %2758 : !pdl.value -> ^bb5157, ^bb1
  ^bb5157:
    %2766 = pdl_interp.get_value_type of %2760 : !pdl.type
    pdl_interp.are_equal %2766, %2746 : !pdl.type -> ^bb5158, ^bb1
  ^bb5158:
    %2767 = pdl_interp.get_value_type of %2765 : !pdl.type
    pdl_interp.are_equal %2767, %2746 : !pdl.type -> ^bb5159, ^bb1
  ^bb5159:
    pdl_interp.record_match @rewriters::@asinh_def_rev(%2745, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.log") -> ^bb1
  ^bb5134:
    pdl_interp.check_operand_count of %2753 is 2 -> ^bb5160, ^bb1
  ^bb5160:
    pdl_interp.check_result_count of %2753 is 1 -> ^bb5161, ^bb1
  ^bb5161:
    %2768 = pdl_interp.get_result 0 of %2753
    pdl_interp.is_not_null %2768 : !pdl.value -> ^bb5162, ^bb1
  ^bb5162:
    pdl_interp.are_equal %2768, %2752 : !pdl.value -> ^bb5163, ^bb1
  ^bb5163:
    %2769 = pdl_interp.get_operand 0 of %2753
    pdl_interp.is_not_null %2769 : !pdl.value -> ^bb5164, ^bb1
  ^bb5164:
    %2770 = pdl_interp.get_defining_op of %2769 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %2770 : !pdl.operation -> ^bb5165, ^bb1
  ^bb5165:
    %2771 = pdl_interp.get_operand 1 of %2753
    %2772 = pdl_interp.get_defining_op of %2771 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %2772 : !pdl.operation -> ^bb5166, ^bb1
  ^bb5166:
    pdl_interp.is_not_null %2771 : !pdl.value -> ^bb5167, ^bb1
  ^bb5167:
    pdl_interp.check_operation_name of %2770 is "arith.mulf" -> ^bb5168, ^bb1
  ^bb5168:
    pdl_interp.check_operand_count of %2770 is 2 -> ^bb5169, ^bb1
  ^bb5169:
    pdl_interp.check_result_count of %2770 is 1 -> ^bb5170, ^bb1
  ^bb5170:
    %2773 = pdl_interp.get_result 0 of %2770
    pdl_interp.is_not_null %2773 : !pdl.value -> ^bb5171, ^bb1
  ^bb5171:
    pdl_interp.are_equal %2773, %2769 : !pdl.value -> ^bb5172, ^bb1
  ^bb5172:
    %2774 = pdl_interp.get_value_type of %2768 : !pdl.type
    pdl_interp.are_equal %2774, %2746 : !pdl.type -> ^bb5173, ^bb1
  ^bb5173:
    pdl_interp.check_operation_name of %2772 is "arith.constant" -> ^bb5174, ^bb1
  ^bb5174:
    pdl_interp.check_operand_count of %2772 is 0 -> ^bb5175, ^bb1
  ^bb5175:
    pdl_interp.check_result_count of %2772 is 1 -> ^bb5176, ^bb1
  ^bb5176:
    %2775 = pdl_interp.get_operand 0 of %2770
    pdl_interp.are_equal %2775, %2745 : !pdl.value -> ^bb5177, ^bb1
  ^bb5177:
    %2776 = pdl_interp.get_operand 1 of %2770
    pdl_interp.are_equal %2776, %2745 : !pdl.value -> ^bb5178, ^bb1
  ^bb5178:
    %2777 = pdl_interp.get_attribute "value" of %2772
    pdl_interp.is_not_null %2777 : !pdl.attribute -> ^bb5179, ^bb1
  ^bb5179:
    pdl_interp.check_attribute %2777 is 1.000000e+00 : f32 -> ^bb5180, ^bb1
  ^bb5180:
    %2778 = pdl_interp.get_result 0 of %2772
    pdl_interp.is_not_null %2778 : !pdl.value -> ^bb5181, ^bb1
  ^bb5181:
    pdl_interp.are_equal %2778, %2771 : !pdl.value -> ^bb5182, ^bb1
  ^bb5182:
    %2779 = pdl_interp.get_value_type of %2773 : !pdl.type
    pdl_interp.are_equal %2779, %2746 : !pdl.type -> ^bb5183, ^bb1
  ^bb5183:
    %2780 = pdl_interp.get_value_type of %2778 : !pdl.type
    pdl_interp.are_equal %2780, %2746 : !pdl.type -> ^bb5184, ^bb1
  ^bb5184:
    pdl_interp.record_match @rewriters::@acosh_def_rev(%2745, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.log") -> ^bb1
  ^bb37:
    pdl_interp.check_operand_count of %0 is 1 -> ^bb5185, ^bb1
  ^bb5185:
    pdl_interp.check_result_count of %0 is 1 -> ^bb5186, ^bb1
  ^bb5186:
    pdl_interp.is_not_null %2 : !pdl.value -> ^bb5187, ^bb1
  ^bb5187:
    pdl_interp.switch_operation_name of %3 to ["arith.constant", "arith.negf", "math.asin", "arith.addf", "arith.subf", "arith.mulf", "math.acos", "math.atan"](^bb5188, ^bb5189, ^bb5190, ^bb5191, ^bb5192, ^bb5193, ^bb5194, ^bb5195) -> ^bb1
  ^bb5188:
    pdl_interp.check_operand_count of %3 is 0 -> ^bb5196, ^bb1
  ^bb5196:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5197, ^bb1
  ^bb5197:
    %2781 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2781 : !pdl.value -> ^bb5198, ^bb1
  ^bb5198:
    pdl_interp.are_equal %2781, %2 : !pdl.value -> ^bb5199, ^bb1
  ^bb5199:
    %2782 = pdl_interp.get_attribute "value" of %3
    pdl_interp.is_not_null %2782 : !pdl.attribute -> ^bb5200, ^bb1
  ^bb5200:
    pdl_interp.check_attribute %2782 is 0.000000e+00 : f32 -> ^bb5201, ^bb1
  ^bb5201:
    %2783 = pdl_interp.get_value_type of %2781 : !pdl.type
    %2784 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2783, %2784 : !pdl.type -> ^bb5202, ^bb1
  ^bb5202:
    pdl_interp.check_type %2783 is f32 -> ^bb5203, ^bb1
  ^bb5203:
    pdl_interp.record_match @rewriters::@sin_0(%0 : !pdl.operation) : benefit(1), loc([]), root("math.sin") -> ^bb1
  ^bb5189:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb5204, ^bb1
  ^bb5204:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5205, ^bb1
  ^bb5205:
    %2785 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2785 : !pdl.value -> ^bb5206, ^bb1
  ^bb5206:
    pdl_interp.are_equal %2785, %2 : !pdl.value -> ^bb5207, ^bb1
  ^bb5207:
    %2786 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2786 : !pdl.value -> ^bb5208, ^bb1
  ^bb5208:
    %2787 = pdl_interp.get_value_type of %2786 : !pdl.type
    %2788 = pdl_interp.get_value_type of %2785 : !pdl.type
    pdl_interp.are_equal %2787, %2788 : !pdl.type -> ^bb5209, ^bb1
  ^bb5209:
    %2789 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2787, %2789 : !pdl.type -> ^bb5210, ^bb1
  ^bb5210:
    pdl_interp.check_type %2787 is f32 -> ^bb5211, ^bb1
  ^bb5211:
    pdl_interp.record_match @rewriters::@sin_neg(%2786, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.sin") -> ^bb1
  ^bb5190:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb5212, ^bb1
  ^bb5212:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5213, ^bb1
  ^bb5213:
    %2790 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2790 : !pdl.value -> ^bb5214, ^bb1
  ^bb5214:
    pdl_interp.are_equal %2790, %2 : !pdl.value -> ^bb5215, ^bb1
  ^bb5215:
    %2791 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2791 : !pdl.value -> ^bb5216, ^bb1
  ^bb5216:
    %2792 = pdl_interp.get_value_type of %2791 : !pdl.type
    %2793 = pdl_interp.get_value_type of %2790 : !pdl.type
    pdl_interp.are_equal %2792, %2793 : !pdl.type -> ^bb5217, ^bb1
  ^bb5217:
    %2794 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2792, %2794 : !pdl.type -> ^bb5218, ^bb1
  ^bb5218:
    pdl_interp.check_type %2792 is f32 -> ^bb5219, ^bb1
  ^bb5219:
    pdl_interp.record_match @rewriters::@sin_asin(%2791, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.sin") -> ^bb1
  ^bb5191:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb5220, ^bb1
  ^bb5220:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5221, ^bb1
  ^bb5221:
    %2795 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2795 : !pdl.value -> ^bb5222, ^bb1
  ^bb5222:
    pdl_interp.are_equal %2795, %2 : !pdl.value -> ^bb5223, ^bb1
  ^bb5223:
    %2796 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2796 : !pdl.value -> ^bb5224, ^bb1
  ^bb5224:
    %2797 = pdl_interp.get_value_type of %2796 : !pdl.type
    %2798 = pdl_interp.get_value_type of %2795 : !pdl.type
    pdl_interp.are_equal %2797, %2798 : !pdl.type -> ^bb5225, ^bb1
  ^bb5225:
    %2799 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2797, %2799 : !pdl.type -> ^bb5226, ^bb1
  ^bb5226:
    pdl_interp.check_type %2797 is f32 -> ^bb5227, ^bb1
  ^bb5227:
    %2800 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2800 : !pdl.value -> ^bb5228, ^bb1
  ^bb5228:
    %2801 = pdl_interp.get_value_type of %2800 : !pdl.type
    pdl_interp.are_equal %2797, %2801 : !pdl.type -> ^bb5229, ^bb1
  ^bb5229:
    pdl_interp.record_match @rewriters::@sin_sum(%2796, %2800, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.sin") -> ^bb1
  ^bb5192:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb5230, ^bb1
  ^bb5230:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5231, ^bb1
  ^bb5231:
    %2802 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2802 : !pdl.value -> ^bb5232, ^bb1
  ^bb5232:
    pdl_interp.are_equal %2802, %2 : !pdl.value -> ^bb5233, ^bb1
  ^bb5233:
    %2803 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2803 : !pdl.value -> ^bb5234, ^bb1
  ^bb5234:
    %2804 = pdl_interp.get_value_type of %2803 : !pdl.type
    %2805 = pdl_interp.get_value_type of %2802 : !pdl.type
    pdl_interp.are_equal %2804, %2805 : !pdl.type -> ^bb5235, ^bb1
  ^bb5235:
    %2806 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2804, %2806 : !pdl.type -> ^bb5236, ^bb1
  ^bb5236:
    pdl_interp.check_type %2804 is f32 -> ^bb5237, ^bb1
  ^bb5237:
    %2807 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2807 : !pdl.value -> ^bb5238, ^bb1
  ^bb5238:
    %2808 = pdl_interp.get_value_type of %2807 : !pdl.type
    pdl_interp.are_equal %2804, %2808 : !pdl.type -> ^bb5239, ^bb1
  ^bb5239:
    pdl_interp.record_match @rewriters::@sin_diff(%2803, %2807, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.sin") -> ^bb1
  ^bb5193:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb5240, ^bb1
  ^bb5240:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5241, ^bb1
  ^bb5241:
    %2809 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2809 : !pdl.value -> ^bb5242, ^bb1
  ^bb5242:
    pdl_interp.are_equal %2809, %2 : !pdl.value -> ^bb5243, ^bb1
  ^bb5243:
    %2810 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2810 : !pdl.value -> ^bb5244, ^bb1
  ^bb5244:
    %2811 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2811 : !pdl.value -> ^bb5245, ^bb1
  ^bb5245:
    %2812 = pdl_interp.get_defining_op of %2810 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %2812 : !pdl.operation -> ^bb5246, ^bb1
  ^bb5246:
    pdl_interp.check_operation_name of %2812 is "arith.constant" -> ^bb5247, ^bb1
  ^bb5247:
    pdl_interp.check_operand_count of %2812 is 0 -> ^bb5248, ^bb1
  ^bb5248:
    pdl_interp.check_result_count of %2812 is 1 -> ^bb5249, ^bb1
  ^bb5249:
    %2813 = pdl_interp.get_result 0 of %2812
    pdl_interp.is_not_null %2813 : !pdl.value -> ^bb5250, ^bb1
  ^bb5250:
    pdl_interp.are_equal %2813, %2810 : !pdl.value -> ^bb5251, ^bb1
  ^bb5251:
    %2814 = pdl_interp.get_attribute "value" of %2812
    pdl_interp.is_not_null %2814 : !pdl.attribute -> ^bb5252, ^bb1
  ^bb5252:
    pdl_interp.switch_attribute %2814 to [2.000000e+00 : f32, 3.000000e+00 : f32](^bb5253, ^bb5254) -> ^bb1
  ^bb5253:
    %2815 = pdl_interp.get_value_type of %2813 : !pdl.type
    %2816 = pdl_interp.get_value_type of %2809 : !pdl.type
    pdl_interp.are_equal %2815, %2816 : !pdl.type -> ^bb5255, ^bb1
  ^bb5255:
    %2817 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2815, %2817 : !pdl.type -> ^bb5256, ^bb1
  ^bb5256:
    pdl_interp.check_type %2815 is f32 -> ^bb5257, ^bb1
  ^bb5257:
    %2818 = pdl_interp.get_value_type of %2811 : !pdl.type
    pdl_interp.are_equal %2815, %2818 : !pdl.type -> ^bb5258, ^bb1
  ^bb5258:
    pdl_interp.record_match @rewriters::@sin_2(%2811, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.sin") -> ^bb1
  ^bb5254:
    %2819 = pdl_interp.get_value_type of %2813 : !pdl.type
    %2820 = pdl_interp.get_value_type of %2809 : !pdl.type
    pdl_interp.are_equal %2819, %2820 : !pdl.type -> ^bb5259, ^bb1
  ^bb5259:
    %2821 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2819, %2821 : !pdl.type -> ^bb5260, ^bb1
  ^bb5260:
    pdl_interp.check_type %2819 is f32 -> ^bb5261, ^bb1
  ^bb5261:
    %2822 = pdl_interp.get_value_type of %2811 : !pdl.type
    pdl_interp.are_equal %2819, %2822 : !pdl.type -> ^bb5262, ^bb1
  ^bb5262:
    pdl_interp.record_match @rewriters::@sin_3(%2811, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.sin") -> ^bb1
  ^bb5194:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb5263, ^bb1
  ^bb5263:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5264, ^bb1
  ^bb5264:
    %2823 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2823 : !pdl.value -> ^bb5265, ^bb1
  ^bb5265:
    pdl_interp.are_equal %2823, %2 : !pdl.value -> ^bb5266, ^bb1
  ^bb5266:
    %2824 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2824 : !pdl.value -> ^bb5267, ^bb1
  ^bb5267:
    %2825 = pdl_interp.get_value_type of %2824 : !pdl.type
    %2826 = pdl_interp.get_value_type of %2823 : !pdl.type
    pdl_interp.are_equal %2825, %2826 : !pdl.type -> ^bb5268, ^bb1
  ^bb5268:
    %2827 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2825, %2827 : !pdl.type -> ^bb5269, ^bb1
  ^bb5269:
    pdl_interp.check_type %2825 is f32 -> ^bb5270, ^bb1
  ^bb5270:
    pdl_interp.record_match @rewriters::@sin_acos(%2824, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.sin") -> ^bb1
  ^bb5195:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb5271, ^bb1
  ^bb5271:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5272, ^bb1
  ^bb5272:
    %2828 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2828 : !pdl.value -> ^bb5273, ^bb1
  ^bb5273:
    pdl_interp.are_equal %2828, %2 : !pdl.value -> ^bb5274, ^bb1
  ^bb5274:
    %2829 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2829 : !pdl.value -> ^bb5275, ^bb1
  ^bb5275:
    %2830 = pdl_interp.get_value_type of %2829 : !pdl.type
    %2831 = pdl_interp.get_value_type of %2828 : !pdl.type
    pdl_interp.are_equal %2830, %2831 : !pdl.type -> ^bb5276, ^bb1
  ^bb5276:
    %2832 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2830, %2832 : !pdl.type -> ^bb5277, ^bb1
  ^bb5277:
    pdl_interp.check_type %2830 is f32 -> ^bb5278, ^bb1
  ^bb5278:
    pdl_interp.record_match @rewriters::@sin_atan(%2829, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.sin") -> ^bb1
  ^bb38:
    pdl_interp.check_operand_count of %0 is 1 -> ^bb5279, ^bb1
  ^bb5279:
    pdl_interp.check_result_count of %0 is 1 -> ^bb5280, ^bb1
  ^bb5280:
    pdl_interp.is_not_null %2 : !pdl.value -> ^bb5281, ^bb1
  ^bb5281:
    pdl_interp.switch_operation_name of %3 to ["arith.constant", "arith.negf", "math.absf", "math.acos", "arith.addf", "arith.subf", "arith.mulf", "math.asin", "math.atan"](^bb5282, ^bb5283, ^bb5284, ^bb5285, ^bb5286, ^bb5287, ^bb5288, ^bb5289, ^bb5290) -> ^bb1
  ^bb5282:
    pdl_interp.check_operand_count of %3 is 0 -> ^bb5291, ^bb1
  ^bb5291:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5292, ^bb1
  ^bb5292:
    %2833 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2833 : !pdl.value -> ^bb5293, ^bb1
  ^bb5293:
    pdl_interp.are_equal %2833, %2 : !pdl.value -> ^bb5294, ^bb1
  ^bb5294:
    %2834 = pdl_interp.get_attribute "value" of %3
    pdl_interp.is_not_null %2834 : !pdl.attribute -> ^bb5295, ^bb1
  ^bb5295:
    pdl_interp.check_attribute %2834 is 0.000000e+00 : f32 -> ^bb5296, ^bb1
  ^bb5296:
    %2835 = pdl_interp.get_value_type of %2833 : !pdl.type
    %2836 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2835, %2836 : !pdl.type -> ^bb5297, ^bb1
  ^bb5297:
    pdl_interp.check_type %2835 is f32 -> ^bb5298, ^bb1
  ^bb5298:
    pdl_interp.record_match @rewriters::@cos_0(%0 : !pdl.operation) : benefit(1), loc([]), root("math.cos") -> ^bb1
  ^bb5283:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb5299, ^bb1
  ^bb5299:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5300, ^bb1
  ^bb5300:
    %2837 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2837 : !pdl.value -> ^bb5301, ^bb1
  ^bb5301:
    pdl_interp.are_equal %2837, %2 : !pdl.value -> ^bb5302, ^bb1
  ^bb5302:
    %2838 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2838 : !pdl.value -> ^bb5303, ^bb1
  ^bb5303:
    %2839 = pdl_interp.get_value_type of %2838 : !pdl.type
    %2840 = pdl_interp.get_value_type of %2837 : !pdl.type
    pdl_interp.are_equal %2839, %2840 : !pdl.type -> ^bb5304, ^bb1
  ^bb5304:
    %2841 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2839, %2841 : !pdl.type -> ^bb5305, ^bb1
  ^bb5305:
    pdl_interp.check_type %2839 is f32 -> ^bb5306, ^bb1
  ^bb5306:
    pdl_interp.record_match @rewriters::@cos_neg(%2838, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.cos") -> ^bb1
  ^bb5284:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb5307, ^bb1
  ^bb5307:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5308, ^bb1
  ^bb5308:
    %2842 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2842 : !pdl.value -> ^bb5309, ^bb1
  ^bb5309:
    pdl_interp.are_equal %2842, %2 : !pdl.value -> ^bb5310, ^bb1
  ^bb5310:
    %2843 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2843 : !pdl.value -> ^bb5311, ^bb1
  ^bb5311:
    %2844 = pdl_interp.get_value_type of %2843 : !pdl.type
    %2845 = pdl_interp.get_value_type of %2842 : !pdl.type
    pdl_interp.are_equal %2844, %2845 : !pdl.type -> ^bb5312, ^bb1
  ^bb5312:
    %2846 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2844, %2846 : !pdl.type -> ^bb5313, ^bb1
  ^bb5313:
    pdl_interp.check_type %2844 is f32 -> ^bb5314, ^bb1
  ^bb5314:
    pdl_interp.record_match @rewriters::@cos_fabs(%2843, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.cos") -> ^bb1
  ^bb5285:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb5315, ^bb1
  ^bb5315:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5316, ^bb1
  ^bb5316:
    %2847 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2847 : !pdl.value -> ^bb5317, ^bb1
  ^bb5317:
    pdl_interp.are_equal %2847, %2 : !pdl.value -> ^bb5318, ^bb1
  ^bb5318:
    %2848 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2848 : !pdl.value -> ^bb5319, ^bb1
  ^bb5319:
    %2849 = pdl_interp.get_value_type of %2848 : !pdl.type
    %2850 = pdl_interp.get_value_type of %2847 : !pdl.type
    pdl_interp.are_equal %2849, %2850 : !pdl.type -> ^bb5320, ^bb1
  ^bb5320:
    %2851 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2849, %2851 : !pdl.type -> ^bb5321, ^bb1
  ^bb5321:
    pdl_interp.check_type %2849 is f32 -> ^bb5322, ^bb1
  ^bb5322:
    pdl_interp.record_match @rewriters::@cos_acos(%2848, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.cos") -> ^bb1
  ^bb5286:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb5323, ^bb1
  ^bb5323:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5324, ^bb1
  ^bb5324:
    %2852 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2852 : !pdl.value -> ^bb5325, ^bb1
  ^bb5325:
    pdl_interp.are_equal %2852, %2 : !pdl.value -> ^bb5326, ^bb1
  ^bb5326:
    %2853 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2853 : !pdl.value -> ^bb5327, ^bb1
  ^bb5327:
    %2854 = pdl_interp.get_value_type of %2853 : !pdl.type
    %2855 = pdl_interp.get_value_type of %2852 : !pdl.type
    pdl_interp.are_equal %2854, %2855 : !pdl.type -> ^bb5328, ^bb1
  ^bb5328:
    %2856 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2854, %2856 : !pdl.type -> ^bb5329, ^bb1
  ^bb5329:
    pdl_interp.check_type %2854 is f32 -> ^bb5330, ^bb1
  ^bb5330:
    %2857 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2857 : !pdl.value -> ^bb5331, ^bb1
  ^bb5331:
    %2858 = pdl_interp.get_value_type of %2857 : !pdl.type
    pdl_interp.are_equal %2854, %2858 : !pdl.type -> ^bb5332, ^bb1
  ^bb5332:
    pdl_interp.record_match @rewriters::@cos_sum(%2853, %2857, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.cos") -> ^bb1
  ^bb5287:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb5333, ^bb1
  ^bb5333:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5334, ^bb1
  ^bb5334:
    %2859 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2859 : !pdl.value -> ^bb5335, ^bb1
  ^bb5335:
    pdl_interp.are_equal %2859, %2 : !pdl.value -> ^bb5336, ^bb1
  ^bb5336:
    %2860 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2860 : !pdl.value -> ^bb5337, ^bb1
  ^bb5337:
    %2861 = pdl_interp.get_value_type of %2860 : !pdl.type
    %2862 = pdl_interp.get_value_type of %2859 : !pdl.type
    pdl_interp.are_equal %2861, %2862 : !pdl.type -> ^bb5338, ^bb1
  ^bb5338:
    %2863 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2861, %2863 : !pdl.type -> ^bb5339, ^bb1
  ^bb5339:
    pdl_interp.check_type %2861 is f32 -> ^bb5340, ^bb1
  ^bb5340:
    %2864 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2864 : !pdl.value -> ^bb5341, ^bb1
  ^bb5341:
    %2865 = pdl_interp.get_value_type of %2864 : !pdl.type
    pdl_interp.are_equal %2861, %2865 : !pdl.type -> ^bb5342, ^bb1
  ^bb5342:
    pdl_interp.record_match @rewriters::@cos_diff(%2860, %2864, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.cos") -> ^bb1
  ^bb5288:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb5343, ^bb1
  ^bb5343:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5344, ^bb1
  ^bb5344:
    %2866 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2866 : !pdl.value -> ^bb5345, ^bb1
  ^bb5345:
    pdl_interp.are_equal %2866, %2 : !pdl.value -> ^bb5346, ^bb1
  ^bb5346:
    %2867 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2867 : !pdl.value -> ^bb5347, ^bb1
  ^bb5347:
    %2868 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2868 : !pdl.value -> ^bb5348, ^bb1
  ^bb5348:
    %2869 = pdl_interp.get_defining_op of %2867 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %2869 : !pdl.operation -> ^bb5349, ^bb1
  ^bb5349:
    pdl_interp.check_operation_name of %2869 is "arith.constant" -> ^bb5350, ^bb1
  ^bb5350:
    pdl_interp.check_operand_count of %2869 is 0 -> ^bb5351, ^bb1
  ^bb5351:
    pdl_interp.check_result_count of %2869 is 1 -> ^bb5352, ^bb1
  ^bb5352:
    %2870 = pdl_interp.get_result 0 of %2869
    pdl_interp.is_not_null %2870 : !pdl.value -> ^bb5353, ^bb1
  ^bb5353:
    pdl_interp.are_equal %2870, %2867 : !pdl.value -> ^bb5354, ^bb1
  ^bb5354:
    %2871 = pdl_interp.get_attribute "value" of %2869
    pdl_interp.is_not_null %2871 : !pdl.attribute -> ^bb5355, ^bb1
  ^bb5355:
    pdl_interp.switch_attribute %2871 to [2.000000e+00 : f32, 3.000000e+00 : f32](^bb5356, ^bb5357) -> ^bb1
  ^bb5356:
    %2872 = pdl_interp.get_value_type of %2870 : !pdl.type
    %2873 = pdl_interp.get_value_type of %2866 : !pdl.type
    pdl_interp.are_equal %2872, %2873 : !pdl.type -> ^bb5358, ^bb1
  ^bb5358:
    %2874 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2872, %2874 : !pdl.type -> ^bb5359, ^bb1
  ^bb5359:
    pdl_interp.check_type %2872 is f32 -> ^bb5360, ^bb1
  ^bb5360:
    %2875 = pdl_interp.get_value_type of %2868 : !pdl.type
    pdl_interp.are_equal %2872, %2875 : !pdl.type -> ^bb5361, ^bb1
  ^bb5361:
    pdl_interp.record_match @rewriters::@cos_2(%2868, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.cos") -> ^bb1
  ^bb5357:
    %2876 = pdl_interp.get_value_type of %2870 : !pdl.type
    %2877 = pdl_interp.get_value_type of %2866 : !pdl.type
    pdl_interp.are_equal %2876, %2877 : !pdl.type -> ^bb5362, ^bb1
  ^bb5362:
    %2878 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2876, %2878 : !pdl.type -> ^bb5363, ^bb1
  ^bb5363:
    pdl_interp.check_type %2876 is f32 -> ^bb5364, ^bb1
  ^bb5364:
    %2879 = pdl_interp.get_value_type of %2868 : !pdl.type
    pdl_interp.are_equal %2876, %2879 : !pdl.type -> ^bb5365, ^bb1
  ^bb5365:
    pdl_interp.record_match @rewriters::@cos_3(%2868, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.cos") -> ^bb1
  ^bb5289:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb5366, ^bb1
  ^bb5366:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5367, ^bb1
  ^bb5367:
    %2880 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2880 : !pdl.value -> ^bb5368, ^bb1
  ^bb5368:
    pdl_interp.are_equal %2880, %2 : !pdl.value -> ^bb5369, ^bb1
  ^bb5369:
    %2881 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2881 : !pdl.value -> ^bb5370, ^bb1
  ^bb5370:
    %2882 = pdl_interp.get_value_type of %2881 : !pdl.type
    %2883 = pdl_interp.get_value_type of %2880 : !pdl.type
    pdl_interp.are_equal %2882, %2883 : !pdl.type -> ^bb5371, ^bb1
  ^bb5371:
    %2884 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2882, %2884 : !pdl.type -> ^bb5372, ^bb1
  ^bb5372:
    pdl_interp.check_type %2882 is f32 -> ^bb5373, ^bb1
  ^bb5373:
    pdl_interp.record_match @rewriters::@cos_asin(%2881, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.cos") -> ^bb1
  ^bb5290:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb5374, ^bb1
  ^bb5374:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5375, ^bb1
  ^bb5375:
    %2885 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2885 : !pdl.value -> ^bb5376, ^bb1
  ^bb5376:
    pdl_interp.are_equal %2885, %2 : !pdl.value -> ^bb5377, ^bb1
  ^bb5377:
    %2886 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2886 : !pdl.value -> ^bb5378, ^bb1
  ^bb5378:
    %2887 = pdl_interp.get_value_type of %2886 : !pdl.type
    %2888 = pdl_interp.get_value_type of %2885 : !pdl.type
    pdl_interp.are_equal %2887, %2888 : !pdl.type -> ^bb5379, ^bb1
  ^bb5379:
    %2889 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2887, %2889 : !pdl.type -> ^bb5380, ^bb1
  ^bb5380:
    pdl_interp.check_type %2887 is f32 -> ^bb5381, ^bb1
  ^bb5381:
    pdl_interp.record_match @rewriters::@cos_atan(%2886, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.cos") -> ^bb1
  ^bb39:
    pdl_interp.check_operand_count of %0 is 1 -> ^bb5382, ^bb1
  ^bb5382:
    pdl_interp.check_result_count of %0 is 1 -> ^bb5383, ^bb1
  ^bb5383:
    pdl_interp.is_not_null %2 : !pdl.value -> ^bb5384, ^bb1
  ^bb5384:
    pdl_interp.switch_operation_name of %3 to ["arith.constant", "arith.negf", "math.atan", "arith.divf", "math.asin", "math.acos"](^bb5385, ^bb5386, ^bb5387, ^bb5388, ^bb5389, ^bb5390) -> ^bb1
  ^bb5385:
    pdl_interp.check_operand_count of %3 is 0 -> ^bb5391, ^bb1
  ^bb5391:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5392, ^bb1
  ^bb5392:
    %2890 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2890 : !pdl.value -> ^bb5393, ^bb1
  ^bb5393:
    pdl_interp.are_equal %2890, %2 : !pdl.value -> ^bb5394, ^bb1
  ^bb5394:
    %2891 = pdl_interp.get_attribute "value" of %3
    pdl_interp.is_not_null %2891 : !pdl.attribute -> ^bb5395, ^bb1
  ^bb5395:
    pdl_interp.check_attribute %2891 is 0.000000e+00 : f32 -> ^bb5396, ^bb1
  ^bb5396:
    %2892 = pdl_interp.get_value_type of %2890 : !pdl.type
    %2893 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2892, %2893 : !pdl.type -> ^bb5397, ^bb1
  ^bb5397:
    pdl_interp.check_type %2892 is f32 -> ^bb5398, ^bb1
  ^bb5398:
    pdl_interp.record_match @rewriters::@tan_0(%0 : !pdl.operation) : benefit(1), loc([]), root("math.tan") -> ^bb1
  ^bb5386:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb5399, ^bb1
  ^bb5399:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5400, ^bb1
  ^bb5400:
    %2894 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2894 : !pdl.value -> ^bb5401, ^bb1
  ^bb5401:
    pdl_interp.are_equal %2894, %2 : !pdl.value -> ^bb5402, ^bb1
  ^bb5402:
    %2895 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2895 : !pdl.value -> ^bb5403, ^bb1
  ^bb5403:
    %2896 = pdl_interp.get_value_type of %2895 : !pdl.type
    %2897 = pdl_interp.get_value_type of %2894 : !pdl.type
    pdl_interp.are_equal %2896, %2897 : !pdl.type -> ^bb5404, ^bb1
  ^bb5404:
    %2898 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2896, %2898 : !pdl.type -> ^bb5405, ^bb1
  ^bb5405:
    pdl_interp.check_type %2896 is f32 -> ^bb5406, ^bb1
  ^bb5406:
    pdl_interp.record_match @rewriters::@tan_neg(%2895, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.tan") -> ^bb1
  ^bb5387:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb5407, ^bb1
  ^bb5407:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5408, ^bb1
  ^bb5408:
    %2899 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2899 : !pdl.value -> ^bb5409, ^bb1
  ^bb5409:
    pdl_interp.are_equal %2899, %2 : !pdl.value -> ^bb5410, ^bb1
  ^bb5410:
    %2900 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2900 : !pdl.value -> ^bb5411, ^bb1
  ^bb5411:
    %2901 = pdl_interp.get_value_type of %2900 : !pdl.type
    %2902 = pdl_interp.get_value_type of %2899 : !pdl.type
    pdl_interp.are_equal %2901, %2902 : !pdl.type -> ^bb5412, ^bb1
  ^bb5412:
    %2903 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2901, %2903 : !pdl.type -> ^bb5413, ^bb1
  ^bb5413:
    pdl_interp.check_type %2901 is f32 -> ^bb5414, ^bb1
  ^bb5414:
    pdl_interp.record_match @rewriters::@tan_atan(%2900, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.tan") -> ^bb1
  ^bb5388:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb5415, ^bb1
  ^bb5415:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5416, ^bb1
  ^bb5416:
    %2904 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2904 : !pdl.value -> ^bb5417, ^bb1
  ^bb5417:
    pdl_interp.are_equal %2904, %2 : !pdl.value -> ^bb5418, ^bb1
  ^bb5418:
    %2905 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2905 : !pdl.value -> ^bb5419, ^bb1
  ^bb5419:
    %2906 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2906 : !pdl.value -> ^bb5420, ^bb5421
  ^bb5421:
    %2907 = pdl_interp.get_value_type of %2905 : !pdl.type
    %2908 = pdl_interp.get_value_type of %2904 : !pdl.type
    pdl_interp.are_equal %2907, %2908 : !pdl.type -> ^bb5422, ^bb1
  ^bb5422:
    %2909 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2907, %2909 : !pdl.type -> ^bb5423, ^bb1
  ^bb5423:
    pdl_interp.check_type %2907 is f32 -> ^bb5424, ^bb1
  ^bb5424:
    %2910 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2910 : !pdl.value -> ^bb5425, ^bb1
  ^bb5425:
    %2911 = pdl_interp.get_defining_op of %2910 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %2911 : !pdl.operation -> ^bb5426, ^bb1
  ^bb5426:
    pdl_interp.check_operation_name of %2911 is "arith.constant" -> ^bb5427, ^bb1
  ^bb5427:
    pdl_interp.check_operand_count of %2911 is 0 -> ^bb5428, ^bb1
  ^bb5428:
    pdl_interp.check_result_count of %2911 is 1 -> ^bb5429, ^bb1
  ^bb5429:
    %2912 = pdl_interp.get_result 0 of %2911
    pdl_interp.is_not_null %2912 : !pdl.value -> ^bb5430, ^bb1
  ^bb5430:
    pdl_interp.are_equal %2912, %2910 : !pdl.value -> ^bb5431, ^bb1
  ^bb5431:
    %2913 = pdl_interp.get_attribute "value" of %2911
    pdl_interp.is_not_null %2913 : !pdl.attribute -> ^bb5432, ^bb1
  ^bb5432:
    pdl_interp.check_attribute %2913 is 2.000000e+00 : f32 -> ^bb5433, ^bb1
  ^bb5433:
    %2914 = pdl_interp.get_value_type of %2912 : !pdl.type
    pdl_interp.are_equal %2914, %2907 : !pdl.type -> ^bb5434, ^bb1
  ^bb5434:
    pdl_interp.record_match @rewriters::@hang_0p_tan_rev(%2905, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.tan") -> ^bb1
  ^bb5420:
    %2915 = pdl_interp.get_defining_op of %2906 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %2915 : !pdl.operation -> ^bb5435, ^bb5421
  ^bb5435:
    %2916 = pdl_interp.get_defining_op of %2905 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %2916 : !pdl.operation -> ^bb5436, ^bb5421
  ^bb5436:
    pdl_interp.check_operation_name of %2915 is "arith.constant" -> ^bb5437, ^bb5421
  ^bb5437:
    pdl_interp.check_operand_count of %2915 is 0 -> ^bb5438, ^bb5421
  ^bb5438:
    pdl_interp.check_result_count of %2915 is 1 -> ^bb5439, ^bb5421
  ^bb5439:
    %2917 = pdl_interp.get_result 0 of %2915
    pdl_interp.is_not_null %2917 : !pdl.value -> ^bb5440, ^bb5421
  ^bb5440:
    pdl_interp.are_equal %2917, %2906 : !pdl.value -> ^bb5441, ^bb5421
  ^bb5441:
    pdl_interp.check_operation_name of %2916 is "arith.negf" -> ^bb5442, ^bb5421
  ^bb5442:
    pdl_interp.check_operand_count of %2916 is 1 -> ^bb5443, ^bb5421
  ^bb5443:
    pdl_interp.check_result_count of %2916 is 1 -> ^bb5444, ^bb5421
  ^bb5444:
    %2918 = pdl_interp.get_result 0 of %2916
    pdl_interp.is_not_null %2918 : !pdl.value -> ^bb5445, ^bb5421
  ^bb5445:
    pdl_interp.are_equal %2918, %2905 : !pdl.value -> ^bb5446, ^bb5421
  ^bb5446:
    %2919 = pdl_interp.get_operand 0 of %2916
    pdl_interp.is_not_null %2919 : !pdl.value -> ^bb5447, ^bb5421
  ^bb5447:
    %2920 = pdl_interp.get_value_type of %2919 : !pdl.type
    %2921 = pdl_interp.get_value_type of %2918 : !pdl.type
    pdl_interp.are_equal %2920, %2921 : !pdl.type -> ^bb5448, ^bb5421
  ^bb5448:
    %2922 = pdl_interp.get_value_type of %2904 : !pdl.type
    pdl_interp.are_equal %2920, %2922 : !pdl.type -> ^bb5449, ^bb5421
  ^bb5449:
    %2923 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2920, %2923 : !pdl.type -> ^bb5450, ^bb5421
  ^bb5450:
    pdl_interp.check_type %2920 is f32 -> ^bb5451, ^bb5421
  ^bb5451:
    %2924 = pdl_interp.get_value_type of %2917 : !pdl.type
    pdl_interp.are_equal %2920, %2924 : !pdl.type -> ^bb5452, ^bb5421
  ^bb5452:
    %2925 = pdl_interp.get_attribute "value" of %2915
    pdl_interp.is_not_null %2925 : !pdl.attribute -> ^bb5453, ^bb5421
  ^bb5453:
    pdl_interp.check_attribute %2925 is 2.000000e+00 : f32 -> ^bb5454, ^bb5421
  ^bb5454:
    pdl_interp.record_match @rewriters::@hang_0m_tan_rev(%2919, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.tan") -> ^bb5421
  ^bb5389:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb5455, ^bb1
  ^bb5455:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5456, ^bb1
  ^bb5456:
    %2926 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2926 : !pdl.value -> ^bb5457, ^bb1
  ^bb5457:
    pdl_interp.are_equal %2926, %2 : !pdl.value -> ^bb5458, ^bb1
  ^bb5458:
    %2927 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2927 : !pdl.value -> ^bb5459, ^bb1
  ^bb5459:
    %2928 = pdl_interp.get_value_type of %2927 : !pdl.type
    %2929 = pdl_interp.get_value_type of %2926 : !pdl.type
    pdl_interp.are_equal %2928, %2929 : !pdl.type -> ^bb5460, ^bb1
  ^bb5460:
    %2930 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2928, %2930 : !pdl.type -> ^bb5461, ^bb1
  ^bb5461:
    pdl_interp.check_type %2928 is f32 -> ^bb5462, ^bb1
  ^bb5462:
    pdl_interp.record_match @rewriters::@tan_asin(%2927, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.tan") -> ^bb1
  ^bb5390:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb5463, ^bb1
  ^bb5463:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5464, ^bb1
  ^bb5464:
    %2931 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2931 : !pdl.value -> ^bb5465, ^bb1
  ^bb5465:
    pdl_interp.are_equal %2931, %2 : !pdl.value -> ^bb5466, ^bb1
  ^bb5466:
    %2932 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2932 : !pdl.value -> ^bb5467, ^bb1
  ^bb5467:
    %2933 = pdl_interp.get_value_type of %2932 : !pdl.type
    %2934 = pdl_interp.get_value_type of %2931 : !pdl.type
    pdl_interp.are_equal %2933, %2934 : !pdl.type -> ^bb5468, ^bb1
  ^bb5468:
    %2935 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2933, %2935 : !pdl.type -> ^bb5469, ^bb1
  ^bb5469:
    pdl_interp.check_type %2933 is f32 -> ^bb5470, ^bb1
  ^bb5470:
    pdl_interp.record_match @rewriters::@tan_acos(%2932, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.tan") -> ^bb1
  ^bb40:
    pdl_interp.check_operand_count of %0 is 2 -> ^bb5471, ^bb1
  ^bb5471:
    pdl_interp.check_result_count of %0 is 1 -> ^bb5472, ^bb1
  ^bb5472:
    %2936 = pdl_interp.get_operand 1 of %0
    %2937 = pdl_interp.get_defining_op of %2936 : !pdl.value {position = "root.operand[1].defining_op"}
    pdl_interp.is_not_null %2937 : !pdl.operation -> ^bb5473, ^bb1
  ^bb5473:
    pdl_interp.is_not_null %2 : !pdl.value -> ^bb5474, ^bb1
  ^bb5474:
    pdl_interp.switch_operation_name of %3 to ["arith.subf", "arith.addf"](^bb5475, ^bb5476) -> ^bb1
  ^bb5475:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb5477, ^bb1
  ^bb5477:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5478, ^bb1
  ^bb5478:
    %2938 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2938 : !pdl.value -> ^bb5479, ^bb1
  ^bb5479:
    pdl_interp.are_equal %2938, %2 : !pdl.value -> ^bb5480, ^bb1
  ^bb5480:
    %2939 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2939 : !pdl.value -> ^bb5481, ^bb1
  ^bb5481:
    pdl_interp.is_not_null %2936 : !pdl.value -> ^bb5482, ^bb1
  ^bb5482:
    %2940 = pdl_interp.get_value_type of %2939 : !pdl.type
    %2941 = pdl_interp.get_value_type of %2938 : !pdl.type
    pdl_interp.are_equal %2940, %2941 : !pdl.type -> ^bb5483, ^bb1
  ^bb5483:
    %2942 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2940, %2942 : !pdl.type -> ^bb5484, ^bb1
  ^bb5484:
    pdl_interp.check_type %2940 is f32 -> ^bb5485, ^bb1
  ^bb5485:
    pdl_interp.check_operation_name of %2937 is "arith.addf" -> ^bb5486, ^bb1
  ^bb5486:
    pdl_interp.check_operand_count of %2937 is 2 -> ^bb5487, ^bb1
  ^bb5487:
    pdl_interp.check_result_count of %2937 is 1 -> ^bb5488, ^bb1
  ^bb5488:
    %2943 = pdl_interp.get_result 0 of %2937
    pdl_interp.is_not_null %2943 : !pdl.value -> ^bb5489, ^bb1
  ^bb5489:
    pdl_interp.are_equal %2943, %2936 : !pdl.value -> ^bb5490, ^bb1
  ^bb5490:
    %2944 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2944 : !pdl.value -> ^bb5491, ^bb1
  ^bb5491:
    %2945 = pdl_interp.get_operand 0 of %2937
    pdl_interp.is_not_null %2945 : !pdl.value -> ^bb5492, ^bb1
  ^bb5492:
    %2946 = pdl_interp.get_defining_op of %2945 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %2946 : !pdl.operation -> ^bb5493, ^bb1
  ^bb5493:
    %2947 = pdl_interp.get_operand 1 of %2937
    %2948 = pdl_interp.get_defining_op of %2947 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %2948 : !pdl.operation -> ^bb5494, ^bb1
  ^bb5494:
    pdl_interp.is_not_null %2947 : !pdl.value -> ^bb5495, ^bb1
  ^bb5495:
    %2949 = pdl_interp.get_value_type of %2943 : !pdl.type
    pdl_interp.are_equal %2940, %2949 : !pdl.type -> ^bb5496, ^bb1
  ^bb5496:
    %2950 = pdl_interp.get_value_type of %2944 : !pdl.type
    pdl_interp.are_equal %2940, %2950 : !pdl.type -> ^bb5497, ^bb1
  ^bb5497:
    pdl_interp.check_operation_name of %2946 is "arith.constant" -> ^bb5498, ^bb1
  ^bb5498:
    pdl_interp.check_operand_count of %2946 is 0 -> ^bb5499, ^bb1
  ^bb5499:
    pdl_interp.check_result_count of %2946 is 1 -> ^bb5500, ^bb1
  ^bb5500:
    %2951 = pdl_interp.get_result 0 of %2946
    pdl_interp.is_not_null %2951 : !pdl.value -> ^bb5501, ^bb1
  ^bb5501:
    pdl_interp.are_equal %2951, %2945 : !pdl.value -> ^bb5502, ^bb1
  ^bb5502:
    pdl_interp.check_operation_name of %2948 is "arith.mulf" -> ^bb5503, ^bb1
  ^bb5503:
    pdl_interp.check_operand_count of %2948 is 2 -> ^bb5504, ^bb1
  ^bb5504:
    pdl_interp.check_result_count of %2948 is 1 -> ^bb5505, ^bb1
  ^bb5505:
    %2952 = pdl_interp.get_result 0 of %2948
    pdl_interp.is_not_null %2952 : !pdl.value -> ^bb5506, ^bb1
  ^bb5506:
    pdl_interp.are_equal %2952, %2947 : !pdl.value -> ^bb5507, ^bb1
  ^bb5507:
    %2953 = pdl_interp.get_attribute "value" of %2946
    pdl_interp.is_not_null %2953 : !pdl.attribute -> ^bb5508, ^bb1
  ^bb5508:
    pdl_interp.check_attribute %2953 is 1.000000e+00 : f32 -> ^bb5509, ^bb1
  ^bb5509:
    %2954 = pdl_interp.get_value_type of %2952 : !pdl.type
    pdl_interp.are_equal %2954, %2940 : !pdl.type -> ^bb5510, ^bb1
  ^bb5510:
    %2955 = pdl_interp.get_value_type of %2951 : !pdl.type
    pdl_interp.are_equal %2955, %2940 : !pdl.type -> ^bb5511, ^bb1
  ^bb5511:
    %2956 = pdl_interp.get_operand 0 of %2948
    pdl_interp.are_equal %2956, %2939 : !pdl.value -> ^bb5512, ^bb1
  ^bb5512:
    %2957 = pdl_interp.get_operand 1 of %2948
    pdl_interp.are_equal %2957, %2944 : !pdl.value -> ^bb5513, ^bb1
  ^bb5513:
    pdl_interp.record_match @rewriters::@diff_atan_rev(%2939, %2944, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.atan2") -> ^bb1
  ^bb5476:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb5514, ^bb1
  ^bb5514:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5515, ^bb1
  ^bb5515:
    %2958 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2958 : !pdl.value -> ^bb5516, ^bb1
  ^bb5516:
    pdl_interp.are_equal %2958, %2 : !pdl.value -> ^bb5517, ^bb1
  ^bb5517:
    %2959 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2959 : !pdl.value -> ^bb5518, ^bb1
  ^bb5518:
    pdl_interp.is_not_null %2936 : !pdl.value -> ^bb5519, ^bb1
  ^bb5519:
    %2960 = pdl_interp.get_value_type of %2959 : !pdl.type
    %2961 = pdl_interp.get_value_type of %2958 : !pdl.type
    pdl_interp.are_equal %2960, %2961 : !pdl.type -> ^bb5520, ^bb1
  ^bb5520:
    %2962 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2960, %2962 : !pdl.type -> ^bb5521, ^bb1
  ^bb5521:
    pdl_interp.check_type %2960 is f32 -> ^bb5522, ^bb1
  ^bb5522:
    pdl_interp.check_operation_name of %2937 is "arith.subf" -> ^bb5523, ^bb1
  ^bb5523:
    pdl_interp.check_operand_count of %2937 is 2 -> ^bb5524, ^bb1
  ^bb5524:
    pdl_interp.check_result_count of %2937 is 1 -> ^bb5525, ^bb1
  ^bb5525:
    %2963 = pdl_interp.get_result 0 of %2937
    pdl_interp.is_not_null %2963 : !pdl.value -> ^bb5526, ^bb1
  ^bb5526:
    pdl_interp.are_equal %2963, %2936 : !pdl.value -> ^bb5527, ^bb1
  ^bb5527:
    %2964 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2964 : !pdl.value -> ^bb5528, ^bb1
  ^bb5528:
    %2965 = pdl_interp.get_operand 0 of %2937
    pdl_interp.is_not_null %2965 : !pdl.value -> ^bb5529, ^bb1
  ^bb5529:
    %2966 = pdl_interp.get_defining_op of %2965 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %2966 : !pdl.operation -> ^bb5530, ^bb1
  ^bb5530:
    %2967 = pdl_interp.get_operand 1 of %2937
    %2968 = pdl_interp.get_defining_op of %2967 : !pdl.value {position = "root.operand[1].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %2968 : !pdl.operation -> ^bb5531, ^bb1
  ^bb5531:
    pdl_interp.is_not_null %2967 : !pdl.value -> ^bb5532, ^bb1
  ^bb5532:
    %2969 = pdl_interp.get_value_type of %2963 : !pdl.type
    pdl_interp.are_equal %2960, %2969 : !pdl.type -> ^bb5533, ^bb1
  ^bb5533:
    %2970 = pdl_interp.get_value_type of %2964 : !pdl.type
    pdl_interp.are_equal %2960, %2970 : !pdl.type -> ^bb5534, ^bb1
  ^bb5534:
    pdl_interp.check_operation_name of %2966 is "arith.constant" -> ^bb5535, ^bb1
  ^bb5535:
    pdl_interp.check_operand_count of %2966 is 0 -> ^bb5536, ^bb1
  ^bb5536:
    pdl_interp.check_result_count of %2966 is 1 -> ^bb5537, ^bb1
  ^bb5537:
    %2971 = pdl_interp.get_result 0 of %2966
    pdl_interp.is_not_null %2971 : !pdl.value -> ^bb5538, ^bb1
  ^bb5538:
    pdl_interp.are_equal %2971, %2965 : !pdl.value -> ^bb5539, ^bb1
  ^bb5539:
    pdl_interp.check_operation_name of %2968 is "arith.mulf" -> ^bb5540, ^bb1
  ^bb5540:
    pdl_interp.check_operand_count of %2968 is 2 -> ^bb5541, ^bb1
  ^bb5541:
    pdl_interp.check_result_count of %2968 is 1 -> ^bb5542, ^bb1
  ^bb5542:
    %2972 = pdl_interp.get_result 0 of %2968
    pdl_interp.is_not_null %2972 : !pdl.value -> ^bb5543, ^bb1
  ^bb5543:
    pdl_interp.are_equal %2972, %2967 : !pdl.value -> ^bb5544, ^bb1
  ^bb5544:
    %2973 = pdl_interp.get_attribute "value" of %2966
    pdl_interp.is_not_null %2973 : !pdl.attribute -> ^bb5545, ^bb1
  ^bb5545:
    pdl_interp.check_attribute %2973 is 1.000000e+00 : f32 -> ^bb5546, ^bb1
  ^bb5546:
    %2974 = pdl_interp.get_value_type of %2972 : !pdl.type
    pdl_interp.are_equal %2974, %2960 : !pdl.type -> ^bb5547, ^bb1
  ^bb5547:
    %2975 = pdl_interp.get_value_type of %2971 : !pdl.type
    pdl_interp.are_equal %2975, %2960 : !pdl.type -> ^bb5548, ^bb1
  ^bb5548:
    %2976 = pdl_interp.get_operand 0 of %2968
    pdl_interp.are_equal %2976, %2959 : !pdl.value -> ^bb5549, ^bb1
  ^bb5549:
    %2977 = pdl_interp.get_operand 1 of %2968
    pdl_interp.are_equal %2977, %2964 : !pdl.value -> ^bb5550, ^bb1
  ^bb5550:
    pdl_interp.record_match @rewriters::@sum_atan_rev(%2959, %2964, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.atan2") -> ^bb1
  ^bb41:
    pdl_interp.check_operand_count of %0 is 1 -> ^bb5551, ^bb1
  ^bb5551:
    pdl_interp.check_result_count of %0 is 1 -> ^bb5552, ^bb1
  ^bb5552:
    pdl_interp.is_not_null %2 : !pdl.value -> ^bb5553, ^bb1
  ^bb5553:
    pdl_interp.check_operation_name of %3 is "arith.negf" -> ^bb5554, ^bb1
  ^bb5554:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb5555, ^bb1
  ^bb5555:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5556, ^bb1
  ^bb5556:
    %2978 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2978 : !pdl.value -> ^bb5557, ^bb1
  ^bb5557:
    pdl_interp.are_equal %2978, %2 : !pdl.value -> ^bb5558, ^bb1
  ^bb5558:
    %2979 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2979 : !pdl.value -> ^bb5559, ^bb1
  ^bb5559:
    %2980 = pdl_interp.get_value_type of %2979 : !pdl.type
    %2981 = pdl_interp.get_value_type of %2978 : !pdl.type
    pdl_interp.are_equal %2980, %2981 : !pdl.type -> ^bb5560, ^bb1
  ^bb5560:
    %2982 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2980, %2982 : !pdl.type -> ^bb5561, ^bb1
  ^bb5561:
    pdl_interp.check_type %2980 is f32 -> ^bb5562, ^bb1
  ^bb5562:
    pdl_interp.record_match @rewriters::@asin_neg(%2979, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.asin") -> ^bb1
  ^bb42:
    pdl_interp.check_operand_count of %0 is 1 -> ^bb5563, ^bb1
  ^bb5563:
    pdl_interp.check_result_count of %0 is 1 -> ^bb5564, ^bb1
  ^bb5564:
    pdl_interp.is_not_null %2 : !pdl.value -> ^bb5565, ^bb1
  ^bb5565:
    pdl_interp.check_operation_name of %3 is "arith.negf" -> ^bb5566, ^bb1
  ^bb5566:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb5567, ^bb1
  ^bb5567:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5568, ^bb1
  ^bb5568:
    %2983 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2983 : !pdl.value -> ^bb5569, ^bb1
  ^bb5569:
    pdl_interp.are_equal %2983, %2 : !pdl.value -> ^bb5570, ^bb1
  ^bb5570:
    %2984 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2984 : !pdl.value -> ^bb5571, ^bb1
  ^bb5571:
    %2985 = pdl_interp.get_value_type of %2984 : !pdl.type
    %2986 = pdl_interp.get_value_type of %2983 : !pdl.type
    pdl_interp.are_equal %2985, %2986 : !pdl.type -> ^bb5572, ^bb1
  ^bb5572:
    %2987 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2985, %2987 : !pdl.type -> ^bb5573, ^bb1
  ^bb5573:
    pdl_interp.check_type %2985 is f32 -> ^bb5574, ^bb1
  ^bb5574:
    pdl_interp.record_match @rewriters::@atan_neg(%2984, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.atan") -> ^bb1
  ^bb43:
    pdl_interp.check_operand_count of %0 is 1 -> ^bb5575, ^bb1
  ^bb5575:
    pdl_interp.check_result_count of %0 is 1 -> ^bb5576, ^bb1
  ^bb5576:
    pdl_interp.is_not_null %2 : !pdl.value -> ^bb5577, ^bb1
  ^bb5577:
    pdl_interp.switch_operation_name of %3 to ["arith.addf", "arith.subf", "arith.mulf", "arith.divf", "arith.negf", "arith.constant", "math.asinh", "math.acosh", "math.atanh"](^bb5578, ^bb5579, ^bb5580, ^bb5581, ^bb5582, ^bb5583, ^bb5584, ^bb5585, ^bb5586) -> ^bb1
  ^bb5578:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb5587, ^bb1
  ^bb5587:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5588, ^bb1
  ^bb5588:
    %2988 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2988 : !pdl.value -> ^bb5589, ^bb1
  ^bb5589:
    pdl_interp.are_equal %2988, %2 : !pdl.value -> ^bb5590, ^bb1
  ^bb5590:
    %2989 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2989 : !pdl.value -> ^bb5591, ^bb1
  ^bb5591:
    %2990 = pdl_interp.get_value_type of %2989 : !pdl.type
    %2991 = pdl_interp.get_value_type of %2988 : !pdl.type
    pdl_interp.are_equal %2990, %2991 : !pdl.type -> ^bb5592, ^bb1
  ^bb5592:
    %2992 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2990, %2992 : !pdl.type -> ^bb5593, ^bb1
  ^bb5593:
    pdl_interp.check_type %2990 is f32 -> ^bb5594, ^bb1
  ^bb5594:
    %2993 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %2993 : !pdl.value -> ^bb5595, ^bb1
  ^bb5595:
    %2994 = pdl_interp.get_value_type of %2993 : !pdl.type
    pdl_interp.are_equal %2990, %2994 : !pdl.type -> ^bb5596, ^bb1
  ^bb5596:
    pdl_interp.record_match @rewriters::@cosh_sum(%2989, %2993, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.cosh") -> ^bb1
  ^bb5579:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb5597, ^bb1
  ^bb5597:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5598, ^bb1
  ^bb5598:
    %2995 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %2995 : !pdl.value -> ^bb5599, ^bb1
  ^bb5599:
    pdl_interp.are_equal %2995, %2 : !pdl.value -> ^bb5600, ^bb1
  ^bb5600:
    %2996 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %2996 : !pdl.value -> ^bb5601, ^bb1
  ^bb5601:
    %2997 = pdl_interp.get_value_type of %2996 : !pdl.type
    %2998 = pdl_interp.get_value_type of %2995 : !pdl.type
    pdl_interp.are_equal %2997, %2998 : !pdl.type -> ^bb5602, ^bb1
  ^bb5602:
    %2999 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %2997, %2999 : !pdl.type -> ^bb5603, ^bb1
  ^bb5603:
    pdl_interp.check_type %2997 is f32 -> ^bb5604, ^bb1
  ^bb5604:
    %3000 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %3000 : !pdl.value -> ^bb5605, ^bb1
  ^bb5605:
    %3001 = pdl_interp.get_value_type of %3000 : !pdl.type
    pdl_interp.are_equal %2997, %3001 : !pdl.type -> ^bb5606, ^bb1
  ^bb5606:
    pdl_interp.record_match @rewriters::@cosh_diff(%2996, %3000, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.cosh") -> ^bb1
  ^bb5580:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb5607, ^bb1
  ^bb5607:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5608, ^bb1
  ^bb5608:
    %3002 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %3002 : !pdl.value -> ^bb5609, ^bb1
  ^bb5609:
    pdl_interp.are_equal %3002, %2 : !pdl.value -> ^bb5610, ^bb1
  ^bb5610:
    %3003 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %3003 : !pdl.value -> ^bb5611, ^bb1
  ^bb5611:
    %3004 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %3004 : !pdl.value -> ^bb5612, ^bb1
  ^bb5612:
    %3005 = pdl_interp.get_defining_op of %3003 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %3005 : !pdl.operation -> ^bb5613, ^bb1
  ^bb5613:
    pdl_interp.check_operation_name of %3005 is "arith.constant" -> ^bb5614, ^bb1
  ^bb5614:
    pdl_interp.check_operand_count of %3005 is 0 -> ^bb5615, ^bb1
  ^bb5615:
    pdl_interp.check_result_count of %3005 is 1 -> ^bb5616, ^bb1
  ^bb5616:
    %3006 = pdl_interp.get_result 0 of %3005
    pdl_interp.is_not_null %3006 : !pdl.value -> ^bb5617, ^bb1
  ^bb5617:
    pdl_interp.are_equal %3006, %3003 : !pdl.value -> ^bb5618, ^bb1
  ^bb5618:
    %3007 = pdl_interp.get_attribute "value" of %3005
    pdl_interp.is_not_null %3007 : !pdl.attribute -> ^bb5619, ^bb1
  ^bb5619:
    pdl_interp.check_attribute %3007 is 2.000000e+00 : f32 -> ^bb5620, ^bb1
  ^bb5620:
    %3008 = pdl_interp.get_value_type of %3006 : !pdl.type
    %3009 = pdl_interp.get_value_type of %3002 : !pdl.type
    pdl_interp.are_equal %3008, %3009 : !pdl.type -> ^bb5621, ^bb1
  ^bb5621:
    %3010 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3008, %3010 : !pdl.type -> ^bb5622, ^bb1
  ^bb5622:
    pdl_interp.check_type %3008 is f32 -> ^bb5623, ^bb1
  ^bb5623:
    %3011 = pdl_interp.get_value_type of %3004 : !pdl.type
    pdl_interp.are_equal %3008, %3011 : !pdl.type -> ^bb5624, ^bb1
  ^bb5624:
    pdl_interp.record_match @rewriters::@cosh_2(%3004, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.cosh") -> ^bb1
  ^bb5581:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb5625, ^bb1
  ^bb5625:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5626, ^bb1
  ^bb5626:
    %3012 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %3012 : !pdl.value -> ^bb5627, ^bb1
  ^bb5627:
    pdl_interp.are_equal %3012, %2 : !pdl.value -> ^bb5628, ^bb1
  ^bb5628:
    %3013 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %3013 : !pdl.value -> ^bb5629, ^bb1
  ^bb5629:
    %3014 = pdl_interp.get_value_type of %3013 : !pdl.type
    %3015 = pdl_interp.get_value_type of %3012 : !pdl.type
    pdl_interp.are_equal %3014, %3015 : !pdl.type -> ^bb5630, ^bb1
  ^bb5630:
    %3016 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3014, %3016 : !pdl.type -> ^bb5631, ^bb1
  ^bb5631:
    pdl_interp.check_type %3014 is f32 -> ^bb5632, ^bb1
  ^bb5632:
    %3017 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %3017 : !pdl.value -> ^bb5633, ^bb1
  ^bb5633:
    %3018 = pdl_interp.get_defining_op of %3017 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %3018 : !pdl.operation -> ^bb5634, ^bb1
  ^bb5634:
    pdl_interp.check_operation_name of %3018 is "arith.constant" -> ^bb5635, ^bb1
  ^bb5635:
    pdl_interp.check_operand_count of %3018 is 0 -> ^bb5636, ^bb1
  ^bb5636:
    pdl_interp.check_result_count of %3018 is 1 -> ^bb5637, ^bb1
  ^bb5637:
    %3019 = pdl_interp.get_result 0 of %3018
    pdl_interp.is_not_null %3019 : !pdl.value -> ^bb5638, ^bb1
  ^bb5638:
    pdl_interp.are_equal %3019, %3017 : !pdl.value -> ^bb5639, ^bb1
  ^bb5639:
    %3020 = pdl_interp.get_attribute "value" of %3018
    pdl_interp.is_not_null %3020 : !pdl.attribute -> ^bb5640, ^bb1
  ^bb5640:
    pdl_interp.check_attribute %3020 is 2.000000e+00 : f32 -> ^bb5641, ^bb1
  ^bb5641:
    %3021 = pdl_interp.get_value_type of %3019 : !pdl.type
    pdl_interp.are_equal %3021, %3014 : !pdl.type -> ^bb5642, ^bb1
  ^bb5642:
    pdl_interp.record_match @rewriters::@cosh_1div2(%3013, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.cosh") -> ^bb1
  ^bb5582:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb5643, ^bb1
  ^bb5643:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5644, ^bb1
  ^bb5644:
    %3022 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %3022 : !pdl.value -> ^bb5645, ^bb1
  ^bb5645:
    pdl_interp.are_equal %3022, %2 : !pdl.value -> ^bb5646, ^bb1
  ^bb5646:
    %3023 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %3023 : !pdl.value -> ^bb5647, ^bb1
  ^bb5647:
    %3024 = pdl_interp.get_value_type of %3023 : !pdl.type
    %3025 = pdl_interp.get_value_type of %3022 : !pdl.type
    pdl_interp.are_equal %3024, %3025 : !pdl.type -> ^bb5648, ^bb1
  ^bb5648:
    %3026 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3024, %3026 : !pdl.type -> ^bb5649, ^bb1
  ^bb5649:
    pdl_interp.check_type %3024 is f32 -> ^bb5650, ^bb1
  ^bb5650:
    pdl_interp.record_match @rewriters::@cosh_neg(%3023, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.cosh") -> ^bb1
  ^bb5583:
    pdl_interp.check_operand_count of %3 is 0 -> ^bb5651, ^bb1
  ^bb5651:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5652, ^bb1
  ^bb5652:
    %3027 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %3027 : !pdl.value -> ^bb5653, ^bb1
  ^bb5653:
    pdl_interp.are_equal %3027, %2 : !pdl.value -> ^bb5654, ^bb1
  ^bb5654:
    %3028 = pdl_interp.get_attribute "value" of %3
    pdl_interp.is_not_null %3028 : !pdl.attribute -> ^bb5655, ^bb1
  ^bb5655:
    pdl_interp.check_attribute %3028 is 0.000000e+00 : f32 -> ^bb5656, ^bb1
  ^bb5656:
    %3029 = pdl_interp.get_value_type of %3027 : !pdl.type
    %3030 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3029, %3030 : !pdl.type -> ^bb5657, ^bb1
  ^bb5657:
    pdl_interp.check_type %3029 is f32 -> ^bb5658, ^bb1
  ^bb5658:
    pdl_interp.record_match @rewriters::@cosh_0(%0 : !pdl.operation) : benefit(1), loc([]), root("math.cosh") -> ^bb1
  ^bb5584:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb5659, ^bb1
  ^bb5659:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5660, ^bb1
  ^bb5660:
    %3031 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %3031 : !pdl.value -> ^bb5661, ^bb1
  ^bb5661:
    pdl_interp.are_equal %3031, %2 : !pdl.value -> ^bb5662, ^bb1
  ^bb5662:
    %3032 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %3032 : !pdl.value -> ^bb5663, ^bb1
  ^bb5663:
    %3033 = pdl_interp.get_value_type of %3032 : !pdl.type
    %3034 = pdl_interp.get_value_type of %3031 : !pdl.type
    pdl_interp.are_equal %3033, %3034 : !pdl.type -> ^bb5664, ^bb1
  ^bb5664:
    %3035 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3033, %3035 : !pdl.type -> ^bb5665, ^bb1
  ^bb5665:
    pdl_interp.check_type %3033 is f32 -> ^bb5666, ^bb1
  ^bb5666:
    pdl_interp.record_match @rewriters::@cosh_asinh(%3032, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.cosh") -> ^bb1
  ^bb5585:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb5667, ^bb1
  ^bb5667:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5668, ^bb1
  ^bb5668:
    %3036 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %3036 : !pdl.value -> ^bb5669, ^bb1
  ^bb5669:
    pdl_interp.are_equal %3036, %2 : !pdl.value -> ^bb5670, ^bb1
  ^bb5670:
    %3037 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %3037 : !pdl.value -> ^bb5671, ^bb1
  ^bb5671:
    %3038 = pdl_interp.get_value_type of %3037 : !pdl.type
    %3039 = pdl_interp.get_value_type of %3036 : !pdl.type
    pdl_interp.are_equal %3038, %3039 : !pdl.type -> ^bb5672, ^bb1
  ^bb5672:
    %3040 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3038, %3040 : !pdl.type -> ^bb5673, ^bb1
  ^bb5673:
    pdl_interp.check_type %3038 is f32 -> ^bb5674, ^bb1
  ^bb5674:
    pdl_interp.record_match @rewriters::@cosh_acosh(%3037, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.cosh") -> ^bb1
  ^bb5586:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb5675, ^bb1
  ^bb5675:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5676, ^bb1
  ^bb5676:
    %3041 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %3041 : !pdl.value -> ^bb5677, ^bb1
  ^bb5677:
    pdl_interp.are_equal %3041, %2 : !pdl.value -> ^bb5678, ^bb1
  ^bb5678:
    %3042 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %3042 : !pdl.value -> ^bb5679, ^bb1
  ^bb5679:
    %3043 = pdl_interp.get_value_type of %3042 : !pdl.type
    %3044 = pdl_interp.get_value_type of %3041 : !pdl.type
    pdl_interp.are_equal %3043, %3044 : !pdl.type -> ^bb5680, ^bb1
  ^bb5680:
    %3045 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3043, %3045 : !pdl.type -> ^bb5681, ^bb1
  ^bb5681:
    pdl_interp.check_type %3043 is f32 -> ^bb5682, ^bb1
  ^bb5682:
    pdl_interp.record_match @rewriters::@cosh_atanh(%3042, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.cosh") -> ^bb1
  ^bb44:
    pdl_interp.check_operand_count of %0 is 1 -> ^bb5683, ^bb1
  ^bb5683:
    pdl_interp.check_result_count of %0 is 1 -> ^bb5684, ^bb1
  ^bb5684:
    pdl_interp.is_not_null %2 : !pdl.value -> ^bb5685, ^bb1
  ^bb5685:
    pdl_interp.switch_operation_name of %3 to ["arith.addf", "arith.subf", "arith.mulf", "arith.divf", "arith.negf", "arith.constant", "math.asinh", "math.acosh", "math.atanh"](^bb5686, ^bb5687, ^bb5688, ^bb5689, ^bb5690, ^bb5691, ^bb5692, ^bb5693, ^bb5694) -> ^bb1
  ^bb5686:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb5695, ^bb1
  ^bb5695:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5696, ^bb1
  ^bb5696:
    %3046 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %3046 : !pdl.value -> ^bb5697, ^bb1
  ^bb5697:
    pdl_interp.are_equal %3046, %2 : !pdl.value -> ^bb5698, ^bb1
  ^bb5698:
    %3047 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %3047 : !pdl.value -> ^bb5699, ^bb1
  ^bb5699:
    %3048 = pdl_interp.get_value_type of %3047 : !pdl.type
    %3049 = pdl_interp.get_value_type of %3046 : !pdl.type
    pdl_interp.are_equal %3048, %3049 : !pdl.type -> ^bb5700, ^bb1
  ^bb5700:
    %3050 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3048, %3050 : !pdl.type -> ^bb5701, ^bb1
  ^bb5701:
    pdl_interp.check_type %3048 is f32 -> ^bb5702, ^bb1
  ^bb5702:
    %3051 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %3051 : !pdl.value -> ^bb5703, ^bb1
  ^bb5703:
    %3052 = pdl_interp.get_value_type of %3051 : !pdl.type
    pdl_interp.are_equal %3048, %3052 : !pdl.type -> ^bb5704, ^bb1
  ^bb5704:
    pdl_interp.record_match @rewriters::@sinh_sum(%3047, %3051, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.sinh") -> ^bb1
  ^bb5687:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb5705, ^bb1
  ^bb5705:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5706, ^bb1
  ^bb5706:
    %3053 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %3053 : !pdl.value -> ^bb5707, ^bb1
  ^bb5707:
    pdl_interp.are_equal %3053, %2 : !pdl.value -> ^bb5708, ^bb1
  ^bb5708:
    %3054 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %3054 : !pdl.value -> ^bb5709, ^bb1
  ^bb5709:
    %3055 = pdl_interp.get_value_type of %3054 : !pdl.type
    %3056 = pdl_interp.get_value_type of %3053 : !pdl.type
    pdl_interp.are_equal %3055, %3056 : !pdl.type -> ^bb5710, ^bb1
  ^bb5710:
    %3057 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3055, %3057 : !pdl.type -> ^bb5711, ^bb1
  ^bb5711:
    pdl_interp.check_type %3055 is f32 -> ^bb5712, ^bb1
  ^bb5712:
    %3058 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %3058 : !pdl.value -> ^bb5713, ^bb1
  ^bb5713:
    %3059 = pdl_interp.get_value_type of %3058 : !pdl.type
    pdl_interp.are_equal %3055, %3059 : !pdl.type -> ^bb5714, ^bb1
  ^bb5714:
    pdl_interp.record_match @rewriters::@sinh_diff(%3054, %3058, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.sinh") -> ^bb1
  ^bb5688:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb5715, ^bb1
  ^bb5715:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5716, ^bb1
  ^bb5716:
    %3060 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %3060 : !pdl.value -> ^bb5717, ^bb1
  ^bb5717:
    pdl_interp.are_equal %3060, %2 : !pdl.value -> ^bb5718, ^bb1
  ^bb5718:
    %3061 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %3061 : !pdl.value -> ^bb5719, ^bb1
  ^bb5719:
    %3062 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %3062 : !pdl.value -> ^bb5720, ^bb1
  ^bb5720:
    %3063 = pdl_interp.get_defining_op of %3061 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %3063 : !pdl.operation -> ^bb5721, ^bb1
  ^bb5721:
    pdl_interp.check_operation_name of %3063 is "arith.constant" -> ^bb5722, ^bb1
  ^bb5722:
    pdl_interp.check_operand_count of %3063 is 0 -> ^bb5723, ^bb1
  ^bb5723:
    pdl_interp.check_result_count of %3063 is 1 -> ^bb5724, ^bb1
  ^bb5724:
    %3064 = pdl_interp.get_result 0 of %3063
    pdl_interp.is_not_null %3064 : !pdl.value -> ^bb5725, ^bb1
  ^bb5725:
    pdl_interp.are_equal %3064, %3061 : !pdl.value -> ^bb5726, ^bb1
  ^bb5726:
    %3065 = pdl_interp.get_attribute "value" of %3063
    pdl_interp.is_not_null %3065 : !pdl.attribute -> ^bb5727, ^bb1
  ^bb5727:
    pdl_interp.check_attribute %3065 is 2.000000e+00 : f32 -> ^bb5728, ^bb1
  ^bb5728:
    %3066 = pdl_interp.get_value_type of %3064 : !pdl.type
    %3067 = pdl_interp.get_value_type of %3060 : !pdl.type
    pdl_interp.are_equal %3066, %3067 : !pdl.type -> ^bb5729, ^bb1
  ^bb5729:
    %3068 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3066, %3068 : !pdl.type -> ^bb5730, ^bb1
  ^bb5730:
    pdl_interp.check_type %3066 is f32 -> ^bb5731, ^bb1
  ^bb5731:
    %3069 = pdl_interp.get_value_type of %3062 : !pdl.type
    pdl_interp.are_equal %3066, %3069 : !pdl.type -> ^bb5732, ^bb1
  ^bb5732:
    pdl_interp.record_match @rewriters::@sinh_2(%3062, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.sinh") -> ^bb1
  ^bb5689:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb5733, ^bb1
  ^bb5733:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5734, ^bb1
  ^bb5734:
    %3070 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %3070 : !pdl.value -> ^bb5735, ^bb1
  ^bb5735:
    pdl_interp.are_equal %3070, %2 : !pdl.value -> ^bb5736, ^bb1
  ^bb5736:
    %3071 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %3071 : !pdl.value -> ^bb5737, ^bb1
  ^bb5737:
    %3072 = pdl_interp.get_value_type of %3071 : !pdl.type
    %3073 = pdl_interp.get_value_type of %3070 : !pdl.type
    pdl_interp.are_equal %3072, %3073 : !pdl.type -> ^bb5738, ^bb1
  ^bb5738:
    %3074 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3072, %3074 : !pdl.type -> ^bb5739, ^bb1
  ^bb5739:
    pdl_interp.check_type %3072 is f32 -> ^bb5740, ^bb1
  ^bb5740:
    %3075 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %3075 : !pdl.value -> ^bb5741, ^bb1
  ^bb5741:
    %3076 = pdl_interp.get_defining_op of %3075 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %3076 : !pdl.operation -> ^bb5742, ^bb1
  ^bb5742:
    pdl_interp.check_operation_name of %3076 is "arith.constant" -> ^bb5743, ^bb1
  ^bb5743:
    pdl_interp.check_operand_count of %3076 is 0 -> ^bb5744, ^bb1
  ^bb5744:
    pdl_interp.check_result_count of %3076 is 1 -> ^bb5745, ^bb1
  ^bb5745:
    %3077 = pdl_interp.get_result 0 of %3076
    pdl_interp.is_not_null %3077 : !pdl.value -> ^bb5746, ^bb1
  ^bb5746:
    pdl_interp.are_equal %3077, %3075 : !pdl.value -> ^bb5747, ^bb1
  ^bb5747:
    %3078 = pdl_interp.get_attribute "value" of %3076
    pdl_interp.is_not_null %3078 : !pdl.attribute -> ^bb5748, ^bb1
  ^bb5748:
    pdl_interp.check_attribute %3078 is 2.000000e+00 : f32 -> ^bb5749, ^bb1
  ^bb5749:
    %3079 = pdl_interp.get_value_type of %3077 : !pdl.type
    pdl_interp.are_equal %3079, %3072 : !pdl.type -> ^bb5750, ^bb1
  ^bb5750:
    pdl_interp.record_match @rewriters::@sinh_1div2(%3071, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.sinh") -> ^bb1
  ^bb5690:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb5751, ^bb1
  ^bb5751:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5752, ^bb1
  ^bb5752:
    %3080 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %3080 : !pdl.value -> ^bb5753, ^bb1
  ^bb5753:
    pdl_interp.are_equal %3080, %2 : !pdl.value -> ^bb5754, ^bb1
  ^bb5754:
    %3081 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %3081 : !pdl.value -> ^bb5755, ^bb1
  ^bb5755:
    %3082 = pdl_interp.get_value_type of %3081 : !pdl.type
    %3083 = pdl_interp.get_value_type of %3080 : !pdl.type
    pdl_interp.are_equal %3082, %3083 : !pdl.type -> ^bb5756, ^bb1
  ^bb5756:
    %3084 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3082, %3084 : !pdl.type -> ^bb5757, ^bb1
  ^bb5757:
    pdl_interp.check_type %3082 is f32 -> ^bb5758, ^bb1
  ^bb5758:
    pdl_interp.record_match @rewriters::@sinh_neg(%3081, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.sinh") -> ^bb1
  ^bb5691:
    pdl_interp.check_operand_count of %3 is 0 -> ^bb5759, ^bb1
  ^bb5759:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5760, ^bb1
  ^bb5760:
    %3085 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %3085 : !pdl.value -> ^bb5761, ^bb1
  ^bb5761:
    pdl_interp.are_equal %3085, %2 : !pdl.value -> ^bb5762, ^bb1
  ^bb5762:
    %3086 = pdl_interp.get_attribute "value" of %3
    pdl_interp.is_not_null %3086 : !pdl.attribute -> ^bb5763, ^bb1
  ^bb5763:
    pdl_interp.check_attribute %3086 is 0.000000e+00 : f32 -> ^bb5764, ^bb1
  ^bb5764:
    %3087 = pdl_interp.get_value_type of %3085 : !pdl.type
    %3088 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3087, %3088 : !pdl.type -> ^bb5765, ^bb1
  ^bb5765:
    pdl_interp.check_type %3087 is f32 -> ^bb5766, ^bb1
  ^bb5766:
    pdl_interp.record_match @rewriters::@sinh_0(%0 : !pdl.operation) : benefit(1), loc([]), root("math.sinh") -> ^bb1
  ^bb5692:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb5767, ^bb1
  ^bb5767:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5768, ^bb1
  ^bb5768:
    %3089 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %3089 : !pdl.value -> ^bb5769, ^bb1
  ^bb5769:
    pdl_interp.are_equal %3089, %2 : !pdl.value -> ^bb5770, ^bb1
  ^bb5770:
    %3090 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %3090 : !pdl.value -> ^bb5771, ^bb1
  ^bb5771:
    %3091 = pdl_interp.get_value_type of %3090 : !pdl.type
    %3092 = pdl_interp.get_value_type of %3089 : !pdl.type
    pdl_interp.are_equal %3091, %3092 : !pdl.type -> ^bb5772, ^bb1
  ^bb5772:
    %3093 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3091, %3093 : !pdl.type -> ^bb5773, ^bb1
  ^bb5773:
    pdl_interp.check_type %3091 is f32 -> ^bb5774, ^bb1
  ^bb5774:
    pdl_interp.record_match @rewriters::@sinh_asinh(%3090, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.sinh") -> ^bb1
  ^bb5693:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb5775, ^bb1
  ^bb5775:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5776, ^bb1
  ^bb5776:
    %3094 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %3094 : !pdl.value -> ^bb5777, ^bb1
  ^bb5777:
    pdl_interp.are_equal %3094, %2 : !pdl.value -> ^bb5778, ^bb1
  ^bb5778:
    %3095 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %3095 : !pdl.value -> ^bb5779, ^bb1
  ^bb5779:
    %3096 = pdl_interp.get_value_type of %3095 : !pdl.type
    %3097 = pdl_interp.get_value_type of %3094 : !pdl.type
    pdl_interp.are_equal %3096, %3097 : !pdl.type -> ^bb5780, ^bb1
  ^bb5780:
    %3098 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3096, %3098 : !pdl.type -> ^bb5781, ^bb1
  ^bb5781:
    pdl_interp.check_type %3096 is f32 -> ^bb5782, ^bb1
  ^bb5782:
    pdl_interp.record_match @rewriters::@sinh_acosh(%3095, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.sinh") -> ^bb1
  ^bb5694:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb5783, ^bb1
  ^bb5783:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5784, ^bb1
  ^bb5784:
    %3099 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %3099 : !pdl.value -> ^bb5785, ^bb1
  ^bb5785:
    pdl_interp.are_equal %3099, %2 : !pdl.value -> ^bb5786, ^bb1
  ^bb5786:
    %3100 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %3100 : !pdl.value -> ^bb5787, ^bb1
  ^bb5787:
    %3101 = pdl_interp.get_value_type of %3100 : !pdl.type
    %3102 = pdl_interp.get_value_type of %3099 : !pdl.type
    pdl_interp.are_equal %3101, %3102 : !pdl.type -> ^bb5788, ^bb1
  ^bb5788:
    %3103 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3101, %3103 : !pdl.type -> ^bb5789, ^bb1
  ^bb5789:
    pdl_interp.check_type %3101 is f32 -> ^bb5790, ^bb1
  ^bb5790:
    pdl_interp.record_match @rewriters::@sinh_atanh(%3100, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.sinh") -> ^bb1
  ^bb45:
    pdl_interp.check_operand_count of %0 is 1 -> ^bb5791, ^bb1
  ^bb5791:
    pdl_interp.check_result_count of %0 is 1 -> ^bb5792, ^bb1
  ^bb5792:
    pdl_interp.is_not_null %2 : !pdl.value -> ^bb5793, ^bb1
  ^bb5793:
    pdl_interp.switch_operation_name of %3 to ["arith.mulf", "arith.divf", "arith.addf", "math.asinh", "math.acosh", "math.atanh"](^bb5794, ^bb5795, ^bb5796, ^bb5797, ^bb5798, ^bb5799) -> ^bb1
  ^bb5794:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb5800, ^bb1
  ^bb5800:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5801, ^bb1
  ^bb5801:
    %3104 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %3104 : !pdl.value -> ^bb5802, ^bb1
  ^bb5802:
    pdl_interp.are_equal %3104, %2 : !pdl.value -> ^bb5803, ^bb1
  ^bb5803:
    %3105 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %3105 : !pdl.value -> ^bb5804, ^bb1
  ^bb5804:
    %3106 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %3106 : !pdl.value -> ^bb5805, ^bb1
  ^bb5805:
    %3107 = pdl_interp.get_defining_op of %3105 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %3107 : !pdl.operation -> ^bb5806, ^bb1
  ^bb5806:
    pdl_interp.check_operation_name of %3107 is "arith.constant" -> ^bb5807, ^bb1
  ^bb5807:
    pdl_interp.check_operand_count of %3107 is 0 -> ^bb5808, ^bb1
  ^bb5808:
    pdl_interp.check_result_count of %3107 is 1 -> ^bb5809, ^bb1
  ^bb5809:
    %3108 = pdl_interp.get_result 0 of %3107
    pdl_interp.is_not_null %3108 : !pdl.value -> ^bb5810, ^bb1
  ^bb5810:
    pdl_interp.are_equal %3108, %3105 : !pdl.value -> ^bb5811, ^bb1
  ^bb5811:
    %3109 = pdl_interp.get_attribute "value" of %3107
    pdl_interp.is_not_null %3109 : !pdl.attribute -> ^bb5812, ^bb1
  ^bb5812:
    pdl_interp.check_attribute %3109 is 2.000000e+00 : f32 -> ^bb5813, ^bb1
  ^bb5813:
    %3110 = pdl_interp.get_value_type of %3108 : !pdl.type
    %3111 = pdl_interp.get_value_type of %3104 : !pdl.type
    pdl_interp.are_equal %3110, %3111 : !pdl.type -> ^bb5814, ^bb1
  ^bb5814:
    %3112 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3110, %3112 : !pdl.type -> ^bb5815, ^bb1
  ^bb5815:
    pdl_interp.check_type %3110 is f32 -> ^bb5816, ^bb1
  ^bb5816:
    %3113 = pdl_interp.get_value_type of %3106 : !pdl.type
    pdl_interp.are_equal %3110, %3113 : !pdl.type -> ^bb5817, ^bb1
  ^bb5817:
    pdl_interp.record_match @rewriters::@tanh_2(%3106, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.tanh") -> ^bb1
  ^bb5795:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb5818, ^bb1
  ^bb5818:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5819, ^bb1
  ^bb5819:
    %3114 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %3114 : !pdl.value -> ^bb5820, ^bb1
  ^bb5820:
    pdl_interp.are_equal %3114, %2 : !pdl.value -> ^bb5821, ^bb1
  ^bb5821:
    %3115 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %3115 : !pdl.value -> ^bb5822, ^bb1
  ^bb5822:
    %3116 = pdl_interp.get_value_type of %3115 : !pdl.type
    %3117 = pdl_interp.get_value_type of %3114 : !pdl.type
    pdl_interp.are_equal %3116, %3117 : !pdl.type -> ^bb5823, ^bb1
  ^bb5823:
    %3118 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3116, %3118 : !pdl.type -> ^bb5824, ^bb1
  ^bb5824:
    pdl_interp.check_type %3116 is f32 -> ^bb5825, ^bb1
  ^bb5825:
    %3119 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %3119 : !pdl.value -> ^bb5826, ^bb1
  ^bb5826:
    %3120 = pdl_interp.get_defining_op of %3119 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %3120 : !pdl.operation -> ^bb5827, ^bb1
  ^bb5827:
    pdl_interp.check_operation_name of %3120 is "arith.constant" -> ^bb5828, ^bb1
  ^bb5828:
    pdl_interp.check_operand_count of %3120 is 0 -> ^bb5829, ^bb1
  ^bb5829:
    pdl_interp.check_result_count of %3120 is 1 -> ^bb5830, ^bb1
  ^bb5830:
    %3121 = pdl_interp.get_result 0 of %3120
    pdl_interp.is_not_null %3121 : !pdl.value -> ^bb5831, ^bb1
  ^bb5831:
    pdl_interp.are_equal %3121, %3119 : !pdl.value -> ^bb5832, ^bb1
  ^bb5832:
    %3122 = pdl_interp.get_attribute "value" of %3120
    pdl_interp.is_not_null %3122 : !pdl.attribute -> ^bb5833, ^bb1
  ^bb5833:
    pdl_interp.check_attribute %3122 is 2.000000e+00 : f32 -> ^bb5834, ^bb1
  ^bb5834:
    %3123 = pdl_interp.get_value_type of %3121 : !pdl.type
    pdl_interp.are_equal %3123, %3116 : !pdl.type -> ^bb5835, ^bb1
  ^bb5835:
    pdl_interp.record_match @rewriters::@tanh_1div2(%3115, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.tanh") -> ^bb1
  ^bb5796:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb5836, ^bb1
  ^bb5836:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5837, ^bb1
  ^bb5837:
    %3124 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %3124 : !pdl.value -> ^bb5838, ^bb1
  ^bb5838:
    pdl_interp.are_equal %3124, %2 : !pdl.value -> ^bb5839, ^bb1
  ^bb5839:
    %3125 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %3125 : !pdl.value -> ^bb5840, ^bb1
  ^bb5840:
    %3126 = pdl_interp.get_value_type of %3125 : !pdl.type
    %3127 = pdl_interp.get_value_type of %3124 : !pdl.type
    pdl_interp.are_equal %3126, %3127 : !pdl.type -> ^bb5841, ^bb1
  ^bb5841:
    %3128 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3126, %3128 : !pdl.type -> ^bb5842, ^bb1
  ^bb5842:
    pdl_interp.check_type %3126 is f32 -> ^bb5843, ^bb1
  ^bb5843:
    %3129 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %3129 : !pdl.value -> ^bb5844, ^bb1
  ^bb5844:
    %3130 = pdl_interp.get_value_type of %3129 : !pdl.type
    pdl_interp.are_equal %3126, %3130 : !pdl.type -> ^bb5845, ^bb1
  ^bb5845:
    pdl_interp.record_match @rewriters::@tanh_sum(%3125, %3129, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.tanh") -> ^bb1
  ^bb5797:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb5846, ^bb1
  ^bb5846:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5847, ^bb1
  ^bb5847:
    %3131 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %3131 : !pdl.value -> ^bb5848, ^bb1
  ^bb5848:
    pdl_interp.are_equal %3131, %2 : !pdl.value -> ^bb5849, ^bb1
  ^bb5849:
    %3132 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %3132 : !pdl.value -> ^bb5850, ^bb1
  ^bb5850:
    %3133 = pdl_interp.get_value_type of %3132 : !pdl.type
    %3134 = pdl_interp.get_value_type of %3131 : !pdl.type
    pdl_interp.are_equal %3133, %3134 : !pdl.type -> ^bb5851, ^bb1
  ^bb5851:
    %3135 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3133, %3135 : !pdl.type -> ^bb5852, ^bb1
  ^bb5852:
    pdl_interp.check_type %3133 is f32 -> ^bb5853, ^bb1
  ^bb5853:
    pdl_interp.record_match @rewriters::@tanh_asinh(%3132, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.tanh") -> ^bb1
  ^bb5798:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb5854, ^bb1
  ^bb5854:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5855, ^bb1
  ^bb5855:
    %3136 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %3136 : !pdl.value -> ^bb5856, ^bb1
  ^bb5856:
    pdl_interp.are_equal %3136, %2 : !pdl.value -> ^bb5857, ^bb1
  ^bb5857:
    %3137 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %3137 : !pdl.value -> ^bb5858, ^bb1
  ^bb5858:
    %3138 = pdl_interp.get_value_type of %3137 : !pdl.type
    %3139 = pdl_interp.get_value_type of %3136 : !pdl.type
    pdl_interp.are_equal %3138, %3139 : !pdl.type -> ^bb5859, ^bb1
  ^bb5859:
    %3140 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3138, %3140 : !pdl.type -> ^bb5860, ^bb1
  ^bb5860:
    pdl_interp.check_type %3138 is f32 -> ^bb5861, ^bb1
  ^bb5861:
    pdl_interp.record_match @rewriters::@tanh_acosh(%3137, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.tanh") -> ^bb1
  ^bb5799:
    pdl_interp.check_operand_count of %3 is 1 -> ^bb5862, ^bb1
  ^bb5862:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5863, ^bb1
  ^bb5863:
    %3141 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %3141 : !pdl.value -> ^bb5864, ^bb1
  ^bb5864:
    pdl_interp.are_equal %3141, %2 : !pdl.value -> ^bb5865, ^bb1
  ^bb5865:
    %3142 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %3142 : !pdl.value -> ^bb5866, ^bb1
  ^bb5866:
    %3143 = pdl_interp.get_value_type of %3142 : !pdl.type
    %3144 = pdl_interp.get_value_type of %3141 : !pdl.type
    pdl_interp.are_equal %3143, %3144 : !pdl.type -> ^bb5867, ^bb1
  ^bb5867:
    %3145 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3143, %3145 : !pdl.type -> ^bb5868, ^bb1
  ^bb5868:
    pdl_interp.check_type %3143 is f32 -> ^bb5869, ^bb1
  ^bb5869:
    pdl_interp.record_match @rewriters::@tanh_atanh(%3142, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.tanh") -> ^bb1
  ^bb46:
    pdl_interp.check_operand_count of %0 is 1 -> ^bb5870, ^bb1
  ^bb5870:
    pdl_interp.check_result_count of %0 is 1 -> ^bb5871, ^bb1
  ^bb5871:
    pdl_interp.is_not_null %2 : !pdl.value -> ^bb5872, ^bb1
  ^bb5872:
    pdl_interp.check_operation_name of %3 is "arith.addf" -> ^bb5873, ^bb1
  ^bb5873:
    pdl_interp.check_operand_count of %3 is 2 -> ^bb5874, ^bb1
  ^bb5874:
    pdl_interp.check_result_count of %3 is 1 -> ^bb5875, ^bb1
  ^bb5875:
    %3146 = pdl_interp.get_result 0 of %3
    pdl_interp.is_not_null %3146 : !pdl.value -> ^bb5876, ^bb1
  ^bb5876:
    pdl_interp.are_equal %3146, %2 : !pdl.value -> ^bb5877, ^bb1
  ^bb5877:
    %3147 = pdl_interp.get_operand 0 of %3
    pdl_interp.is_not_null %3147 : !pdl.value -> ^bb5878, ^bb1
  ^bb5878:
    %3148 = pdl_interp.get_operand 1 of %3
    pdl_interp.is_not_null %3148 : !pdl.value -> ^bb5879, ^bb1
  ^bb5879:
    %3149 = pdl_interp.get_defining_op of %3148 : !pdl.value {position = "root.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %3149 : !pdl.operation -> ^bb5880, ^bb1
  ^bb5880:
    %3150 = pdl_interp.get_defining_op of %3147 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %3150 : !pdl.operation -> ^bb5881, ^bb1
  ^bb5881:
    pdl_interp.check_operation_name of %3149 is "arith.constant" -> ^bb5882, ^bb1
  ^bb5882:
    pdl_interp.check_operand_count of %3149 is 0 -> ^bb5883, ^bb1
  ^bb5883:
    pdl_interp.check_result_count of %3149 is 1 -> ^bb5884, ^bb1
  ^bb5884:
    %3151 = pdl_interp.get_result 0 of %3149
    pdl_interp.is_not_null %3151 : !pdl.value -> ^bb5885, ^bb1
  ^bb5885:
    pdl_interp.are_equal %3151, %3148 : !pdl.value -> ^bb5886, ^bb1
  ^bb5886:
    pdl_interp.check_operation_name of %3150 is "arith.mulf" -> ^bb5887, ^bb1
  ^bb5887:
    pdl_interp.check_operand_count of %3150 is 2 -> ^bb5888, ^bb1
  ^bb5888:
    pdl_interp.check_result_count of %3150 is 1 -> ^bb5889, ^bb1
  ^bb5889:
    %3152 = pdl_interp.get_result 0 of %3150
    pdl_interp.is_not_null %3152 : !pdl.value -> ^bb5890, ^bb1
  ^bb5890:
    pdl_interp.are_equal %3152, %3147 : !pdl.value -> ^bb5891, ^bb1
  ^bb5891:
    %3153 = pdl_interp.get_operand 0 of %3150
    pdl_interp.is_not_null %3153 : !pdl.value -> ^bb5892, ^bb1
  ^bb5892:
    %3154 = pdl_interp.get_attribute "value" of %3149
    pdl_interp.is_not_null %3154 : !pdl.attribute -> ^bb5893, ^bb1
  ^bb5893:
    pdl_interp.check_attribute %3154 is 1.000000e+00 : f32 -> ^bb5894, ^bb1
  ^bb5894:
    %3155 = pdl_interp.get_defining_op of %3153 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %3155 : !pdl.operation -> ^bb5895, ^bb1
  ^bb5895:
    %3156 = pdl_interp.get_operand 1 of %3150
    %3157 = pdl_interp.get_defining_op of %3156 : !pdl.value {position = "root.operand[0].defining_op.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %3157 : !pdl.operation -> ^bb5896, ^bb1
  ^bb5896:
    pdl_interp.check_operation_name of %3155 is "arith.constant" -> ^bb5897, ^bb1
  ^bb5897:
    pdl_interp.check_operand_count of %3155 is 0 -> ^bb5898, ^bb1
  ^bb5898:
    pdl_interp.check_result_count of %3155 is 1 -> ^bb5899, ^bb1
  ^bb5899:
    %3158 = pdl_interp.get_result 0 of %3155
    pdl_interp.is_not_null %3158 : !pdl.value -> ^bb5900, ^bb1
  ^bb5900:
    pdl_interp.are_equal %3158, %3153 : !pdl.value -> ^bb5901, ^bb1
  ^bb5901:
    pdl_interp.is_not_null %3156 : !pdl.value -> ^bb5902, ^bb1
  ^bb5902:
    pdl_interp.check_operation_name of %3157 is "arith.mulf" -> ^bb5903, ^bb1
  ^bb5903:
    pdl_interp.check_operand_count of %3157 is 2 -> ^bb5904, ^bb1
  ^bb5904:
    pdl_interp.check_result_count of %3157 is 1 -> ^bb5905, ^bb1
  ^bb5905:
    %3159 = pdl_interp.get_result 0 of %3157
    pdl_interp.is_not_null %3159 : !pdl.value -> ^bb5906, ^bb1
  ^bb5906:
    pdl_interp.are_equal %3159, %3156 : !pdl.value -> ^bb5907, ^bb1
  ^bb5907:
    %3160 = pdl_interp.get_operand 0 of %3157
    pdl_interp.is_not_null %3160 : !pdl.value -> ^bb5908, ^bb1
  ^bb5908:
    %3161 = pdl_interp.get_operand 1 of %3157
    pdl_interp.are_equal %3160, %3161 : !pdl.value -> ^bb5909, ^bb1
  ^bb5909:
    %3162 = pdl_interp.get_attribute "value" of %3155
    pdl_interp.is_not_null %3162 : !pdl.attribute -> ^bb5910, ^bb1
  ^bb5910:
    pdl_interp.check_attribute %3162 is 2.000000e+00 : f32 -> ^bb5911, ^bb1
  ^bb5911:
    %3163 = pdl_interp.get_value_type of %3158 : !pdl.type
    %3164 = pdl_interp.get_value_type of %3160 : !pdl.type
    pdl_interp.are_equal %3163, %3164 : !pdl.type -> ^bb5912, ^bb1
  ^bb5912:
    %3165 = pdl_interp.get_value_type of %3159 : !pdl.type
    pdl_interp.are_equal %3163, %3165 : !pdl.type -> ^bb5913, ^bb1
  ^bb5913:
    %3166 = pdl_interp.get_value_type of %3152 : !pdl.type
    pdl_interp.are_equal %3163, %3166 : !pdl.type -> ^bb5914, ^bb1
  ^bb5914:
    %3167 = pdl_interp.get_value_type of %3146 : !pdl.type
    pdl_interp.are_equal %3163, %3167 : !pdl.type -> ^bb5915, ^bb1
  ^bb5915:
    %3168 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3163, %3168 : !pdl.type -> ^bb5916, ^bb1
  ^bb5916:
    pdl_interp.check_type %3163 is f32 -> ^bb5917, ^bb1
  ^bb5917:
    %3169 = pdl_interp.get_value_type of %3151 : !pdl.type
    pdl_interp.are_equal %3163, %3169 : !pdl.type -> ^bb5918, ^bb1
  ^bb5918:
    pdl_interp.record_match @rewriters::@asinh_2(%3160, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.acosh") -> ^bb1
  ^bb2:
    pdl_interp.check_operand_count of %0 is 2 -> ^bb5919, ^bb23
  ^bb5919:
    pdl_interp.check_result_count of %0 is 1 -> ^bb5920, ^bb23
  ^bb5920:
    %3170 = pdl_interp.get_operand 0 of %0
    pdl_interp.is_not_null %3170 : !pdl.value -> ^bb5921, ^bb5922
  ^bb5922:
    %3171 = pdl_interp.get_operand 1 of %0
    %3172 = pdl_interp.get_defining_op of %3171 : !pdl.value {position = "root.operand[1].defining_op"}
    pdl_interp.is_not_null %3172 : !pdl.operation -> ^bb5923, ^bb23
  ^bb5923:
    %3173 = pdl_interp.get_operand 0 of %0
    pdl_interp.is_not_null %3173 : !pdl.value -> ^bb5924, ^bb23
  ^bb5924:
    pdl_interp.is_not_null %3171 : !pdl.value -> ^bb5925, ^bb23
  ^bb5925:
    pdl_interp.switch_operation_name of %3172 to ["arith.addf", "arith.subf", "arith.constant", "arith.mulf", "arith.negf", "arith.divf"](^bb5926, ^bb5927, ^bb5928, ^bb5929, ^bb5930, ^bb5931) -> ^bb23
  ^bb5926:
    pdl_interp.check_operand_count of %3172 is 2 -> ^bb5932, ^bb23
  ^bb5932:
    pdl_interp.check_result_count of %3172 is 1 -> ^bb5933, ^bb23
  ^bb5933:
    %3174 = pdl_interp.get_result 0 of %3172
    pdl_interp.is_not_null %3174 : !pdl.value -> ^bb5934, ^bb23
  ^bb5934:
    pdl_interp.are_equal %3174, %3171 : !pdl.value -> ^bb5935, ^bb23
  ^bb5935:
    %3175 = pdl_interp.get_operand 0 of %3172
    pdl_interp.is_not_null %3175 : !pdl.value -> ^bb5936, ^bb23
  ^bb5936:
    %3176 = pdl_interp.get_operand 1 of %3172
    pdl_interp.is_not_null %3176 : !pdl.value -> ^bb5937, ^bb23
  ^bb5937:
    %3177 = pdl_interp.get_value_type of %3173 : !pdl.type
    %3178 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3177, %3178 : !pdl.type -> ^bb5938, ^bb23
  ^bb5938:
    pdl_interp.check_type %3177 is f32 -> ^bb5939, ^bb23
  ^bb5939:
    %3179 = pdl_interp.get_value_type of %3174 : !pdl.type
    pdl_interp.are_equal %3179, %3177 : !pdl.type -> ^bb5940, ^bb23
  ^bb5940:
    %3180 = pdl_interp.get_value_type of %3175 : !pdl.type
    pdl_interp.are_equal %3180, %3177 : !pdl.type -> ^bb5941, ^bb23
  ^bb5941:
    %3181 = pdl_interp.get_value_type of %3176 : !pdl.type
    pdl_interp.are_equal %3181, %3177 : !pdl.type -> ^bb5942, ^bb23
  ^bb5942:
    pdl_interp.record_match @rewriters::@associate_addradd(%3173, %3175, %3176, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb23
  ^bb5927:
    pdl_interp.check_operand_count of %3172 is 2 -> ^bb5943, ^bb23
  ^bb5943:
    pdl_interp.check_result_count of %3172 is 1 -> ^bb5944, ^bb23
  ^bb5944:
    %3182 = pdl_interp.get_result 0 of %3172
    pdl_interp.is_not_null %3182 : !pdl.value -> ^bb5945, ^bb23
  ^bb5945:
    pdl_interp.are_equal %3182, %3171 : !pdl.value -> ^bb5946, ^bb23
  ^bb5946:
    %3183 = pdl_interp.get_operand 0 of %3172
    pdl_interp.is_not_null %3183 : !pdl.value -> ^bb5947, ^bb23
  ^bb5947:
    %3184 = pdl_interp.get_operand 1 of %3172
    pdl_interp.is_not_null %3184 : !pdl.value -> ^bb5948, ^bb23
  ^bb5948:
    %3185 = pdl_interp.get_value_type of %3173 : !pdl.type
    %3186 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3185, %3186 : !pdl.type -> ^bb5949, ^bb23
  ^bb5949:
    pdl_interp.check_type %3185 is f32 -> ^bb5950, ^bb23
  ^bb5950:
    %3187 = pdl_interp.get_value_type of %3182 : !pdl.type
    pdl_interp.are_equal %3187, %3185 : !pdl.type -> ^bb5951, ^bb23
  ^bb5951:
    %3188 = pdl_interp.get_value_type of %3183 : !pdl.type
    pdl_interp.are_equal %3188, %3185 : !pdl.type -> ^bb5952, ^bb23
  ^bb5952:
    %3189 = pdl_interp.get_value_type of %3184 : !pdl.type
    pdl_interp.are_equal %3189, %3185 : !pdl.type -> ^bb5953, ^bb23
  ^bb5953:
    pdl_interp.record_match @rewriters::@associate_addr_(%3173, %3183, %3184, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb23
  ^bb5928:
    pdl_interp.check_operand_count of %3172 is 0 -> ^bb5954, ^bb23
  ^bb5954:
    pdl_interp.check_result_count of %3172 is 1 -> ^bb5955, ^bb23
  ^bb5955:
    %3190 = pdl_interp.get_result 0 of %3172
    pdl_interp.is_not_null %3190 : !pdl.value -> ^bb5956, ^bb23
  ^bb5956:
    pdl_interp.are_equal %3190, %3171 : !pdl.value -> ^bb5957, ^bb23
  ^bb5957:
    %3191 = pdl_interp.get_value_type of %3173 : !pdl.type
    %3192 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3191, %3192 : !pdl.type -> ^bb5958, ^bb23
  ^bb5958:
    pdl_interp.check_type %3191 is f32 -> ^bb5959, ^bb23
  ^bb5959:
    %3193 = pdl_interp.get_value_type of %3190 : !pdl.type
    pdl_interp.are_equal %3193, %3191 : !pdl.type -> ^bb5960, ^bb23
  ^bb5960:
    %3194 = pdl_interp.get_attribute "value" of %3172
    pdl_interp.is_not_null %3194 : !pdl.attribute -> ^bb5961, ^bb23
  ^bb5961:
    pdl_interp.check_attribute %3194 is 0.000000e+00 : f32 -> ^bb5962, ^bb23
  ^bb5962:
    pdl_interp.record_match @rewriters::@add_rgt_identity(%3173, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb23
  ^bb5929:
    pdl_interp.check_operand_count of %3172 is 2 -> ^bb5963, ^bb23
  ^bb5963:
    pdl_interp.check_result_count of %3172 is 1 -> ^bb5964, ^bb23
  ^bb5964:
    %3195 = pdl_interp.get_result 0 of %3172
    pdl_interp.is_not_null %3195 : !pdl.value -> ^bb5965, ^bb23
  ^bb5965:
    pdl_interp.are_equal %3195, %3171 : !pdl.value -> ^bb5966, ^bb23
  ^bb5966:
    %3196 = pdl_interp.get_operand 0 of %3172
    pdl_interp.is_not_null %3196 : !pdl.value -> ^bb5967, ^bb23
  ^bb5967:
    %3197 = pdl_interp.get_value_type of %3173 : !pdl.type
    %3198 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3197, %3198 : !pdl.type -> ^bb5968, ^bb5969
  ^bb5969:
    %3199 = pdl_interp.get_defining_op of %3196 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %3199 : !pdl.operation -> ^bb5970, ^bb5971
  ^bb5971:
    %3200 = pdl_interp.get_operand 1 of %3172
    pdl_interp.is_not_null %3200 : !pdl.value -> ^bb5972, ^bb23
  ^bb5972:
    %3201 = pdl_interp.get_value_type of %3173 : !pdl.type
    %3202 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3201, %3202 : !pdl.type -> ^bb5973, ^bb23
  ^bb5973:
    pdl_interp.check_type %3201 is f32 -> ^bb5974, ^bb23
  ^bb5974:
    %3203 = pdl_interp.get_value_type of %3195 : !pdl.type
    pdl_interp.are_equal %3203, %3201 : !pdl.type -> ^bb5975, ^bb23
  ^bb5975:
    %3204 = pdl_interp.get_value_type of %3196 : !pdl.type
    pdl_interp.are_equal %3204, %3201 : !pdl.type -> ^bb5976, ^bb23
  ^bb5976:
    %3205 = pdl_interp.get_value_type of %3200 : !pdl.type
    pdl_interp.are_equal %3205, %3201 : !pdl.type -> ^bb5977, ^bb23
  ^bb5977:
    pdl_interp.record_match @rewriters::@fp_cancel_sign_sub_inv(%3196, %3200, %3173, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb23
  ^bb5970:
    %3206 = pdl_interp.get_operand 1 of %3172
    pdl_interp.is_not_null %3206 : !pdl.value -> ^bb5978, ^bb5971
  ^bb5978:
    %3207 = pdl_interp.get_value_type of %3173 : !pdl.type
    %3208 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3207, %3208 : !pdl.type -> ^bb5979, ^bb5971
  ^bb5979:
    pdl_interp.check_type %3207 is f32 -> ^bb5980, ^bb5971
  ^bb5980:
    pdl_interp.check_operation_name of %3199 is "arith.negf" -> ^bb5981, ^bb5971
  ^bb5981:
    pdl_interp.check_operand_count of %3199 is 1 -> ^bb5982, ^bb5971
  ^bb5982:
    pdl_interp.check_result_count of %3199 is 1 -> ^bb5983, ^bb5971
  ^bb5983:
    %3209 = pdl_interp.get_result 0 of %3199
    pdl_interp.is_not_null %3209 : !pdl.value -> ^bb5984, ^bb5971
  ^bb5984:
    pdl_interp.are_equal %3209, %3196 : !pdl.value -> ^bb5985, ^bb5971
  ^bb5985:
    %3210 = pdl_interp.get_value_type of %3195 : !pdl.type
    pdl_interp.are_equal %3210, %3207 : !pdl.type -> ^bb5986, ^bb5971
  ^bb5986:
    %3211 = pdl_interp.get_operand 0 of %3199
    pdl_interp.is_not_null %3211 : !pdl.value -> ^bb5987, ^bb5971
  ^bb5987:
    %3212 = pdl_interp.get_value_type of %3206 : !pdl.type
    pdl_interp.are_equal %3212, %3207 : !pdl.type -> ^bb5988, ^bb5971
  ^bb5988:
    %3213 = pdl_interp.get_value_type of %3209 : !pdl.type
    pdl_interp.are_equal %3213, %3207 : !pdl.type -> ^bb5989, ^bb5971
  ^bb5989:
    %3214 = pdl_interp.get_value_type of %3211 : !pdl.type
    pdl_interp.are_equal %3214, %3207 : !pdl.type -> ^bb5990, ^bb5971
  ^bb5990:
    pdl_interp.record_match @rewriters::@fp_cancel_sub_sign(%3211, %3206, %3173, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb5971
  ^bb5968:
    pdl_interp.check_type %3197 is f32 -> ^bb5991, ^bb5969
  ^bb5991:
    %3215 = pdl_interp.get_value_type of %3195 : !pdl.type
    pdl_interp.are_equal %3215, %3197 : !pdl.type -> ^bb5992, ^bb5969
  ^bb5992:
    %3216 = pdl_interp.get_value_type of %3196 : !pdl.type
    pdl_interp.are_equal %3216, %3197 : !pdl.type -> ^bb5993, ^bb5969
  ^bb5993:
    %3217 = pdl_interp.get_operand 1 of %3172
    pdl_interp.are_equal %3217, %3173 : !pdl.value -> ^bb5994, ^bb5969
  ^bb5994:
    pdl_interp.record_match @rewriters::@distribute_rgt1_in(%3196, %3173, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb5969
  ^bb5930:
    pdl_interp.check_operand_count of %3172 is 1 -> ^bb5995, ^bb23
  ^bb5995:
    pdl_interp.check_result_count of %3172 is 1 -> ^bb5996, ^bb23
  ^bb5996:
    %3218 = pdl_interp.get_result 0 of %3172
    pdl_interp.is_not_null %3218 : !pdl.value -> ^bb5997, ^bb23
  ^bb5997:
    pdl_interp.are_equal %3218, %3171 : !pdl.value -> ^bb5998, ^bb23
  ^bb5998:
    %3219 = pdl_interp.get_operand 0 of %3172
    pdl_interp.is_not_null %3219 : !pdl.value -> ^bb5999, ^bb23
  ^bb5999:
    %3220 = pdl_interp.get_value_type of %3173 : !pdl.type
    %3221 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3220, %3221 : !pdl.type -> ^bb6000, ^bb23
  ^bb6000:
    pdl_interp.check_type %3220 is f32 -> ^bb6001, ^bb23
  ^bb6001:
    %3222 = pdl_interp.get_value_type of %3218 : !pdl.type
    pdl_interp.are_equal %3222, %3220 : !pdl.type -> ^bb6002, ^bb23
  ^bb6002:
    %3223 = pdl_interp.get_value_type of %3219 : !pdl.type
    pdl_interp.are_equal %3223, %3220 : !pdl.type -> ^bb6003, ^bb23
  ^bb6003:
    pdl_interp.record_match @rewriters::@sub_flip_reverse(%3173, %3219, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb23
  ^bb5931:
    pdl_interp.check_operand_count of %3172 is 2 -> ^bb6004, ^bb23
  ^bb6004:
    pdl_interp.check_result_count of %3172 is 1 -> ^bb6005, ^bb23
  ^bb6005:
    %3224 = pdl_interp.get_result 0 of %3172
    pdl_interp.is_not_null %3224 : !pdl.value -> ^bb6006, ^bb23
  ^bb6006:
    pdl_interp.are_equal %3224, %3171 : !pdl.value -> ^bb6007, ^bb23
  ^bb6007:
    %3225 = pdl_interp.get_operand 0 of %3172
    pdl_interp.is_not_null %3225 : !pdl.value -> ^bb6008, ^bb23
  ^bb6008:
    %3226 = pdl_interp.get_operand 1 of %3172
    pdl_interp.is_not_null %3226 : !pdl.value -> ^bb6009, ^bb23
  ^bb6009:
    %3227 = pdl_interp.get_value_type of %3173 : !pdl.type
    %3228 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3227, %3228 : !pdl.type -> ^bb6010, ^bb23
  ^bb6010:
    pdl_interp.check_type %3227 is f32 -> ^bb6011, ^bb23
  ^bb6011:
    %3229 = pdl_interp.get_value_type of %3224 : !pdl.type
    pdl_interp.are_equal %3229, %3227 : !pdl.type -> ^bb6012, ^bb23
  ^bb6012:
    %3230 = pdl_interp.get_value_type of %3225 : !pdl.type
    pdl_interp.are_equal %3230, %3227 : !pdl.type -> ^bb6013, ^bb23
  ^bb6013:
    %3231 = pdl_interp.get_value_type of %3226 : !pdl.type
    pdl_interp.are_equal %3231, %3227 : !pdl.type -> ^bb6014, ^bb23
  ^bb6014:
    pdl_interp.record_match @rewriters::@add_to_fraction(%3173, %3226, %3225, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb23
  ^bb5921:
    %3232 = pdl_interp.get_operand 1 of %0
    pdl_interp.is_not_null %3232 : !pdl.value -> ^bb6015, ^bb6016
  ^bb6016:
    %3233 = pdl_interp.get_value_type of %3170 : !pdl.type
    %3234 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3233, %3234 : !pdl.type -> ^bb6017, ^bb5922
  ^bb6017:
    pdl_interp.check_type %3233 is f32 -> ^bb6018, ^bb5922
  ^bb6018:
    %3235 = pdl_interp.get_operand 1 of %0
    pdl_interp.are_equal %3170, %3235 : !pdl.value -> ^bb6019, ^bb5922
  ^bb6019:
    pdl_interp.record_match @rewriters::@count_2(%3170, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb5922
  ^bb6015:
    %3236 = pdl_interp.get_value_type of %3170 : !pdl.type
    %3237 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3236, %3237 : !pdl.type -> ^bb6020, ^bb6016
  ^bb6020:
    pdl_interp.check_type %3236 is f32 -> ^bb6021, ^bb6016
  ^bb6021:
    %3238 = pdl_interp.get_value_type of %3232 : !pdl.type
    pdl_interp.are_equal %3236, %3238 : !pdl.type -> ^bb6022, ^bb6016
  ^bb6022:
    pdl_interp.record_match @rewriters::@add_flip(%3232, %3170, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb6023
  ^bb6023:
    pdl_interp.record_match @rewriters::@add_commutative(%3232, %3170, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.addf") -> ^bb6016
  ^bb3:
    pdl_interp.check_operand_count of %0 is 2 -> ^bb6024, ^bb23
  ^bb6024:
    pdl_interp.check_result_count of %0 is 1 -> ^bb6025, ^bb23
  ^bb6025:
    %3239 = pdl_interp.get_operand 0 of %0
    pdl_interp.is_not_null %3239 : !pdl.value -> ^bb6026, ^bb6027
  ^bb6027:
    %3240 = pdl_interp.get_operand 1 of %0
    %3241 = pdl_interp.get_defining_op of %3240 : !pdl.value {position = "root.operand[1].defining_op"}
    pdl_interp.is_not_null %3241 : !pdl.operation -> ^bb6028, ^bb23
  ^bb6028:
    %3242 = pdl_interp.get_operand 0 of %0
    pdl_interp.is_not_null %3242 : !pdl.value -> ^bb6029, ^bb23
  ^bb6029:
    pdl_interp.is_not_null %3240 : !pdl.value -> ^bb6030, ^bb23
  ^bb6030:
    pdl_interp.switch_operation_name of %3241 to ["arith.mulf", "arith.divf", "arith.constant", "arith.addf", "arith.negf", "math.log"](^bb6031, ^bb6032, ^bb6033, ^bb6034, ^bb6035, ^bb6036) -> ^bb23
  ^bb6031:
    pdl_interp.check_operand_count of %3241 is 2 -> ^bb6037, ^bb23
  ^bb6037:
    pdl_interp.check_result_count of %3241 is 1 -> ^bb6038, ^bb23
  ^bb6038:
    %3243 = pdl_interp.get_result 0 of %3241
    pdl_interp.is_not_null %3243 : !pdl.value -> ^bb6039, ^bb23
  ^bb6039:
    pdl_interp.are_equal %3243, %3240 : !pdl.value -> ^bb6040, ^bb23
  ^bb6040:
    %3244 = pdl_interp.get_operand 0 of %3241
    pdl_interp.is_not_null %3244 : !pdl.value -> ^bb6041, ^bb6042
  ^bb6042:
    %3245 = pdl_interp.get_value_type of %3242 : !pdl.type
    %3246 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3245, %3246 : !pdl.type -> ^bb6043, ^bb23
  ^bb6043:
    pdl_interp.check_type %3245 is f32 -> ^bb6044, ^bb23
  ^bb6044:
    %3247 = pdl_interp.get_value_type of %3243 : !pdl.type
    pdl_interp.are_equal %3247, %3245 : !pdl.type -> ^bb6045, ^bb23
  ^bb6045:
    %3248 = pdl_interp.get_operand 1 of %3241
    pdl_interp.are_equal %3248, %3242 : !pdl.value -> ^bb6046, ^bb23
  ^bb6046:
    %3249 = pdl_interp.get_operand 0 of %3241
    pdl_interp.are_equal %3249, %3242 : !pdl.value -> ^bb6047, ^bb23
  ^bb6047:
    pdl_interp.record_match @rewriters::@cube_unmult(%3242, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb23
  ^bb6041:
    %3250 = pdl_interp.get_operand 1 of %3241
    pdl_interp.is_not_null %3250 : !pdl.value -> ^bb6048, ^bb6042
  ^bb6048:
    %3251 = pdl_interp.get_value_type of %3242 : !pdl.type
    %3252 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3251, %3252 : !pdl.type -> ^bb6049, ^bb6042
  ^bb6049:
    pdl_interp.check_type %3251 is f32 -> ^bb6050, ^bb6042
  ^bb6050:
    %3253 = pdl_interp.get_value_type of %3243 : !pdl.type
    pdl_interp.are_equal %3253, %3251 : !pdl.type -> ^bb6051, ^bb6042
  ^bb6051:
    %3254 = pdl_interp.get_value_type of %3244 : !pdl.type
    pdl_interp.are_equal %3254, %3251 : !pdl.type -> ^bb6052, ^bb6042
  ^bb6052:
    %3255 = pdl_interp.get_value_type of %3250 : !pdl.type
    pdl_interp.are_equal %3255, %3251 : !pdl.type -> ^bb6053, ^bb6042
  ^bb6053:
    pdl_interp.record_match @rewriters::@associate_mulrmul(%3242, %3244, %3250, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb6042
  ^bb6032:
    pdl_interp.check_operand_count of %3241 is 2 -> ^bb6054, ^bb23
  ^bb6054:
    pdl_interp.check_result_count of %3241 is 1 -> ^bb6055, ^bb23
  ^bb6055:
    %3256 = pdl_interp.get_result 0 of %3241
    pdl_interp.is_not_null %3256 : !pdl.value -> ^bb6056, ^bb23
  ^bb6056:
    pdl_interp.are_equal %3256, %3240 : !pdl.value -> ^bb6057, ^bb23
  ^bb6057:
    %3257 = pdl_interp.get_operand 0 of %3241
    pdl_interp.is_not_null %3257 : !pdl.value -> ^bb6058, ^bb23
  ^bb6058:
    %3258 = pdl_interp.get_operand 1 of %3241
    pdl_interp.is_not_null %3258 : !pdl.value -> ^bb6059, ^bb6060
  ^bb6060:
    %3259 = pdl_interp.get_defining_op of %3257 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %3259 : !pdl.operation -> ^bb6061, ^bb23
  ^bb6061:
    %3260 = pdl_interp.get_value_type of %3242 : !pdl.type
    %3261 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3260, %3261 : !pdl.type -> ^bb6062, ^bb6063
  ^bb6063:
    %3262 = pdl_interp.get_operand 1 of %3241
    pdl_interp.is_not_null %3262 : !pdl.value -> ^bb6064, ^bb23
  ^bb6064:
    %3263 = pdl_interp.get_value_type of %3242 : !pdl.type
    %3264 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3263, %3264 : !pdl.type -> ^bb6065, ^bb23
  ^bb6065:
    pdl_interp.check_type %3263 is f32 -> ^bb6066, ^bb23
  ^bb6066:
    pdl_interp.check_operation_name of %3259 is "arith.constant" -> ^bb6067, ^bb23
  ^bb6067:
    pdl_interp.check_operand_count of %3259 is 0 -> ^bb6068, ^bb23
  ^bb6068:
    pdl_interp.check_result_count of %3259 is 1 -> ^bb6069, ^bb23
  ^bb6069:
    %3265 = pdl_interp.get_result 0 of %3259
    pdl_interp.is_not_null %3265 : !pdl.value -> ^bb6070, ^bb23
  ^bb6070:
    pdl_interp.are_equal %3265, %3257 : !pdl.value -> ^bb6071, ^bb23
  ^bb6071:
    %3266 = pdl_interp.get_value_type of %3256 : !pdl.type
    pdl_interp.are_equal %3266, %3263 : !pdl.type -> ^bb6072, ^bb23
  ^bb6072:
    %3267 = pdl_interp.get_value_type of %3262 : !pdl.type
    pdl_interp.are_equal %3267, %3263 : !pdl.type -> ^bb6073, ^bb23
  ^bb6073:
    %3268 = pdl_interp.get_attribute "value" of %3259
    pdl_interp.is_not_null %3268 : !pdl.attribute -> ^bb6074, ^bb23
  ^bb6074:
    pdl_interp.check_attribute %3268 is 1.000000e+00 : f32 -> ^bb6075, ^bb23
  ^bb6075:
    %3269 = pdl_interp.get_value_type of %3265 : !pdl.type
    pdl_interp.are_equal %3269, %3263 : !pdl.type -> ^bb6076, ^bb23
  ^bb6076:
    pdl_interp.record_match @rewriters::@mult_flip_rev(%3242, %3262, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb23
  ^bb6062:
    pdl_interp.check_type %3260 is f32 -> ^bb6077, ^bb6063
  ^bb6077:
    pdl_interp.check_operation_name of %3259 is "arith.constant" -> ^bb6078, ^bb6063
  ^bb6078:
    pdl_interp.check_operand_count of %3259 is 0 -> ^bb6079, ^bb6063
  ^bb6079:
    pdl_interp.check_result_count of %3259 is 1 -> ^bb6080, ^bb6063
  ^bb6080:
    %3270 = pdl_interp.get_result 0 of %3259
    pdl_interp.is_not_null %3270 : !pdl.value -> ^bb6081, ^bb6063
  ^bb6081:
    pdl_interp.are_equal %3270, %3257 : !pdl.value -> ^bb6082, ^bb6063
  ^bb6082:
    %3271 = pdl_interp.get_value_type of %3256 : !pdl.type
    pdl_interp.are_equal %3271, %3260 : !pdl.type -> ^bb6083, ^bb6063
  ^bb6083:
    %3272 = pdl_interp.get_attribute "value" of %3259
    pdl_interp.is_not_null %3272 : !pdl.attribute -> ^bb6084, ^bb6063
  ^bb6084:
    pdl_interp.check_attribute %3272 is 1.000000e+00 : f32 -> ^bb6085, ^bb6063
  ^bb6085:
    %3273 = pdl_interp.get_value_type of %3270 : !pdl.type
    pdl_interp.are_equal %3273, %3260 : !pdl.type -> ^bb6086, ^bb6063
  ^bb6086:
    %3274 = pdl_interp.get_operand 1 of %3241
    pdl_interp.are_equal %3274, %3242 : !pdl.value -> ^bb6087, ^bb6063
  ^bb6087:
    pdl_interp.record_match @rewriters::@rgt_mult_inverse(%0 : !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb6063
  ^bb6059:
    %3275 = pdl_interp.get_value_type of %3242 : !pdl.type
    %3276 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3275, %3276 : !pdl.type -> ^bb6088, ^bb6060
  ^bb6088:
    pdl_interp.check_type %3275 is f32 -> ^bb6089, ^bb6060
  ^bb6089:
    %3277 = pdl_interp.get_value_type of %3256 : !pdl.type
    pdl_interp.are_equal %3277, %3275 : !pdl.type -> ^bb6090, ^bb6060
  ^bb6090:
    %3278 = pdl_interp.get_value_type of %3257 : !pdl.type
    pdl_interp.are_equal %3278, %3275 : !pdl.type -> ^bb6091, ^bb6060
  ^bb6091:
    %3279 = pdl_interp.get_value_type of %3258 : !pdl.type
    pdl_interp.are_equal %3279, %3275 : !pdl.type -> ^bb6092, ^bb6060
  ^bb6092:
    pdl_interp.record_match @rewriters::@associate_mulrdiv(%3242, %3257, %3258, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb6060
  ^bb6033:
    pdl_interp.check_operand_count of %3241 is 0 -> ^bb6093, ^bb23
  ^bb6093:
    pdl_interp.check_result_count of %3241 is 1 -> ^bb6094, ^bb23
  ^bb6094:
    %3280 = pdl_interp.get_result 0 of %3241
    pdl_interp.is_not_null %3280 : !pdl.value -> ^bb6095, ^bb23
  ^bb6095:
    pdl_interp.are_equal %3280, %3240 : !pdl.value -> ^bb6096, ^bb23
  ^bb6096:
    %3281 = pdl_interp.get_value_type of %3242 : !pdl.type
    %3282 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3281, %3282 : !pdl.type -> ^bb6097, ^bb23
  ^bb6097:
    pdl_interp.check_type %3281 is f32 -> ^bb6098, ^bb23
  ^bb6098:
    %3283 = pdl_interp.get_value_type of %3280 : !pdl.type
    pdl_interp.are_equal %3283, %3281 : !pdl.type -> ^bb6099, ^bb23
  ^bb6099:
    %3284 = pdl_interp.get_attribute "value" of %3241
    pdl_interp.is_not_null %3284 : !pdl.attribute -> ^bb6100, ^bb23
  ^bb6100:
    pdl_interp.switch_attribute %3284 to [0.000000e+00 : f32, 1.000000e+00 : f32](^bb6101, ^bb6102) -> ^bb23
  ^bb6101:
    pdl_interp.record_match @rewriters::@mul0_rgt(%0 : !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb23
  ^bb6102:
    pdl_interp.record_match @rewriters::@mul_rgt_identity(%3242, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb23
  ^bb6034:
    pdl_interp.check_operand_count of %3241 is 2 -> ^bb6103, ^bb23
  ^bb6103:
    pdl_interp.check_result_count of %3241 is 1 -> ^bb6104, ^bb23
  ^bb6104:
    %3285 = pdl_interp.get_result 0 of %3241
    pdl_interp.is_not_null %3285 : !pdl.value -> ^bb6105, ^bb23
  ^bb6105:
    pdl_interp.are_equal %3285, %3240 : !pdl.value -> ^bb6106, ^bb23
  ^bb6106:
    %3286 = pdl_interp.get_operand 0 of %3241
    pdl_interp.is_not_null %3286 : !pdl.value -> ^bb6107, ^bb23
  ^bb6107:
    %3287 = pdl_interp.get_operand 1 of %3241
    pdl_interp.is_not_null %3287 : !pdl.value -> ^bb6108, ^bb23
  ^bb6108:
    %3288 = pdl_interp.get_value_type of %3242 : !pdl.type
    %3289 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3288, %3289 : !pdl.type -> ^bb6109, ^bb23
  ^bb6109:
    pdl_interp.check_type %3288 is f32 -> ^bb6110, ^bb23
  ^bb6110:
    %3290 = pdl_interp.get_value_type of %3285 : !pdl.type
    pdl_interp.are_equal %3290, %3288 : !pdl.type -> ^bb6111, ^bb23
  ^bb6111:
    %3291 = pdl_interp.get_value_type of %3286 : !pdl.type
    pdl_interp.are_equal %3291, %3288 : !pdl.type -> ^bb6112, ^bb23
  ^bb6112:
    %3292 = pdl_interp.get_value_type of %3287 : !pdl.type
    pdl_interp.are_equal %3292, %3288 : !pdl.type -> ^bb6113, ^bb23
  ^bb6113:
    pdl_interp.record_match @rewriters::@distribute_rgt_in(%3286, %3242, %3287, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb6114
  ^bb6114:
    pdl_interp.record_match @rewriters::@distribute_lft_in(%3242, %3286, %3287, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb23
  ^bb6035:
    pdl_interp.check_operand_count of %3241 is 1 -> ^bb6115, ^bb23
  ^bb6115:
    pdl_interp.check_result_count of %3241 is 1 -> ^bb6116, ^bb23
  ^bb6116:
    %3293 = pdl_interp.get_result 0 of %3241
    pdl_interp.is_not_null %3293 : !pdl.value -> ^bb6117, ^bb23
  ^bb6117:
    pdl_interp.are_equal %3293, %3240 : !pdl.value -> ^bb6118, ^bb23
  ^bb6118:
    %3294 = pdl_interp.get_operand 0 of %3241
    pdl_interp.is_not_null %3294 : !pdl.value -> ^bb6119, ^bb23
  ^bb6119:
    %3295 = pdl_interp.get_value_type of %3242 : !pdl.type
    %3296 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3295, %3296 : !pdl.type -> ^bb6120, ^bb23
  ^bb6120:
    pdl_interp.check_type %3295 is f32 -> ^bb6121, ^bb23
  ^bb6121:
    %3297 = pdl_interp.get_value_type of %3293 : !pdl.type
    pdl_interp.are_equal %3297, %3295 : !pdl.type -> ^bb6122, ^bb23
  ^bb6122:
    %3298 = pdl_interp.get_value_type of %3294 : !pdl.type
    pdl_interp.are_equal %3298, %3295 : !pdl.type -> ^bb6123, ^bb23
  ^bb6123:
    pdl_interp.record_match @rewriters::@distribute_rgt_neg_out(%3242, %3294, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb23
  ^bb6036:
    pdl_interp.check_operand_count of %3241 is 1 -> ^bb6124, ^bb23
  ^bb6124:
    pdl_interp.check_result_count of %3241 is 1 -> ^bb6125, ^bb23
  ^bb6125:
    %3299 = pdl_interp.get_result 0 of %3241
    pdl_interp.is_not_null %3299 : !pdl.value -> ^bb6126, ^bb23
  ^bb6126:
    pdl_interp.are_equal %3299, %3240 : !pdl.value -> ^bb6127, ^bb23
  ^bb6127:
    %3300 = pdl_interp.get_operand 0 of %3241
    pdl_interp.is_not_null %3300 : !pdl.value -> ^bb6128, ^bb23
  ^bb6128:
    %3301 = pdl_interp.get_value_type of %3242 : !pdl.type
    %3302 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3301, %3302 : !pdl.type -> ^bb6129, ^bb23
  ^bb6129:
    pdl_interp.check_type %3301 is f32 -> ^bb6130, ^bb23
  ^bb6130:
    %3303 = pdl_interp.get_value_type of %3299 : !pdl.type
    pdl_interp.are_equal %3303, %3301 : !pdl.type -> ^bb6131, ^bb23
  ^bb6131:
    %3304 = pdl_interp.get_value_type of %3300 : !pdl.type
    pdl_interp.are_equal %3304, %3301 : !pdl.type -> ^bb6132, ^bb23
  ^bb6132:
    pdl_interp.record_match @rewriters::@log_pow_rev(%3300, %3242, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb23
  ^bb6026:
    %3305 = pdl_interp.get_operand 1 of %0
    pdl_interp.is_not_null %3305 : !pdl.value -> ^bb6133, ^bb6134
  ^bb6134:
    %3306 = pdl_interp.get_value_type of %3239 : !pdl.type
    %3307 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3306, %3307 : !pdl.type -> ^bb6135, ^bb6027
  ^bb6135:
    pdl_interp.check_type %3306 is f32 -> ^bb6136, ^bb6027
  ^bb6136:
    %3308 = pdl_interp.get_operand 1 of %0
    pdl_interp.are_equal %3239, %3308 : !pdl.value -> ^bb6137, ^bb6027
  ^bb6137:
    pdl_interp.record_match @rewriters::@pow2(%3239, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb6138
  ^bb6138:
    pdl_interp.record_match @rewriters::@sqr_neg_rev(%3239, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb6139
  ^bb6139:
    pdl_interp.record_match @rewriters::@sqr_abs_rev(%3239, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb6027
  ^bb6133:
    %3309 = pdl_interp.get_value_type of %3239 : !pdl.type
    %3310 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3309, %3310 : !pdl.type -> ^bb6140, ^bb6134
  ^bb6140:
    pdl_interp.check_type %3309 is f32 -> ^bb6141, ^bb6134
  ^bb6141:
    %3311 = pdl_interp.get_value_type of %3305 : !pdl.type
    pdl_interp.are_equal %3309, %3311 : !pdl.type -> ^bb6142, ^bb6134
  ^bb6142:
    pdl_interp.record_match @rewriters::@mul_commutative(%3305, %3239, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.mulf") -> ^bb6134
  ^bb4:
    pdl_interp.check_operand_count of %0 is 2 -> ^bb6143, ^bb23
  ^bb6143:
    pdl_interp.check_result_count of %0 is 1 -> ^bb6144, ^bb23
  ^bb6144:
    %3312 = pdl_interp.get_operand 1 of %0
    %3313 = pdl_interp.get_defining_op of %3312 : !pdl.value {position = "root.operand[1].defining_op"}
    pdl_interp.is_not_null %3313 : !pdl.operation -> ^bb6145, ^bb6146
  ^bb6146:
    %3314 = pdl_interp.get_operand 0 of %0
    pdl_interp.is_not_null %3314 : !pdl.value -> ^bb6147, ^bb23
  ^bb6147:
    %3315 = pdl_interp.get_value_type of %3314 : !pdl.type
    %3316 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3315, %3316 : !pdl.type -> ^bb6148, ^bb6149
  ^bb6149:
    %3317 = pdl_interp.get_operand 1 of %0
    pdl_interp.is_not_null %3317 : !pdl.value -> ^bb6150, ^bb23
  ^bb6150:
    %3318 = pdl_interp.get_value_type of %3314 : !pdl.type
    %3319 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3318, %3319 : !pdl.type -> ^bb6151, ^bb23
  ^bb6151:
    pdl_interp.check_type %3318 is f32 -> ^bb6152, ^bb23
  ^bb6152:
    %3320 = pdl_interp.get_value_type of %3317 : !pdl.type
    pdl_interp.are_equal %3318, %3320 : !pdl.type -> ^bb6153, ^bb23
  ^bb6153:
    pdl_interp.record_match @rewriters::@sub_negate_rev(%3317, %3314, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb6154
  ^bb6154:
    pdl_interp.record_match @rewriters::@sub_flip(%3317, %3314, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb23
  ^bb6148:
    pdl_interp.check_type %3315 is f32 -> ^bb6155, ^bb6149
  ^bb6155:
    %3321 = pdl_interp.get_operand 1 of %0
    pdl_interp.are_equal %3314, %3321 : !pdl.value -> ^bb6156, ^bb6149
  ^bb6156:
    pdl_interp.record_match @rewriters::@add_inverses(%0 : !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb6149
  ^bb6145:
    %3322 = pdl_interp.get_operand 0 of %0
    pdl_interp.is_not_null %3322 : !pdl.value -> ^bb6157, ^bb6146
  ^bb6157:
    pdl_interp.is_not_null %3312 : !pdl.value -> ^bb6158, ^bb6146
  ^bb6158:
    pdl_interp.switch_operation_name of %3313 to ["arith.addf", "arith.subf", "arith.constant", "arith.mulf", "arith.negf", "arith.divf"](^bb6159, ^bb6160, ^bb6161, ^bb6162, ^bb6163, ^bb6164) -> ^bb6146
  ^bb6159:
    pdl_interp.check_operand_count of %3313 is 2 -> ^bb6165, ^bb6146
  ^bb6165:
    pdl_interp.check_result_count of %3313 is 1 -> ^bb6166, ^bb6146
  ^bb6166:
    %3323 = pdl_interp.get_result 0 of %3313
    pdl_interp.is_not_null %3323 : !pdl.value -> ^bb6167, ^bb6146
  ^bb6167:
    pdl_interp.are_equal %3323, %3312 : !pdl.value -> ^bb6168, ^bb6146
  ^bb6168:
    %3324 = pdl_interp.get_operand 0 of %3313
    pdl_interp.is_not_null %3324 : !pdl.value -> ^bb6169, ^bb6146
  ^bb6169:
    %3325 = pdl_interp.get_operand 1 of %3313
    pdl_interp.is_not_null %3325 : !pdl.value -> ^bb6170, ^bb6146
  ^bb6170:
    %3326 = pdl_interp.get_value_type of %3322 : !pdl.type
    %3327 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3326, %3327 : !pdl.type -> ^bb6171, ^bb6146
  ^bb6171:
    pdl_interp.check_type %3326 is f32 -> ^bb6172, ^bb6146
  ^bb6172:
    %3328 = pdl_interp.get_value_type of %3323 : !pdl.type
    pdl_interp.are_equal %3328, %3326 : !pdl.type -> ^bb6173, ^bb6146
  ^bb6173:
    %3329 = pdl_interp.get_value_type of %3324 : !pdl.type
    pdl_interp.are_equal %3329, %3326 : !pdl.type -> ^bb6174, ^bb6146
  ^bb6174:
    %3330 = pdl_interp.get_value_type of %3325 : !pdl.type
    pdl_interp.are_equal %3330, %3326 : !pdl.type -> ^bb6175, ^bb6146
  ^bb6175:
    pdl_interp.record_match @rewriters::@associatesub_radd(%3322, %3324, %3325, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb6146
  ^bb6160:
    pdl_interp.check_operand_count of %3313 is 2 -> ^bb6176, ^bb6146
  ^bb6176:
    pdl_interp.check_result_count of %3313 is 1 -> ^bb6177, ^bb6146
  ^bb6177:
    %3331 = pdl_interp.get_result 0 of %3313
    pdl_interp.is_not_null %3331 : !pdl.value -> ^bb6178, ^bb6146
  ^bb6178:
    pdl_interp.are_equal %3331, %3312 : !pdl.value -> ^bb6179, ^bb6146
  ^bb6179:
    %3332 = pdl_interp.get_operand 0 of %3313
    pdl_interp.is_not_null %3332 : !pdl.value -> ^bb6180, ^bb6146
  ^bb6180:
    %3333 = pdl_interp.get_operand 1 of %3313
    pdl_interp.is_not_null %3333 : !pdl.value -> ^bb6181, ^bb6146
  ^bb6181:
    %3334 = pdl_interp.get_value_type of %3322 : !pdl.type
    %3335 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3334, %3335 : !pdl.type -> ^bb6182, ^bb6146
  ^bb6182:
    pdl_interp.check_type %3334 is f32 -> ^bb6183, ^bb6146
  ^bb6183:
    %3336 = pdl_interp.get_value_type of %3331 : !pdl.type
    pdl_interp.are_equal %3336, %3334 : !pdl.type -> ^bb6184, ^bb6146
  ^bb6184:
    %3337 = pdl_interp.get_value_type of %3332 : !pdl.type
    pdl_interp.are_equal %3337, %3334 : !pdl.type -> ^bb6185, ^bb6146
  ^bb6185:
    %3338 = pdl_interp.get_value_type of %3333 : !pdl.type
    pdl_interp.are_equal %3338, %3334 : !pdl.type -> ^bb6186, ^bb6146
  ^bb6186:
    pdl_interp.record_match @rewriters::@associatesub_r_(%3322, %3332, %3333, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb6146
  ^bb6161:
    pdl_interp.check_operand_count of %3313 is 0 -> ^bb6187, ^bb6146
  ^bb6187:
    pdl_interp.check_result_count of %3313 is 1 -> ^bb6188, ^bb6146
  ^bb6188:
    %3339 = pdl_interp.get_result 0 of %3313
    pdl_interp.is_not_null %3339 : !pdl.value -> ^bb6189, ^bb6146
  ^bb6189:
    pdl_interp.are_equal %3339, %3312 : !pdl.value -> ^bb6190, ^bb6146
  ^bb6190:
    %3340 = pdl_interp.get_value_type of %3322 : !pdl.type
    %3341 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3340, %3341 : !pdl.type -> ^bb6191, ^bb6146
  ^bb6191:
    pdl_interp.check_type %3340 is f32 -> ^bb6192, ^bb6146
  ^bb6192:
    %3342 = pdl_interp.get_value_type of %3339 : !pdl.type
    pdl_interp.are_equal %3342, %3340 : !pdl.type -> ^bb6193, ^bb6146
  ^bb6193:
    %3343 = pdl_interp.get_attribute "value" of %3313
    pdl_interp.is_not_null %3343 : !pdl.attribute -> ^bb6194, ^bb6146
  ^bb6194:
    pdl_interp.check_attribute %3343 is 0.000000e+00 : f32 -> ^bb6195, ^bb6146
  ^bb6195:
    pdl_interp.record_match @rewriters::@sub_rgt_identity(%3322, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb6146
  ^bb6162:
    pdl_interp.check_operand_count of %3313 is 2 -> ^bb6196, ^bb6146
  ^bb6196:
    pdl_interp.check_result_count of %3313 is 1 -> ^bb6197, ^bb6146
  ^bb6197:
    %3344 = pdl_interp.get_result 0 of %3313
    pdl_interp.is_not_null %3344 : !pdl.value -> ^bb6198, ^bb6146
  ^bb6198:
    pdl_interp.are_equal %3344, %3312 : !pdl.value -> ^bb6199, ^bb6146
  ^bb6199:
    %3345 = pdl_interp.get_operand 0 of %3313
    pdl_interp.is_not_null %3345 : !pdl.value -> ^bb6200, ^bb6146
  ^bb6200:
    %3346 = pdl_interp.get_defining_op of %3345 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %3346 : !pdl.operation -> ^bb6201, ^bb6202
  ^bb6202:
    %3347 = pdl_interp.get_operand 1 of %3313
    pdl_interp.is_not_null %3347 : !pdl.value -> ^bb6203, ^bb6146
  ^bb6203:
    %3348 = pdl_interp.get_value_type of %3322 : !pdl.type
    %3349 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3348, %3349 : !pdl.type -> ^bb6204, ^bb6146
  ^bb6204:
    pdl_interp.check_type %3348 is f32 -> ^bb6205, ^bb6146
  ^bb6205:
    %3350 = pdl_interp.get_value_type of %3344 : !pdl.type
    pdl_interp.are_equal %3350, %3348 : !pdl.type -> ^bb6206, ^bb6146
  ^bb6206:
    %3351 = pdl_interp.get_value_type of %3345 : !pdl.type
    pdl_interp.are_equal %3351, %3348 : !pdl.type -> ^bb6207, ^bb6146
  ^bb6207:
    %3352 = pdl_interp.get_value_type of %3347 : !pdl.type
    pdl_interp.are_equal %3352, %3348 : !pdl.type -> ^bb6208, ^bb6146
  ^bb6208:
    pdl_interp.record_match @rewriters::@fp_cancel_sub_sign_inv(%3345, %3347, %3322, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb6146
  ^bb6201:
    %3353 = pdl_interp.get_operand 1 of %3313
    pdl_interp.is_not_null %3353 : !pdl.value -> ^bb6209, ^bb6202
  ^bb6209:
    %3354 = pdl_interp.get_value_type of %3322 : !pdl.type
    %3355 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3354, %3355 : !pdl.type -> ^bb6210, ^bb6202
  ^bb6210:
    pdl_interp.check_type %3354 is f32 -> ^bb6211, ^bb6202
  ^bb6211:
    pdl_interp.check_operation_name of %3346 is "arith.negf" -> ^bb6212, ^bb6202
  ^bb6212:
    pdl_interp.check_operand_count of %3346 is 1 -> ^bb6213, ^bb6202
  ^bb6213:
    pdl_interp.check_result_count of %3346 is 1 -> ^bb6214, ^bb6202
  ^bb6214:
    %3356 = pdl_interp.get_result 0 of %3346
    pdl_interp.is_not_null %3356 : !pdl.value -> ^bb6215, ^bb6202
  ^bb6215:
    pdl_interp.are_equal %3356, %3345 : !pdl.value -> ^bb6216, ^bb6202
  ^bb6216:
    %3357 = pdl_interp.get_value_type of %3344 : !pdl.type
    pdl_interp.are_equal %3357, %3354 : !pdl.type -> ^bb6217, ^bb6202
  ^bb6217:
    %3358 = pdl_interp.get_operand 0 of %3346
    pdl_interp.is_not_null %3358 : !pdl.value -> ^bb6218, ^bb6202
  ^bb6218:
    %3359 = pdl_interp.get_value_type of %3353 : !pdl.type
    pdl_interp.are_equal %3359, %3354 : !pdl.type -> ^bb6219, ^bb6202
  ^bb6219:
    %3360 = pdl_interp.get_value_type of %3356 : !pdl.type
    pdl_interp.are_equal %3360, %3354 : !pdl.type -> ^bb6220, ^bb6202
  ^bb6220:
    %3361 = pdl_interp.get_value_type of %3358 : !pdl.type
    pdl_interp.are_equal %3361, %3354 : !pdl.type -> ^bb6221, ^bb6202
  ^bb6221:
    pdl_interp.record_match @rewriters::@fp_cancel_sign_sub(%3358, %3353, %3322, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb6202
  ^bb6163:
    pdl_interp.check_operand_count of %3313 is 1 -> ^bb6222, ^bb6146
  ^bb6222:
    pdl_interp.check_result_count of %3313 is 1 -> ^bb6223, ^bb6146
  ^bb6223:
    %3362 = pdl_interp.get_result 0 of %3313
    pdl_interp.is_not_null %3362 : !pdl.value -> ^bb6224, ^bb6146
  ^bb6224:
    pdl_interp.are_equal %3362, %3312 : !pdl.value -> ^bb6225, ^bb6146
  ^bb6225:
    %3363 = pdl_interp.get_operand 0 of %3313
    pdl_interp.is_not_null %3363 : !pdl.value -> ^bb6226, ^bb6146
  ^bb6226:
    %3364 = pdl_interp.get_value_type of %3322 : !pdl.type
    %3365 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3364, %3365 : !pdl.type -> ^bb6227, ^bb6146
  ^bb6227:
    pdl_interp.check_type %3364 is f32 -> ^bb6228, ^bb6146
  ^bb6228:
    %3366 = pdl_interp.get_value_type of %3362 : !pdl.type
    pdl_interp.are_equal %3366, %3364 : !pdl.type -> ^bb6229, ^bb6146
  ^bb6229:
    %3367 = pdl_interp.get_value_type of %3363 : !pdl.type
    pdl_interp.are_equal %3367, %3364 : !pdl.type -> ^bb6230, ^bb6146
  ^bb6230:
    pdl_interp.record_match @rewriters::@add_flip_rev(%3322, %3363, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb6146
  ^bb6164:
    pdl_interp.check_operand_count of %3313 is 2 -> ^bb6231, ^bb6146
  ^bb6231:
    pdl_interp.check_result_count of %3313 is 1 -> ^bb6232, ^bb6146
  ^bb6232:
    %3368 = pdl_interp.get_result 0 of %3313
    pdl_interp.is_not_null %3368 : !pdl.value -> ^bb6233, ^bb6146
  ^bb6233:
    pdl_interp.are_equal %3368, %3312 : !pdl.value -> ^bb6234, ^bb6146
  ^bb6234:
    %3369 = pdl_interp.get_operand 0 of %3313
    pdl_interp.is_not_null %3369 : !pdl.value -> ^bb6235, ^bb6146
  ^bb6235:
    %3370 = pdl_interp.get_operand 1 of %3313
    pdl_interp.is_not_null %3370 : !pdl.value -> ^bb6236, ^bb6146
  ^bb6236:
    %3371 = pdl_interp.get_value_type of %3322 : !pdl.type
    %3372 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3371, %3372 : !pdl.type -> ^bb6237, ^bb6146
  ^bb6237:
    pdl_interp.check_type %3371 is f32 -> ^bb6238, ^bb6146
  ^bb6238:
    %3373 = pdl_interp.get_value_type of %3368 : !pdl.type
    pdl_interp.are_equal %3373, %3371 : !pdl.type -> ^bb6239, ^bb6146
  ^bb6239:
    %3374 = pdl_interp.get_value_type of %3369 : !pdl.type
    pdl_interp.are_equal %3374, %3371 : !pdl.type -> ^bb6240, ^bb6146
  ^bb6240:
    %3375 = pdl_interp.get_value_type of %3370 : !pdl.type
    pdl_interp.are_equal %3375, %3371 : !pdl.type -> ^bb6241, ^bb6146
  ^bb6241:
    pdl_interp.record_match @rewriters::@sub_to_fraction(%3322, %3370, %3369, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.subf") -> ^bb6146
  ^bb5:
    pdl_interp.check_operand_count of %0 is 2 -> ^bb6242, ^bb23
  ^bb6242:
    pdl_interp.check_result_count of %0 is 1 -> ^bb6243, ^bb23
  ^bb6243:
    %3376 = pdl_interp.get_operand 1 of %0
    %3377 = pdl_interp.get_defining_op of %3376 : !pdl.value {position = "root.operand[1].defining_op"}
    pdl_interp.is_not_null %3377 : !pdl.operation -> ^bb6244, ^bb6245
  ^bb6245:
    %3378 = pdl_interp.get_operand 0 of %0
    pdl_interp.is_not_null %3378 : !pdl.value -> ^bb6246, ^bb23
  ^bb6246:
    %3379 = pdl_interp.get_value_type of %3378 : !pdl.type
    %3380 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3379, %3380 : !pdl.type -> ^bb6247, ^bb6248
  ^bb6248:
    %3381 = pdl_interp.get_operand 1 of %0
    pdl_interp.is_not_null %3381 : !pdl.value -> ^bb6249, ^bb23
  ^bb6249:
    %3382 = pdl_interp.get_value_type of %3378 : !pdl.type
    %3383 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3382, %3383 : !pdl.type -> ^bb6250, ^bb23
  ^bb6250:
    pdl_interp.check_type %3382 is f32 -> ^bb6251, ^bb23
  ^bb6251:
    %3384 = pdl_interp.get_value_type of %3381 : !pdl.type
    pdl_interp.are_equal %3382, %3384 : !pdl.type -> ^bb6252, ^bb23
  ^bb6252:
    pdl_interp.record_match @rewriters::@frac_2neg(%3378, %3381, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb6253
  ^bb6253:
    pdl_interp.record_match @rewriters::@mult_flip(%3381, %3378, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb23
  ^bb6247:
    pdl_interp.check_type %3379 is f32 -> ^bb6254, ^bb6248
  ^bb6254:
    %3385 = pdl_interp.get_operand 1 of %0
    pdl_interp.are_equal %3378, %3385 : !pdl.value -> ^bb6255, ^bb6248
  ^bb6255:
    pdl_interp.record_match @rewriters::@mul_inverses(%0 : !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb6248
  ^bb6244:
    %3386 = pdl_interp.get_operand 0 of %0
    pdl_interp.is_not_null %3386 : !pdl.value -> ^bb6256, ^bb6245
  ^bb6256:
    pdl_interp.is_not_null %3376 : !pdl.value -> ^bb6257, ^bb6245
  ^bb6257:
    pdl_interp.switch_operation_name of %3377 to ["arith.mulf", "arith.divf", "arith.constant", "arith.negf", "math.absf", "math.sqrt"](^bb6258, ^bb6259, ^bb6260, ^bb6261, ^bb6262, ^bb6263) -> ^bb6245
  ^bb6258:
    pdl_interp.check_operand_count of %3377 is 2 -> ^bb6264, ^bb6245
  ^bb6264:
    pdl_interp.check_result_count of %3377 is 1 -> ^bb6265, ^bb6245
  ^bb6265:
    %3387 = pdl_interp.get_result 0 of %3377
    pdl_interp.is_not_null %3387 : !pdl.value -> ^bb6266, ^bb6245
  ^bb6266:
    pdl_interp.are_equal %3387, %3376 : !pdl.value -> ^bb6267, ^bb6245
  ^bb6267:
    %3388 = pdl_interp.get_operand 0 of %3377
    pdl_interp.is_not_null %3388 : !pdl.value -> ^bb6268, ^bb6245
  ^bb6268:
    %3389 = pdl_interp.get_operand 1 of %3377
    pdl_interp.is_not_null %3389 : !pdl.value -> ^bb6269, ^bb6245
  ^bb6269:
    %3390 = pdl_interp.get_value_type of %3386 : !pdl.type
    %3391 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3390, %3391 : !pdl.type -> ^bb6270, ^bb6245
  ^bb6270:
    pdl_interp.check_type %3390 is f32 -> ^bb6271, ^bb6245
  ^bb6271:
    %3392 = pdl_interp.get_value_type of %3387 : !pdl.type
    pdl_interp.are_equal %3392, %3390 : !pdl.type -> ^bb6272, ^bb6245
  ^bb6272:
    %3393 = pdl_interp.get_value_type of %3388 : !pdl.type
    pdl_interp.are_equal %3393, %3390 : !pdl.type -> ^bb6273, ^bb6245
  ^bb6273:
    %3394 = pdl_interp.get_value_type of %3389 : !pdl.type
    pdl_interp.are_equal %3394, %3390 : !pdl.type -> ^bb6274, ^bb6245
  ^bb6274:
    pdl_interp.record_match @rewriters::@associate_divrmul(%3386, %3388, %3389, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb6245
  ^bb6259:
    pdl_interp.check_operand_count of %3377 is 2 -> ^bb6275, ^bb6245
  ^bb6275:
    pdl_interp.check_result_count of %3377 is 1 -> ^bb6276, ^bb6245
  ^bb6276:
    %3395 = pdl_interp.get_result 0 of %3377
    pdl_interp.is_not_null %3395 : !pdl.value -> ^bb6277, ^bb6245
  ^bb6277:
    pdl_interp.are_equal %3395, %3376 : !pdl.value -> ^bb6278, ^bb6245
  ^bb6278:
    %3396 = pdl_interp.get_operand 0 of %3377
    pdl_interp.is_not_null %3396 : !pdl.value -> ^bb6279, ^bb6245
  ^bb6279:
    %3397 = pdl_interp.get_operand 1 of %3377
    pdl_interp.is_not_null %3397 : !pdl.value -> ^bb6280, ^bb6245
  ^bb6280:
    %3398 = pdl_interp.get_value_type of %3386 : !pdl.type
    %3399 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3398, %3399 : !pdl.type -> ^bb6281, ^bb6245
  ^bb6281:
    pdl_interp.check_type %3398 is f32 -> ^bb6282, ^bb6245
  ^bb6282:
    %3400 = pdl_interp.get_value_type of %3395 : !pdl.type
    pdl_interp.are_equal %3400, %3398 : !pdl.type -> ^bb6283, ^bb6245
  ^bb6283:
    %3401 = pdl_interp.get_value_type of %3396 : !pdl.type
    pdl_interp.are_equal %3401, %3398 : !pdl.type -> ^bb6284, ^bb6245
  ^bb6284:
    %3402 = pdl_interp.get_value_type of %3397 : !pdl.type
    pdl_interp.are_equal %3402, %3398 : !pdl.type -> ^bb6285, ^bb6245
  ^bb6285:
    pdl_interp.record_match @rewriters::@associate_divrdiv(%3386, %3396, %3397, %0 : !pdl.value, !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb6245
  ^bb6260:
    pdl_interp.check_operand_count of %3377 is 0 -> ^bb6286, ^bb6245
  ^bb6286:
    pdl_interp.check_result_count of %3377 is 1 -> ^bb6287, ^bb6245
  ^bb6287:
    %3403 = pdl_interp.get_result 0 of %3377
    pdl_interp.is_not_null %3403 : !pdl.value -> ^bb6288, ^bb6245
  ^bb6288:
    pdl_interp.are_equal %3403, %3376 : !pdl.value -> ^bb6289, ^bb6245
  ^bb6289:
    %3404 = pdl_interp.get_value_type of %3386 : !pdl.type
    %3405 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3404, %3405 : !pdl.type -> ^bb6290, ^bb6245
  ^bb6290:
    pdl_interp.check_type %3404 is f32 -> ^bb6291, ^bb6245
  ^bb6291:
    %3406 = pdl_interp.get_value_type of %3403 : !pdl.type
    pdl_interp.are_equal %3406, %3404 : !pdl.type -> ^bb6292, ^bb6245
  ^bb6292:
    %3407 = pdl_interp.get_attribute "value" of %3377
    pdl_interp.is_not_null %3407 : !pdl.attribute -> ^bb6293, ^bb6245
  ^bb6293:
    pdl_interp.check_attribute %3407 is 1.000000e+00 : f32 -> ^bb6294, ^bb6245
  ^bb6294:
    pdl_interp.record_match @rewriters::@div_rgt_identity(%3386, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb6245
  ^bb6261:
    pdl_interp.check_operand_count of %3377 is 1 -> ^bb6295, ^bb6245
  ^bb6295:
    pdl_interp.check_result_count of %3377 is 1 -> ^bb6296, ^bb6245
  ^bb6296:
    %3408 = pdl_interp.get_result 0 of %3377
    pdl_interp.is_not_null %3408 : !pdl.value -> ^bb6297, ^bb6245
  ^bb6297:
    pdl_interp.are_equal %3408, %3376 : !pdl.value -> ^bb6298, ^bb6245
  ^bb6298:
    %3409 = pdl_interp.get_operand 0 of %3377
    pdl_interp.is_not_null %3409 : !pdl.value -> ^bb6299, ^bb6245
  ^bb6299:
    %3410 = pdl_interp.get_value_type of %3386 : !pdl.type
    %3411 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3410, %3411 : !pdl.type -> ^bb6300, ^bb6245
  ^bb6300:
    pdl_interp.check_type %3410 is f32 -> ^bb6301, ^bb6245
  ^bb6301:
    %3412 = pdl_interp.get_value_type of %3408 : !pdl.type
    pdl_interp.are_equal %3412, %3410 : !pdl.type -> ^bb6302, ^bb6245
  ^bb6302:
    %3413 = pdl_interp.get_value_type of %3409 : !pdl.type
    pdl_interp.are_equal %3413, %3410 : !pdl.type -> ^bb6303, ^bb6245
  ^bb6303:
    pdl_interp.record_match @rewriters::@distribute_frac_neg2(%3386, %3409, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb6245
  ^bb6262:
    pdl_interp.check_operand_count of %3377 is 1 -> ^bb6304, ^bb6245
  ^bb6304:
    pdl_interp.check_result_count of %3377 is 1 -> ^bb6305, ^bb6245
  ^bb6305:
    %3414 = pdl_interp.get_result 0 of %3377
    pdl_interp.is_not_null %3414 : !pdl.value -> ^bb6306, ^bb6245
  ^bb6306:
    pdl_interp.are_equal %3414, %3376 : !pdl.value -> ^bb6307, ^bb6245
  ^bb6307:
    %3415 = pdl_interp.get_value_type of %3386 : !pdl.type
    %3416 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3415, %3416 : !pdl.type -> ^bb6308, ^bb6245
  ^bb6308:
    pdl_interp.check_type %3415 is f32 -> ^bb6309, ^bb6245
  ^bb6309:
    %3417 = pdl_interp.get_value_type of %3414 : !pdl.type
    pdl_interp.are_equal %3417, %3415 : !pdl.type -> ^bb6310, ^bb6245
  ^bb6310:
    %3418 = pdl_interp.get_operand 0 of %3377
    pdl_interp.are_equal %3418, %3386 : !pdl.value -> ^bb6311, ^bb6245
  ^bb6311:
    pdl_interp.record_match @rewriters::@fabs_rhs_div(%3386, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb6245
  ^bb6263:
    pdl_interp.check_operand_count of %3377 is 1 -> ^bb6312, ^bb6245
  ^bb6312:
    pdl_interp.check_result_count of %3377 is 1 -> ^bb6313, ^bb6245
  ^bb6313:
    %3419 = pdl_interp.get_result 0 of %3377
    pdl_interp.is_not_null %3419 : !pdl.value -> ^bb6314, ^bb6245
  ^bb6314:
    pdl_interp.are_equal %3419, %3376 : !pdl.value -> ^bb6315, ^bb6245
  ^bb6315:
    %3420 = pdl_interp.get_operand 0 of %3377
    pdl_interp.is_not_null %3420 : !pdl.value -> ^bb6316, ^bb6245
  ^bb6316:
    %3421 = pdl_interp.get_defining_op of %3420 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %3421 : !pdl.operation -> ^bb6317, ^bb6245
  ^bb6317:
    %3422 = pdl_interp.get_value_type of %3386 : !pdl.type
    %3423 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3422, %3423 : !pdl.type -> ^bb6318, ^bb6245
  ^bb6318:
    pdl_interp.check_type %3422 is f32 -> ^bb6319, ^bb6245
  ^bb6319:
    pdl_interp.switch_operation_name of %3421 to ["arith.subf", "arith.addf"](^bb6320, ^bb6321) -> ^bb6245
  ^bb6320:
    pdl_interp.check_operand_count of %3421 is 2 -> ^bb6322, ^bb6245
  ^bb6322:
    pdl_interp.check_result_count of %3421 is 1 -> ^bb6323, ^bb6245
  ^bb6323:
    %3424 = pdl_interp.get_result 0 of %3421
    pdl_interp.is_not_null %3424 : !pdl.value -> ^bb6324, ^bb6245
  ^bb6324:
    pdl_interp.are_equal %3424, %3420 : !pdl.value -> ^bb6325, ^bb6245
  ^bb6325:
    %3425 = pdl_interp.get_value_type of %3419 : !pdl.type
    pdl_interp.are_equal %3425, %3422 : !pdl.type -> ^bb6326, ^bb6245
  ^bb6326:
    %3426 = pdl_interp.get_operand 0 of %3421
    %3427 = pdl_interp.get_defining_op of %3426 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %3427 : !pdl.operation -> ^bb6327, ^bb6245
  ^bb6327:
    pdl_interp.is_not_null %3426 : !pdl.value -> ^bb6328, ^bb6245
  ^bb6328:
    pdl_interp.check_operation_name of %3427 is "arith.constant" -> ^bb6329, ^bb6245
  ^bb6329:
    pdl_interp.check_operand_count of %3427 is 0 -> ^bb6330, ^bb6245
  ^bb6330:
    pdl_interp.check_result_count of %3427 is 1 -> ^bb6331, ^bb6245
  ^bb6331:
    %3428 = pdl_interp.get_result 0 of %3427
    pdl_interp.is_not_null %3428 : !pdl.value -> ^bb6332, ^bb6245
  ^bb6332:
    pdl_interp.are_equal %3428, %3426 : !pdl.value -> ^bb6333, ^bb6245
  ^bb6333:
    %3429 = pdl_interp.get_operand 1 of %3421
    %3430 = pdl_interp.get_defining_op of %3429 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %3430 : !pdl.operation -> ^bb6334, ^bb6245
  ^bb6334:
    %3431 = pdl_interp.get_value_type of %3424 : !pdl.type
    pdl_interp.are_equal %3431, %3422 : !pdl.type -> ^bb6335, ^bb6245
  ^bb6335:
    pdl_interp.is_not_null %3429 : !pdl.value -> ^bb6336, ^bb6245
  ^bb6336:
    pdl_interp.check_operation_name of %3430 is "arith.mulf" -> ^bb6337, ^bb6245
  ^bb6337:
    pdl_interp.check_operand_count of %3430 is 2 -> ^bb6338, ^bb6245
  ^bb6338:
    pdl_interp.check_result_count of %3430 is 1 -> ^bb6339, ^bb6245
  ^bb6339:
    %3432 = pdl_interp.get_attribute "value" of %3427
    pdl_interp.is_not_null %3432 : !pdl.attribute -> ^bb6340, ^bb6245
  ^bb6340:
    pdl_interp.check_attribute %3432 is 1.000000e+00 : f32 -> ^bb6341, ^bb6245
  ^bb6341:
    %3433 = pdl_interp.get_result 0 of %3430
    pdl_interp.is_not_null %3433 : !pdl.value -> ^bb6342, ^bb6245
  ^bb6342:
    pdl_interp.are_equal %3433, %3429 : !pdl.value -> ^bb6343, ^bb6245
  ^bb6343:
    %3434 = pdl_interp.get_operand 0 of %3430
    pdl_interp.are_equal %3434, %3386 : !pdl.value -> ^bb6344, ^bb6245
  ^bb6344:
    %3435 = pdl_interp.get_operand 1 of %3430
    pdl_interp.are_equal %3435, %3386 : !pdl.value -> ^bb6345, ^bb6245
  ^bb6345:
    %3436 = pdl_interp.get_value_type of %3428 : !pdl.type
    pdl_interp.are_equal %3436, %3422 : !pdl.type -> ^bb6346, ^bb6245
  ^bb6346:
    %3437 = pdl_interp.get_value_type of %3433 : !pdl.type
    pdl_interp.are_equal %3437, %3422 : !pdl.type -> ^bb6347, ^bb6245
  ^bb6347:
    pdl_interp.record_match @rewriters::@sinh_atanh_rev(%3386, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb6348
  ^bb6348:
    pdl_interp.record_match @rewriters::@tan_asin_rev(%3386, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb6245
  ^bb6321:
    pdl_interp.check_operand_count of %3421 is 2 -> ^bb6349, ^bb6245
  ^bb6349:
    pdl_interp.check_result_count of %3421 is 1 -> ^bb6350, ^bb6245
  ^bb6350:
    %3438 = pdl_interp.get_result 0 of %3421
    pdl_interp.is_not_null %3438 : !pdl.value -> ^bb6351, ^bb6245
  ^bb6351:
    pdl_interp.are_equal %3438, %3420 : !pdl.value -> ^bb6352, ^bb6245
  ^bb6352:
    %3439 = pdl_interp.get_value_type of %3419 : !pdl.type
    pdl_interp.are_equal %3439, %3422 : !pdl.type -> ^bb6353, ^bb6245
  ^bb6353:
    %3440 = pdl_interp.get_operand 0 of %3421
    %3441 = pdl_interp.get_defining_op of %3440 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op.operand[0].defining_op"}
    pdl_interp.is_not_null %3441 : !pdl.operation -> ^bb6354, ^bb6245
  ^bb6354:
    pdl_interp.is_not_null %3440 : !pdl.value -> ^bb6355, ^bb6245
  ^bb6355:
    pdl_interp.check_operation_name of %3441 is "arith.constant" -> ^bb6356, ^bb6245
  ^bb6356:
    pdl_interp.check_operand_count of %3441 is 0 -> ^bb6357, ^bb6245
  ^bb6357:
    pdl_interp.check_result_count of %3441 is 1 -> ^bb6358, ^bb6245
  ^bb6358:
    %3442 = pdl_interp.get_result 0 of %3441
    pdl_interp.is_not_null %3442 : !pdl.value -> ^bb6359, ^bb6245
  ^bb6359:
    pdl_interp.are_equal %3442, %3440 : !pdl.value -> ^bb6360, ^bb6245
  ^bb6360:
    %3443 = pdl_interp.get_operand 1 of %3421
    %3444 = pdl_interp.get_defining_op of %3443 : !pdl.value {position = "root.operand[1].defining_op.operand[0].defining_op.operand[1].defining_op"}
    pdl_interp.is_not_null %3444 : !pdl.operation -> ^bb6361, ^bb6245
  ^bb6361:
    %3445 = pdl_interp.get_value_type of %3438 : !pdl.type
    pdl_interp.are_equal %3445, %3422 : !pdl.type -> ^bb6362, ^bb6245
  ^bb6362:
    pdl_interp.is_not_null %3443 : !pdl.value -> ^bb6363, ^bb6245
  ^bb6363:
    pdl_interp.check_operation_name of %3444 is "arith.mulf" -> ^bb6364, ^bb6245
  ^bb6364:
    pdl_interp.check_operand_count of %3444 is 2 -> ^bb6365, ^bb6245
  ^bb6365:
    pdl_interp.check_result_count of %3444 is 1 -> ^bb6366, ^bb6245
  ^bb6366:
    %3446 = pdl_interp.get_attribute "value" of %3441
    pdl_interp.is_not_null %3446 : !pdl.attribute -> ^bb6367, ^bb6245
  ^bb6367:
    pdl_interp.check_attribute %3446 is 1.000000e+00 : f32 -> ^bb6368, ^bb6245
  ^bb6368:
    %3447 = pdl_interp.get_result 0 of %3444
    pdl_interp.is_not_null %3447 : !pdl.value -> ^bb6369, ^bb6245
  ^bb6369:
    pdl_interp.are_equal %3447, %3443 : !pdl.value -> ^bb6370, ^bb6245
  ^bb6370:
    %3448 = pdl_interp.get_operand 0 of %3444
    pdl_interp.are_equal %3448, %3386 : !pdl.value -> ^bb6371, ^bb6245
  ^bb6371:
    %3449 = pdl_interp.get_operand 1 of %3444
    pdl_interp.are_equal %3449, %3386 : !pdl.value -> ^bb6372, ^bb6245
  ^bb6372:
    %3450 = pdl_interp.get_value_type of %3442 : !pdl.type
    pdl_interp.are_equal %3450, %3422 : !pdl.type -> ^bb6373, ^bb6245
  ^bb6373:
    %3451 = pdl_interp.get_value_type of %3447 : !pdl.type
    pdl_interp.are_equal %3451, %3422 : !pdl.type -> ^bb6374, ^bb6245
  ^bb6374:
    pdl_interp.record_match @rewriters::@tanh_asinh_rev(%3386, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb6375
  ^bb6375:
    pdl_interp.record_match @rewriters::@sin_atan_rev(%3386, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("arith.divf") -> ^bb6245
  ^bb6:
    pdl_interp.check_operand_count of %0 is 0 -> ^bb6376, ^bb23
  ^bb6376:
    pdl_interp.check_result_count of %0 is 1 -> ^bb6377, ^bb23
  ^bb6377:
    %3452 = pdl_interp.get_attribute "value" of %0
    pdl_interp.is_not_null %3452 : !pdl.attribute -> ^bb6378, ^bb23
  ^bb6378:
    pdl_interp.switch_attribute %3452 to [2.000000e+00 : f32, 1.000000e+00 : f32, 0.000000e+00 : f32](^bb6379, ^bb6380, ^bb6381) -> ^bb23
  ^bb6379:
    %3453 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.check_type %3453 is f32 -> ^bb6382, ^bb23
  ^bb6382:
    pdl_interp.record_match @rewriters::@_2_split(%0 : !pdl.operation) : benefit(1), loc([]), root("arith.constant") -> ^bb23
  ^bb6380:
    %3454 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.check_type %3454 is f32 -> ^bb6383, ^bb23
  ^bb6383:
    pdl_interp.record_match @rewriters::@cosh_0_rev(%0 : !pdl.operation) : benefit(1), loc([]), root("arith.constant") -> ^bb6384
  ^bb6384:
    pdl_interp.record_match @rewriters::@_1_exp(%0 : !pdl.operation) : benefit(1), loc([]), root("arith.constant") -> ^bb6385
  ^bb6385:
    pdl_interp.record_match @rewriters::@_1_split(%0 : !pdl.operation) : benefit(1), loc([]), root("arith.constant") -> ^bb23
  ^bb6381:
    %3455 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.check_type %3455 is f32 -> ^bb6386, ^bb23
  ^bb6386:
    pdl_interp.record_match @rewriters::@sinh_0_rev(%0 : !pdl.operation) : benefit(1), loc([]), root("arith.constant") -> ^bb23
  ^bb7:
    pdl_interp.check_operand_count of %0 is 1 -> ^bb6387, ^bb23
  ^bb6387:
    pdl_interp.check_result_count of %0 is 1 -> ^bb6388, ^bb23
  ^bb6388:
    %3456 = pdl_interp.get_operand 0 of %0
    pdl_interp.is_not_null %3456 : !pdl.value -> ^bb6389, ^bb23
  ^bb6389:
    %3457 = pdl_interp.get_value_type of %3456 : !pdl.type
    %3458 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3457, %3458 : !pdl.type -> ^bb6390, ^bb23
  ^bb6390:
    pdl_interp.check_type %3457 is f32 -> ^bb6391, ^bb23
  ^bb6391:
    pdl_interp.record_match @rewriters::@neg_fabs(%3456, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.absf") -> ^bb6392
  ^bb6392:
    pdl_interp.record_match @rewriters::@rem_sqrt_square_rev(%3456, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.absf") -> ^bb23
  ^bb8:
    pdl_interp.check_operand_count of %0 is 1 -> ^bb6393, ^bb23
  ^bb6393:
    pdl_interp.check_result_count of %0 is 1 -> ^bb6394, ^bb23
  ^bb6394:
    %3459 = pdl_interp.get_operand 0 of %0
    pdl_interp.is_not_null %3459 : !pdl.value -> ^bb6395, ^bb23
  ^bb6395:
    %3460 = pdl_interp.get_value_type of %3459 : !pdl.type
    %3461 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3460, %3461 : !pdl.type -> ^bb6396, ^bb23
  ^bb6396:
    pdl_interp.check_type %3460 is f32 -> ^bb6397, ^bb23
  ^bb6397:
    pdl_interp.record_match @rewriters::@pow1div2(%3459, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.sqrt") -> ^bb6398
  ^bb6398:
    pdl_interp.record_match @rewriters::@sqrt_fabs_rev(%3459, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.sqrt") -> ^bb23
  ^bb9:
    pdl_interp.check_operand_count of %0 is 2 -> ^bb6399, ^bb23
  ^bb6399:
    pdl_interp.check_result_count of %0 is 1 -> ^bb6400, ^bb23
  ^bb6400:
    %3462 = pdl_interp.get_operand 1 of %0
    %3463 = pdl_interp.get_defining_op of %3462 : !pdl.value {position = "root.operand[1].defining_op"}
    pdl_interp.is_not_null %3463 : !pdl.operation -> ^bb6401, ^bb23
  ^bb6401:
    %3464 = pdl_interp.get_operand 0 of %0
    pdl_interp.is_not_null %3464 : !pdl.value -> ^bb6402, ^bb23
  ^bb6402:
    pdl_interp.is_not_null %3462 : !pdl.value -> ^bb6403, ^bb23
  ^bb6403:
    pdl_interp.switch_operation_name of %3463 to ["arith.negf", "math.absf"](^bb6404, ^bb6405) -> ^bb23
  ^bb6404:
    pdl_interp.check_operand_count of %3463 is 1 -> ^bb6406, ^bb23
  ^bb6406:
    pdl_interp.check_result_count of %3463 is 1 -> ^bb6407, ^bb23
  ^bb6407:
    %3465 = pdl_interp.get_result 0 of %3463
    pdl_interp.is_not_null %3465 : !pdl.value -> ^bb6408, ^bb23
  ^bb6408:
    pdl_interp.are_equal %3465, %3462 : !pdl.value -> ^bb6409, ^bb23
  ^bb6409:
    %3466 = pdl_interp.get_operand 0 of %3463
    pdl_interp.is_not_null %3466 : !pdl.value -> ^bb6410, ^bb23
  ^bb6410:
    %3467 = pdl_interp.get_value_type of %3464 : !pdl.type
    %3468 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3467, %3468 : !pdl.type -> ^bb6411, ^bb23
  ^bb6411:
    pdl_interp.check_type %3467 is f32 -> ^bb6412, ^bb23
  ^bb6412:
    %3469 = pdl_interp.get_value_type of %3465 : !pdl.type
    pdl_interp.are_equal %3469, %3467 : !pdl.type -> ^bb6413, ^bb23
  ^bb6413:
    %3470 = pdl_interp.get_value_type of %3466 : !pdl.type
    pdl_interp.are_equal %3470, %3467 : !pdl.type -> ^bb6414, ^bb23
  ^bb6414:
    pdl_interp.record_match @rewriters::@copysign_neg(%3464, %3466, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.copysign") -> ^bb23
  ^bb6405:
    pdl_interp.check_operand_count of %3463 is 1 -> ^bb6415, ^bb23
  ^bb6415:
    pdl_interp.check_result_count of %3463 is 1 -> ^bb6416, ^bb23
  ^bb6416:
    %3471 = pdl_interp.get_result 0 of %3463
    pdl_interp.is_not_null %3471 : !pdl.value -> ^bb6417, ^bb23
  ^bb6417:
    pdl_interp.are_equal %3471, %3462 : !pdl.value -> ^bb6418, ^bb23
  ^bb6418:
    %3472 = pdl_interp.get_operand 0 of %3463
    pdl_interp.is_not_null %3472 : !pdl.value -> ^bb6419, ^bb23
  ^bb6419:
    %3473 = pdl_interp.get_value_type of %3464 : !pdl.type
    %3474 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3473, %3474 : !pdl.type -> ^bb6420, ^bb23
  ^bb6420:
    pdl_interp.check_type %3473 is f32 -> ^bb6421, ^bb23
  ^bb6421:
    %3475 = pdl_interp.get_value_type of %3471 : !pdl.type
    pdl_interp.are_equal %3475, %3473 : !pdl.type -> ^bb6422, ^bb23
  ^bb6422:
    %3476 = pdl_interp.get_value_type of %3472 : !pdl.type
    pdl_interp.are_equal %3476, %3473 : !pdl.type -> ^bb6423, ^bb23
  ^bb6423:
    pdl_interp.record_match @rewriters::@copysign_fabs(%3464, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.copysign") -> ^bb23
  ^bb10:
    pdl_interp.check_operand_count of %0 is 2 -> ^bb6424, ^bb23
  ^bb6424:
    pdl_interp.check_result_count of %0 is 1 -> ^bb6425, ^bb23
  ^bb6425:
    %3477 = pdl_interp.get_operand 1 of %0
    %3478 = pdl_interp.get_defining_op of %3477 : !pdl.value {position = "root.operand[1].defining_op"}
    pdl_interp.is_not_null %3478 : !pdl.operation -> ^bb6426, ^bb23
  ^bb6426:
    %3479 = pdl_interp.get_operand 0 of %0
    pdl_interp.is_not_null %3479 : !pdl.value -> ^bb6427, ^bb23
  ^bb6427:
    pdl_interp.is_not_null %3477 : !pdl.value -> ^bb6428, ^bb23
  ^bb6428:
    pdl_interp.check_operation_name of %3478 is "arith.constant" -> ^bb6429, ^bb23
  ^bb6429:
    pdl_interp.check_operand_count of %3478 is 0 -> ^bb6430, ^bb23
  ^bb6430:
    pdl_interp.check_result_count of %3478 is 1 -> ^bb6431, ^bb23
  ^bb6431:
    %3480 = pdl_interp.get_result 0 of %3478
    pdl_interp.is_not_null %3480 : !pdl.value -> ^bb6432, ^bb23
  ^bb6432:
    pdl_interp.are_equal %3480, %3477 : !pdl.value -> ^bb6433, ^bb23
  ^bb6433:
    %3481 = pdl_interp.get_value_type of %3479 : !pdl.type
    %3482 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3481, %3482 : !pdl.type -> ^bb6434, ^bb23
  ^bb6434:
    pdl_interp.check_type %3481 is f32 -> ^bb6435, ^bb23
  ^bb6435:
    %3483 = pdl_interp.get_value_type of %3480 : !pdl.type
    pdl_interp.are_equal %3483, %3481 : !pdl.type -> ^bb6436, ^bb23
  ^bb6436:
    %3484 = pdl_interp.get_attribute "value" of %3478
    pdl_interp.is_not_null %3484 : !pdl.attribute -> ^bb6437, ^bb23
  ^bb6437:
    pdl_interp.switch_attribute %3484 to [3.000000e+00 : f32, -1.000000e+00 : f32, 1.000000e+00 : f32, 0.000000e+00 : f32, 5.000000e-01 : f32, 2.000000e+00 : f32, 0.333333343 : f32](^bb6438, ^bb6439, ^bb6440, ^bb6441, ^bb6442, ^bb6443, ^bb6444) -> ^bb23
  ^bb6438:
    pdl_interp.record_match @rewriters::@unpow3(%3479, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.powf") -> ^bb6445
  ^bb6445:
    pdl_interp.record_match @rewriters::@cube_mult(%3479, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.powf") -> ^bb23
  ^bb6439:
    pdl_interp.record_match @rewriters::@unpow_1(%3479, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.powf") -> ^bb23
  ^bb6440:
    pdl_interp.record_match @rewriters::@unpow1(%3479, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.powf") -> ^bb23
  ^bb6441:
    pdl_interp.record_match @rewriters::@unpow0(%0 : !pdl.operation) : benefit(1), loc([]), root("math.powf") -> ^bb23
  ^bb6442:
    pdl_interp.record_match @rewriters::@unpow1div2(%3479, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.powf") -> ^bb23
  ^bb6443:
    pdl_interp.record_match @rewriters::@unpow2(%3479, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.powf") -> ^bb23
  ^bb6444:
    pdl_interp.record_match @rewriters::@unpow1div3(%3479, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.powf") -> ^bb23
  ^bb11:
    pdl_interp.check_operand_count of %0 is 2 -> ^bb6446, ^bb23
  ^bb6446:
    pdl_interp.check_result_count of %0 is 1 -> ^bb6447, ^bb23
  ^bb6447:
    %3485 = pdl_interp.get_operand 0 of %0
    pdl_interp.is_not_null %3485 : !pdl.value -> ^bb6448, ^bb23
  ^bb6448:
    %3486 = pdl_interp.get_operand 1 of %0
    pdl_interp.is_not_null %3486 : !pdl.value -> ^bb6449, ^bb23
  ^bb6449:
    %3487 = pdl_interp.get_value_type of %3485 : !pdl.type
    %3488 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3487, %3488 : !pdl.type -> ^bb6450, ^bb23
  ^bb6450:
    pdl_interp.check_type %3487 is f32 -> ^bb6451, ^bb23
  ^bb6451:
    %3489 = pdl_interp.get_value_type of %3486 : !pdl.type
    pdl_interp.are_equal %3487, %3489 : !pdl.type -> ^bb6452, ^bb23
  ^bb6452:
    pdl_interp.record_match @rewriters::@fmin_swap(%3486, %3485, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("fmin") -> ^bb23
  ^bb12:
    pdl_interp.check_operand_count of %0 is 2 -> ^bb6453, ^bb23
  ^bb6453:
    pdl_interp.check_result_count of %0 is 1 -> ^bb6454, ^bb23
  ^bb6454:
    %3490 = pdl_interp.get_operand 0 of %0
    pdl_interp.is_not_null %3490 : !pdl.value -> ^bb6455, ^bb23
  ^bb6455:
    %3491 = pdl_interp.get_operand 1 of %0
    pdl_interp.is_not_null %3491 : !pdl.value -> ^bb6456, ^bb23
  ^bb6456:
    %3492 = pdl_interp.get_value_type of %3490 : !pdl.type
    %3493 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3492, %3493 : !pdl.type -> ^bb6457, ^bb23
  ^bb6457:
    pdl_interp.check_type %3492 is f32 -> ^bb6458, ^bb23
  ^bb6458:
    %3494 = pdl_interp.get_value_type of %3491 : !pdl.type
    pdl_interp.are_equal %3492, %3494 : !pdl.type -> ^bb6459, ^bb23
  ^bb6459:
    pdl_interp.record_match @rewriters::@fmax_swap(%3491, %3490, %0 : !pdl.value, !pdl.value, !pdl.operation) : benefit(1), loc([]), root("fmax") -> ^bb23
  ^bb13:
    pdl_interp.check_operand_count of %0 is 1 -> ^bb6460, ^bb23
  ^bb6460:
    pdl_interp.check_result_count of %0 is 1 -> ^bb6461, ^bb23
  ^bb6461:
    %3495 = pdl_interp.get_operand 0 of %0
    pdl_interp.is_not_null %3495 : !pdl.value -> ^bb6462, ^bb23
  ^bb6462:
    %3496 = pdl_interp.get_value_type of %3495 : !pdl.type
    %3497 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3496, %3497 : !pdl.type -> ^bb6463, ^bb23
  ^bb6463:
    pdl_interp.check_type %3496 is f32 -> ^bb6464, ^bb23
  ^bb6464:
    pdl_interp.record_match @rewriters::@sinh_add_cosh_rev(%3495, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.exp") -> ^bb6465
  ^bb6465:
    pdl_interp.record_match @rewriters::@exp_fabs(%3495, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.exp") -> ^bb23
  ^bb14:
    pdl_interp.check_operand_count of %0 is 1 -> ^bb6466, ^bb23
  ^bb6466:
    pdl_interp.check_result_count of %0 is 1 -> ^bb6467, ^bb23
  ^bb6467:
    %3498 = pdl_interp.get_operand 0 of %0
    pdl_interp.is_not_null %3498 : !pdl.value -> ^bb6468, ^bb23
  ^bb6468:
    %3499 = pdl_interp.get_value_type of %3498 : !pdl.type
    %3500 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3499, %3500 : !pdl.type -> ^bb6469, ^bb23
  ^bb6469:
    pdl_interp.check_type %3499 is f32 -> ^bb6470, ^bb23
  ^bb6470:
    pdl_interp.record_match @rewriters::@pow1div3(%3498, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.cbrt") -> ^bb23
  ^bb15:
    pdl_interp.check_operand_count of %0 is 1 -> ^bb6471, ^bb23
  ^bb6471:
    pdl_interp.check_result_count of %0 is 1 -> ^bb6472, ^bb23
  ^bb6472:
    %3501 = pdl_interp.get_operand 0 of %0
    pdl_interp.is_not_null %3501 : !pdl.value -> ^bb6473, ^bb23
  ^bb6473:
    %3502 = pdl_interp.get_value_type of %3501 : !pdl.type
    %3503 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3502, %3503 : !pdl.type -> ^bb6474, ^bb23
  ^bb6474:
    pdl_interp.check_type %3502 is f32 -> ^bb6475, ^bb23
  ^bb6475:
    pdl_interp.record_match @rewriters::@cos_fabs_rev(%3501, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.cos") -> ^bb6476
  ^bb6476:
    pdl_interp.record_match @rewriters::@cos_neg_rev(%3501, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.cos") -> ^bb23
  ^bb16:
    pdl_interp.check_operand_count of %0 is 1 -> ^bb6477, ^bb23
  ^bb6477:
    pdl_interp.check_result_count of %0 is 1 -> ^bb6478, ^bb23
  ^bb6478:
    %3504 = pdl_interp.get_operand 0 of %0
    pdl_interp.is_not_null %3504 : !pdl.value -> ^bb6479, ^bb23
  ^bb6479:
    %3505 = pdl_interp.get_value_type of %3504 : !pdl.type
    %3506 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3505, %3506 : !pdl.type -> ^bb6480, ^bb23
  ^bb6480:
    pdl_interp.check_type %3505 is f32 -> ^bb6481, ^bb23
  ^bb6481:
    pdl_interp.record_match @rewriters::@tan_quot(%3504, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.tan") -> ^bb23
  ^bb17:
    pdl_interp.check_operand_count of %0 is 1 -> ^bb6482, ^bb23
  ^bb6482:
    pdl_interp.check_result_count of %0 is 1 -> ^bb6483, ^bb23
  ^bb6483:
    %3507 = pdl_interp.get_operand 0 of %0
    pdl_interp.is_not_null %3507 : !pdl.value -> ^bb6484, ^bb23
  ^bb6484:
    %3508 = pdl_interp.get_value_type of %3507 : !pdl.type
    %3509 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3508, %3509 : !pdl.type -> ^bb6485, ^bb23
  ^bb6485:
    pdl_interp.check_type %3508 is f32 -> ^bb6486, ^bb23
  ^bb6486:
    pdl_interp.record_match @rewriters::@sinh_def(%3507, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.sinh") -> ^bb23
  ^bb18:
    pdl_interp.check_operand_count of %0 is 1 -> ^bb6487, ^bb23
  ^bb6487:
    pdl_interp.check_result_count of %0 is 1 -> ^bb6488, ^bb23
  ^bb6488:
    %3510 = pdl_interp.get_operand 0 of %0
    pdl_interp.is_not_null %3510 : !pdl.value -> ^bb6489, ^bb23
  ^bb6489:
    %3511 = pdl_interp.get_value_type of %3510 : !pdl.type
    %3512 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3511, %3512 : !pdl.type -> ^bb6490, ^bb23
  ^bb6490:
    pdl_interp.check_type %3511 is f32 -> ^bb6491, ^bb23
  ^bb6491:
    pdl_interp.record_match @rewriters::@cosh_neg_rev(%3510, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.cosh") -> ^bb6492
  ^bb6492:
    pdl_interp.record_match @rewriters::@cosh_def(%3510, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.cosh") -> ^bb23
  ^bb19:
    pdl_interp.check_operand_count of %0 is 1 -> ^bb6493, ^bb23
  ^bb6493:
    pdl_interp.check_result_count of %0 is 1 -> ^bb6494, ^bb23
  ^bb6494:
    %3513 = pdl_interp.get_operand 0 of %0
    pdl_interp.is_not_null %3513 : !pdl.value -> ^bb6495, ^bb23
  ^bb6495:
    %3514 = pdl_interp.get_value_type of %3513 : !pdl.type
    %3515 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3514, %3515 : !pdl.type -> ^bb6496, ^bb23
  ^bb6496:
    pdl_interp.check_type %3514 is f32 -> ^bb6497, ^bb23
  ^bb6497:
    pdl_interp.record_match @rewriters::@tanh_def_c(%3513, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.tanh") -> ^bb6498
  ^bb6498:
    pdl_interp.record_match @rewriters::@tanh_def_b(%3513, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.tanh") -> ^bb6499
  ^bb6499:
    pdl_interp.record_match @rewriters::@tanh_def_a(%3513, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.tanh") -> ^bb23
  ^bb20:
    pdl_interp.check_operand_count of %0 is 1 -> ^bb6500, ^bb23
  ^bb6500:
    pdl_interp.check_result_count of %0 is 1 -> ^bb6501, ^bb23
  ^bb6501:
    %3516 = pdl_interp.get_operand 0 of %0
    pdl_interp.is_not_null %3516 : !pdl.value -> ^bb6502, ^bb23
  ^bb6502:
    %3517 = pdl_interp.get_value_type of %3516 : !pdl.type
    %3518 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3517, %3518 : !pdl.type -> ^bb6503, ^bb23
  ^bb6503:
    pdl_interp.check_type %3517 is f32 -> ^bb6504, ^bb23
  ^bb6504:
    pdl_interp.record_match @rewriters::@asinh_def(%3516, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.asinh") -> ^bb23
  ^bb21:
    pdl_interp.check_operand_count of %0 is 1 -> ^bb6505, ^bb23
  ^bb6505:
    pdl_interp.check_result_count of %0 is 1 -> ^bb6506, ^bb23
  ^bb6506:
    %3519 = pdl_interp.get_operand 0 of %0
    pdl_interp.is_not_null %3519 : !pdl.value -> ^bb6507, ^bb23
  ^bb6507:
    %3520 = pdl_interp.get_value_type of %3519 : !pdl.type
    %3521 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3520, %3521 : !pdl.type -> ^bb6508, ^bb23
  ^bb6508:
    pdl_interp.check_type %3520 is f32 -> ^bb6509, ^bb23
  ^bb6509:
    pdl_interp.record_match @rewriters::@acosh_def(%3519, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.acosh") -> ^bb23
  ^bb22:
    pdl_interp.check_operand_count of %0 is 1 -> ^bb6510, ^bb23
  ^bb6510:
    pdl_interp.check_result_count of %0 is 1 -> ^bb6511, ^bb23
  ^bb6511:
    %3522 = pdl_interp.get_operand 0 of %0
    pdl_interp.is_not_null %3522 : !pdl.value -> ^bb6512, ^bb23
  ^bb6512:
    %3523 = pdl_interp.get_value_type of %3522 : !pdl.type
    %3524 = pdl_interp.get_value_type of %1 : !pdl.type
    pdl_interp.are_equal %3523, %3524 : !pdl.type -> ^bb6513, ^bb23
  ^bb6513:
    pdl_interp.check_type %3523 is f32 -> ^bb6514, ^bb23
  ^bb6514:
    pdl_interp.record_match @rewriters::@atanh_def(%3522, %0 : !pdl.value, !pdl.operation) : benefit(1), loc([]), root("math.atanh") -> ^bb23
  }
  builtin.module @rewriters {
    pdl_interp.func @cos_diff_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.subf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.cos"(%5 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %2 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cos_sin_sum(%0 : !pdl.operation) {
      %1 = pdl_interp.create_attribute 1.000000e+00 : f32
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.constant" {"value" = %1} -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %0 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sin_sum_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.addf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.sin"(%5 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %2 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cosh_2_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 2.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.mulf"(%5, %0 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "math.cosh"(%7 : !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %1 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sinh_sum_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.addf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.sinh"(%5 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %2 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cosh_sum_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.addf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.cosh"(%5 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %2 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @_1_add_cos(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.sin"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.sin"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.mulf"(%4, %6 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_operation "arith.negf"(%8 : !pdl.value) -> (%2 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      pdl_interp.replace %1 with (%10 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @_1_add_sin(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.cos"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.cos"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.mulf"(%4, %6 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_operation "arith.negf"(%8 : !pdl.value) -> (%2 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      pdl_interp.replace %1 with (%10 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @distribute_rgt_out(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.operation) {
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.addf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.mulf"(%2, %6 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %3 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @distribute_lft_out(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.operation) {
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.addf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.mulf"(%2, %6 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %3 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @difference_of_sqrsub_1(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 1.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.addf"(%0, %5 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_attribute 1.000000e+00 : f32
      %9 = pdl_interp.create_operation "arith.constant" {"value" = %8} -> (%3 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      %11 = pdl_interp.create_operation "arith.subf"(%0, %10 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %12 = pdl_interp.get_result 0 of %11
      %13 = pdl_interp.create_operation "arith.mulf"(%7, %12 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %14 = pdl_interp.get_result 0 of %13
      pdl_interp.replace %1 with (%14 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @distribute_neg_out(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.addf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.negf"(%5 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %2 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sum_square_pow_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.addf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_attribute 2.000000e+00 : f32
      %7 = pdl_interp.create_operation "arith.constant" {"value" = %6} -> (%3 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_operation "math.powf"(%5, %8 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      pdl_interp.replace %2 with (%10 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sub_square_pow_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.subf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_attribute 2.000000e+00 : f32
      %7 = pdl_interp.create_operation "arith.constant" {"value" = %6} -> (%3 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_operation "math.powf"(%5, %8 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      pdl_interp.replace %2 with (%10 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @div_add_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.operation) {
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.addf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.divf"(%6, %2 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %3 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @common_denominator(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.value, %4 : !pdl.operation) {
      %5 = pdl_interp.create_type f32
      %6 = pdl_interp.create_operation "arith.mulf"(%0, %1 : !pdl.value, !pdl.value) -> (%5 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.mulf"(%2, %3 : !pdl.value, !pdl.value) -> (%5 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "arith.addf"(%7, %9 : !pdl.value, !pdl.value) -> (%5 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "arith.mulf"(%3, %1 : !pdl.value, !pdl.value) -> (%5 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      %14 = pdl_interp.create_operation "arith.divf"(%11, %13 : !pdl.value, !pdl.value) -> (%5 : !pdl.type)
      %15 = pdl_interp.get_result 0 of %14
      pdl_interp.replace %4 with (%15 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @frac_add(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.value, %4 : !pdl.operation) {
      %5 = pdl_interp.create_type f32
      %6 = pdl_interp.create_operation "arith.mulf"(%0, %1 : !pdl.value, !pdl.value) -> (%5 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.mulf"(%2, %3 : !pdl.value, !pdl.value) -> (%5 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "arith.addf"(%7, %9 : !pdl.value, !pdl.value) -> (%5 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "arith.mulf"(%2, %1 : !pdl.value, !pdl.value) -> (%5 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      %14 = pdl_interp.create_operation "arith.divf"(%11, %13 : !pdl.value, !pdl.value) -> (%5 : !pdl.type)
      %15 = pdl_interp.get_result 0 of %14
      pdl_interp.replace %4 with (%15 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sum_cubes(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.mulf"(%0, %0 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.mulf"(%1, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.mulf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "arith.subf"(%7, %9 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "arith.addf"(%5, %11 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      %14 = pdl_interp.create_operation "arith.addf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %15 = pdl_interp.get_result 0 of %14
      %16 = pdl_interp.create_operation "arith.mulf"(%13, %15 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %17 = pdl_interp.get_result 0 of %16
      pdl_interp.replace %2 with (%17 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sum_log(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.mulf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.log"(%5 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %2 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sum_sin(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_attribute 2.000000e+00 : f32
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.constant" {"value" = %3} -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.addf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_attribute 2.000000e+00 : f32
      %10 = pdl_interp.create_operation "arith.constant" {"value" = %9} -> (%4 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "arith.divf"(%8, %11 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      %14 = pdl_interp.create_operation "math.sin"(%13 : !pdl.value) -> (%4 : !pdl.type)
      %15 = pdl_interp.get_result 0 of %14
      %16 = pdl_interp.create_operation "arith.subf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %17 = pdl_interp.get_result 0 of %16
      %18 = pdl_interp.create_attribute 2.000000e+00 : f32
      %19 = pdl_interp.create_operation "arith.constant" {"value" = %18} -> (%4 : !pdl.type)
      %20 = pdl_interp.get_result 0 of %19
      %21 = pdl_interp.create_operation "arith.divf"(%17, %20 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %22 = pdl_interp.get_result 0 of %21
      %23 = pdl_interp.create_operation "math.cos"(%22 : !pdl.value) -> (%4 : !pdl.type)
      %24 = pdl_interp.get_result 0 of %23
      %25 = pdl_interp.create_operation "arith.mulf"(%15, %24 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %26 = pdl_interp.get_result 0 of %25
      %27 = pdl_interp.create_operation "arith.mulf"(%6, %26 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %28 = pdl_interp.get_result 0 of %27
      pdl_interp.replace %2 with (%28 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sum_cos(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_attribute 2.000000e+00 : f32
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.constant" {"value" = %3} -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.addf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_attribute 2.000000e+00 : f32
      %10 = pdl_interp.create_operation "arith.constant" {"value" = %9} -> (%4 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "arith.divf"(%8, %11 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      %14 = pdl_interp.create_operation "math.cos"(%13 : !pdl.value) -> (%4 : !pdl.type)
      %15 = pdl_interp.get_result 0 of %14
      %16 = pdl_interp.create_operation "arith.subf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %17 = pdl_interp.get_result 0 of %16
      %18 = pdl_interp.create_attribute 2.000000e+00 : f32
      %19 = pdl_interp.create_operation "arith.constant" {"value" = %18} -> (%4 : !pdl.type)
      %20 = pdl_interp.get_result 0 of %19
      %21 = pdl_interp.create_operation "arith.divf"(%17, %20 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %22 = pdl_interp.get_result 0 of %21
      %23 = pdl_interp.create_operation "math.cos"(%22 : !pdl.value) -> (%4 : !pdl.type)
      %24 = pdl_interp.get_result 0 of %23
      %25 = pdl_interp.create_operation "arith.mulf"(%15, %24 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %26 = pdl_interp.get_result 0 of %25
      %27 = pdl_interp.create_operation "arith.mulf"(%6, %26 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %28 = pdl_interp.get_result 0 of %27
      pdl_interp.replace %2 with (%28 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sum_atan(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.addf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_attribute 1.000000e+00 : f32
      %7 = pdl_interp.create_operation "arith.constant" {"value" = %6} -> (%3 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_operation "arith.mulf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      %11 = pdl_interp.create_operation "arith.subf"(%8, %10 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %12 = pdl_interp.get_result 0 of %11
      %13 = pdl_interp.create_operation "math.atan2"(%5, %12 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %14 = pdl_interp.get_result 0 of %13
      pdl_interp.replace %2 with (%14 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sqr_cos_a_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.cos"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.cos"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.mulf"(%4, %6 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %1 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sinh_add_cosh(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.exp"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %1 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sum_cosh(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_attribute 2.000000e+00 : f32
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.constant" {"value" = %3} -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.addf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_attribute 2.000000e+00 : f32
      %10 = pdl_interp.create_operation "arith.constant" {"value" = %9} -> (%4 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "arith.divf"(%8, %11 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      %14 = pdl_interp.create_operation "math.cosh"(%13 : !pdl.value) -> (%4 : !pdl.type)
      %15 = pdl_interp.get_result 0 of %14
      %16 = pdl_interp.create_operation "arith.subf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %17 = pdl_interp.get_result 0 of %16
      %18 = pdl_interp.create_attribute 2.000000e+00 : f32
      %19 = pdl_interp.create_operation "arith.constant" {"value" = %18} -> (%4 : !pdl.type)
      %20 = pdl_interp.get_result 0 of %19
      %21 = pdl_interp.create_operation "arith.divf"(%17, %20 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %22 = pdl_interp.get_result 0 of %21
      %23 = pdl_interp.create_operation "math.cosh"(%22 : !pdl.value) -> (%4 : !pdl.type)
      %24 = pdl_interp.get_result 0 of %23
      %25 = pdl_interp.create_operation "arith.mulf"(%15, %24 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %26 = pdl_interp.get_result 0 of %25
      %27 = pdl_interp.create_operation "arith.mulf"(%6, %26 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %28 = pdl_interp.get_result 0 of %27
      pdl_interp.replace %2 with (%28 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cosh_undef(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 2.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.cosh"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.mulf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %1 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sum_sinh(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_attribute 2.000000e+00 : f32
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.constant" {"value" = %3} -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.addf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_attribute 2.000000e+00 : f32
      %10 = pdl_interp.create_operation "arith.constant" {"value" = %9} -> (%4 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "arith.divf"(%8, %11 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      %14 = pdl_interp.create_operation "math.sinh"(%13 : !pdl.value) -> (%4 : !pdl.type)
      %15 = pdl_interp.get_result 0 of %14
      %16 = pdl_interp.create_operation "arith.subf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %17 = pdl_interp.get_result 0 of %16
      %18 = pdl_interp.create_attribute 2.000000e+00 : f32
      %19 = pdl_interp.create_operation "arith.constant" {"value" = %18} -> (%4 : !pdl.type)
      %20 = pdl_interp.get_result 0 of %19
      %21 = pdl_interp.create_operation "arith.divf"(%17, %20 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %22 = pdl_interp.get_result 0 of %21
      %23 = pdl_interp.create_operation "math.cosh"(%22 : !pdl.value) -> (%4 : !pdl.type)
      %24 = pdl_interp.get_result 0 of %23
      %25 = pdl_interp.create_operation "arith.mulf"(%15, %24 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %26 = pdl_interp.get_result 0 of %25
      %27 = pdl_interp.create_operation "arith.mulf"(%6, %26 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %28 = pdl_interp.get_result 0 of %27
      pdl_interp.replace %2 with (%28 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @associate_addladd(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.operation) {
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.addf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.addf"(%2, %6 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %3 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @associate_addl_(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.operation) {
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.subf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.subf"(%2, %6 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %3 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @add_lft_identity(%0 : !pdl.value, %1 : !pdl.operation) {
      pdl_interp.replace %1 with (%0 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @distribute_lft1_in(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_attribute 1.000000e+00 : f32
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.constant" {"value" = %3} -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.addf"(%0, %6 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_operation "arith.mulf"(%8, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      pdl_interp.replace %2 with (%10 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sub_1_cos(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.sin"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.sin"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.mulf"(%4, %6 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_operation "arith.negf"(%8 : !pdl.value) -> (%2 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      pdl_interp.replace %1 with (%10 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sub_1_sin(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.cos"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.cos"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.mulf"(%4, %6 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_operation "arith.negf"(%8 : !pdl.value) -> (%2 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      pdl_interp.replace %1 with (%10 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @_3_sin(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 3.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.mulf"(%5, %0 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "math.sin"(%7 : !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %1 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cos_sum_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.addf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.cos"(%5 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %2 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @_2_cos(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 2.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.mulf"(%5, %0 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "math.cos"(%7 : !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %1 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sin_diff_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.subf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.sin"(%5 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %2 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @_3_cos(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 3.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.mulf"(%5, %0 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "math.cos"(%7 : !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %1 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cosh_diff_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.subf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.cosh"(%5 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %2 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sinh_cosh(%0 : !pdl.operation) {
      %1 = pdl_interp.create_attribute 1.000000e+00 : f32
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.constant" {"value" = %1} -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %0 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sinh_diff_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.subf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.sinh"(%5 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %2 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @difference_of_squares(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.addf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.subf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.mulf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %2 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @distribute_rgt_outsub_(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.operation) {
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.subf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.mulf"(%2, %6 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %3 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @distribute_lft_outsub_(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.operation) {
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.subf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.mulf"(%2, %6 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %3 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @difference_of_sqr_1(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 1.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.addf"(%0, %5 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_attribute 1.000000e+00 : f32
      %9 = pdl_interp.create_operation "arith.constant" {"value" = %8} -> (%3 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      %11 = pdl_interp.create_operation "arith.subf"(%0, %10 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %12 = pdl_interp.get_result 0 of %11
      %13 = pdl_interp.create_operation "arith.mulf"(%7, %12 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %14 = pdl_interp.get_result 0 of %13
      pdl_interp.replace %1 with (%14 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @difference_cubes(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.mulf"(%0, %0 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.mulf"(%1, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.mulf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "arith.addf"(%7, %9 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "arith.addf"(%5, %11 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      %14 = pdl_interp.create_operation "arith.subf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %15 = pdl_interp.get_result 0 of %14
      %16 = pdl_interp.create_operation "arith.mulf"(%13, %15 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %17 = pdl_interp.get_result 0 of %16
      pdl_interp.replace %2 with (%17 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @frac_sub(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.value, %4 : !pdl.operation) {
      %5 = pdl_interp.create_type f32
      %6 = pdl_interp.create_operation "arith.mulf"(%0, %1 : !pdl.value, !pdl.value) -> (%5 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.mulf"(%2, %3 : !pdl.value, !pdl.value) -> (%5 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "arith.subf"(%7, %9 : !pdl.value, !pdl.value) -> (%5 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "arith.mulf"(%2, %1 : !pdl.value, !pdl.value) -> (%5 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      %14 = pdl_interp.create_operation "arith.divf"(%11, %13 : !pdl.value, !pdl.value) -> (%5 : !pdl.type)
      %15 = pdl_interp.get_result 0 of %14
      pdl_interp.replace %4 with (%15 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sub_div(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.operation) {
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.subf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.divf"(%6, %2 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %3 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @diff_log(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.divf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.log"(%5 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %2 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sqr_cos_b_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.cos"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.cos"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.mulf"(%4, %6 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %1 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @_1_sub_sin(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.cos"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.cos"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.mulf"(%4, %6 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %1 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sqr_sin_b_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.sin"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.sin"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.mulf"(%4, %6 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %1 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @_1_sub_cos(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.sin"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.sin"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.mulf"(%4, %6 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %1 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sqr_sin_a_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.sin"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.sin"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.mulf"(%4, %6 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %1 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @diff_sin(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_attribute 2.000000e+00 : f32
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.constant" {"value" = %3} -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.subf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_attribute 2.000000e+00 : f32
      %10 = pdl_interp.create_operation "arith.constant" {"value" = %9} -> (%4 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "arith.divf"(%8, %11 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      %14 = pdl_interp.create_operation "math.sin"(%13 : !pdl.value) -> (%4 : !pdl.type)
      %15 = pdl_interp.get_result 0 of %14
      %16 = pdl_interp.create_operation "arith.addf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %17 = pdl_interp.get_result 0 of %16
      %18 = pdl_interp.create_attribute 2.000000e+00 : f32
      %19 = pdl_interp.create_operation "arith.constant" {"value" = %18} -> (%4 : !pdl.type)
      %20 = pdl_interp.get_result 0 of %19
      %21 = pdl_interp.create_operation "arith.divf"(%17, %20 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %22 = pdl_interp.get_result 0 of %21
      %23 = pdl_interp.create_operation "math.cos"(%22 : !pdl.value) -> (%4 : !pdl.type)
      %24 = pdl_interp.get_result 0 of %23
      %25 = pdl_interp.create_operation "arith.mulf"(%15, %24 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %26 = pdl_interp.get_result 0 of %25
      %27 = pdl_interp.create_operation "arith.mulf"(%6, %26 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %28 = pdl_interp.get_result 0 of %27
      pdl_interp.replace %2 with (%28 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @diff_cos(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_attribute -2.000000e+00 : f32
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.constant" {"value" = %3} -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.subf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_attribute 2.000000e+00 : f32
      %10 = pdl_interp.create_operation "arith.constant" {"value" = %9} -> (%4 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "arith.divf"(%8, %11 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      %14 = pdl_interp.create_operation "math.sin"(%13 : !pdl.value) -> (%4 : !pdl.type)
      %15 = pdl_interp.get_result 0 of %14
      %16 = pdl_interp.create_operation "arith.addf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %17 = pdl_interp.get_result 0 of %16
      %18 = pdl_interp.create_attribute 2.000000e+00 : f32
      %19 = pdl_interp.create_operation "arith.constant" {"value" = %18} -> (%4 : !pdl.type)
      %20 = pdl_interp.get_result 0 of %19
      %21 = pdl_interp.create_operation "arith.divf"(%17, %20 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %22 = pdl_interp.get_result 0 of %21
      %23 = pdl_interp.create_operation "math.sin"(%22 : !pdl.value) -> (%4 : !pdl.type)
      %24 = pdl_interp.get_result 0 of %23
      %25 = pdl_interp.create_operation "arith.mulf"(%15, %24 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %26 = pdl_interp.get_result 0 of %25
      %27 = pdl_interp.create_operation "arith.mulf"(%6, %26 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %28 = pdl_interp.get_result 0 of %27
      pdl_interp.replace %2 with (%28 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @diff_atan(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.subf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_attribute 1.000000e+00 : f32
      %7 = pdl_interp.create_operation "arith.constant" {"value" = %6} -> (%3 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_operation "arith.mulf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      %11 = pdl_interp.create_operation "arith.addf"(%8, %10 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %12 = pdl_interp.get_result 0 of %11
      %13 = pdl_interp.create_operation "math.atan2"(%5, %12 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %14 = pdl_interp.get_result 0 of %13
      pdl_interp.replace %2 with (%14 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sinhsub__cosh(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.negf"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.exp"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @diff_cosh(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_attribute 2.000000e+00 : f32
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.constant" {"value" = %3} -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.addf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_attribute 2.000000e+00 : f32
      %10 = pdl_interp.create_operation "arith.constant" {"value" = %9} -> (%4 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "arith.divf"(%8, %11 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      %14 = pdl_interp.create_operation "math.sinh"(%13 : !pdl.value) -> (%4 : !pdl.type)
      %15 = pdl_interp.get_result 0 of %14
      %16 = pdl_interp.create_operation "arith.subf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %17 = pdl_interp.get_result 0 of %16
      %18 = pdl_interp.create_attribute 2.000000e+00 : f32
      %19 = pdl_interp.create_operation "arith.constant" {"value" = %18} -> (%4 : !pdl.type)
      %20 = pdl_interp.get_result 0 of %19
      %21 = pdl_interp.create_operation "arith.divf"(%17, %20 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %22 = pdl_interp.get_result 0 of %21
      %23 = pdl_interp.create_operation "math.sinh"(%22 : !pdl.value) -> (%4 : !pdl.type)
      %24 = pdl_interp.get_result 0 of %23
      %25 = pdl_interp.create_operation "arith.mulf"(%15, %24 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %26 = pdl_interp.get_result 0 of %25
      %27 = pdl_interp.create_operation "arith.mulf"(%6, %26 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %28 = pdl_interp.get_result 0 of %27
      pdl_interp.replace %2 with (%28 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sinh_undef(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 2.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.sinh"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.mulf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %1 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @diff_sinh(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_attribute 2.000000e+00 : f32
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.constant" {"value" = %3} -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.addf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_attribute 2.000000e+00 : f32
      %10 = pdl_interp.create_operation "arith.constant" {"value" = %9} -> (%4 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "arith.divf"(%8, %11 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      %14 = pdl_interp.create_operation "math.cosh"(%13 : !pdl.value) -> (%4 : !pdl.type)
      %15 = pdl_interp.get_result 0 of %14
      %16 = pdl_interp.create_operation "arith.subf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %17 = pdl_interp.get_result 0 of %16
      %18 = pdl_interp.create_attribute 2.000000e+00 : f32
      %19 = pdl_interp.create_operation "arith.constant" {"value" = %18} -> (%4 : !pdl.type)
      %20 = pdl_interp.get_result 0 of %19
      %21 = pdl_interp.create_operation "arith.divf"(%17, %20 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %22 = pdl_interp.get_result 0 of %21
      %23 = pdl_interp.create_operation "math.sinh"(%22 : !pdl.value) -> (%4 : !pdl.type)
      %24 = pdl_interp.get_result 0 of %23
      %25 = pdl_interp.create_operation "arith.mulf"(%15, %24 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %26 = pdl_interp.get_result 0 of %25
      %27 = pdl_interp.create_operation "arith.mulf"(%6, %26 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %28 = pdl_interp.get_result 0 of %27
      pdl_interp.replace %2 with (%28 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @associatesub_ladd(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.operation) {
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.subf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.addf"(%2, %6 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %3 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @associatesub_l_(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.operation) {
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.addf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.subf"(%2, %6 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %3 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sub0_neg(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.negf"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %1 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @rem_3cbrt_lft(%0 : !pdl.value, %1 : !pdl.operation) {
      pdl_interp.replace %1 with (%0 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @unswap_sqr(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.mulf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.mulf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.mulf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %2 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @swap_sqr(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.mulf"(%0, %0 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.mulf"(%1, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.mulf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %2 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @pow_prod_up(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.operation) {
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.addf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "math.powf"(%2, %6 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %3 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @pow_prod_down(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.operation) {
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.mulf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "math.powf"(%6, %2 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %3 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cube_prod_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.mulf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_attribute 3.000000e+00 : f32
      %7 = pdl_interp.create_operation "arith.constant" {"value" = %6} -> (%3 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_operation "math.powf"(%5, %8 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      pdl_interp.replace %2 with (%10 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @pow_sqr(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_attribute 2.000000e+00 : f32
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.constant" {"value" = %3} -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.mulf"(%6, %0 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_operation "math.powf"(%1, %8 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      pdl_interp.replace %2 with (%10 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @difference_cubes_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_attribute 3.000000e+00 : f32
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.constant" {"value" = %3} -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "math.powf"(%0, %6 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_attribute 3.000000e+00 : f32
      %10 = pdl_interp.create_operation "arith.constant" {"value" = %9} -> (%4 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "math.powf"(%1, %11 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      %14 = pdl_interp.create_operation "arith.subf"(%8, %13 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %15 = pdl_interp.get_result 0 of %14
      pdl_interp.replace %2 with (%15 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sum_cubes_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_attribute 3.000000e+00 : f32
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.constant" {"value" = %3} -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "math.powf"(%0, %6 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_attribute 3.000000e+00 : f32
      %10 = pdl_interp.create_operation "arith.constant" {"value" = %9} -> (%4 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "math.powf"(%1, %11 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      %14 = pdl_interp.create_operation "arith.addf"(%8, %13 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %15 = pdl_interp.get_result 0 of %14
      pdl_interp.replace %2 with (%15 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @difference_of_squares_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.mulf"(%0, %0 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.mulf"(%1, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.subf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %2 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @difference_of_sqr_1_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.mulf"(%0, %0 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_attribute 1.000000e+00 : f32
      %6 = pdl_interp.create_operation "arith.constant" {"value" = %5} -> (%2 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.subf"(%4, %7 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %1 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @difference_of_sqrsub_1_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.mulf"(%0, %0 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_attribute -1.000000e+00 : f32
      %6 = pdl_interp.create_operation "arith.constant" {"value" = %5} -> (%2 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.addf"(%4, %7 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %1 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @frac_times(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.value, %4 : !pdl.operation) {
      %5 = pdl_interp.create_type f32
      %6 = pdl_interp.create_operation "arith.mulf"(%0, %1 : !pdl.value, !pdl.value) -> (%5 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.mulf"(%2, %3 : !pdl.value, !pdl.value) -> (%5 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "arith.divf"(%7, %9 : !pdl.value, !pdl.value) -> (%5 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      pdl_interp.replace %4 with (%11 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sqrt_unprod(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.mulf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.sqrt"(%5 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %2 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @rem_square_sqrt(%0 : !pdl.value, %1 : !pdl.operation) {
      pdl_interp.replace %1 with (%0 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sqr_neg(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.mulf"(%0, %0 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %1 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @mul_fabs(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.mulf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.absf"(%5 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %2 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sqr_abs(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.mulf"(%0, %0 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %1 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @rem_3cbrt_rft(%0 : !pdl.value, %1 : !pdl.operation) {
      pdl_interp.replace %1 with (%0 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cbrt_unprod(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.mulf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.cbrt"(%5 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %2 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @exp_lft_sqr_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 2.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.mulf"(%0, %5 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "math.exp"(%7 : !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %1 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @prod_exp(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.addf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.exp"(%5 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %2 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sin_mult(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.subf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.cos"(%5 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.addf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "math.cos"(%9 : !pdl.value) -> (%3 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "arith.subf"(%7, %11 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      %14 = pdl_interp.create_attribute 2.000000e+00 : f32
      %15 = pdl_interp.create_operation "arith.constant" {"value" = %14} -> (%3 : !pdl.type)
      %16 = pdl_interp.get_result 0 of %15
      %17 = pdl_interp.create_operation "arith.divf"(%13, %16 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %18 = pdl_interp.get_result 0 of %17
      pdl_interp.replace %2 with (%18 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sqr_sin_b(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 1.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.cos"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "math.cos"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "arith.mulf"(%7, %9 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "arith.subf"(%5, %11 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      pdl_interp.replace %1 with (%13 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sqr_sin_a(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 5.000000e-01 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_attribute 5.000000e-01 : f32
      %7 = pdl_interp.create_operation "arith.constant" {"value" = %6} -> (%3 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_attribute 2.000000e+00 : f32
      %10 = pdl_interp.create_operation "arith.constant" {"value" = %9} -> (%3 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "arith.mulf"(%11, %0 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      %14 = pdl_interp.create_operation "math.cos"(%13 : !pdl.value) -> (%3 : !pdl.type)
      %15 = pdl_interp.get_result 0 of %14
      %16 = pdl_interp.create_operation "arith.mulf"(%8, %15 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %17 = pdl_interp.get_result 0 of %16
      %18 = pdl_interp.create_operation "arith.subf"(%5, %17 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %19 = pdl_interp.get_result 0 of %18
      pdl_interp.replace %1 with (%19 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sin_cos_mult(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.subf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.sin"(%5 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.addf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "math.sin"(%9 : !pdl.value) -> (%3 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "arith.addf"(%7, %11 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      %14 = pdl_interp.create_attribute 2.000000e+00 : f32
      %15 = pdl_interp.create_operation "arith.constant" {"value" = %14} -> (%3 : !pdl.type)
      %16 = pdl_interp.get_result 0 of %15
      %17 = pdl_interp.create_operation "arith.divf"(%13, %16 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %18 = pdl_interp.get_result 0 of %17
      pdl_interp.replace %2 with (%18 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cos_mult(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.addf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.cos"(%5 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.subf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "math.cos"(%9 : !pdl.value) -> (%3 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "arith.addf"(%7, %11 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      %14 = pdl_interp.create_attribute 2.000000e+00 : f32
      %15 = pdl_interp.create_operation "arith.constant" {"value" = %14} -> (%3 : !pdl.type)
      %16 = pdl_interp.get_result 0 of %15
      %17 = pdl_interp.create_operation "arith.divf"(%13, %16 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %18 = pdl_interp.get_result 0 of %17
      pdl_interp.replace %2 with (%18 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sqr_cos_b(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 1.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.sin"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "math.sin"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "arith.mulf"(%7, %9 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "arith.subf"(%5, %11 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      pdl_interp.replace %1 with (%13 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @_1_sub_sin_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 1.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.sin"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "math.sin"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "arith.mulf"(%7, %9 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "arith.subf"(%5, %11 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      pdl_interp.replace %1 with (%13 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sqr_cos_a(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 5.000000e-01 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_attribute 5.000000e-01 : f32
      %7 = pdl_interp.create_operation "arith.constant" {"value" = %6} -> (%3 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_attribute 2.000000e+00 : f32
      %10 = pdl_interp.create_operation "arith.constant" {"value" = %9} -> (%3 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "arith.mulf"(%11, %0 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      %14 = pdl_interp.create_operation "math.cos"(%13 : !pdl.value) -> (%3 : !pdl.type)
      %15 = pdl_interp.get_result 0 of %14
      %16 = pdl_interp.create_operation "arith.mulf"(%8, %15 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %17 = pdl_interp.get_result 0 of %16
      %18 = pdl_interp.create_operation "arith.addf"(%5, %17 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %19 = pdl_interp.get_result 0 of %18
      pdl_interp.replace %1 with (%19 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @diff_sin_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "math.sin"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.sin"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.subf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %2 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sum_sin_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "math.sin"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.sin"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.addf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %2 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @_2_sin(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 2.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.mulf"(%5, %0 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "math.sin"(%7 : !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %1 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @diff_cos_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "math.cos"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.cos"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.subf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %2 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sum_cos_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "math.cos"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.cos"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.addf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %2 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @diff_cosh_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "math.cosh"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.cosh"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.subf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %2 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sum_sinh_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "math.sinh"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.sinh"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.addf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %2 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sinh_2_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 2.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.mulf"(%5, %0 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "math.sinh"(%7 : !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %1 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @diff_sinh_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "math.sinh"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.sinh"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.subf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %2 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sum_cosh_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "math.cosh"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.cosh"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.addf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %2 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sinh_undef_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.exp"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "arith.negf"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "math.exp"(%6 : !pdl.value) -> (%2 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_operation "arith.subf"(%4, %8 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      pdl_interp.replace %1 with (%10 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cosh_undef_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.exp"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "arith.negf"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "math.exp"(%6 : !pdl.value) -> (%2 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_operation "arith.addf"(%4, %8 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      pdl_interp.replace %1 with (%10 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @acosh_2_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 2.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.mulf"(%0, %0 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.mulf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_attribute 1.000000e+00 : f32
      %11 = pdl_interp.create_operation "arith.constant" {"value" = %10} -> (%3 : !pdl.type)
      %12 = pdl_interp.get_result 0 of %11
      %13 = pdl_interp.create_operation "arith.subf"(%9, %12 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %14 = pdl_interp.get_result 0 of %13
      %15 = pdl_interp.create_operation "math.acosh"(%14 : !pdl.value) -> (%3 : !pdl.type)
      %16 = pdl_interp.get_result 0 of %15
      pdl_interp.replace %1 with (%16 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @pow3(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 3.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.powf"(%0, %5 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %1 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @associate_mullmul(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.operation) {
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.mulf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.mulf"(%2, %6 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %3 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @lft_mult_inverse(%0 : !pdl.operation) {
      %1 = pdl_interp.create_attribute 1.000000e+00 : f32
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.constant" {"value" = %1} -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %0 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @associate_mulldiv(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.operation) {
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.mulf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.divf"(%6, %2 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %3 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @mul0_lft(%0 : !pdl.operation) {
      %1 = pdl_interp.create_attribute 0.000000e+00 : f32
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.constant" {"value" = %1} -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %0 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @mul_lft_identity(%0 : !pdl.value, %1 : !pdl.operation) {
      pdl_interp.replace %1 with (%0 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @mul_1_neg(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.negf"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %1 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @count_2_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.addf"(%0, %0 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %1 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @distribute_lft_neg_out(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.mulf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.negf"(%5 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %2 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sum_to_mult_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.addf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      pdl_interp.replace %2 with (%5 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sub_to_mult_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.subf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      pdl_interp.replace %2 with (%5 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @pow_plus(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_attribute 1.000000e+00 : f32
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.constant" {"value" = %3} -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.addf"(%0, %6 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_operation "math.powf"(%1, %8 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      pdl_interp.replace %2 with (%10 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @div_flip_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.divf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      pdl_interp.replace %2 with (%5 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @remove_double_div(%0 : !pdl.value, %1 : !pdl.operation) {
      pdl_interp.replace %1 with (%0 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @rec_exp(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.negf"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.exp"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @pow_flip(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.negf"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.powf"(%1, %5 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %2 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cos_atan_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.atan"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.cos"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cosh_atanh_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.atanh"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.cosh"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @_2_tan(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 2.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.mulf"(%5, %0 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "math.tan"(%7 : !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %1 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @tanh_2_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 2.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.mulf"(%5, %0 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "math.tanh"(%7 : !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %1 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @times_frac(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.value, %4 : !pdl.operation) {
      %5 = pdl_interp.create_type f32
      %6 = pdl_interp.create_operation "arith.divf"(%0, %1 : !pdl.value, !pdl.value) -> (%5 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.divf"(%2, %3 : !pdl.value, !pdl.value) -> (%5 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "arith.mulf"(%7, %9 : !pdl.value, !pdl.value) -> (%5 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      pdl_interp.replace %4 with (%11 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @hang_0m_tan(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.negf"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_attribute 2.000000e+00 : f32
      %6 = pdl_interp.create_operation "arith.constant" {"value" = %5} -> (%2 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.divf"(%4, %7 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "math.tan"(%9 : !pdl.value) -> (%2 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      pdl_interp.replace %1 with (%11 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @frac_2neg_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.divf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      pdl_interp.replace %2 with (%5 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cbrt_div_cbrt2(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 1.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.copysign"(%5, %0 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %1 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @div_fabs(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.divf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.absf"(%5 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %2 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sqrt_undiv(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.divf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.sqrt"(%5 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %2 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @pow_div(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.operation) {
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.subf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "math.powf"(%2, %6 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %3 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cube_div_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.divf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_attribute 3.000000e+00 : f32
      %7 = pdl_interp.create_operation "arith.constant" {"value" = %6} -> (%3 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_operation "math.powf"(%5, %8 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      pdl_interp.replace %2 with (%10 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cbrt_undiv(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.divf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.cbrt"(%5 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %2 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cbrt_div_cbrt(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 1.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.copysign"(%5, %0 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %1 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @div_exp(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.subf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.exp"(%5 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %2 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @hang_0p_tan(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 2.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.divf"(%0, %5 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "math.tan"(%7 : !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %1 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @quot_tan(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.tan"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %1 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @hang_p0_tan(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 2.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.divf"(%0, %5 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "math.tan"(%7 : !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %1 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @hang_m0_tan(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.negf"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_attribute 2.000000e+00 : f32
      %6 = pdl_interp.create_operation "arith.constant" {"value" = %5} -> (%2 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.divf"(%4, %7 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "math.tan"(%9 : !pdl.value) -> (%2 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      pdl_interp.replace %1 with (%11 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @hang_m_tan(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.subf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_attribute 2.000000e+00 : f32
      %7 = pdl_interp.create_operation "arith.constant" {"value" = %6} -> (%3 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_operation "arith.divf"(%5, %8 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      %11 = pdl_interp.create_operation "math.tan"(%10 : !pdl.value) -> (%3 : !pdl.type)
      %12 = pdl_interp.get_result 0 of %11
      pdl_interp.replace %2 with (%12 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @tanh_def_b_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.tanh"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %1 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @tanh_def_c_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.tanh"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %1 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @tanh_undef(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.tanh"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %1 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sin_mult_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "math.sin"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.sin"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.mulf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %2 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sinh_def_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.sinh"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %1 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @tanh_1div2mul_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 2.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.divf"(%0, %5 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "math.tanh"(%7 : !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %1 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @hang_p_tan(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.addf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_attribute 2.000000e+00 : f32
      %7 = pdl_interp.create_operation "arith.constant" {"value" = %6} -> (%3 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_operation "arith.divf"(%5, %8 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      %11 = pdl_interp.create_operation "math.tan"(%10 : !pdl.value) -> (%3 : !pdl.type)
      %12 = pdl_interp.get_result 0 of %11
      pdl_interp.replace %2 with (%12 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @tanh_sum_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.addf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.tanh"(%5 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %2 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @tan_sum_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.addf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.tan"(%5 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %2 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cos_mult_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "math.cos"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.cos"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.mulf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %2 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sin_cos_mult_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "math.sin"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.cos"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.mulf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %2 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cosh_def_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.cosh"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %1 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @tanh_1div2_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 2.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.divf"(%0, %5 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "math.tanh"(%7 : !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %1 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sinh_1div2_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 2.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.divf"(%0, %5 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "math.sinh"(%7 : !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %1 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @atanh_def_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.atanh"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %1 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @associate_divldiv(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.operation) {
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.mulf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.divf"(%2, %6 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %3 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @associate_divlmul(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.operation) {
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.divf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.mulf"(%2, %6 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %3 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @div0(%0 : !pdl.operation) {
      %1 = pdl_interp.create_attribute 0.000000e+00 : f32
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.constant" {"value" = %1} -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %0 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @inv_pow(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute -1.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.powf"(%0, %5 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %1 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @distribute_frac_neg(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.divf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.negf"(%5 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %2 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @div_add(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.operation) {
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.divf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.divf"(%2, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_operation "arith.addf"(%6, %8 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      pdl_interp.replace %3 with (%10 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @add_to_fraction_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.operation) {
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.divf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.addf"(%2, %6 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %3 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @div_sub(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.operation) {
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.divf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.divf"(%2, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_operation "arith.subf"(%6, %8 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      pdl_interp.replace %3 with (%10 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sub_to_fraction_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.operation) {
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.divf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.subf"(%2, %6 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %3 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @fabs_lhs_div(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 1.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.copysign"(%5, %0 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %1 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @fabs_cbrt_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.cbrt"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "arith.divf"(%4, %0 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "math.absf"(%6 : !pdl.value) -> (%2 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %1 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @tan_acos_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.acos"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.tan"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @remove_double_neg(%0 : !pdl.value, %1 : !pdl.operation) {
      pdl_interp.replace %1 with (%0 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @distribute_lft_neg_in(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.negf"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.mulf"(%5, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %2 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @distribute_rgt_neg_in(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.negf"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.mulf"(%1, %5 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %2 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @distribute_neg_in(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.negf"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.negf"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.addf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %2 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @distribute_neg_frac(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.negf"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.divf"(%5, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %2 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @distribute_neg_frac2(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.negf"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.divf"(%1, %5 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %2 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sub_negate(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.subf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      pdl_interp.replace %2 with (%5 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @neg_copysign(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.negf"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.copysign"(%1, %5 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %2 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cube_neg_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.negf"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_attribute 3.000000e+00 : f32
      %6 = pdl_interp.create_operation "arith.constant" {"value" = %5} -> (%2 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "math.powf"(%4, %7 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %1 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cbrt_neg_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.negf"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.cbrt"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @neg_log(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 1.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.divf"(%5, %0 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "math.log"(%7 : !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %1 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sin_neg_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.negf"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.sin"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @tan_neg_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.negf"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.tan"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @asin_neg_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.negf"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.asin"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @atan_neg_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.negf"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.atan"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sinh_neg_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.negf"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.sinh"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sqrt_pow2(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_attribute 2.000000e+00 : f32
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.constant" {"value" = %3} -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.divf"(%0, %6 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_operation "math.powf"(%1, %8 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      pdl_interp.replace %2 with (%10 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @pow_cbrt(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_attribute 3.000000e+00 : f32
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.constant" {"value" = %3} -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.divf"(%0, %6 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_operation "math.powf"(%1, %8 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      pdl_interp.replace %2 with (%10 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @pow_base_1(%0 : !pdl.operation) {
      %1 = pdl_interp.create_attribute 1.000000e+00 : f32
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.constant" {"value" = %1} -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %0 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @pow_base_0(%0 : !pdl.operation) {
      %1 = pdl_interp.create_attribute 0.000000e+00 : f32
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.constant" {"value" = %1} -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %0 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @pow_exp(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.mulf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.exp"(%5 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %2 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sum_square_pow(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_attribute 2.000000e+00 : f32
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.constant" {"value" = %3} -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "math.powf"(%0, %6 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_attribute 2.000000e+00 : f32
      %10 = pdl_interp.create_operation "arith.constant" {"value" = %9} -> (%4 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "arith.mulf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      %14 = pdl_interp.create_operation "arith.mulf"(%11, %13 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %15 = pdl_interp.get_result 0 of %14
      %16 = pdl_interp.create_operation "arith.addf"(%8, %15 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %17 = pdl_interp.get_result 0 of %16
      %18 = pdl_interp.create_attribute 2.000000e+00 : f32
      %19 = pdl_interp.create_operation "arith.constant" {"value" = %18} -> (%4 : !pdl.type)
      %20 = pdl_interp.get_result 0 of %19
      %21 = pdl_interp.create_operation "math.powf"(%1, %20 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %22 = pdl_interp.get_result 0 of %21
      %23 = pdl_interp.create_operation "arith.addf"(%17, %22 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %24 = pdl_interp.get_result 0 of %23
      pdl_interp.replace %2 with (%24 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sub_square_pow(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_attribute 2.000000e+00 : f32
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.constant" {"value" = %3} -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "math.powf"(%0, %6 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_attribute 2.000000e+00 : f32
      %10 = pdl_interp.create_operation "arith.constant" {"value" = %9} -> (%4 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "arith.mulf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      %14 = pdl_interp.create_operation "arith.mulf"(%11, %13 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %15 = pdl_interp.get_result 0 of %14
      %16 = pdl_interp.create_operation "arith.subf"(%8, %15 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %17 = pdl_interp.get_result 0 of %16
      %18 = pdl_interp.create_attribute 2.000000e+00 : f32
      %19 = pdl_interp.create_operation "arith.constant" {"value" = %18} -> (%4 : !pdl.type)
      %20 = pdl_interp.get_result 0 of %19
      %21 = pdl_interp.create_operation "math.powf"(%1, %20 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %22 = pdl_interp.get_result 0 of %21
      %23 = pdl_interp.create_operation "arith.addf"(%17, %22 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %24 = pdl_interp.get_result 0 of %23
      pdl_interp.replace %2 with (%24 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @rem_cube_cbrt(%0 : !pdl.value, %1 : !pdl.operation) {
      pdl_interp.replace %1 with (%0 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cube_neg(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 3.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.powf"(%0, %5 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.negf"(%7 : !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %1 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cube_prod(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_attribute 3.000000e+00 : f32
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.constant" {"value" = %3} -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "math.powf"(%0, %6 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_attribute 3.000000e+00 : f32
      %10 = pdl_interp.create_operation "arith.constant" {"value" = %9} -> (%4 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "math.powf"(%1, %11 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      %14 = pdl_interp.create_operation "arith.mulf"(%8, %13 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %15 = pdl_interp.get_result 0 of %14
      pdl_interp.replace %2 with (%15 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cube_div(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_attribute 3.000000e+00 : f32
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.constant" {"value" = %3} -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "math.powf"(%0, %6 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_attribute 3.000000e+00 : f32
      %10 = pdl_interp.create_operation "arith.constant" {"value" = %9} -> (%4 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "math.powf"(%1, %11 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      %14 = pdl_interp.create_operation "arith.divf"(%8, %13 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %15 = pdl_interp.get_result 0 of %14
      pdl_interp.replace %2 with (%15 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @exp_lft_cube_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 3.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.mulf"(%0, %5 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "math.exp"(%7 : !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %1 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sqrt_prod(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "math.absf"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.sqrt"(%5 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "math.absf"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "math.sqrt"(%9 : !pdl.value) -> (%3 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "arith.mulf"(%7, %11 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      pdl_interp.replace %2 with (%13 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @rem_sqrt_square(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.absf"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %1 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sqrt_cbrt(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.sqrt"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.cbrt"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cosh_1div2_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 2.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.divf"(%0, %5 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "math.cosh"(%7 : !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %1 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sqrt_div(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "math.absf"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.sqrt"(%5 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "math.absf"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "math.sqrt"(%9 : !pdl.value) -> (%3 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "arith.divf"(%7, %11 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      pdl_interp.replace %2 with (%13 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @exp_sqrt_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 2.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.divf"(%0, %5 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "math.exp"(%7 : !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %1 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cos_asin_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.asin"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.cos"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sin_acos_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.acos"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.sin"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cosh_asinh_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.asinh"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.cosh"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cbrt_sqrt(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.cbrt"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.sqrt"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cbrt_pow(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_attribute 3.000000e+00 : f32
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.constant" {"value" = %3} -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.divf"(%0, %6 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_operation "math.powf"(%1, %8 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      pdl_interp.replace %2 with (%10 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @rem_cbrt_cube(%0 : !pdl.value, %1 : !pdl.operation) {
      pdl_interp.replace %1 with (%0 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cbrt_prod(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "math.cbrt"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.cbrt"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.mulf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %2 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cbrt_div(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "math.cbrt"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.cbrt"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.divf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %2 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cbrt_neg(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.cbrt"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "arith.negf"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cbrt_fabs(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.cbrt"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.absf"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @exp_cbrt_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 3.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.divf"(%0, %5 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "math.exp"(%7 : !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %1 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @fabs_fabs(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.absf"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %1 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @fabs_sub(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.subf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.absf"(%5 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %2 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @fabs_add(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "math.absf"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.absf"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.addf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %2 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @fabs_neg(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.absf"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %1 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @fabs_mul(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "math.absf"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.absf"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.mulf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %2 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @fabs_sqr(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.mulf"(%0, %0 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %1 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @fabs_cbrt(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.cbrt"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "arith.divf"(%4, %0 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @fabs_div(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "math.absf"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.absf"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.divf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %2 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sqrt_fabs(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.sqrt"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %1 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @fabs_copysign(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.absf"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %1 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cbrt_fabs_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.absf"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.cbrt"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @fabs_exp(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.exp"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %1 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @copysign_other_neg(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "math.copysign"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      pdl_interp.replace %2 with (%5 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @copysign_other_fabs(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "math.copysign"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      pdl_interp.replace %2 with (%5 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @rem_exp_log(%0 : !pdl.value, %1 : !pdl.operation) {
      pdl_interp.replace %1 with (%0 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @exp_0(%0 : !pdl.operation) {
      %1 = pdl_interp.create_attribute 1.000000e+00 : f32
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.constant" {"value" = %1} -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %0 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @exp_sum(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "math.exp"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.exp"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.mulf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %2 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @exp_neg(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 1.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.exp"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.divf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %1 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sinhsub__cosh_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.cosh"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.sinh"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.subf"(%4, %6 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %1 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @exp_diff(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "math.exp"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.exp"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.divf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %2 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @exp_to_pow(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "math.powf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      pdl_interp.replace %2 with (%5 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @exp_lft_sqr(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.exp"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.exp"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.mulf"(%4, %6 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %1 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @exp_lft_cube(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.exp"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_attribute 3.000000e+00 : f32
      %6 = pdl_interp.create_operation "arith.constant" {"value" = %5} -> (%2 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "math.powf"(%4, %7 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %1 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @exp_prod(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "math.exp"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.powf"(%5, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %2 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @exp_sqrt(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.exp"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.sqrt"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @exp_cbrt(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.exp"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.cbrt"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @rem_log_exp(%0 : !pdl.value, %1 : !pdl.operation) {
      pdl_interp.replace %1 with (%0 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @log_div(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "math.absf"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.log"(%5 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "math.absf"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "math.log"(%9 : !pdl.value) -> (%3 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "arith.subf"(%7, %11 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      pdl_interp.replace %2 with (%13 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @log_rec(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.log"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "arith.negf"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @log_prod(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "math.absf"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.log"(%5 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "math.absf"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "math.log"(%9 : !pdl.value) -> (%3 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "arith.addf"(%7, %11 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      pdl_interp.replace %2 with (%13 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @asinh_def_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.asinh"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %1 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @acosh_def_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.acosh"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %1 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sin_0(%0 : !pdl.operation) {
      %1 = pdl_interp.create_attribute 0.000000e+00 : f32
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.constant" {"value" = %1} -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %0 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sin_neg(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.sin"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "arith.negf"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sin_asin(%0 : !pdl.value, %1 : !pdl.operation) {
      pdl_interp.replace %1 with (%0 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sin_sum(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "math.sin"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.cos"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.mulf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "math.cos"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "math.sin"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      %14 = pdl_interp.create_operation "arith.mulf"(%11, %13 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %15 = pdl_interp.get_result 0 of %14
      %16 = pdl_interp.create_operation "arith.addf"(%9, %15 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %17 = pdl_interp.get_result 0 of %16
      pdl_interp.replace %2 with (%17 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sin_diff(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "math.sin"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.cos"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.mulf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "math.cos"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "math.sin"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      %14 = pdl_interp.create_operation "arith.mulf"(%11, %13 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %15 = pdl_interp.get_result 0 of %14
      %16 = pdl_interp.create_operation "arith.subf"(%9, %15 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %17 = pdl_interp.get_result 0 of %16
      pdl_interp.replace %2 with (%17 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sin_2(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 2.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.sin"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "math.cos"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "arith.mulf"(%7, %9 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "arith.mulf"(%5, %11 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      pdl_interp.replace %1 with (%13 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sin_3(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 3.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.sin"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.mulf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_attribute 4.000000e+00 : f32
      %11 = pdl_interp.create_operation "arith.constant" {"value" = %10} -> (%3 : !pdl.type)
      %12 = pdl_interp.get_result 0 of %11
      %13 = pdl_interp.create_operation "math.sin"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %14 = pdl_interp.get_result 0 of %13
      %15 = pdl_interp.create_attribute 3.000000e+00 : f32
      %16 = pdl_interp.create_operation "arith.constant" {"value" = %15} -> (%3 : !pdl.type)
      %17 = pdl_interp.get_result 0 of %16
      %18 = pdl_interp.create_operation "math.powf"(%14, %17 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %19 = pdl_interp.get_result 0 of %18
      %20 = pdl_interp.create_operation "arith.mulf"(%12, %19 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %21 = pdl_interp.get_result 0 of %20
      %22 = pdl_interp.create_operation "arith.subf"(%9, %21 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %23 = pdl_interp.get_result 0 of %22
      pdl_interp.replace %1 with (%23 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sin_acos(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 1.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.mulf"(%0, %0 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.subf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "math.sqrt"(%9 : !pdl.value) -> (%3 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      pdl_interp.replace %1 with (%11 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sin_atan(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 1.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.mulf"(%0, %0 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.addf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "math.sqrt"(%9 : !pdl.value) -> (%3 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "arith.divf"(%0, %11 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      pdl_interp.replace %1 with (%13 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cos_0(%0 : !pdl.operation) {
      %1 = pdl_interp.create_attribute 1.000000e+00 : f32
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.constant" {"value" = %1} -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %0 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cos_neg(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.cos"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %1 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cos_fabs(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.cos"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %1 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cos_acos(%0 : !pdl.value, %1 : !pdl.operation) {
      pdl_interp.replace %1 with (%0 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cos_sum(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "math.cos"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.cos"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.mulf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "math.sin"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "math.sin"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      %14 = pdl_interp.create_operation "arith.mulf"(%11, %13 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %15 = pdl_interp.get_result 0 of %14
      %16 = pdl_interp.create_operation "arith.subf"(%9, %15 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %17 = pdl_interp.get_result 0 of %16
      pdl_interp.replace %2 with (%17 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cos_diff(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "math.cos"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.cos"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.mulf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "math.sin"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "math.sin"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      %14 = pdl_interp.create_operation "arith.mulf"(%11, %13 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %15 = pdl_interp.get_result 0 of %14
      %16 = pdl_interp.create_operation "arith.addf"(%9, %15 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %17 = pdl_interp.get_result 0 of %16
      pdl_interp.replace %2 with (%17 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cos_2(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.cos"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.cos"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.mulf"(%4, %6 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_operation "math.sin"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      %11 = pdl_interp.create_operation "math.sin"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %12 = pdl_interp.get_result 0 of %11
      %13 = pdl_interp.create_operation "arith.mulf"(%10, %12 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %14 = pdl_interp.get_result 0 of %13
      %15 = pdl_interp.create_operation "arith.subf"(%8, %14 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %16 = pdl_interp.get_result 0 of %15
      pdl_interp.replace %1 with (%16 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cos_3(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 4.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.cos"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_attribute 3.000000e+00 : f32
      %9 = pdl_interp.create_operation "arith.constant" {"value" = %8} -> (%3 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      %11 = pdl_interp.create_operation "math.powf"(%7, %10 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %12 = pdl_interp.get_result 0 of %11
      %13 = pdl_interp.create_operation "arith.mulf"(%5, %12 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %14 = pdl_interp.get_result 0 of %13
      %15 = pdl_interp.create_attribute 3.000000e+00 : f32
      %16 = pdl_interp.create_operation "arith.constant" {"value" = %15} -> (%3 : !pdl.type)
      %17 = pdl_interp.get_result 0 of %16
      %18 = pdl_interp.create_operation "math.cos"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %19 = pdl_interp.get_result 0 of %18
      %20 = pdl_interp.create_operation "arith.mulf"(%17, %19 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %21 = pdl_interp.get_result 0 of %20
      %22 = pdl_interp.create_operation "arith.subf"(%14, %21 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %23 = pdl_interp.get_result 0 of %22
      pdl_interp.replace %1 with (%23 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cos_asin(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 1.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.mulf"(%0, %0 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.subf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "math.sqrt"(%9 : !pdl.value) -> (%3 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      pdl_interp.replace %1 with (%11 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cos_atan(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 1.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_attribute 1.000000e+00 : f32
      %7 = pdl_interp.create_operation "arith.constant" {"value" = %6} -> (%3 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_operation "arith.mulf"(%0, %0 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      %11 = pdl_interp.create_operation "arith.addf"(%8, %10 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %12 = pdl_interp.get_result 0 of %11
      %13 = pdl_interp.create_operation "math.sqrt"(%12 : !pdl.value) -> (%3 : !pdl.type)
      %14 = pdl_interp.get_result 0 of %13
      %15 = pdl_interp.create_operation "arith.divf"(%5, %14 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %16 = pdl_interp.get_result 0 of %15
      pdl_interp.replace %1 with (%16 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @tan_0(%0 : !pdl.operation) {
      %1 = pdl_interp.create_attribute 0.000000e+00 : f32
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.constant" {"value" = %1} -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %0 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @tan_neg(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.tan"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "arith.negf"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @tan_atan(%0 : !pdl.value, %1 : !pdl.operation) {
      pdl_interp.replace %1 with (%0 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @hang_0p_tan_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.sin"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_attribute 1.000000e+00 : f32
      %6 = pdl_interp.create_operation "arith.constant" {"value" = %5} -> (%2 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "math.cos"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "arith.addf"(%7, %9 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "arith.divf"(%4, %11 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      pdl_interp.replace %1 with (%13 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @hang_0m_tan_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.sin"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "arith.negf"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_attribute 1.000000e+00 : f32
      %8 = pdl_interp.create_operation "arith.constant" {"value" = %7} -> (%2 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "math.cos"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "arith.addf"(%9, %11 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      %14 = pdl_interp.create_operation "arith.divf"(%6, %13 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %15 = pdl_interp.get_result 0 of %14
      pdl_interp.replace %1 with (%15 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @tan_asin(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 1.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.mulf"(%0, %0 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.subf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "math.sqrt"(%9 : !pdl.value) -> (%3 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "arith.divf"(%0, %11 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      pdl_interp.replace %1 with (%13 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @tan_acos(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 1.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.mulf"(%0, %0 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.subf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "math.sqrt"(%9 : !pdl.value) -> (%3 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "arith.divf"(%11, %0 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      pdl_interp.replace %1 with (%13 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @diff_atan_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "math.atan"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.atan"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.subf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %2 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sum_atan_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "math.atan"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.atan"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.addf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %2 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @asin_neg(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.asin"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "arith.negf"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @atan_neg(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.atan"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "arith.negf"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cosh_sum(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "math.cosh"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.cosh"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.mulf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "math.sinh"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "math.sinh"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      %14 = pdl_interp.create_operation "arith.mulf"(%11, %13 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %15 = pdl_interp.get_result 0 of %14
      %16 = pdl_interp.create_operation "arith.addf"(%9, %15 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %17 = pdl_interp.get_result 0 of %16
      pdl_interp.replace %2 with (%17 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cosh_diff(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "math.cosh"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.cosh"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.mulf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "math.sinh"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "math.sinh"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      %14 = pdl_interp.create_operation "arith.mulf"(%11, %13 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %15 = pdl_interp.get_result 0 of %14
      %16 = pdl_interp.create_operation "arith.subf"(%9, %15 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %17 = pdl_interp.get_result 0 of %16
      pdl_interp.replace %2 with (%17 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cosh_2(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.sinh"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.sinh"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.mulf"(%4, %6 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_operation "math.cosh"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      %11 = pdl_interp.create_operation "math.cosh"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %12 = pdl_interp.get_result 0 of %11
      %13 = pdl_interp.create_operation "arith.mulf"(%10, %12 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %14 = pdl_interp.get_result 0 of %13
      %15 = pdl_interp.create_operation "arith.addf"(%8, %14 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %16 = pdl_interp.get_result 0 of %15
      pdl_interp.replace %1 with (%16 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cosh_1div2(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.cosh"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_attribute 1.000000e+00 : f32
      %6 = pdl_interp.create_operation "arith.constant" {"value" = %5} -> (%2 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.addf"(%4, %7 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_attribute 2.000000e+00 : f32
      %11 = pdl_interp.create_operation "arith.constant" {"value" = %10} -> (%2 : !pdl.type)
      %12 = pdl_interp.get_result 0 of %11
      %13 = pdl_interp.create_operation "arith.divf"(%9, %12 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %14 = pdl_interp.get_result 0 of %13
      %15 = pdl_interp.create_operation "math.sqrt"(%14 : !pdl.value) -> (%2 : !pdl.type)
      %16 = pdl_interp.get_result 0 of %15
      pdl_interp.replace %1 with (%16 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cosh_neg(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.cosh"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %1 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cosh_0(%0 : !pdl.operation) {
      %1 = pdl_interp.create_attribute 1.000000e+00 : f32
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.constant" {"value" = %1} -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %0 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cosh_asinh(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.mulf"(%0, %0 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_attribute 1.000000e+00 : f32
      %6 = pdl_interp.create_operation "arith.constant" {"value" = %5} -> (%2 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.addf"(%4, %7 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "math.sqrt"(%9 : !pdl.value) -> (%2 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      pdl_interp.replace %1 with (%11 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cosh_acosh(%0 : !pdl.value, %1 : !pdl.operation) {
      pdl_interp.replace %1 with (%0 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cosh_atanh(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 1.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_attribute 1.000000e+00 : f32
      %7 = pdl_interp.create_operation "arith.constant" {"value" = %6} -> (%3 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_operation "arith.mulf"(%0, %0 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      %11 = pdl_interp.create_operation "arith.subf"(%8, %10 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %12 = pdl_interp.get_result 0 of %11
      %13 = pdl_interp.create_operation "math.sqrt"(%12 : !pdl.value) -> (%3 : !pdl.type)
      %14 = pdl_interp.get_result 0 of %13
      %15 = pdl_interp.create_operation "arith.divf"(%5, %14 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %16 = pdl_interp.get_result 0 of %15
      pdl_interp.replace %1 with (%16 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sinh_sum(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "math.sinh"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.cosh"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.mulf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "math.cosh"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "math.sinh"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      %14 = pdl_interp.create_operation "arith.mulf"(%11, %13 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %15 = pdl_interp.get_result 0 of %14
      %16 = pdl_interp.create_operation "arith.addf"(%9, %15 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %17 = pdl_interp.get_result 0 of %16
      pdl_interp.replace %2 with (%17 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sinh_diff(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "math.sinh"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.cosh"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.mulf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "math.cosh"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "math.sinh"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      %14 = pdl_interp.create_operation "arith.mulf"(%11, %13 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %15 = pdl_interp.get_result 0 of %14
      %16 = pdl_interp.create_operation "arith.subf"(%9, %15 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %17 = pdl_interp.get_result 0 of %16
      pdl_interp.replace %2 with (%17 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sinh_2(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 2.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.sinh"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "math.cosh"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "arith.mulf"(%7, %9 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "arith.mulf"(%5, %11 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      pdl_interp.replace %1 with (%13 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sinh_1div2(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.sinh"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_attribute 2.000000e+00 : f32
      %6 = pdl_interp.create_operation "arith.constant" {"value" = %5} -> (%2 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "math.cosh"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_attribute 1.000000e+00 : f32
      %11 = pdl_interp.create_operation "arith.constant" {"value" = %10} -> (%2 : !pdl.type)
      %12 = pdl_interp.get_result 0 of %11
      %13 = pdl_interp.create_operation "arith.addf"(%9, %12 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %14 = pdl_interp.get_result 0 of %13
      %15 = pdl_interp.create_operation "arith.mulf"(%7, %14 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %16 = pdl_interp.get_result 0 of %15
      %17 = pdl_interp.create_operation "math.sqrt"(%16 : !pdl.value) -> (%2 : !pdl.type)
      %18 = pdl_interp.get_result 0 of %17
      %19 = pdl_interp.create_operation "arith.divf"(%4, %18 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %20 = pdl_interp.get_result 0 of %19
      pdl_interp.replace %1 with (%20 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sinh_neg(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.sinh"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "arith.negf"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sinh_0(%0 : !pdl.operation) {
      %1 = pdl_interp.create_attribute 0.000000e+00 : f32
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.constant" {"value" = %1} -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %0 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sinh_asinh(%0 : !pdl.value, %1 : !pdl.operation) {
      pdl_interp.replace %1 with (%0 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sinh_acosh(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.mulf"(%0, %0 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_attribute 1.000000e+00 : f32
      %6 = pdl_interp.create_operation "arith.constant" {"value" = %5} -> (%2 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.subf"(%4, %7 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "math.sqrt"(%9 : !pdl.value) -> (%2 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      pdl_interp.replace %1 with (%11 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sinh_atanh(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 1.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.mulf"(%0, %0 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.subf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "math.sqrt"(%9 : !pdl.value) -> (%3 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "arith.divf"(%0, %11 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      pdl_interp.replace %1 with (%13 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @tanh_2(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 2.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.tanh"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.mulf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_attribute 1.000000e+00 : f32
      %11 = pdl_interp.create_operation "arith.constant" {"value" = %10} -> (%3 : !pdl.type)
      %12 = pdl_interp.get_result 0 of %11
      %13 = pdl_interp.create_operation "math.tanh"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %14 = pdl_interp.get_result 0 of %13
      %15 = pdl_interp.create_operation "math.tanh"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %16 = pdl_interp.get_result 0 of %15
      %17 = pdl_interp.create_operation "arith.mulf"(%14, %16 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %18 = pdl_interp.get_result 0 of %17
      %19 = pdl_interp.create_operation "arith.addf"(%12, %18 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %20 = pdl_interp.get_result 0 of %19
      %21 = pdl_interp.create_operation "arith.divf"(%9, %20 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %22 = pdl_interp.get_result 0 of %21
      pdl_interp.replace %1 with (%22 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @tanh_1div2(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.sinh"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.cosh"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_attribute 1.000000e+00 : f32
      %8 = pdl_interp.create_operation "arith.constant" {"value" = %7} -> (%2 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "arith.addf"(%6, %9 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "arith.divf"(%4, %11 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      pdl_interp.replace %1 with (%13 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @tanh_sum(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "math.tanh"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.tanh"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.addf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_attribute 1.000000e+00 : f32
      %11 = pdl_interp.create_operation "arith.constant" {"value" = %10} -> (%3 : !pdl.type)
      %12 = pdl_interp.get_result 0 of %11
      %13 = pdl_interp.create_operation "math.tanh"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %14 = pdl_interp.get_result 0 of %13
      %15 = pdl_interp.create_operation "math.tanh"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %16 = pdl_interp.get_result 0 of %15
      %17 = pdl_interp.create_operation "arith.mulf"(%14, %16 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %18 = pdl_interp.get_result 0 of %17
      %19 = pdl_interp.create_operation "arith.addf"(%12, %18 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %20 = pdl_interp.get_result 0 of %19
      %21 = pdl_interp.create_operation "arith.divf"(%9, %20 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %22 = pdl_interp.get_result 0 of %21
      pdl_interp.replace %2 with (%22 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @tanh_asinh(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 1.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.mulf"(%0, %0 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.addf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "math.sqrt"(%9 : !pdl.value) -> (%3 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "arith.divf"(%0, %11 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      pdl_interp.replace %1 with (%13 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @tanh_acosh(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.mulf"(%0, %0 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_attribute 1.000000e+00 : f32
      %6 = pdl_interp.create_operation "arith.constant" {"value" = %5} -> (%2 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.subf"(%4, %7 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "math.sqrt"(%9 : !pdl.value) -> (%2 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "arith.divf"(%11, %0 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      pdl_interp.replace %1 with (%13 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @tanh_atanh(%0 : !pdl.value, %1 : !pdl.operation) {
      pdl_interp.replace %1 with (%0 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @asinh_2(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 2.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.absf"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "math.asinh"(%7 : !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "arith.mulf"(%5, %9 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      pdl_interp.replace %1 with (%11 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @associate_addradd(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.operation) {
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.addf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.addf"(%6, %2 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %3 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @associate_addr_(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.operation) {
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.addf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.subf"(%6, %2 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %3 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @add_rgt_identity(%0 : !pdl.value, %1 : !pdl.operation) {
      pdl_interp.replace %1 with (%0 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @fp_cancel_sign_sub_inv(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.operation) {
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.negf"(%0 : !pdl.value) -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.mulf"(%6, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_operation "arith.subf"(%2, %8 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      pdl_interp.replace %3 with (%10 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @fp_cancel_sub_sign(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.operation) {
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.mulf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.subf"(%2, %6 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %3 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @distribute_rgt1_in(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_attribute 1.000000e+00 : f32
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.constant" {"value" = %3} -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.addf"(%0, %6 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_operation "arith.mulf"(%8, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      pdl_interp.replace %2 with (%10 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sub_flip_reverse(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.subf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      pdl_interp.replace %2 with (%5 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @add_to_fraction(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.operation) {
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.mulf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.addf"(%6, %2 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_operation "arith.divf"(%8, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      pdl_interp.replace %3 with (%10 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @count_2(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 2.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.mulf"(%5, %0 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %1 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @add_commutative(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.addf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      pdl_interp.replace %2 with (%5 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @add_flip(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.negf"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.subf"(%1, %5 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %2 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cube_unmult(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 3.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.powf"(%0, %5 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %1 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @associate_mulrmul(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.operation) {
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.mulf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.mulf"(%6, %2 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %3 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @mult_flip_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.divf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      pdl_interp.replace %2 with (%5 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @rgt_mult_inverse(%0 : !pdl.operation) {
      %1 = pdl_interp.create_attribute 1.000000e+00 : f32
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.constant" {"value" = %1} -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %0 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @associate_mulrdiv(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.operation) {
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.mulf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.divf"(%6, %2 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %3 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @mul0_rgt(%0 : !pdl.operation) {
      %1 = pdl_interp.create_attribute 0.000000e+00 : f32
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.constant" {"value" = %1} -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %0 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @mul_rgt_identity(%0 : !pdl.value, %1 : !pdl.operation) {
      pdl_interp.replace %1 with (%0 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @distribute_lft_in(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.operation) {
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.mulf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.mulf"(%0, %2 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_operation "arith.addf"(%6, %8 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      pdl_interp.replace %3 with (%10 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @distribute_rgt_in(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.operation) {
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.mulf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.mulf"(%2, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_operation "arith.addf"(%6, %8 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      pdl_interp.replace %3 with (%10 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @distribute_rgt_neg_out(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.mulf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.negf"(%5 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %2 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @log_pow_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "math.powf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.log"(%5 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %2 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sqr_abs_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.absf"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.absf"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.mulf"(%4, %6 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %1 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sqr_neg_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.negf"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "arith.negf"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.mulf"(%4, %6 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %1 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @pow2(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 2.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.powf"(%0, %5 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %1 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @mul_commutative(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.mulf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      pdl_interp.replace %2 with (%5 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sub_flip(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.negf"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.addf"(%1, %5 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %2 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sub_negate_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.subf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.negf"(%5 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %2 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @add_inverses(%0 : !pdl.operation) {
      %1 = pdl_interp.create_attribute 0.000000e+00 : f32
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.constant" {"value" = %1} -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %0 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @associatesub_radd(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.operation) {
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.subf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.subf"(%6, %2 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %3 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @associatesub_r_(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.operation) {
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.subf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.addf"(%6, %2 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %3 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sub_rgt_identity(%0 : !pdl.value, %1 : !pdl.operation) {
      pdl_interp.replace %1 with (%0 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @fp_cancel_sub_sign_inv(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.operation) {
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.negf"(%0 : !pdl.value) -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.mulf"(%6, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_operation "arith.addf"(%2, %8 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      pdl_interp.replace %3 with (%10 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @fp_cancel_sign_sub(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.operation) {
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.mulf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.addf"(%2, %6 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %3 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @add_flip_rev(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.addf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      pdl_interp.replace %2 with (%5 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sub_to_fraction(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.operation) {
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.mulf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.subf"(%6, %2 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_operation "arith.divf"(%8, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      pdl_interp.replace %3 with (%10 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @mult_flip(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_attribute 1.000000e+00 : f32
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.constant" {"value" = %3} -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.divf"(%6, %0 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_operation "arith.mulf"(%1, %8 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      pdl_interp.replace %2 with (%10 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @frac_2neg(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.negf"(%0 : !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.negf"(%1 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.divf"(%5, %7 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %2 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @mul_inverses(%0 : !pdl.operation) {
      %1 = pdl_interp.create_attribute 1.000000e+00 : f32
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.constant" {"value" = %1} -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %0 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @associate_divrmul(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.operation) {
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.divf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.divf"(%6, %2 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %3 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @associate_divrdiv(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.value, %3 : !pdl.operation) {
      %4 = pdl_interp.create_type f32
      %5 = pdl_interp.create_operation "arith.divf"(%0, %1 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.mulf"(%6, %2 : !pdl.value, !pdl.value) -> (%4 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %3 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @div_rgt_identity(%0 : !pdl.value, %1 : !pdl.operation) {
      pdl_interp.replace %1 with (%0 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @distribute_frac_neg2(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.divf"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.negf"(%5 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %2 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @fabs_rhs_div(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 1.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.copysign"(%5, %0 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %1 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @tan_asin_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.asin"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.tan"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sinh_atanh_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.atanh"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.sinh"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sin_atan_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.atan"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.sin"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @tanh_asinh_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.asinh"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.tanh"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @_2_split(%0 : !pdl.operation) {
      %1 = pdl_interp.create_attribute 1.000000e+00 : f32
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.constant" {"value" = %1} -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_attribute 1.000000e+00 : f32
      %6 = pdl_interp.create_operation "arith.constant" {"value" = %5} -> (%2 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.addf"(%4, %7 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %0 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @_1_split(%0 : !pdl.operation) {
      %1 = pdl_interp.create_attribute 2.000000e+00 : f32
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.constant" {"value" = %1} -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_attribute 5.000000e-01 : f32
      %6 = pdl_interp.create_operation "arith.constant" {"value" = %5} -> (%2 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.mulf"(%4, %7 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      pdl_interp.replace %0 with (%9 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @_1_exp(%0 : !pdl.operation) {
      %1 = pdl_interp.create_attribute 0.000000e+00 : f32
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.constant" {"value" = %1} -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.exp"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %0 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cosh_0_rev(%0 : !pdl.operation) {
      %1 = pdl_interp.create_attribute 0.000000e+00 : f32
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.constant" {"value" = %1} -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.cosh"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %0 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sinh_0_rev(%0 : !pdl.operation) {
      %1 = pdl_interp.create_attribute 0.000000e+00 : f32
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.constant" {"value" = %1} -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.sinh"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %0 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @rem_sqrt_square_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.mulf"(%0, %0 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.sqrt"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @neg_fabs(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.negf"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.absf"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sqrt_fabs_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.sqrt"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.absf"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @pow1div2(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 5.000000e-01 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.powf"(%0, %5 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %1 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @copysign_neg(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "math.copysign"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.negf"(%5 : !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %2 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @copysign_fabs(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.absf"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %1 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cube_mult(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.mulf"(%0, %0 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "arith.mulf"(%0, %4 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @unpow3(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.mulf"(%0, %0 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "arith.mulf"(%4, %0 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @unpow_1(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 1.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.divf"(%5, %0 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %1 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @unpow1(%0 : !pdl.value, %1 : !pdl.operation) {
      pdl_interp.replace %1 with (%0 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @unpow0(%0 : !pdl.operation) {
      %1 = pdl_interp.create_attribute 1.000000e+00 : f32
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.constant" {"value" = %1} -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %0 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @unpow1div2(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.sqrt"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %1 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @unpow2(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.mulf"(%0, %0 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %1 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @unpow1div3(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.cbrt"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      pdl_interp.replace %1 with (%4 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @fmin_swap(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "fmin"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      pdl_interp.replace %2 with (%5 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @fmax_swap(%0 : !pdl.value, %1 : !pdl.value, %2 : !pdl.operation) {
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "fmax"(%0, %1 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      pdl_interp.replace %2 with (%5 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @exp_fabs(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.exp"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.absf"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sinh_add_cosh_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.cosh"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.sinh"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.addf"(%4, %6 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %1 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @pow1div3(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 0.333333343 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "math.powf"(%0, %5 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      pdl_interp.replace %1 with (%7 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cos_neg_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.negf"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.cos"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cos_fabs_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.absf"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.cos"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @tan_quot(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.sin"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.cos"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "arith.divf"(%4, %6 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      pdl_interp.replace %1 with (%8 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @sinh_def(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.exp"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "arith.negf"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "math.exp"(%6 : !pdl.value) -> (%2 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_operation "arith.subf"(%4, %8 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      %11 = pdl_interp.create_attribute 2.000000e+00 : f32
      %12 = pdl_interp.create_operation "arith.constant" {"value" = %11} -> (%2 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      %14 = pdl_interp.create_operation "arith.divf"(%10, %13 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %15 = pdl_interp.get_result 0 of %14
      pdl_interp.replace %1 with (%15 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cosh_def(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.exp"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "arith.negf"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "math.exp"(%6 : !pdl.value) -> (%2 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_operation "arith.addf"(%4, %8 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      %11 = pdl_interp.create_attribute 2.000000e+00 : f32
      %12 = pdl_interp.create_operation "arith.constant" {"value" = %11} -> (%2 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      %14 = pdl_interp.create_operation "arith.divf"(%10, %13 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %15 = pdl_interp.get_result 0 of %14
      pdl_interp.replace %1 with (%15 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @cosh_neg_rev(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.negf"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "math.cosh"(%4 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      pdl_interp.replace %1 with (%6 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @tanh_def_a(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "math.exp"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_operation "arith.negf"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %6 = pdl_interp.get_result 0 of %5
      %7 = pdl_interp.create_operation "math.exp"(%6 : !pdl.value) -> (%2 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_operation "arith.subf"(%4, %8 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      %11 = pdl_interp.create_operation "math.exp"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %12 = pdl_interp.get_result 0 of %11
      %13 = pdl_interp.create_operation "arith.negf"(%0 : !pdl.value) -> (%2 : !pdl.type)
      %14 = pdl_interp.get_result 0 of %13
      %15 = pdl_interp.create_operation "math.exp"(%14 : !pdl.value) -> (%2 : !pdl.type)
      %16 = pdl_interp.get_result 0 of %15
      %17 = pdl_interp.create_operation "arith.addf"(%12, %16 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %18 = pdl_interp.get_result 0 of %17
      %19 = pdl_interp.create_operation "arith.divf"(%10, %18 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %20 = pdl_interp.get_result 0 of %19
      pdl_interp.replace %1 with (%20 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @tanh_def_b(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 2.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.mulf"(%5, %0 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "math.exp"(%7 : !pdl.value) -> (%3 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_attribute 1.000000e+00 : f32
      %11 = pdl_interp.create_operation "arith.constant" {"value" = %10} -> (%3 : !pdl.type)
      %12 = pdl_interp.get_result 0 of %11
      %13 = pdl_interp.create_operation "arith.subf"(%9, %12 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %14 = pdl_interp.get_result 0 of %13
      %15 = pdl_interp.create_attribute 2.000000e+00 : f32
      %16 = pdl_interp.create_operation "arith.constant" {"value" = %15} -> (%3 : !pdl.type)
      %17 = pdl_interp.get_result 0 of %16
      %18 = pdl_interp.create_operation "arith.mulf"(%17, %0 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %19 = pdl_interp.get_result 0 of %18
      %20 = pdl_interp.create_operation "math.exp"(%19 : !pdl.value) -> (%3 : !pdl.type)
      %21 = pdl_interp.get_result 0 of %20
      %22 = pdl_interp.create_attribute 1.000000e+00 : f32
      %23 = pdl_interp.create_operation "arith.constant" {"value" = %22} -> (%3 : !pdl.type)
      %24 = pdl_interp.get_result 0 of %23
      %25 = pdl_interp.create_operation "arith.addf"(%21, %24 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %26 = pdl_interp.get_result 0 of %25
      %27 = pdl_interp.create_operation "arith.divf"(%14, %26 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %28 = pdl_interp.get_result 0 of %27
      pdl_interp.replace %1 with (%28 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @tanh_def_c(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 1.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_attribute -2.000000e+00 : f32
      %7 = pdl_interp.create_operation "arith.constant" {"value" = %6} -> (%3 : !pdl.type)
      %8 = pdl_interp.get_result 0 of %7
      %9 = pdl_interp.create_operation "arith.mulf"(%8, %0 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      %11 = pdl_interp.create_operation "math.exp"(%10 : !pdl.value) -> (%3 : !pdl.type)
      %12 = pdl_interp.get_result 0 of %11
      %13 = pdl_interp.create_operation "arith.subf"(%5, %12 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %14 = pdl_interp.get_result 0 of %13
      %15 = pdl_interp.create_attribute 1.000000e+00 : f32
      %16 = pdl_interp.create_operation "arith.constant" {"value" = %15} -> (%3 : !pdl.type)
      %17 = pdl_interp.get_result 0 of %16
      %18 = pdl_interp.create_attribute -2.000000e+00 : f32
      %19 = pdl_interp.create_operation "arith.constant" {"value" = %18} -> (%3 : !pdl.type)
      %20 = pdl_interp.get_result 0 of %19
      %21 = pdl_interp.create_operation "arith.mulf"(%20, %0 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %22 = pdl_interp.get_result 0 of %21
      %23 = pdl_interp.create_operation "math.exp"(%22 : !pdl.value) -> (%3 : !pdl.type)
      %24 = pdl_interp.get_result 0 of %23
      %25 = pdl_interp.create_operation "arith.addf"(%17, %24 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %26 = pdl_interp.get_result 0 of %25
      %27 = pdl_interp.create_operation "arith.divf"(%14, %26 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %28 = pdl_interp.get_result 0 of %27
      pdl_interp.replace %1 with (%28 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @asinh_def(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.mulf"(%0, %0 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_attribute 1.000000e+00 : f32
      %6 = pdl_interp.create_operation "arith.constant" {"value" = %5} -> (%2 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.addf"(%4, %7 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "math.sqrt"(%9 : !pdl.value) -> (%2 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "arith.addf"(%0, %11 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      %14 = pdl_interp.create_operation "math.log"(%13 : !pdl.value) -> (%2 : !pdl.type)
      %15 = pdl_interp.get_result 0 of %14
      pdl_interp.replace %1 with (%15 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @acosh_def(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_type f32
      %3 = pdl_interp.create_operation "arith.mulf"(%0, %0 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %4 = pdl_interp.get_result 0 of %3
      %5 = pdl_interp.create_attribute 1.000000e+00 : f32
      %6 = pdl_interp.create_operation "arith.constant" {"value" = %5} -> (%2 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_operation "arith.subf"(%4, %7 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %9 = pdl_interp.get_result 0 of %8
      %10 = pdl_interp.create_operation "math.sqrt"(%9 : !pdl.value) -> (%2 : !pdl.type)
      %11 = pdl_interp.get_result 0 of %10
      %12 = pdl_interp.create_operation "arith.addf"(%0, %11 : !pdl.value, !pdl.value) -> (%2 : !pdl.type)
      %13 = pdl_interp.get_result 0 of %12
      %14 = pdl_interp.create_operation "math.log"(%13 : !pdl.value) -> (%2 : !pdl.type)
      %15 = pdl_interp.get_result 0 of %14
      pdl_interp.replace %1 with (%15 : !pdl.value)
      pdl_interp.finalize
    }
    pdl_interp.func @atanh_def(%0 : !pdl.value, %1 : !pdl.operation) {
      %2 = pdl_interp.create_attribute 1.000000e+00 : f32
      %3 = pdl_interp.create_type f32
      %4 = pdl_interp.create_operation "arith.constant" {"value" = %2} -> (%3 : !pdl.type)
      %5 = pdl_interp.get_result 0 of %4
      %6 = pdl_interp.create_operation "arith.addf"(%5, %0 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %7 = pdl_interp.get_result 0 of %6
      %8 = pdl_interp.create_attribute 1.000000e+00 : f32
      %9 = pdl_interp.create_operation "arith.constant" {"value" = %8} -> (%3 : !pdl.type)
      %10 = pdl_interp.get_result 0 of %9
      %11 = pdl_interp.create_operation "arith.subf"(%10, %0 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %12 = pdl_interp.get_result 0 of %11
      %13 = pdl_interp.create_operation "arith.divf"(%7, %12 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %14 = pdl_interp.get_result 0 of %13
      %15 = pdl_interp.create_operation "math.log"(%14 : !pdl.value) -> (%3 : !pdl.type)
      %16 = pdl_interp.get_result 0 of %15
      %17 = pdl_interp.create_attribute 2.000000e+00 : f32
      %18 = pdl_interp.create_operation "arith.constant" {"value" = %17} -> (%3 : !pdl.type)
      %19 = pdl_interp.get_result 0 of %18
      %20 = pdl_interp.create_operation "arith.divf"(%16, %19 : !pdl.value, !pdl.value) -> (%3 : !pdl.type)
      %21 = pdl_interp.get_result 0 of %20
      pdl_interp.replace %1 with (%21 : !pdl.value)
      pdl_interp.finalize
    }
  }
}

