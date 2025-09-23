// RUN: true

// Support file for apply_eqsat_pdl_extra_file.mlir

// x * 0 -> 0
pdl.pattern : benefit(1) {
  %x = pdl.operand
  %type = pdl.type
  %zero = pdl.attribute = 0 : i32
  %constop = pdl.operation "arith.constant" {"value" = %zero} -> (%type : !pdl.type)
  %const = pdl.result 0 of %constop
  %mulop = pdl.operation "arith.muli" (%x, %const : !pdl.value, !pdl.value) -> (%type : !pdl.type)
  pdl.rewrite %mulop {
    pdl.replace %mulop with %constop
  }
}

// x - x -> 0
pdl.pattern : benefit(1) {
  %x = pdl.operand
  %type = pdl.type
  %subop = pdl.operation "arith.subi" (%x, %x : !pdl.value, !pdl.value) -> (%type : !pdl.type)
  pdl.rewrite %subop {
    %zero = pdl.attribute = 0 : i32
    %constop = pdl.operation "arith.constant" {"value" = %zero} -> (%type : !pdl.type)
    pdl.replace %subop with %constop
  }
}
