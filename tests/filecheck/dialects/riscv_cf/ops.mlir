// RUN: XDSL_ROUNDTRIP

%0, %1, %2, %3, %4, %5 = "test.op"() : () -> (!riscv.reg<a0>, !riscv.reg<a1>, !riscv.reg<a2>, !riscv.reg<a3>, !riscv.reg<a2>, !riscv.reg<a3>)

"test.op"() ({
    riscv_cf.beq %0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>, ^then(%2 : !riscv.reg<a2>, %3 : !riscv.reg<a3>), ^else0(%4 : !riscv.reg<a2>, %5 : !riscv.reg<a3>)
  ^else0(%e00 : !riscv.reg<a2>, %e01 : !riscv.reg<a3>):
    riscv.label "else0"
    riscv_cf.bne %0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>, ^then(%2 : !riscv.reg<a2>, %3 : !riscv.reg<a3>), ^else1(%4 : !riscv.reg<a2>, %5 : !riscv.reg<a3>) attributes {"comment" = "comment"}
  ^else1(%e10 : !riscv.reg<a2>, %e11 : !riscv.reg<a3>):
    riscv.label "else1"
    riscv_cf.blt %0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>, ^then(%2 : !riscv.reg<a2>, %3 : !riscv.reg<a3>), ^else2(%4 : !riscv.reg<a2>, %5 : !riscv.reg<a3>) attributes {"hello" = "world"}
  ^else2(%e20 : !riscv.reg<a2>, %e21 : !riscv.reg<a3>):
    riscv.label "else2"
    riscv_cf.bge %0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>, ^then(%2 : !riscv.reg<a2>, %3 : !riscv.reg<a3>), ^else3(%4 : !riscv.reg<a2>, %5 : !riscv.reg<a3>)
  ^else3(%e30 : !riscv.reg<a2>, %e31 : !riscv.reg<a3>):
    riscv.label "else3"
    riscv_cf.bltu %0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>, ^then(%2 : !riscv.reg<a2>, %3 : !riscv.reg<a3>), ^else4(%4 : !riscv.reg<a2>, %5 : !riscv.reg<a3>)
  ^else4(%e40 : !riscv.reg<a2>, %e41 : !riscv.reg<a3>):
    riscv.label "else4"
    riscv_cf.bgeu %0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>, ^then(%2 : !riscv.reg<a2>, %3 : !riscv.reg<a3>), ^else5(%4 : !riscv.reg<a2>, %5 : !riscv.reg<a3>)
  ^else5(%e50 : !riscv.reg<a2>, %e51 : !riscv.reg<a3>):
    riscv.label "else5"
    riscv_cf.branch ^then(%2 : !riscv.reg<a2>, %3 : !riscv.reg<a3>) attributes {"hello" = "world"}
  ^then(%t0 : !riscv.reg<a2>, %t1 : !riscv.reg<a3>):
    riscv.label "then"
    riscv_cf.j ^then(%2 : !riscv.reg<a2>, %3 : !riscv.reg<a3>) attributes {"hello" = "world"}
}) : () -> ()

// CHECK:       builtin.module {
// CHECK-NEXT:    %0, %1, %2, %3, %4, %5 = "test.op"() : () -> (!riscv.reg<a0>, !riscv.reg<a1>, !riscv.reg<a2>, !riscv.reg<a3>, !riscv.reg<a2>, !riscv.reg<a3>)
// CHECK-NEXT:    "test.op"() ({
// CHECK-NEXT:      riscv_cf.beq %0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>, ^{{.+}}(%2 : !riscv.reg<a2>, %3 : !riscv.reg<a3>), ^{{.+}}(%4 : !riscv.reg<a2>, %5 : !riscv.reg<a3>)
// CHECK-NEXT:    ^{{.+}}(%{{.+}} : !riscv.reg<a2>, %{{.+}} : !riscv.reg<a3>):
// CHECK-NEXT:      riscv.label "else0"
// CHECK-NEXT:      riscv_cf.bne %0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>, ^{{.+}}(%2 : !riscv.reg<a2>, %3 : !riscv.reg<a3>), ^{{.+}}(%4 : !riscv.reg<a2>, %5 : !riscv.reg<a3>) attributes {comment = "comment"}
// CHECK-NEXT:    ^{{.+}}(%{{.+}} : !riscv.reg<a2>, %{{.+}} : !riscv.reg<a3>):
// CHECK-NEXT:      riscv.label "else1"
// CHECK-NEXT:      riscv_cf.blt %0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>, ^{{.+}}(%2 : !riscv.reg<a2>, %3 : !riscv.reg<a3>), ^{{.+}}(%4 : !riscv.reg<a2>, %5 : !riscv.reg<a3>) attributes {hello = "world"}
// CHECK-NEXT:    ^{{.+}}(%{{.+}} : !riscv.reg<a2>, %{{.+}} : !riscv.reg<a3>):
// CHECK-NEXT:      riscv.label "else2"
// CHECK-NEXT:      riscv_cf.bge %0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>, ^{{.+}}(%2 : !riscv.reg<a2>, %3 : !riscv.reg<a3>), ^{{.+}}(%4 : !riscv.reg<a2>, %5 : !riscv.reg<a3>)
// CHECK-NEXT:    ^{{.+}}(%{{.+}} : !riscv.reg<a2>, %{{.+}} : !riscv.reg<a3>):
// CHECK-NEXT:      riscv.label "else3"
// CHECK-NEXT:      riscv_cf.bltu %0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>, ^{{.+}}(%2 : !riscv.reg<a2>, %3 : !riscv.reg<a3>), ^{{.+}}(%4 : !riscv.reg<a2>, %5 : !riscv.reg<a3>)
// CHECK-NEXT:    ^{{.+}}(%{{.+}} : !riscv.reg<a2>, %{{.+}} : !riscv.reg<a3>):
// CHECK-NEXT:      riscv.label "else4"
// CHECK-NEXT:      riscv_cf.bgeu %0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>, ^{{.+}}(%2 : !riscv.reg<a2>, %3 : !riscv.reg<a3>), ^{{.+}}(%4 : !riscv.reg<a2>, %5 : !riscv.reg<a3>)
// CHECK-NEXT:    ^{{.+}}(%{{.+}} : !riscv.reg<a2>, %{{.+}} : !riscv.reg<a3>):
// CHECK-NEXT:      riscv.label "else5"
// CHECK-NEXT:      riscv_cf.branch ^{{.+}}(%2 : !riscv.reg<a2>, %3 : !riscv.reg<a3>) attributes {hello = "world"}
// CHECK-NEXT:    ^{{.+}}(%{{.+}} : !riscv.reg<a2>, %{{.+}} : !riscv.reg<a3>):
// CHECK-NEXT:      riscv.label "then"
// CHECK-NEXT:      riscv_cf.j ^{{.+}}(%2 : !riscv.reg<a2>, %3 : !riscv.reg<a3>) attributes {hello = "world"}
// CHECK-NEXT:    }) : () -> ()
// CHECK-NEXT:  }

// -----

%0, %1, %2, %3, %4, %5 = "test.op"() : () -> (!riscv.reg<a0>, !riscv.reg<a1>, !riscv.freg<ft2>, !riscv.freg<ft3>, !riscv.freg<ft2>, !riscv.freg<ft3>)

"test.op"() ({
    riscv_cf.beq %0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>, ^then(%2 : !riscv.freg<ft2>, %3 : !riscv.freg<ft3>), ^else0(%4 : !riscv.freg<ft2>, %5 : !riscv.freg<ft3>)
  ^else0(%e00 : !riscv.freg<ft2>, %e01 : !riscv.freg<ft3>):
    riscv.label "else0"
    riscv_cf.bne %0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>, ^then(%2 : !riscv.freg<ft2>, %3 : !riscv.freg<ft3>), ^else1(%4 : !riscv.freg<ft2>, %5 : !riscv.freg<ft3>) attributes {"comment" = "comment"}
  ^else1(%e10 : !riscv.freg<ft2>, %e11 : !riscv.freg<ft3>):
    riscv.label "else1"
    riscv_cf.blt %0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>, ^then(%2 : !riscv.freg<ft2>, %3 : !riscv.freg<ft3>), ^else2(%4 : !riscv.freg<ft2>, %5 : !riscv.freg<ft3>) attributes {"hello" = "world"}
  ^else2(%e20 : !riscv.freg<ft2>, %e21 : !riscv.freg<ft3>):
    riscv.label "else2"
    riscv_cf.bge %0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>, ^then(%2 : !riscv.freg<ft2>, %3 : !riscv.freg<ft3>), ^else3(%4 : !riscv.freg<ft2>, %5 : !riscv.freg<ft3>)
  ^else3(%e30 : !riscv.freg<ft2>, %e31 : !riscv.freg<ft3>):
    riscv.label "else3"
    riscv_cf.bltu %0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>, ^then(%2 : !riscv.freg<ft2>, %3 : !riscv.freg<ft3>), ^else4(%4 : !riscv.freg<ft2>, %5 : !riscv.freg<ft3>)
  ^else4(%e40 : !riscv.freg<ft2>, %e41 : !riscv.freg<ft3>):
    riscv.label "else4"
    riscv_cf.bgeu %0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>, ^then(%2 : !riscv.freg<ft2>, %3 : !riscv.freg<ft3>), ^else5(%4 : !riscv.freg<ft2>, %5 : !riscv.freg<ft3>)
  ^else5(%e50 : !riscv.freg<ft2>, %e51 : !riscv.freg<ft3>):
    riscv.label "else5"
    riscv_cf.branch ^then(%2 : !riscv.freg<ft2>, %3 : !riscv.freg<ft3>) attributes {"hello" = "world"}
  ^then(%t0 : !riscv.freg<ft2>, %t1 : !riscv.freg<ft3>):
    riscv.label "then"
    riscv_cf.j ^then(%2 : !riscv.freg<ft2>, %3 : !riscv.freg<ft3>) attributes {"hello" = "world"}
}) : () -> ()

// CHECK:       builtin.module {
// CHECK-NEXT:    %0, %1, %2, %3, %4, %5 = "test.op"() : () -> (!riscv.reg<a0>, !riscv.reg<a1>, !riscv.freg<ft2>, !riscv.freg<ft3>, !riscv.freg<ft2>, !riscv.freg<ft3>)
// CHECK-NEXT:    "test.op"() ({
// CHECK-NEXT:      riscv_cf.beq %0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>, ^{{.+}}(%2 : !riscv.freg<ft2>, %3 : !riscv.freg<ft3>), ^{{.+}}(%4 : !riscv.freg<ft2>, %5 : !riscv.freg<ft3>)
// CHECK-NEXT:    ^{{.+}}(%{{.+}} : !riscv.freg<ft2>, %{{.+}} : !riscv.freg<ft3>):
// CHECK-NEXT:      riscv.label "else0"
// CHECK-NEXT:      riscv_cf.bne %0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>, ^{{.+}}(%2 : !riscv.freg<ft2>, %3 : !riscv.freg<ft3>), ^{{.+}}(%4 : !riscv.freg<ft2>, %5 : !riscv.freg<ft3>) attributes {comment = "comment"}
// CHECK-NEXT:    ^{{.+}}(%{{.+}} : !riscv.freg<ft2>, %{{.+}} : !riscv.freg<ft3>):
// CHECK-NEXT:      riscv.label "else1"
// CHECK-NEXT:      riscv_cf.blt %0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>, ^{{.+}}(%2 : !riscv.freg<ft2>, %3 : !riscv.freg<ft3>), ^{{.+}}(%4 : !riscv.freg<ft2>, %5 : !riscv.freg<ft3>) attributes {hello = "world"}
// CHECK-NEXT:    ^{{.+}}(%{{.+}} : !riscv.freg<ft2>, %{{.+}} : !riscv.freg<ft3>):
// CHECK-NEXT:      riscv.label "else2"
// CHECK-NEXT:      riscv_cf.bge %0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>, ^{{.+}}(%2 : !riscv.freg<ft2>, %3 : !riscv.freg<ft3>), ^{{.+}}(%4 : !riscv.freg<ft2>, %5 : !riscv.freg<ft3>)
// CHECK-NEXT:    ^{{.+}}(%{{.+}} : !riscv.freg<ft2>, %{{.+}} : !riscv.freg<ft3>):
// CHECK-NEXT:      riscv.label "else3"
// CHECK-NEXT:      riscv_cf.bltu %0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>, ^{{.+}}(%2 : !riscv.freg<ft2>, %3 : !riscv.freg<ft3>), ^{{.+}}(%4 : !riscv.freg<ft2>, %5 : !riscv.freg<ft3>)
// CHECK-NEXT:    ^{{.+}}(%{{.+}} : !riscv.freg<ft2>, %{{.+}} : !riscv.freg<ft3>):
// CHECK-NEXT:      riscv.label "else4"
// CHECK-NEXT:      riscv_cf.bgeu %0 : !riscv.reg<a0>, %1 : !riscv.reg<a1>, ^{{.+}}(%2 : !riscv.freg<ft2>, %3 : !riscv.freg<ft3>), ^{{.+}}(%4 : !riscv.freg<ft2>, %5 : !riscv.freg<ft3>)
// CHECK-NEXT:    ^{{.+}}(%{{.+}} : !riscv.freg<ft2>, %{{.+}} : !riscv.freg<ft3>):
// CHECK-NEXT:      riscv.label "else5"
// CHECK-NEXT:      riscv_cf.branch ^{{.+}}(%2 : !riscv.freg<ft2>, %3 : !riscv.freg<ft3>) attributes {hello = "world"}
// CHECK-NEXT:    ^{{.+}}(%{{.+}} : !riscv.freg<ft2>, %{{.+}} : !riscv.freg<ft3>):
// CHECK-NEXT:      riscv.label "then"
// CHECK-NEXT:      riscv_cf.j ^{{.+}}(%2 : !riscv.freg<ft2>, %3 : !riscv.freg<ft3>) attributes {hello = "world"}
// CHECK-NEXT:    }) : () -> ()
// CHECK-NEXT:  }
