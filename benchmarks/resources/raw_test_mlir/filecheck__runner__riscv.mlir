// RUN: xdsl-run %s --verbose | filecheck %s

riscv_func.func @malloc(!riscv.reg) -> !riscv.reg

riscv_func.func @putchar(!riscv.reg) -> !riscv.reg

riscv_func.func @free(!riscv.reg) -> ()

riscv_func.func @main() -> () {
    %33 = riscv.li 33 : !riscv.reg
    %newline = riscv.li 10 : !riscv.reg
    %ptr = riscv_func.call @malloc(%33) : (!riscv.reg) -> !riscv.reg
    riscv.sw %ptr, %33, 0 : (!riscv.reg, !riscv.reg) -> ()

    %res = riscv.lw %ptr, 0 : (!riscv.reg) -> !riscv.reg

    %nothing = riscv_func.call @putchar(%res) : (!riscv.reg) -> !riscv.reg
    %nothing2 = riscv_func.call @putchar(%newline) : (!riscv.reg) -> !riscv.reg

    riscv_func.call @free(%ptr) : (!riscv.reg) -> ()

    riscv_func.return
}

// CHECK: !
// CHECK: result: ()
