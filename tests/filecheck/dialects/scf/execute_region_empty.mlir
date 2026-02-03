// RUN: xdsl-opt %s --verify-diagnostics | filecheck %s

// CHECK:      Operation does not verify: scf.execute_region op region needs to have at least one block
func.func @execute_region_empty_region() {
    scf.execute_region {
    }
    func.return
}
