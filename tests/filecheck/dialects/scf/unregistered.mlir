// RUN: xdsl-opt %s --print-op-generic --allow-unregistered-dialect | xdsl-opt --split-input-file | filecheck %s

  func.func @for_unregistered() {
    %lb = arith.constant 0 : index
    %ub = arith.constant 42 : index
    %s = arith.constant 3 : index
    scf.for %iv = %lb to %ub step %s {
      "unregistered_op"() : () -> ()
      scf.yield
    }
    func.return
  }
