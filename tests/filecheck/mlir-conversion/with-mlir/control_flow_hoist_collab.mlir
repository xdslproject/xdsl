// RUN: xdsl-opt -p "mlir-opt[loop-invariant-code-motion]" %s | filecheck %s --check-prefix WITHOUT
// RUN: xdsl-opt -p "control-flow-hoist,mlir-opt[loop-invariant-code-motion]" %s | filecheck %s --check-prefix WITH

func.func @nested_loop_invariant(%n : index) {
    %0 = arith.constant 0 : index
    %1 = arith.constant 1 : index
    %100 = arith.constant 100 : index
    scf.for %i = %0 to %100 step %1 {
        %cond = "test.op"() : () -> (i1)
        %thing = scf.if %cond -> (index) {
            // This is loop invariant
            // Also is nested in conditional
            // Contradictory local intuitions :S
            // MLIR really want to keep operations nested if they only occur on one
            // branch, which locally makes sense!
            %n100 = arith.muli %n, %100 : index
            scf.yield %n100 :index
        } else {
            scf.yield %n : index
        }
        "test.op"(%thing) : (index) -> ()
        scf.yield
    }
    return
}

// Currently, loop-invariant-code-motion does not help.

// WITHOUT:         func.func @nested_loop_invariant(%arg0 : index) {
// WITHOUT-NEXT:      %0 = arith.constant 0 : index
// WITHOUT-NEXT:      %1 = arith.constant 1 : index
// WITHOUT-NEXT:      %2 = arith.constant 100 : index
// WITHOUT-NEXT:      scf.for %arg1 = %0 to %2 step %1 {
// WITHOUT-NEXT:        %3 = "test.op"() : () -> i1
// WITHOUT-NEXT:        %4 = scf.if %3 -> (index) {
// WITHOUT-NEXT:          %5 = arith.muli %arg0, %2 : index
// WITHOUT-NEXT:          scf.yield %5 : index
// WITHOUT-NEXT:        } else {
// WITHOUT-NEXT:          scf.yield %arg0 : index
// WITHOUT-NEXT:        }
// WITHOUT-NEXT:        "test.op"(%4) : (index) -> ()
// WITHOUT-NEXT:      }
// WITHOUT-NEXT:      func.return
// WITHOUT-NEXT:    }

// With control-flow-hoist, we force things to bubble up at loop-level, to get further hoisted as loop invariant.
// MLIR already provides the opposit control-flow-sink to put whatever was *invariant* back in the right branches
// if so desired

// WITH:         func.func @nested_loop_invariant(%arg0 : index) {
// WITH-NEXT:      %0 = arith.constant 0 : index
// WITH-NEXT:      %1 = arith.constant 1 : index
// WITH-NEXT:      %2 = arith.constant 100 : index
// WITH-NEXT:      %3 = arith.muli %arg0, %2 : index
// WITH-NEXT:      scf.for %arg1 = %0 to %2 step %1 {
// WITH-NEXT:        %4 = "test.op"() : () -> i1
// WITH-NEXT:        %5 = scf.if %4 -> (index) {
// WITH-NEXT:          scf.yield %3 : index
// WITH-NEXT:        } else {
// WITH-NEXT:          scf.yield %arg0 : index
// WITH-NEXT:        }
// WITH-NEXT:        "test.op"(%5) : (index) -> ()
// WITH-NEXT:      }
// WITH-NEXT:      func.return
// WITH-NEXT:    }
