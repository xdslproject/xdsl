

func.func @basic() {
    "test.op"() {"before_op"} : () -> ()

    %cond = "test.op"() {"specialize_on_vals"= [true]} : () -> i1

    "scf.if"(%cond) ({
        "test.op"() {"inside_if"} : () -> ()
        scf.yield
    }, {
        scf.yield
    }) : (i1) -> ()

    "test.op"() {"after_op"} : () -> ()

    func.return
}


// after:



// func.func @basic() {
//     "test.op"() {"before_op"} : () -> ()
//
//     %cond = "test.op"() {"specialize_on_vals": [true]} : () -> i1
//     %true = arith.constant true : i1
//     %cond_matches_spec = arith.cmpi eq, %cond, $true : (i1, i1) -> i1
//
//     scf.if (%cond_matches_spec) {
//         func.call @basic_spec_1()
//     } else {
//         scf.if(%cond) {
//             "test.op"() {"inside_if"} : () -> ()
//         }
//
//         "test.op"() {"after_op"} : () -> ()
//     }
// }
//
// func.func @basic_specialized() {
//     %cond = arith.constant true : i1
//
//     scf.if(%cond) {
//         "test.op"() {"inside_if"} : () -> ()
//     }
//
//     "test.op"() {"after_op"} : () -> ()
// }