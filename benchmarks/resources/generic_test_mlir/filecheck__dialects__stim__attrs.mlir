"builtin.module"() ({
  "test.op"() {qubit = !stim.qubit<0>, qubitcoord = #stim.qubit_coord<(0,0), !stim.qubit<0>>} : () -> ()
  %0 = "test.op"() : () -> !stim.qubit<0>
}) : () -> ()
