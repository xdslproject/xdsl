
func.func @example(%T : index, %u_init : !stencil.field<[-4,68]x[-4,68]xf64>, %v_init : !stencil.field<[-4,68]x[-4,68]xf64>) {
    %0 = arith.constant 0 : index
    %1 = arith.constant 1 : index

    // Iterate %N time, take initial buffers
    scf.for %time = %0 to %T step %1 iter_args(%u = %u_init, %v = %v_init) -> (!stencil.field<[-4,68]x[-4,68]xf64>, !stencil.field<[-4,68]x[-4,68]xf64>) {

        // Load input
        %ut = stencil.load %u : !stencil.field<[-4,68]x[-4,68]xf64> -> !stencil.temp<?x?xf64>

        // Value-semantics compute
        %vt = stencil.apply(%uarg = %ut : !stencil.temp<?x?xf64>) -> (!stencil.temp<?x?xf64>) {
            %left = stencil.access %uarg[-1, 0] : !stencil.temp<?x?xf64>
            %center = stencil.access %uarg[-1, 0] : !stencil.temp<?x?xf64>
            %right = stencil.access %uarg[-1, 0] : !stencil.temp<?x?xf64>
            %value = func.call @compute(%left, %center, %right) : (f64, f64, f64) -> f64
            stencil.return %value : f64
        }

        // Store outputs
        stencil.store %vt to %v ([0, 0] : [64, 64]) : !stencil.temp<?x?xf64> to !stencil.field<[-4,68]x[-4,68]xf64>

        // Swap buffers
        scf.yield %v, %u : !stencil.field<[-4,68]x[-4,68]xf64>, !stencil.field<[-4,68]x[-4,68]xf64>

    }
    return
}
