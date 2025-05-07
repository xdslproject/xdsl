// RUN: xdsl-opt -p canonicalize-dmp %s | filecheck %s

builtin.module {
    %ref = "test.op"() : () -> (!stencil.field<[0,1024]x[0,1024]xf32>)
    %val = "test.op"() : () -> (!stencil.temp<[0,1024]x[0,1024]xf32>)

    "dmp.swap"(%ref) {
        strategy = #dmp.grid_slice_2d<#dmp.topo<2x2>, false>,
        swaps = [
            #dmp.exchange<at [4, 0] size [100, 4] source offset [0, 4] to [-1, 0]>,
            #dmp.exchange<at [4, 104] size [100, 0] source offset [0, -4] to [1, 0]>
        ]
    } : (!stencil.field<[0,1024]x[0,1024]xf32>) -> ()

    // CHECK: "dmp.swap"(%ref) {strategy = #dmp.grid_slice_2d<#dmp.topo<2x2>, false>, swaps = [#dmp.exchange<at [4, 0] size [100, 4] source offset [0, 4] to [-1, 0]>]} : (!stencil.field<[0,1024]x[0,1024]xf32>) -> ()

    %swap_val = "dmp.swap"(%val) {
        strategy = #dmp.grid_slice_2d<#dmp.topo<2x2>, false>,
        swaps = [
            #dmp.exchange<at [4, 0] size [100, 0] source offset [0, 4] to [-1, 0]>,
            #dmp.exchange<at [4, 104] size [100, 0] source offset [0, -4] to [1, 0]>
        ]
    } : (!stencil.temp<[0,1024]x[0,1024]xf32>) -> (!stencil.temp<[0,1024]x[0,1024]xf32>)

    "test.op"(%swap_val) : (!stencil.temp<[0,1024]x[0,1024]xf32>) -> ()

    // this op should be completely removed since both exchanges are empty, so we expect the next op to be a test.op
    // and its operand replaced by the unswapped input
    // CHECK-NEXT: "test.op"(%val) : (!stencil.temp<[0,1024]x[0,1024]xf32>) -> ()
}
