// RUN: xdsl-opt --print-op-generic %s | xdsl-opt | filecheck %s

builtin.module {
    %ref = "test.op"() : () -> (memref<1024x1024xf32>)

    "dmp.swap"(%ref) {
        grid = #dmp.grid<2x2>,
        swaps = [
            #dmp.exchange_decl<at [4, 0] size [100, 4] source offset [0, 4] to [-1, 0]>,
            #dmp.exchange_decl<at [4, 104] size [100, 4] source offset [0, -4] to [1, 0]>
        ]
    } : (memref<1024x1024xf32>) -> ()

    // CHECK: "dmp.swap"(%ref) {"grid" = #dmp.grid<2x2>, "swaps" = [#dmp.exchange_decl<at [4, 0] size [100, 4] source offset [0, 4] to [-1, 0]>, #dmp.exchange_decl<at [4, 104] size [100, 4] source offset [0, -4] to [1, 0]>]} : (memref<1024x1024xf32>) -> ()
}
