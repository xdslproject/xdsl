"builtin.module"() ({
  %0 = "test.op"() : () -> !stencil.field<[0,1024]x[0,1024]xf32>
  %1 = "test.op"() : () -> !stencil.temp<[0,1024]x[0,1024]xf32>
  "dmp.swap"(%0) {strategy = #dmp.grid_slice_2d<#dmp.topo<2x2>, false>, swaps = [#dmp.exchange<at [4, 0] size [100, 4] source offset [0, 4] to [-1, 0]>, #dmp.exchange<at [4, 104] size [100, 4] source offset [0, -4] to [1, 0]>]} : (!stencil.field<[0,1024]x[0,1024]xf32>) -> ()
  %2 = "dmp.swap"(%1) {strategy = #dmp.grid_slice_2d<#dmp.topo<2x2>, false>, swaps = [#dmp.exchange<at [4, 0] size [100, 2] source offset [0, 4] to [-1, 0]>, #dmp.exchange<at [4, 104] size [100, 2] source offset [0, -4] to [1, 0]>]} : (!stencil.temp<[0,1024]x[0,1024]xf32>) -> !stencil.temp<[0,1024]x[0,1024]xf32>
}) : () -> ()
