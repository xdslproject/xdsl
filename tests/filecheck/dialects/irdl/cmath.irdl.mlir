// RUN: XDSL_AUTO_ROUNDTRIP

builtin.module {
  
  irdl.dialect @cmath {

    irdl.type @complex {
      %0 = irdl.is f32
      %1 = irdl.is f64
      %2 = irdl.any_of(%0, %1)
      irdl.parameters(%2)
    }

    irdl.operation @norm {
      %0 = irdl.any
      %1 = irdl.parametric @complex<%0>
      irdl.operands(%1)
      irdl.results(%0)
    }

    irdl.operation @mul {
      %0 = irdl.is f32
      %1 = irdl.is f64
      %2 = irdl.any_of(%0, %1)
      %3 = irdl.parametric @complex<%2>
      irdl.operands(%3, %3)
      irdl.results(%3)
    }

  }
}
