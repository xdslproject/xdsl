// RUN: XDSL_AUTO_ROUNDTRIP

// Types that have cyclic references.
builtin.module {
  
  irdl.dialect @testd {
    
    irdl.type @self_referencing {
      %0 = irdl.any
      %1 = irdl.parametric @self_referencing<%0>
      %2 = irdl.is i32
      %3 = irdl.any_of(%1, %2)
      irdl.parameters(%3)
    }

    irdl.type @type1 {
      %0 = irdl.any
      %1 = irdl.parametric @type2<%0>
      %2 = irdl.is i32
      %3 = irdl.any_of(%1, %2)
      irdl.parameters(%3)
    }

    irdl.type @type2 {
        %0 = irdl.any
        %1 = irdl.parametric @type1<%0>
        %2 = irdl.is i32
        %3 = irdl.any_of(%1, %2)
        irdl.parameters(%3)
    }
  }
}
