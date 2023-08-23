// RUN:XDSL_AUTO_ROUNDTRIP

builtin.module {

  irdl.dialect @testd {
    irdl.type @parametric {
      %0 = irdl.any
      irdl.parameters(%0)
    }
    
    irdl.attribute @parametric_attr {
      %0 = irdl.any
      irdl.parameters(%0)
    }
    
    irdl.type @attr_in_type_out {
      %0 = irdl.any
      irdl.parameters(%0)
    }
    
    irdl.operation @eq {
      %0 = irdl.is i32
      irdl.results(%0)
    }
    
    irdl.operation @anyof {
      %0 = irdl.is i32
      %1 = irdl.is i64
      %2 = irdl.any_of(%0, %1)
      irdl.results(%2)
    }
    
    irdl.operation @all_of {
      %0 = irdl.is i32
      %1 = irdl.is i64
      %2 = irdl.any_of(%0, %1)
      %3 = irdl.all_of(%2, %1)
      irdl.results(%3)
    }
    
    irdl.operation @any {
      %0 = irdl.any
      irdl.results(%0)
    }
    
    irdl.operation @dynbase {
      %0 = irdl.any
      %1 = irdl.parametric @parametric<%0>
      irdl.results(%1)
    }
    
    irdl.operation @dynparams {
      %0 = irdl.is i32
      %1 = irdl.is i64
      %2 = irdl.any_of(%0, %1)
      %3 = irdl.parametric @parametric<%2>
      irdl.results(%3)
    }
    
    irdl.operation @constraint_vars {
      %0 = irdl.is i32
      %1 = irdl.is i64
      %2 = irdl.any_of(%0, %1)
      irdl.results(%2, %2)
    }
  }
}
