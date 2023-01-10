pdl.pattern @stencilInlining : benefit(1) {
    %producer_apply : !operation = pdl.operation "stencil.apply"()
    %producer_result = result 0 of %producer_apply  // The result index should actually be unspecified

    %consumer_apply : !operation = pdl.operation "stencil.apply"(%producer_result)
    // All inter region matching has to happen in C++.
    // Problem, apply_native_constraint can not return handles, so we have to rematch the accessOp in the
    // consumer region during the rewrite
    pdl.apply_native_constraint "check_access_in_stencil_apply_region" (%producer_apply, %consumer_apply)

    rewrite %consumer_apply {
        apply_native_rewrite "StencilInlining"(%root : !pdl.operation)
    }
}