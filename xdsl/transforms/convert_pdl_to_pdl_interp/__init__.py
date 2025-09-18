"""
The pdl-to-pdl-interp conversion works by translating declarative [pdl patterns][xdsl.dialects.pdl] into a set of
[predicates][xdsl.transforms.convert_pdl_to_pdl_interp.predicate] that are then converted to imperative code in
the [pdl_interp dialect][xdsl.dialects.pdl_interp].
"""
