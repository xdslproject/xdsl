from xdsl.ir import Block

import xdsl.dialects.pdl as pdl
from xdsl.dialects.builtin import StringAttr

type_type = pdl.TypeType()
attribute_type = pdl.AttributeType()


def test_build_ops():

    block = Block.from_arg_types([type_type, attribute_type])
    arg0, arg1, = block.args

    anc = pdl.ApplyNativeConstraintOp.get('anc', [arg0])

    assert anc.attributes['name'] == StringAttr('anc')
    assert anc.args == (arg0, )

    anr = pdl.ApplyNativeRewriteOp.get('anr', [arg0], [attribute_type])

    assert anr.attributes['name'] == StringAttr('anr')
    assert anr.args == (arg0, )
    assert len(anr.results) == 1
    assert [r.typ for r in anr.results] == [attribute_type]

    r = pdl.RewriteOp.get('r',
                          root=None,
                          external_args=[arg0, arg1],
                          body=None)

    assert r.attributes['name'] == StringAttr('r')
    assert r.externalArgs == (arg0, arg1)
    assert len(r.results) == 0
