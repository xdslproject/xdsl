from xdsl.ir import Block, Operation, SSAValue

import xdsl.dialects.pdl as pdl
from xdsl.dialects.builtin import StringAttr

type_type = pdl.TypeType()
attribute_type = pdl.AttributeType()


def test_build_ops():

    def func(*args: SSAValue) -> list[Operation]:
        arg0, arg1, = args

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

        return [anc, anr, r]

    _block = Block.from_callable([type_type, attribute_type], func)

    # lb = Constant.from_int_and_width(0, IndexType())
    # ub = Constant.from_int_and_width(42, IndexType())
    # step = Constant.from_int_and_width(3, IndexType())
    # carried = Constant.from_int_and_width(1, IndexType())
    # bodyblock = Block.from_arg_types([IndexType()])
    # body = Region.from_block_list([bodyblock])
    # f = For.get(lb, ub, step, [carried], body)

    # assert f.lb is lb.result
    # assert f.ub is ub.result
    # assert f.step is step.result
    # assert f.iter_args == tuple([carried.result])
    # assert f.body is body

    # assert len(f.results) == 1
    # assert f.results[0].typ == carried.result.typ
    # assert f.operands == (lb.result, ub.result, step.result, carried.result)
    # assert f.regions == [body]
    # assert f.attributes == {}