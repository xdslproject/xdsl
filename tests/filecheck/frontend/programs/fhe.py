from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext
from xdsl.frontend.template import template
from xdsl.frontend.dialects.fhe import Secret

from typing import List

p = FrontendProgram()
with CodeContext(p):

    @template('L')
    def hammingdist(x: Secret[List[int]], y: Secret[List[int]],
                    L: int) -> Secret[int]:
        sum: int = 0
        for i in range(L):
            sum += (x[i] - y[i]) ^ 2
        return sum


p.compile({'L': [4]})
print(p.xdsl())

from Pyfhel import Pyfhel

HE = Pyfhel()
HE.contextGen(scheme='bgv', n=2**14)
HE.keyGen()
ctxt1 = HE.encryptInt([1, 2, 3, 4])
ctxt2 = HE.encryptInt([4, 3, 2, 1])

ctxt3 = p.run('hammingdist', {'x': ctxt1, 'y': ctxt2})
