from typing import List
from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext
from xdsl.frontend.dialects.builtin import ui8, ui32, ui64


def witness():
    pass


p = FrontendProgram()
with CodeContext(p):

    # yapf: disable
    ks = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
        0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
        0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ] # how to type hint a large constant array

    hs = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ]
    # yapf: enable

    M32: ui32 = 0xFFFFFFFF
    H = hs[:]
    k = ks[:]

    def ror(x: ui32, y: ui32):
        return ((x >> y) | (x << (32 - y))) & M32

    def maj(x: ui32, y: ui32, z: ui32):
        return (x & y) ^ (x & z) ^ (y & z)

    def ch(x: ui32, y: ui32, z: ui32):
        return (x & y) ^ ((~x) & z)

    def compress(c: list[ui8]):  # how to hint the length of array
        w: List[ui64] = [0] * 64
        w[0:16] = [
            ((c[i] << 24) + (c[i + 1] << 16) + (c[i + 2] << 8) + c[i + 3])
            for i in range(0, len(c), 4)
        ]

        for i in range(16, 64):
            s0 = ror(w[i - 15], 7) ^ ror(w[i - 15], 18) ^ (w[i - 15] >> 3)
            s1 = ror(w[i - 2], 17) ^ ror(w[i - 2], 19) ^ (w[i - 2] >> 10)
            w[i] = (w[i - 16] + s0 + w[i - 7] + s1) & M32

        a, b, c, d, e, f, g, h = H

        for i in range(64):
            s0 = ror(a, 2) ^ ror(a, 13) ^ ror(a, 22)
            t2 = s0 + maj(a, b, c)
            s1 = ror(e, 6) ^ ror(e, 11) ^ ror(e, 25)
            t1 = h + s1 + ch(e, f, g) + k[i] + w[i]

            h = g
            g = f
            f = e
            e = (d + t1) & M32
            d = c
            c = b
            b = a
            a = (t1 + t2) & M32

        for i, (x, y) in enumerate(zip(H, [a, b, c, d, e, f, g, h])):
            H[i] = (x + y) & M32

    def update(m: list[ui8]):
        if m is None or len(m) == 0:
            return

        #self.mlen += len(m)
        #m = self.buf + m

        for i in range(0, len(m) // 64):
            compress(m[64 * i:64 * (i + 1)])

        #self.buf = m[len(m) - (len(m) % 64):]
        return h

    def Digest(data: list[ui8]):
        return update(data)

    def main():
        data = []  # input type?
        a = Digest(data)
        print(a)


p.compile(
)  #TODO: (short-term) add option to add field, e.g. p.compile(field=BN254)
#p.compile(field=BN254, int=ui64)