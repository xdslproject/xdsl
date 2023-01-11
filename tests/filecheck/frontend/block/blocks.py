# RUN: python %s | filecheck %s

from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext
from xdsl.frontend.dialects.frontend import block

p = FrontendProgram()
with CodeContext(p):
    
    def foo(arg: int):
        @block
        def bb0(x: int):
            return

    # def llvm_loop():
    #     @block
    #     def entry():
    #         header()
    #         # llvm.br(header())

    #     @block
    #     def header():
    #         cond = True
    #         body() if cond else exit()
    #         # llvm.br(cond, body(), exit())

    #     @block
    #     def body():
    #         header()
    #         # llvm.br(header())

    #     @block
    #     def exit():
    #         return

p.compile()
print(p.xdsl())

