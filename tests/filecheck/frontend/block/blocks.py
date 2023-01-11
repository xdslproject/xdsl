# RUN: python %s | filecheck %s

from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext
from xdsl.frontend.dialects.frontend import block

p = FrontendProgram()
with CodeContext(p):

    def foo1(y: int):
        bb1(y)

        @block
        def bb1(x: int):
            return

    def foo2(x: int):
        return

    
    def foo(x: int):
        y: int = 2
        bb1(y)

        @block
        def bb1(x: int):
            return

    def bar(a: int):
        bb0(a, a)

        @block
        def bb0(x: int, y: int):
            return

    # def llvm_loop():
    #     @block
    #     def entry():
    #         header()
    #         # llvm.br("header"))

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

p.compile(desymref=False)
print(p.xdsl())

# MLIR_OPT_PATH = "../llvm-project/build/bin/mlir-opt"
# mlir_output = p.mlir_roundtrip(MLIR_OPT_PATH, mlir_opt_args=["--verify-each"])
# print(mlir_output)
