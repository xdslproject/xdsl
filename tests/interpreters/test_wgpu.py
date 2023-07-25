from contextlib import redirect_stdout
from io import StringIO

from xdsl.dialects import arith, builtin, func, gpu, memref, printf
from xdsl.interpreter import Interpreter
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.experimental.wgpu import WGPUFunctions
from xdsl.interpreters.memref import MemrefFunctions
from xdsl.interpreters.printf import PrintfFunctions
from xdsl.ir import MLContext
from xdsl.parser import Parser


def test_init():
    mlir_source = """
builtin.module attributes {gpu.container_module} {
  "gpu.module"() ({
    "gpu.func"() ({
    ^0(%arg : memref<4x4xindex>):
      %0 = "arith.constant"() {"value" = 2 : index} : () -> index
      %1 = "gpu.global_id"() {"dimension" = #gpu<dim x>} : () -> index
      %2 = "gpu.global_id"() {"dimension" = #gpu<dim y>} : () -> index
      %3 = "arith.constant"() {"value" = 4 : index} : () -> index
      %4 = "arith.muli"(%1, %3) : (index, index) -> index
      %5 = "arith.addi"(%4, %2) : (index, index) -> index

      "memref.store"(%5, %arg, %1, %2) {"nontemporal" = false} : (index, memref<4x4xindex>, index, index) -> ()
      "gpu.return"() : () -> ()
    }) {"function_type" = (memref<4x4xindex>) -> (),
        "gpu.kernel",
        "sym_name" = "fill"
       } : () -> ()
    "gpu.func"() ({
    ^0(%arg : memref<4x4xindex>):
      %0 = "arith.constant"() {"value" = 1 : index} : () -> index
      %1 = "gpu.global_id"() {"dimension" = #gpu<dim x>} : () -> index
      %2 = "gpu.global_id"() {"dimension" = #gpu<dim y>} : () -> index
      %3 = "memref.load"(%arg, %1, %2) {"nontemporal" = false} : (memref<4x4xindex>, index, index) -> (index)
      %4 = "arith.addi"(%3, %0) : (index, index) -> index
      "memref.store"(%4, %arg, %1, %2) {"nontemporal" = false} : (index, memref<4x4xindex>, index, index) -> ()
      "gpu.return"() : () -> ()
    }) {"function_type" = (memref<4x4xindex>) -> (),
        "gpu.kernel",
        "sym_name" = "inc"
       } : () -> ()
    "gpu.module_end"() : () -> ()
  }) {"sym_name" = "gpu"} : () -> ()

  func.func @main() -> index {
    %four = "arith.constant"() {"value" = 4 : index} : () -> index
    %one = "arith.constant"() {"value" = 1 : index} : () -> index
    %memref = "memref.alloc"() {"alignment" = 0 : i64, "operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<4x4xindex>
    "gpu.launch_func"(%four, %four, %one, %one, %one, %one, %memref) {"operand_segment_sizes" = array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 1>, "kernel" = @gpu::@fill} : (index, index, index, index, index, index, memref<4x4xindex>) -> ()
    "gpu.launch_func"(%four, %four, %one, %one, %one, %one, %memref) {"operand_segment_sizes" = array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 1>, "kernel" = @gpu::@inc} : (index, index, index, index, index, index, memref<4x4xindex>) -> ()
    printf.print_format "Result : {}", %memref : memref<4x4xindex>
  }
}
"""
    context = MLContext()
    context.register_dialect(arith.Arith)
    context.register_dialect(memref.MemRef)
    context.register_dialect(builtin.Builtin)
    context.register_dialect(gpu.GPU)
    context.register_dialect(func.Func)
    context.register_dialect(printf.Printf)
    parser = Parser(context, mlir_source)
    module = parser.parse_module()

    interpreter = Interpreter(module)
    interpreter.register_implementations(ArithFunctions)
    interpreter.register_implementations(MemrefFunctions)
    interpreter.register_implementations(WGPUFunctions)
    interpreter.register_implementations(PrintfFunctions)
    f = StringIO("")
    with redirect_stdout(f):
        interpreter.call_op("main", ())
    assert (
        "Result : [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]"
        in f.getvalue()
    )
