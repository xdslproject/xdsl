from xdsl.ir import *
from xdsl.parser import *
from xdsl.printer import *
from xdsl.dialects.irdl import *
from xdsl.dialects.builtin import *
from xdsl.xdsl_opt_main import *
from xdsl.printer import Printer
# from xdsl.mlir_converter import MLIRConverter


class Translation:

    ctx: MLContext

    # stream

    # def __init__(self, file="tests/filecheck/cmath.xdsl"):
    #     self.ctx = MLContext()
    #     self.register_the_dialects()
    #     # file_name = "tests/filecheck/cmath.xdsl"
    #     # Translation.run(self, file)

    # prog
    def print_module(self, module):
        """
        To test AnyTypeConstraint and DynTypeBaseConstraintAttr,
        replace file to be opened with 'tests/filecheck/translation_ops.xdsl'
        """

        # file = open("tests/filecheck/cmath.xdsl")
        # remove
        # def parse_xdsl_file(self, f: IOBase):
        #     file = open(f)
        #     input_str = file.read()
        #     parser = Parser(self.ctx, input_str)
        #     module = parser.parse_op()
        #     if not (isinstance(module, ModuleOp)):
        #         raise Exception(
        #             "Expected module or program as toplevel operation")
        #     return module

        # my_op = parse_xdsl_file(self, file_name)
        my_op.walk(lambda di: Translation.print_di_definition(self, di)
                   if isinstance(di, DialectOp) else None)

    def register_the_dialects(self):
        irdl = IRDL(self.ctx)
        builtin = Builtin(self.ctx)

    def print_type_definition(self, type: TypeOp):
        print(f"  {type.name} {type.attributes['type_name'].data} {{")
        type.walk(lambda param: self.print_parameters_definition(param)
                  if isinstance(param, ParametersOp) else None)
        print(f"  }}")

    def print_attr_constraint(self, f):
        if isinstance(f, AnyOfTypeConstraintAttr):
            print(f"AnyOf", end="<")
            for i in range(len(f.params.data) - 1):
                self.print_attr_constraint(f.params.data[i])
                print(", ", end="")
            self.print_attr_constraint(f.params.data[len(f.params.data) - 1])
            print(">", end="")
        elif isinstance(f, EqTypeConstraintAttr):
            p = Printer()
            # converter = MLIRConverter(self.ctx)
            # print(converter.convert_type(f.type))
            p.print(f.type)
        elif isinstance(f, AnyTypeConstraintAttr):
            print(f"Any", end="")
        elif isinstance(f, DynTypeBaseConstraintAttr):
            print(f.type_name.data, end='')
        elif isinstance(f, DynTypeParamsConstraintAttr):
            print(f.type_name.data, end='')
            print("<", end="")
            for i in range(len(f.params.data)):
                self.print_attr_constraint(f.params.data[i])
            print(">", end="")
        elif isinstance(f, TypeParamsConstraintAttr):
            print(f"{f.type_name}")
        elif isinstance(f, NamedTypeConstraintAttr):
            print(f.type_name.data, end='')
            print(":", end=" ")
            self.print_attr_constraint(f.params_constraints)
        else:
            raise Exception(f"Invalid Constraint: {type(f)} ")

    def print_parameters_definition(self, param_op: ParametersOp):
        print(f"    {param_op.name}", end="(")
        for param in param_op.constraints.data:
            print(f"{param.type_name.data}: ", end="")
            self.print_attr_constraint(param.params_constraints)
        print(")")

    def print_operation_definition(self, operation: OperationOp):
        print(f"  {operation.name} {operation.attributes['name'].data} {{")

        op_list = []
        operation.walk(lambda operand_def: op_list.append(operand_def)
                       if isinstance(operand_def, OperandsOp) else None)

        self.print_operand_definition(op_list)

        operation.walk(lambda res: self.print_result_definition(res)
                       if isinstance(res, ResultsOp) else None)

        print(f"  }}")

    def print_result_definition(self, result_op: ResultsOp):
        print(f"    {result_op.name}", end="(")
        for result in result_op.attributes['constraints'].data:
            self.print_attr_constraint(result)
        print(")")

    def print_operand_definition(self, op_list=List[OperandsOp]):
        for i in range(len(op_list)):
            if i == 0:
                print(f"    {op_list[i].name}", end="(")
            for ops in op_list[i].attributes['constraints'].data:
                self.print_attr_constraint(ops)
                print(", ", end='') if i != len(op_list) - 1 else ''
        print(")")

    def print_di_definition(self, di: DialectOp):
        print(f"{di.name} {di.attributes['dialect_name'].data} {{")

        di.walk(lambda type: self.print_type_definition(type)
                if isinstance(type, TypeOp) else None)

        di.walk(lambda op: self.print_operation_definition(op)
                if isinstance(op, OperationOp) else None)
        print(f"}}")


# if __name__ == "__main__":
#     t = Translation()
