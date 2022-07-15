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

    def __init__(self):
        self.ctx = MLContext()
        self.register_the_dialects()
        cmath_loc = open("tests/filecheck/cmath.xdsl")
        translation_loc = open("tests/filecheck/translation_ops.xdsl")
        my_op = self.parse_xdsl_file(translation_loc)
        my_op.walk(lambda di: self.print_di_definition(di)
                   if isinstance(di, DialectOp) else None)
        # my_op.walk(lambda type: self.print_type_definition(type) if isinstance(type, TypeOp) else None)
        # my_op.walk(lambda op: self.print_operation_definition(op) if isinstance(op, OperationOp) else None)

    def print_type_definition(self, op: TypeOp):
        print(f"  {op.name} {op.attributes['type_name'].data} {{")
        op.walk(lambda param: self.print_parameters_definition(param)
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

        # operation.walk(lambda operand_def: self.print_operand_definition(
        #     op_list) if isinstance(operand_def, OperandsOp) else None)

        self.print_operand_definition(op_list)

        operation.walk(lambda res: self.print_res_definition(res)
                       if isinstance(res, ResultsOp) else None)

        print(f"  }}")

    def print_res_definition(self, res: ResultsOp):
        print(f"    {res.name}", end="(")
        for result in res.attributes['constraints'].data:
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

    def register_the_dialects(self):
        irdl = IRDL(self.ctx)
        builtin = Builtin(self.ctx)

    def parse_xdsl_file(self, f: IOBase):
        input_str = f.read()
        parser = Parser(self.ctx, input_str)
        module = parser.parse_op()
        if not (isinstance(module, ModuleOp)):
            raise Exception("Expected module or program as toplevel operation")
        return module


if __name__ == "__main__":
    t = Translation()
