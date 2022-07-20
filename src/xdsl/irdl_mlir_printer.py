from xdsl.ir import MLContext
from xdsl.printer import Printer
from xdsl.dialects.irdl import *
from dataclasses import dataclass

# remb to look at and remove all the comments


@dataclass
class IRDLPrinter:

    ctx: MLContext

    #stream in all prints to output to a file

    def print_module(self, module):
        print('module{')
        module.walk(lambda di: IRDLPrinter.print_dialect_definition(self, di)
                    if isinstance(di, DialectOp) else None)
        print('}')
        # adjust the indentation accordingly later since module is added

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
        print(f"    {ParametersOp.name}", end="(")
        for param in param_op.constraints.data:
            print(f"{param.type_name.data}: ", end="")
            self.print_attr_constraint(param.params_constraints)
        print(")")

    def print_operation_definition(self, operation: OperationOp):
        print(f"  {OperationOp.name} {operation.attributes['name'].data} {{")
        '''
        Checking for existence of operands
        '''
        operand_list = []
        operation.walk(lambda operand_def: operand_list.append(operand_def)
                       if isinstance(operand_def, OperandsOp) else None)
        if operand_list:
            self.print_operand_definition(operand_list)
        '''
        Checking for existence of results
        '''
        result_list = []
        operation.walk(lambda res: result_list.append(res)
                       if isinstance(res, ResultsOp) else None)
        if result_list:
            self.print_result_definition(result_list)

        print(f"  }}")

    def print_result_definition(self, res_list=list[ResultsOp]):

        # OLD VERSION
        # print(f"    {ResultsOp.name}", end="(")
        # for result in result_op.attributes['constraints'].data:
        #     self.print_attr_constraint(result)
        # print(")")

        print(f"    {ResultsOp.name}", end="(")
        for i in range(len(res_list)):
            for result in res_list[i].attributes['constraints'].data:
                self.print_attr_constraint(result)
                print(", ", end='') if i != len(res_list) - 1 else ''
        print(")")

    def print_operand_definition(self, op_list=list[OperandsOp]):
        print(f"    {OperandsOp.name}", end="(")
        for i in range(len(op_list)):
            for ops in op_list[i].attributes['constraints'].data:
                self.print_attr_constraint(ops)
                print(", ", end='') if i != len(op_list) - 1 else ''
        print(")")

        # MORE READABLE VERSION but circumvent i
        # for operation in op_list:
        #     for constraint in operation.attributes['constraints'].data:
        #         self.print_attr_constraint(constraint)
        #         print(", ", end='') if i != len(op_list) - 1 else ''
        # print(")")

        # OLD VERSION
        # for i in range(len(op_list)):
        #     if i == 0:
        #         print(f"    {op_list[i].name}", end="(")
        #     for ops in op_list[i].attributes['constraints'].data:
        #         self.print_attr_constraint(ops)
        #         print(", ", end='') if i != len(op_list) - 1 else ''
        # print(")")

    def print_dialect_definition(self, di: DialectOp):
        print(f"{di.name} {di.attributes['dialect_name'].data} {{")

        di.walk(lambda type: self.print_type_definition(type)
                if isinstance(type, TypeOp) else None)

        di.walk(lambda op: self.print_operation_definition(op)
                if isinstance(op, OperationOp) else None)
        print(f"}}")
