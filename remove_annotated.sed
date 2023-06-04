s/([a-zA-Z0-9]+): Annotated\[OpResult, ([a-zA-Z0-9]+)\]/\1: OpResult = result_def(\2)/g
s/([a-zA-Z0-9]+): Annotated\[VarOpResult, ([a-zA-Z0-9]+)\]/\1: VarOpResult = var_result_def(\2)/g
s/([a-zA-Z0-9]+): Annotated\[OptOpResult, ([a-zA-Z0-9]+)\]/\1: OptOpResult = opt_result_def(\2)/g
s/([a-zA-Z0-9]+): Annotated\[Operand, ([a-zA-Z0-9]+)\]/\1: Operand = operand_def(\2)/g
s/([a-zA-Z0-9]+): Annotated\[VarOperand, ([a-zA-Z0-9]+)\]/\1: VarOperand = var_operand_def(\2)/g
s/([a-zA-Z0-9]+): Annotated\[OptOperand, ([a-zA-Z0-9]+)\]/\1: OptOperand = opt_operand_def(\2)/g
