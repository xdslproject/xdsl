s/([a-zA-Z0-9]+): Annotated\[OpResult, (.+)\]/\1: OpResult = result_def(\2)/g
s/([a-zA-Z0-9]+): Annotated\[VarOpResult, (.+)\]/\1: VarOpResult = var_result_def(\2)/g
s/([a-zA-Z0-9]+): Annotated\[OptOpResult, (.+)\]/\1: OptOpResult = opt_result_def(\2)/g

s/([a-zA-Z0-9]+): OpResult$/\1: OpResult = result_def()/g
s/([a-zA-Z0-9]+): VarOpResult$/\1: VarOpResult = var_result_def()/g
s/([a-zA-Z0-9]+): OptOpResult$/\1: OptOpResult = opt_result_def()/g

s/([a-zA-Z0-9]+): Annotated\[Operand, (.+)\]/\1: Operand = operand_def(\2)/g
s/([a-zA-Z0-9]+): Annotated\[VarOperand, (.+)\]/\1: VarOperand = var_operand_def(\2)/g
s/([a-zA-Z0-9]+): Annotated\[OptOperand, (.+)\]/\1: OptOperand = opt_operand_def(\2)/g
