s/(.+): Annotated\[OpResult, (.+)\]/\1: OpResult = result_def(\2)/g
s/(.+): Annotated\[VarOpResult, (.+)\]/\1: VarOpResult = var_result_def(\2)/g
s/(.+): Annotated\[OptOpResult, (.+)\]/\1: OptOpResult = opt_result_def(\2)/g

s/(.+): OpResult$/\1: OpResult = result_def()/g
s/(.+): VarOpResult$/\1: VarOpResult = var_result_def()/g
s/(.+): OptOpResult$/\1: OptOpResult = opt_result_def()/g

s/(.+): Annotated\[Operand, (.+)\]/\1: Operand = operand_def(\2)/g
s/(.+): Annotated\[VarOperand, (.+)\]/\1: VarOperand = var_operand_def(\2)/g
s/(.+): Annotated\[OptOperand, (.+)\]/\1: OptOperand = opt_operand_def(\2)/g

s/(.+): Operand$/\1: Operand = operand_def()/g
s/(.+): VarOperand$/\1: VarOperand = var_operand_def()/g
s/(.+): OptOperand$/\1: OptOperand = opt_operand_def()/g
