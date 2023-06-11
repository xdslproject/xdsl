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

s/(.+): OpAttr\[(.+)\]$/\1: \2 = attr_def(\2)/g
s/(.+): OptOpAttr\[(.+)\]$/\1: \2 | None = opt_attr_def(\2)/g

s/(.+): Region$/\1: Region = region_def()/g
s/(.+): VarRegion$/\1: VarRegion = var_region_def()/g
s/(.+): OptRegion$/\1: OptRegion = opt_region_def()/g

s/(.+): SingleBlockRegion$/\1: Region = region_def("single_block")/g
s/(.+): VarSingleBlockRegion$/\1: VarRegion = var_region_def("single_block")/g
s/(.+): OptSingleBlockRegion$/\1: OptRegion = opt_region_def("single_block")/g

s/(.+): Successor$/\1: Successor = successor_def()/g
s/(.+): VarSuccessor$/\1: VarSuccessor = var_successor_def()/g
s/(.+): OptSuccessor$/\1: OptSuccessor = opt_successor_def()/g
