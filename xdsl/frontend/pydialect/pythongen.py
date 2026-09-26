from dataclasses import dataclass, field
from typing import Sequence

from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.py.ops import (
	FuncOp,
	CallOp,
	CastOp,
	ConstantOp,
	PassOp,
	ReturnOp,
	AssertOp,
)
from xdsl.ir.core import Block, BlockArgument, Operation, SSAValue


def get_func_name(s:str):
	return s.removeprefix('"').removesuffix('"')


def trim_double_quotes(s:str) -> str:
	return s.removeprefix('"').removesuffix('"')

@dataclass(eq=False, repr=False)
class PythonGenerator:
	depth = 0 # indentation level
	_VAR_PREFIX = "xdsl_var_"
	_ssa_values: dict[SSAValue, str] = field(
		default_factory=dict[SSAValue, str], init=False
	)

	_ssa_names: list[dict[str, int]] = field(
		default_factory=lambda: [dict[str, int]()], init=False
	)

	_next_valid_name_id: list[int] = field(default_factory=lambda: [0], init=False)


	@property
	def ssa_names(self):
		return self._ssa_names[-1]

	def _get_new_valid_name_id(self) -> str:
		self._next_valid_name_id[-1] += 1
		return str(self._next_valid_name_id[-1] - 1)


	def get_ssa_name(self, value: SSAValue) -> str:
		"""
		Print an SSA value in the printer. This assigns a name to the value if the value
		does not have one in the current printing context.
		If the value has a name hint, it will use it as a prefix, and otherwise assign
		a number as the name. Numbers are assigned in order.

		Returns the name used for printing the value.
		"""

		if value in self._ssa_values:
			name = self._ssa_values[value]
			
		elif value.name_hint:
			curr_ind = self.ssa_names.get(value.name_hint, 0)
			suffix = f"_{curr_ind}" if curr_ind != 0 else ""
			name = f"{value.name_hint}{suffix}"
			self._ssa_values[value] = name
			self.ssa_names[value.name_hint] = curr_ind + 1
		else:
			name = self._get_new_valid_name_id()
			self._ssa_values[value] = name

		return f"{self._VAR_PREFIX}{name}"

	def gen_identation(self):
		return "\t" * self.depth

	def arg_decl_list_to_python(self, b:Block | None) -> str:
		if b is None:
			return ""
		
		res = ""
		arg_list:tuple[BlockArgument, ...] = b.args

		for i, arg in enumerate(arg_list):
			if i:
				res +=", "
			name = self.get_ssa_name(arg)
			type_hint = arg.type.get_type()

			res += f"{name}:{type_hint}"
		return res

	# def arg_list_to_python(self, b:Sequence[SSAValue] | None) -> str:
		# if b is None:
		# 	return ""
		
		# res = ""
		# arg_list:tuple[BlockArgument, ...] = b.args

		# for i, arg in enumerate(arg_list):
		# 	if i:
		# 		res +=", "
		# 	name = self.get_ssa_name(arg)
		# 	type_hint = arg.type.get_type()

		# 	res += f"{name}:{type_hint}"
		# return res

	def gen_python_module(self, module:ModuleOp) -> str:
		first_block = module.body.first_block
		if first_block is None:
			return ""
		
		first_op = first_block.first_op
		return "" if first_op is None else self.gen_python_generic(first_op)

	def gen_python_generic(self, py:Operation) -> str:
		match py:
			case FuncOp():
				return self.gen_python_function(py)

			case ConstantOp():
				return self.gen_python_constant(py)

			case CallOp():
				return self.gen_python_call(py)

			case ReturnOp():
				return self.gen_python_return(py)


			case _:
				raise NotImplementedError(f"{type(py)} not supported")


	def gen_python_function(self, o:FuncOp) -> str:
		fundef = f"{self.gen_identation()}def {trim_double_quotes(o.sym_name.__str__())}({self.arg_decl_list_to_python(o.body.first_block)}) -> {o.result_types}:\n"

		self.depth += 1
		for body in o.body.walk():
			fundef += f"{self.gen_python_generic(body)}\n"
		self.depth -= 1

		return fundef

	def gen_python_constant(self, o:ConstantOp) -> str:
		return f"{self.gen_identation()}{self.get_ssa_name(o.result)}:{o.result_types[0].data} = {o.get_value()}"

	def gen_python_arg_list(self, args:Sequence[SSAValue | Operation]) -> str:
		res:str = ""

		for i, arg in enumerate(args):
			assert isinstance(arg, SSAValue)
			if i:
				res += ", "

			res += self.get_ssa_name(arg)

		return res

	def gen_python_call(self, o:CallOp) -> str:
		caller:str = self.get_ssa_name(o.caller) if o.caller else ""
		return f"{self.gen_identation()}{self.get_ssa_name(o.res)}:{o.result_types[0].data} = {caller}{"." if caller != "" else ""}{o.sym_name.data}({self.gen_python_arg_list(o.arguments)})"

	def gen_python_return(self, o:ReturnOp):
		return f"{self.gen_identation()}return {self.get_ssa_name(o.input)}"