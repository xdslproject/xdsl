from enum import Enum, auto

from xdsl.dialects import builtin, func


class NodeType(Enum):
    BUF = auto()
    SCALAR = auto()


class Node:
    def __init__(self, node_func: func.FuncOp):
        self.name = node_func.sym_name.data
        self.pred = []
        self.succ = []
        self.n_args = len(node_func.function_type.inputs)

        self.arg_types = [
            NodeType.BUF if isinstance(arg, builtin.MemRefType) else NodeType.SCALAR
            for arg in node_func.function_type.inputs
        ]

    def get_buf_arg_indices(self):
        return [
            idx
            for idx, arg_type in enumerate(self.arg_types)
            if arg_type == NodeType.BUF
        ]

    def get_scalar_arg_indices(self):
        return [
            idx
            for idx, arg_type in enumerate(self.arg_types)
            if arg_type == NodeType.SCALAR
        ]


class Graph:
    def __init__(self, subnodes):
        self.subnodes = subnodes

        self.subnodes_per_node = dict()
        for subnode in subnodes:
            if "SUBNODE" in subnode.name:
                node_name = subnode.name.split("_")[0]
                if node_name not in self.subnodes_per_node:
                    self.subnodes_per_node[node_name] = []
                self.subnodes_per_node[node_name].append(subnode)
            else:
                self.subnodes_per_node[subnode.name] = [subnode]

        # Vertical splitting induces a linear pipeline across the subnodes. This needs an artificial dependency between successive stages.
        for node_name, subnodes in self.subnodes_per_node.items():
            if len(subnodes) > 1:
                for idx in range(len(subnodes) - 1):
                    subnodes[idx].succ.append(subnodes[idx + 1])
                    subnodes[idx + 1].pred.append(subnodes[idx])

    @staticmethod
    def generate_graph(program: builtin.ModuleOp):
        node_funcs = [
            program_op
            for program_op in program.walk()
            if isinstance(program_op, func.FuncOp)
            and "sub_node" in program_op.attributes
        ]

        nodes_lst = []
        for node_func in node_funcs:
            node = Node(node_func)
            nodes_lst.append(node)

        graph = Graph(nodes_lst)

        return graph

    def get_subnode_names(self):
        return [subnode.name for subnode in self.subnodes]

    def get_subnode_by_name(self, name):
        return [subnode for subnode in self.subnodes if subnode.name == name][0]
