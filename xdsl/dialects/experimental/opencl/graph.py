from xdsl.dialects import builtin, func


class Node:
    def __init__(self, name):
        self.name = name
        self.pred = []
        self.succ = []


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
            node = Node(node_func.sym_name.data)
            nodes_lst.append(node)

        graph = Graph(nodes_lst)

        return graph
