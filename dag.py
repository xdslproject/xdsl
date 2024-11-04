import marimo

__generated_with = "0.9.14"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(mo):
    input = mo.ui.code_editor(value="""\
    func.func @hello() -> index {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c8 = arith.constant 8 : index
        %c64 = arith.constant 64 : index

        scf.for %16 = %c0 to %c64 step %c8 {
            scf.for %17 = %c0 to %c8 step %c1 {
            %18 = arith.constant 8 : index
            %19 = arith.addi %16, %17 : index
            %20 = builtin.unrealized_conversion_cast %19 : index to !riscv.reg
            "test.op"(%20) : (!riscv.reg) -> ()
            }
        }
    }
    """)
    # func.func @hello(%a : i32, %b : i32) -> i32 {
    #     %a_sq = arith.muli %a, %a : i32
    #     %b_sq = arith.muli %b, %b : i32
    #     %res = arith.subi %a_sq, %b_sq : i32
    #     func.return %res : i32
    # }
    # func.func @hello(%n : i32) -> i32 {
    #     %two = arith.constant 0 : i32
    #     %res = arith.addi %two, %n : i32

    #     func.return %res : i32
    # }
    input
    return (input,)


@app.cell
def __():
    from xdsl.parser import Parser
    from xdsl.interactive.passes import get_new_registered_context
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.interactive.passes import iter_condensed_passes
    from xdsl.utils.hashable_module import HashableModule

    from xdsl.dialects.builtin import UnrealizedConversionCastOp
    from xdsl.dialects.func import FuncOp
    from xdsl.dialects.scf import For
    return (
        For,
        FuncOp,
        HashableModule,
        ModuleOp,
        Parser,
        UnrealizedConversionCastOp,
        get_new_registered_context,
        iter_condensed_passes,
    )


@app.cell
def __(Parser, get_new_registered_context, input):
    ctx = get_new_registered_context()
    input_module = Parser(ctx, input.value).parse_module()

    print(input_module)
    return ctx, input_module


@app.cell
def __():
    from dataclasses import dataclass, field
    from typing import Hashable
    return Hashable, dataclass, field


@app.cell
def __(input):
    input
    return


@app.cell
def __(
    For,
    FuncOp,
    HashableModule,
    ModuleOp,
    UnrealizedConversionCastOp,
    input_module,
    iter_condensed_passes,
):
    import networkx as nx
    import random

    from typing import Callable

    G = nx.MultiDiGraph()

    root = HashableModule(input_module)
    queue = [root]
    visited = set()

    filters: dict[str, Callable[[ModuleOp], bool]] = {}

    def has_no_casts(module: ModuleOp):
        return not any(isinstance(op, UnrealizedConversionCastOp) for op in module.walk())

    filters["riscv-allocate-registers"] = has_no_casts
    filters["canonicalize"] = has_no_casts
    filters["convert-riscv-scf-to-riscv-cf"] = has_no_casts

    def can_lower_arith(module: ModuleOp):
        """
        Only lower arith after lowering func and scf, since they commute.
        """
        return not any(isinstance(op, FuncOp) or isinstance(op, For) for op in module.walk())

    filters["convert-arith-to-riscv"] = can_lower_arith

    def can_lower_scf(module: ModuleOp):
        """
        Only lower arith after lowering func, since they commute.
        """
        return not any(isinstance(op, FuncOp) for op in module.walk())

    filters["convert-scf-to-riscv-scf"] = can_lower_scf

    while queue and len(visited) < 5000:
        source = queue.pop()
        if source in visited:
            continue
        visited.add(source)
        for available_pass, t in iter_condensed_passes(source.module):
            if available_pass.display_name in filters:
                if not filters[available_pass.display_name](source.module):
                    continue
            target = HashableModule(t)
            # I'm not sure we want to exclude duplicate edges here
            # if G.has_edge(source, target):
            #     continue
            G.add_edge(source, target, name=available_pass.display_name, weight=random.uniform(0.1, 1.0))
            if target not in visited:
                queue.append(target)
    return (
        Callable,
        G,
        available_pass,
        can_lower_arith,
        can_lower_scf,
        filters,
        has_no_casts,
        nx,
        queue,
        random,
        root,
        source,
        t,
        target,
        visited,
    )


@app.cell
def __(G, mo, nx, root):
    import plotly.graph_objects as go

    import plotly.colors as pc

    color_scale = pc.sequential.Blues

    print(len(color_scale))

    assert nx.is_directed_acyclic_graph(G), "topo sort breaks otherwise"

    all_nodes = tuple(G.nodes())
    node_index_by_node = {n:i for i, n in enumerate(all_nodes)}

    layers = dict(enumerate(nx.bfs_layers(G, root)))
    x_scale = len(layers) + 2.0
    y_scales = tuple(len(layers[i]) + 2.0 for i in range(len(layers)))

    node_x = [-1] * len(all_nodes)
    node_y = [-1] * len(all_nodes)

    for layer_index, nodes in layers.items():
        x = (layer_index + 1.0) / x_scale
        y_scale = y_scales[layer_index]
        for node_index, node in enumerate(nodes):
            y = (node_index + 1.0) / y_scale

            node_x[node_index_by_node[node]] = x
            node_y[node_index_by_node[node]] = y
            # print(node_index_by_node[node], x, y, node)

    for i, node in enumerate(nx.topological_sort(G)):
        node_index = node_index_by_node[node]
        node_x[node_index] = (i + 1.0) / (len(all_nodes) + 2.0)

    # print(node_x)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Number of ops',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    edge_traces = []

    for edge_source, edge_target, edge_data in G.edges(data=True):
        n0 = node_index_by_node[edge_source]
        n1 = node_index_by_node[edge_target]
        x0, y0 = node_x[n0], node_y[n0]
        x1, y1 = node_x[n1], node_y[n1]
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(
                width=edge_data['weight'],  # Set the width to the edge weight
                color=color_scale[int(edge_data['weight'] * 9)]
            ),
            hoverinfo='none',
            mode='lines'
        )
        edge_traces.append(edge_trace)

    # edge_x = []
    # edge_y = []
    # edge_weights = [] 
    # for edge in G.edges(data=True):
    #     n0 = node_index_by_node[edge[0]]
    #     n1 = node_index_by_node[edge[1]]
    #     x0, y0 = node_x[n0], node_y[n0]
    #     x1, y1 = node_x[n1], node_y[n1]
    #     edge_x.append(x0)
    #     edge_x.append(x1)
    #     edge_x.append(None)
    #     edge_y.append(y0)
    #     edge_y.append(y1)
    #     edge_y.append(None)
    #     edge_weights.append(edge[2]['weight'])

    # edge_trace = go.Scatter(
    #     x=edge_x, y=edge_y,
    #     line=dict(width=edge_weights, color='#888'),
    #     hoverinfo='none',
    #     mode='lines')


    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        module = all_nodes[node].module
        num_ops = 0
        for op in module.walk():
            num_ops += 1
        node_adjacencies.append(num_ops)
        node_text.append(str(module).replace("\n", "<br>"))

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[*edge_traces, node_trace],
                 layout=go.Layout(
                    title=str(G), # 'Network graph made with Python',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    mo.ui.plotly(fig)
    return (
        adjacencies,
        all_nodes,
        color_scale,
        edge_data,
        edge_source,
        edge_target,
        edge_trace,
        edge_traces,
        fig,
        go,
        i,
        layer_index,
        layers,
        module,
        n0,
        n1,
        node,
        node_adjacencies,
        node_index,
        node_index_by_node,
        node_text,
        node_trace,
        node_x,
        node_y,
        nodes,
        num_ops,
        op,
        pc,
        x,
        x0,
        x1,
        x_scale,
        y,
        y0,
        y1,
        y_scale,
        y_scales,
    )


@app.cell
def __(G):
    node_index_by_module = {n: i for i, n in enumerate(G.nodes())}
    return (node_index_by_module,)


@app.cell(disabled=True)
def __(G, nx, root):
    for n in G.nodes:
        print(n.module)
        paths = nx.all_simple_edge_paths(G, root, n)

        for path in paths:
           print("Path :: " + ','.join(e[2] for e in path))
    str(G)
    return n, path, paths


@app.cell
def __(G, HashableModule, nx):
    # Calculate the number of nodes in the tree equivalent to a dag

    def equivalent_tree_node_count(dag: nx.Graph) -> int:
        if not nx.is_directed_acyclic_graph(dag):
            raise ValueError("Expected DAG")

        count_by_node: dict[HashableModule, int] = {}

        for i, generation in enumerate(nx.topological_generations(dag)):
            if not i:
                # First generation, assign 1 to root node
                assert len(generation) == 1, "Only one root node allowed"
                count_by_node[generation[0]] = 1
                continue

            # For each generation, we know the number of equivalent nodes of the parents
            # The number of nodes is the sum of counts of parents
            for node in generation:
                count_by_node[node] = sum(count_by_node[parent] for parent in dag.predecessors(node))

        return sum(count_by_node.values())

    equivalent_tree_node_count(G)
    return (equivalent_tree_node_count,)


if __name__ == "__main__":
    app.run()
