import marimo

__generated_with = "0.2.5"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __(mo):
    input = mo.ui.code_editor(value="""\
    func.func @hello(%n : i32) -> i32 {
        %two = arith.constant 0 : i32
        %res = arith.addi %two, %n : i32
        func.return %res : i32
    }
    """)
    input
    return input,


@app.cell
def __(input):
    from xdsl.parser import Parser
    from xdsl.interactive.passes import get_new_registered_context

    ctx = get_new_registered_context()
    input_module = Parser(ctx, input.value).parse_module()

    print(input_module)
    return Parser, ctx, get_new_registered_context, input_module


@app.cell
def __():
    from dataclasses import dataclass, field
    from typing import Hashable
    return Hashable, dataclass, field


@app.cell
def __():
    from xdsl.dialects.builtin import ModuleOp
    return ModuleOp,


@app.cell
def __(input):
    input
    return


@app.cell
def __(G, mo, nx, root):
    import plotly.graph_objects as go

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
            print(node_index_by_node[node], x, y, node)

    print(node_x)

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

    edge_x = []
    edge_y = []
    for edge in G.edges():
        n0 = node_index_by_node[edge[0]]
        n1 = node_index_by_node[edge[1]]
        x0, y0 = node_x[n0], node_y[n0]
        x1, y1 = node_x[n1], node_y[n1]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')


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

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title=str(G), # 'Network graph made with Python',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        # text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    mo.ui.plotly(fig)
    return (
        adjacencies,
        all_nodes,
        edge,
        edge_trace,
        edge_x,
        edge_y,
        fig,
        go,
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
def __(input_module):
    from xdsl.interactive.passes import iter_condensed_passes
    from xdsl.utils.hashable_module import HashableModule

    import networkx as nx

    G = nx.MultiDiGraph()

    root = HashableModule(input_module)
    queue = [root]
    visited = set()

    while queue:
        source = queue.pop()
        if source in visited:
            continue
        visited.add(source)
        for available_pass, t in iter_condensed_passes(source.module):
            target = HashableModule(t)
            G.add_edge(source, target, available_pass.display_name)
            if target not in visited:
                queue.append(target)
    return (
        G,
        HashableModule,
        available_pass,
        iter_condensed_passes,
        nx,
        queue,
        root,
        source,
        t,
        target,
        visited,
    )


@app.cell
def __(G):
    node_index_by_module = {n: i for i, n in enumerate(G.nodes())}


    return node_index_by_module,


@app.cell
def __(G, nx, root):
    for n in G.nodes:
        print(n.module)
        paths = nx.all_simple_edge_paths(G, root, n)

        for path in paths:
           print("Path :: " + ','.join(e[2] for e in path))
    str(G)
    return n, path, paths


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
