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
def __(mo):
    import plotly.graph_objects as go

    import networkx as nx

    G = nx.random_geometric_graph(200, 0.125)
    while not nx.is_connected(G):
        G = nx.random_geometric_graph(200, 0.125)


    node_x = [-1] * len(G.nodes())
    node_y = [-1] * len(G.nodes())
    for i, node in enumerate(G.nodes()):
        x, y = G.nodes[node]['pos']
        node_x[i] = (x)
        node_y[i] = (y)

        
    node_index_by_node = {n:i for i, n in enumerate(G.nodes())}

    # g_root = G.nodes()[0]

    node_x[node_index_by_node[0]] = 0.0
    node_y[node_index_by_node[0]] = 0.5

    layers = dict(enumerate(nx.bfs_layers(G, 0)))
    x_scale = len(layers) + 2.0
    y_scales = tuple(len(layers[i]) + 2.0 for i in range(len(layers)))

    for layer_index, nodes in layers.items():
        x = (layer_index + 1.0) / x_scale
        y_scale = y_scales[layer_index]
        for node_index, node in enumerate(nodes):
            y = (node_index + 1.0) / y_scale
            node_x[node] = x
            node_y[node] = y

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
                title='Node Connections',
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
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(f"pos: { G.nodes[node]['pos']}"+str(len(adjacencies[1])))

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title='Network graph made with Python',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    mo.ui.plotly(fig)
    return (
        G,
        adjacencies,
        edge,
        edge_trace,
        edge_x,
        edge_y,
        fig,
        go,
        i,
        layer_index,
        layers,
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
        nx,
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
def __(input_module, nx):
    from xdsl.interactive.passes import iter_condensed_passes
    from xdsl.utils.hashable_module import HashableModule

    bla = nx.MultiDiGraph()

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
            bla.add_edge(source, target, available_pass.display_name)
            if target not in visited:
                queue.append(target)
    return (
        HashableModule,
        available_pass,
        bla,
        iter_condensed_passes,
        queue,
        root,
        source,
        t,
        target,
        visited,
    )


@app.cell
def __(bla):
    node_index_by_module = {n: i for i, n in enumerate(bla.nodes())}


    return node_index_by_module,


@app.cell
def __(bla, nx, root):
    for n in bla.nodes:
        print(n.module)
        paths = nx.all_simple_edge_paths(bla, root, n)

        for path in paths:
           print("Path :: " + ','.join(e[2] for e in path))
    str(bla)
    return n, path, paths


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
