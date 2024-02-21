import marimo

__generated_with = "0.2.5"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __(mo):
    input = mo.ui.text_area()
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
def __(Hashable, Hasher, ModuleOp, dataclass):
    @dataclass(frozen=True)
    class ModuleStructure(Hashable):
        module: ModuleOp

        def __eq__(self, other: object) -> bool:
            return isinstance(other, ModuleStructure) and self.module.is_structurally_equivalent(other.module)

        def __hash__(self) -> int:
            hasher = Hasher()
            for op in self.module.walk():
                hasher.combine(op.name)
            return hasher.hash

    assert ModuleStructure(ModuleOp([])) == ModuleStructure(ModuleOp([]))
    assert hash(ModuleStructure(ModuleOp([]))) == hash(ModuleStructure(ModuleOp([])))
    return ModuleStructure,


@app.cell
def __():
    from xdsl.utils.hasher import Hasher
    return Hasher,


@app.cell
def __(Hasher):
    h = Hasher()
    h.combine(1)
    h.combine(2)
    print(h.hash)

    j = Hasher(seed=2654435770)
    j.combine(2)
    print(j.hash)
    return h, j


@app.cell
def __(mo):
    import plotly.graph_objects as go

    import networkx as nx

    G = nx.random_geometric_graph(200, 0.125)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
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

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

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

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append('# of connections: '+str(len(adjacencies[1])))

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title='<br>Network graph made with Python',
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
        node,
        node_adjacencies,
        node_text,
        node_trace,
        node_x,
        node_y,
        nx,
        x,
        x0,
        x1,
        y,
        y0,
        y1,
    )


if __name__ == "__main__":
    app.run()
