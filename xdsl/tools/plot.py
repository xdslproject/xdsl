import time
import plotly.graph_objects as go
# import plotly.plotly as py
import pandas as pd
import plotly.figure_factory as FF
import chart_studio.plotly as py
from IPython.display import display
import plotly.express as px
import itertools
from plotly.subplots import make_subplots
import math

import dash
from dash import Dash, dcc, html, Input, Output

timings = pd.read_csv('bench_results_07.11.22.csv', sep=";")
# Same as below but we have access to the len so we can compute a ratio
# timings = pd.read_csv('bench_results_03.11.22.csv', sep=";")
# This one has the ifs using each other as operands
# timings = pd.read_csv('bench_results_02.11.22.csv', sep=";")
# timings = pd.read_csv('bench_results_31.10.22.csv', sep=";")
# timings = pd.read_csv('bench_results_reps.csv', sep=";")
# timings = pd.read_csv('bench_results_30.10.22.csv', sep=";")

################### One Figure that shows all info #####################

everything_fig = px.line(timings,
                         x="opsize",
                         y="time",
                         color="pass",
                         symbol="nesting")
everything_fig.update_traces(marker={'size': 15})

# Edit the layout
everything_fig.update_layout(title='Boolean Folding and If Inlining',
                             xaxis_title='Number of Operations',
                             yaxis_title='Runtime (s)')

# everything_fig.show()

##################### One Figure with many plots #######################

# Helpers
op_sizes = timings["opsize"].unique()
#[10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]
sizes = timings["nesting"].unique()
#[0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]

passes = timings["pass"].unique()

# If we have repetitions of experiments with the same config use the median
timings = timings.groupby(['opsize', 'nesting', 'pass']).median().reset_index()

all_configs = list(itertools.product(sizes, passes))
DEFAULT_PLOTLY_COLORS = [
    'rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)',
    'rgb(214, 39, 40)', 'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
    'rgb(227, 119, 194)', 'rgb(127, 127, 127)', 'rgb(188, 189, 34)',
    'rgb(23, 190, 207)'
]

clone_colors = ['#1f78b4', '#a6cee3']  # blue
composable_colors = ['#33a02c', '#b2df8a']  # green
no_backtracking_colors = ['#636363', '#bdbdbd']  # gray
marker_symbols = ["circle", "square"]
rows = 3
cols = 4

# multi_fig = make_subplots(
#     rows=rows,
#     cols=cols,
#     subplot_titles=tuple([
#         f"nesting={nesting}" for nesting in
#         [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]
#     ]),
#     specs=[[{
#         "secondary_y": True
#     }, {
#         "secondary_y": True
#     }, {
#         "secondary_y": True
#     }, {
#         "secondary_y": True
#     }],
#            [{
#                "secondary_y": True
#            }, {
#                "secondary_y": True
#            }, {
#                "secondary_y": True
#            }, {
#                "secondary_y": True
#            }],
#            [{
#                "secondary_y": True
#            }, {
#                "secondary_y": True
#            }, {
#                "secondary_y": True
#            }, {
#                "secondary_y": True
#            }]],
#     horizontal_spacing=0.05,
#     vertical_spacing=0.05)

# idx = 0
# for nesting, pass_name in all_configs:
#     if idx >= rows * cols:
#         break
#     colors = clone_colors if pass_name == "bool-nest-clone" else composable_colors
#     graph_name = "cloning" if pass_name == "bool-nest-clone" else "composable"

#     print(f"{nesting}, {pass_name}")
#     data = timings[(timings["nesting"] == nesting)
#                    & (timings["pass"] == pass_name) &
#                    (timings["opsize"] < 10000)]
#     print(f"row: {int(idx/cols)+1}, col: {(idx%cols)+1}")
#     multi_fig.add_trace(go.Scatter(x=data["opsize"],
#                                    y=data["time"],
#                                    name=f"{graph_name} runtime (s)",
#                                    marker=dict(color=colors[0])),
#                         row=int(idx / cols) + 1,
#                         col=(idx % cols) + 1,
#                         secondary_y=False)

#     multi_fig.add_trace(
#         go.Scatter(x=data["opsize"],
#                    y=data["memory"],
#                    name=f"{graph_name} memory (MB)",
#                    marker=dict(color=colors[1])),
#         row=(int(idx / cols) + 1),
#         col=((idx % cols) + 1),
#         secondary_y=True,
#     )
#     if pass_name == all_configs[-1][-1]:
#         idx += 1

# multi_fig.show()

##################### Interactive App #######################

app = dash.Dash()
app.layout = html.Div([
    dcc.Graph(id="graph"),
    html.P("What should be displayed on the x-axis:"),
    dcc.Dropdown(["vary op_size", "vary nesting", "vary op locality"],
                 "vary op_size",
                 id='xaxis-dropdown'),
    html.P("Ratio of nested operations: (when x-axis=opsize) "),
    dcc.Slider(step=None,
               marks={idx: f'{i}'
                      for idx, i in enumerate(sizes)},
               value=0.0,
               id="nesting-slider"),
    html.P("Max plotted opsize: (when x-axis=opsize)"),
    dcc.Dropdown(op_sizes, 3000, id='max_opsizes-dropdown'),
    html.P("Fixed opsize: (when x-axis=nesting)"),
    dcc.Dropdown(op_sizes, 1000, id='opsizes-dropdown'),
    dcc.Checklist(['x in log scale', 'y in log scale'],
                  inline=True,
                  id='log-scale-checklist'),
])


def get_plot(nesting: float, opsize: int, max_opsize: int, xaxis_mode: str,
             log_scale_x: bool, log_scale_y: bool):
    if xaxis_mode == "vary op_size":
        title = ""#f"nesting={nesting}"
    else:
        title = ""#f"Program size: {opsize} operations"
    # title = ""
    fig = make_subplots(rows=1,
                        cols=1,
                        subplot_titles=tuple([title]),
                        specs=[[{
                            "secondary_y": True
                        }]])

    for pass_name in passes:
        match pass_name:
            case "bool-nest-clone":
                colors = clone_colors
                graph_name = "naive cloning"
            case "bool-nest-composable":
                colors = composable_colors
                graph_name = "immutable (ours)"
            case "bool-nest-no-backtracking":
                colors = no_backtracking_colors
                graph_name = "destructive"# rewriting"

        if xaxis_mode == "vary op_size":
            data = timings[(timings["nesting"] == nesting)
                           & (timings["pass"] == pass_name) &
                           (timings["opsize"] <= max_opsize)]
            x_data = data["opsize"]
            # time_data = data.groupby("opsize")["time"].median()
            # print(time_data)
            xaxis_title = "Number of Operations"
        elif xaxis_mode == "vary nesting":
            data = timings[(timings["opsize"] == opsize)
                           & (timings["pass"] == pass_name)]
            x_data = data["nesting"]
            xaxis_title = "Number of Operations"
        elif xaxis_mode == "vary op locality":
            data = timings[(timings["opsize"] == opsize)
                           & (timings["pass"] == pass_name)]
            # data.sort_values(by="localityMean", inplace=True)
            data = data.drop_duplicates(subset="localityMean")
            x_data = data["localityMean"]# / data["len"]


            # Fixed rewriting localities I extracted from programs:
            # localityMax; localityMean; localityMedian; localityStdev; opcount

            # Open Earth compiler stuff:
            # see /home/martin/development/phd/projects/papers/xdsl_elevate/evaluation/open_earth_compiler/open-earth-compiler/test/Examples/fvtp2d_generic.mlir
            # 34;14.901907356948229;15;6.701371824499192;367
            fig.add_vline(x=14.901907356948229, line_width=1,
                          line_color="red")  # bert attention layer locality
            fig.add_annotation(x=math.log(14.901907356948229, 10), y=math.log(3.5, 10),
                                text="Climate<br>stencil",
                                showarrow=False,
                                arrowhead=1,
                                # yshift=10,
                                xshift=-25,
                                )
            # see /home/martin/development/phd/projects/papers/xdsl_elevate/evaluation/open_earth_compiler/open-earth-compiler/test/Examples/hadvuv5th_generic.mlir
            #46;12.99250936329588;12;7.544849614770229;267

            # Machine Learning stuff:
            # see /home/martin/development/phd/projects/papers/xdsl_elevate/evaluation/mlmodels/test_data/models/mlir/attention_generic.mlir
            # 27;12.673076923076923;13.0;6.360790288321573;52

            # see /home/martin/development/phd/projects/papers/xdsl_elevate/evaluation/mlmodels/bert/bert_1/model_generic.tmp
            # 98;47.15243902439025;50.0;27.26232360774099;164

            # see /home/martin/development/phd/projects/papers/xdsl_elevate/evaluation/mlmodels/bert/bert_10/model_generic.tmp
            # 800;398.9819227608874;405;233.45582671719256;1217

            xaxis_title = "Mean operation use dependencies"  #"Mean of how many other ops have to be touched when an op is rewritten (Operation locality?)"

            # fig.add_vline(x=12.673076923076923, line_width=1,
            #               line_color="red")  # bert attention layer locality

            # fig.add_vline(x=47.15243902439025, line_width=1,
            #               line_color="red")  # bert small complete locality

            # fig.add_vline(x=12.673076923076923, line_width=1,
            #               line_color="red")  # bert attention layer locality

            fig.add_vline(x=47.15243902439025, line_width=1,
                          line_color="red")  # bert small complete locality
            fig.add_annotation(x=math.log(47.15243902439025, 10), y=math.log(3.5, 10),
                        text="BERT<br>small",
                        showarrow=False,
                        arrowhead=1,
                        # yshift=10,
                        xshift=-20,
                        )

            #   annotation_text="bert attention layer op locality",
            #   annotation_position="top left")
        else:
            raise ValueError("Unknown xaxis mode")

        fig.add_trace(go.Scatter(x=x_data,
                                 y=data["time"]/data["len"]*data["opsize"],
                                 name=f"{graph_name}",
                                 marker=dict(color=colors[0]),
                                 marker_symbol=marker_symbols[0],#marker_line_width=1
                                #  legendgroup="Runtime",
                                #  legendgrouptitle_text="Runtime",
                                 showlegend=True),
                      secondary_y=False)

        fig.add_trace(go.Scatter(x=x_data,
                                 y=data["memory"]/data["len"]*data["opsize"],
                                 name=f"{graph_name}",
                                 marker=dict(color=colors[1]),
                                 marker_symbol=marker_symbols[1],#marker_line_width=0.5
                                #  legendgroup="Memory",
                                #  legendgrouptitle_text="Memory",
                                 showlegend=True),
                      secondary_y=True)

        fig.update_layout(title='',
                          xaxis_title=xaxis_title,
                          yaxis_title='runtime (s)')
        fig.update_yaxes(title_text="memory (MB)", secondary_y=True)

        if log_scale_x:
            fig.update_xaxes(type="log")
        else:
            fig.update_xaxes(type="linear")
        if log_scale_y:
            fig.update_yaxes(type="log")
        else:
            fig.update_yaxes(type="linear")

        # fig.update_layout(legend=dict(
        #     yanchor="top",
        #     y=0.99,
        #     xanchor="left",
        #     x=0.01
        # ))
        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            # itemwidth=30,
            xanchor="right",
            x=0.9,
            font=dict(
                # family="Courier",
                size=16,
                # color="black"
            ),
        ), legend_title_text='Runtime')
        fig.update_layout(template="plotly_white")
        fig.update_xaxes(title_font=dict(size=18))#, family='Courier', color='crimson'))
        fig.update_yaxes(title_font=dict(size=18), title_standoff=0, secondary_y=False)#, family='Courier', color='crimson'))
        fig.update_yaxes(title_font=dict(size=18), title_standoff=10, secondary_y=True)#, family='Courier', color='crimson'))
    return fig


nesting = 0.0
opsize = 1000
max_opsize = 30000
log_scale_x = False
log_scale_y = False
xaxis_mode = "vary op_size"

# saving twice with time in between as a workaroung to a stupid Mathjax box that pops up in the pdf

# "vary nesting"
# "vary op locality"
# "vary op_size"
get_plot(0.6, 1000, 1000, "vary op locality", True, True).write_image("fig1.pdf")
time.sleep(1)
get_plot(0.6, 1000, 1000, "vary op locality", True, True).write_image("rewriting_use_case_op_use_dep_scaling_log_log.pdf")
get_plot(0.6, 1000, 1000, "vary op locality", False, False).write_image("rewriting_use_case_op_use_dep_scaling_lin_lin.pdf")
get_plot(0.6, 3000, 3000, "vary op_size", True, True).write_image("rewriting_use_case_num_ops_scaling.pdf")


@app.callback(Output("graph", "figure"), 
              Output("nesting-slider", "disabled"),
              Output("opsizes-dropdown", "disabled"),
              Output("max_opsizes-dropdown", "disabled"),
              Input("nesting-slider", "value"),
              Input("opsizes-dropdown", "value"),
              Input("max_opsizes-dropdown", "value"),
              Input("xaxis-dropdown", "value"),
              Input("log-scale-checklist", "value"))
def update_figure(selected_nesting, selected_op_size, selected_max_opsize,
                  selected_xaxis_mode, log_scale):
    global nesting
    global max_opsize
    global log_scale_x
    global log_scale_y
    global opsize
    global xaxis_mode

    slider_disabled = False
    fixed_opsize_disabled = False
    max_op_size_disabled = False

    if selected_nesting is not None:
        nesting = sizes[selected_nesting]
    if selected_op_size is not None:
        opsize = selected_op_size
    if selected_max_opsize is not None:
        max_opsize = selected_max_opsize
    if log_scale is not None:
        log_scale_x = 'x in log scale' in log_scale
        log_scale_y = 'y in log scale' in log_scale
    if selected_xaxis_mode is not None:
        xaxis_mode = selected_xaxis_mode
        match xaxis_mode:
            case "vary nesting":
                slider_disabled = True
                fixed_opsize_disabled = False
                max_op_size_disabled = True
            case "vary op locality":
                slider_disabled = True
                fixed_opsize_disabled = False
                max_op_size_disabled = True
            case "vary op_size":
                slider_disabled = False
                fixed_opsize_disabled = True
                max_op_size_disabled = False
            case _:
                raise ValueError("Unknown xaxis mode")


    return get_plot(nesting, opsize, max_opsize, xaxis_mode, log_scale_x,
                    log_scale_y), slider_disabled, fixed_opsize_disabled, max_op_size_disabled


app.run_server(host='0.0.0.0', port=8060, debug=True,
               use_reloader=False)  # Turn off reloader if inside Jupyter