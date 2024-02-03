import marimo

__generated_with = "0.1.77"
app = marimo.App()


@app.cell
def __():
    import marimo as mo

    return (mo,)


@app.cell
def __():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch_mlir

    @torch.jit.script
    class MyModel(nn.Module):
        def forward(self, x, y):
            return x + y

    torch.manual_seed(0)
    torch_model = MyModel()
    x = torch.randn(3)
    y = torch.randn(3)

    print(x)
    print(y)
    print(torch_model.forward(x, y))

    module = torch_mlir.compile(torch_model, (x, y), output_type="LINALG_ON_TENSORS")
    module
    return F, MyModel, module, nn, torch, torch_mlir, torch_model, x, y


@app.cell
def __(torch, torch_model, x, y):
    onnx_program = torch.onnx.dynamo_export(torch_model, x, y)
    return (onnx_program,)


@app.cell
def __(onnx_program):
    onnx_program.model_proto.graph
    return


@app.cell
def __(onnx_program):
    from xdsl.frontend.onnx import build_module as bm

    onnx_module = bm(onnx_program.model_proto.graph)

    str(onnx_module)
    return bm, onnx_module


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
