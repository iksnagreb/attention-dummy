# attention-dummy
Quickly generate PyTorch/Brevitas Scaled Dot-Product Attention dummy operators
for exploring [QONNX](https://github.com/fastmachinelearning/qonnx) and
[FINN](https://github.com/Xilinx/finn) graph transformations

## Setup
Install the dependencies listed in the `requirements.txt`, for example via pip
(these are mostly for debugging and orchestrating the builds, FINN comes with
its own dependencies bundled in a docker image):
```
pip install -r requirements.txt
```

Clone and checkout the feature branch of FINN combining all necessary additions
and modifications to FINN related to the attention feature and remember the path
you cloned into:
```
git clone https://github.com/iksnagreb/finn.git@v0.10/merge/attention
```

Add the [attention-hlslib](https://github.com/iksnagreb/attention-hlslib)
dependency manually to your FINN installation (note that this is a private
repository for now, you may have to request access):
```
cd <path-to-finn>/deps/
git clone https://github.com/iksnagreb/attention-hlslib.git
```

## Running FINN Builds
The whole export, build and evaluation pipeline is managed by
[dvc](https://github.com/iterative/dvc) and is executed as follows (see
`dvc.yaml` and `params.yaml` for configuration options and stage dependencies):
```
FINN=<path-to-finn> dvc repro
```
It is possible to specify all FINN related environment variables (e.g. the
`FINN_HOST_BUILD_DIR`) as usual. It might be necessary to pull existing model
and build artifacts into the local dvc cache first, though dvc should try to
reproduce these if not available:
```
dvc pull
```
All output artifacts, i.e., the exported model ONNX files, verification inputs,
FINN build logs, reports and the generated bitstream and driver script can be
found in the `build` directory.
