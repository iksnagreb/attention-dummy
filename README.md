# attention-dummy
Quickly generate PyTorch/Brevitas Scaled Dot-Product Attention dummy operators
for exploring QONNX and FINN graph transformations

## Setup and Running the Code
To just try dummy model export, streamlining and detection of the attention
operator pattern, without actually building the FPGA accelerator, install the
`requirements.txt`. Now execute `attention.py` (to generate a dummy model),
`transform.py` (to try cleanup and streamlining) and `infer.py` (to detect and
parallelize the attention heads) in this order. Each of these stages generates
a `.onnx` file which can be viewed using [netron](https://github.com/lutzroeder/netron).

### Running Dataflow Builds
To run dataflow builds, install the finn docker container from the
feature branch combining streamlining and attention operator support (WIP):
```
git clone https://github.com/iksnagreb/finn.git@merge/attention
```
Additionally, add the [attention-hlslib](https://github.com/iksnagreb/attention-hlslib)
dependency manually to your finn installation (note that this is a private
repository for now, you may have to request access):
```
cd <path-to-finn>/deps/
git clone https://github.com/iksnagreb/attention-hlslib.git
```
Checkout the most recent feature branch of the attention-hlslib, e.g.:
```
cd <path-to-finn>/deps/attention-hlslib
git checkout masked-softmax
```
Now execute dataflow builds generating bitfile and driver to be run on an FPGA
as configured in `build.py` from within your finn installation:
```
FINN_HOST_BUILD_DIR=/<absolute-path...>/attention-dummy/build ./run-docker.sh build_custom /<absolute-path...>/attention-dummy/
```
Logs, reports, intermediate model graphs and the deployment package can be found
in the `attention-build` subdirectory. Code generation, simulation and synthesis
outputs will be stored in the `build` directory.