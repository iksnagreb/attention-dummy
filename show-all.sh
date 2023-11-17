#!/bin/bash

# Collect all ONNX graph files
for onnx in *.onnx; do
    # Show using netron in browser
    netron --browse $onnx &
    # Wait some time to avoid port confict between netron instances
    sleep 0.1
done;
