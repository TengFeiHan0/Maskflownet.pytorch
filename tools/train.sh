#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=4 main.py --config configs/MaskFlowNet.yaml --launcher pytorch