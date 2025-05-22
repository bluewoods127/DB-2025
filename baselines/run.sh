#!/bin/bash

source activate dpok_4
python3 -u run_RS.py \
--cache_dir ".cache/" \
--ir_weight 0.2 \
--prompt_file "dataset/test_prompts.json" \
--t2i_path 'path_to_checkpoint/pytorch_lora_weights.bin' \
--vila_path 'path_to_checkpoint/pytorch_lora_weights.bin'


