#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

input_jsonl_path="path_to_data/mteval-test26-official-ces_Latn-deu_Latn.jsonl"

python3 cot_tagged_span_annotation.py \
	--input_json $input_jsonl_path \
