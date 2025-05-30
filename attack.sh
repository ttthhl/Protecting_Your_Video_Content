python attack.py \
    --cfg-path eval_configs/video_llama_eval_only_vl.yaml \
    --model_type llama_v2 \
    --type eos \
    --gt_file path/csv \
    --video_dir path/video \
    --output_dir save_path \
    --iter 500 \
    --max_modify 16 \
    --step 1 \
    --weight_clip 1 \
    --weight_llm 1

python attack.py \
    --cfg-path eval_configs/video_llama_eval_only_vl.yaml \
    --model_type llama_v2 \
    --type eos2 \
    --gt_file path/csv \
    --video_dir path/video \
    --output_dir save_path \
    --iter 500 \
    --max_modify 16 \
    --step 1 \
    --weight_clip 1 \
    --weight_llm 1

python attack.py \
    --cfg-path eval_configs/video_llama_eval_only_vl.yaml \
    --model_type llama_v2 \
    --type random \
    --gt_file path/csv \
    --video_dir path/video \
    --output_dir save_path \
    --iter 200 \
    --max_modify 16 \
    --step 1 \
    --weight_clip 1 \
    --weight_llm 1

python attack.py \
    --cfg-path eval_configs/video_llama_eval_only_vl.yaml \
    --model_type llama_v2 \
    --type train \
    --gt_file path/csv \
    --video_dir path/video \
    --output_dir save_path \
    --iter 200 \
    --max_modify 16 \
    --step 1 \
    --weight_clip 1 \
    --weight_llm 1