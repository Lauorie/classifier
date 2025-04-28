# 批量推理
CUDA_VISIBLE_DEVICES=7 \
python inference.py \
    --model_path /root/app/finetune_bert/fine_tuned_m3_model_trainer/checkpoint-1600 \
    --input_json "/root/app/rag_data/paper_rewrite/data_20250410/paper_rewrite_2000_cn_aigc.json" \
    --output_json "/root/app/rag_data/paper_rewrite/data_20250410/paper_rewrite_2000_cn_aigc_classified.json" \
    --text_field "verify_text" \
    --output_field "from_source" \
    --max_length 1024 \
    --batch_size 32 \
    --device cuda