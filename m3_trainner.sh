CUDA_VISIBLE_DEVICES=7 \
nohup python m3_trainner.py \
    --model_name_or_path /root/app/models/models--TencentBAC--Conan-embedding-v1 \
    --data_path "/root/app/rag_data/paper_rewrite/from_zhaominjie/2015ago/papers_cn_1500_rewrite_all_models_sft_16444.json" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --learning_rate 2e-5 \
    --max_length 512 \
    --logging_steps 50 \
    --eval_steps 200 \
    --save_steps 200 \
    --fp16 \
    --gradient_accumulation_steps 2 \
    --output_dir "./fine_tuned_conan_model_trainer"  > /root/app/finetune_bert/conan_trainner.log 2>&1 &
