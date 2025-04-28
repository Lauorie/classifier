# 单条推理
CUDA_VISIBLE_DEVICES=7 \
python inference.py \
    --model_path /root/app/finetune_bert/fine_tuned_m3_model_trainer/checkpoint-1600 \
    --max_length 1024 \
    --device cuda \
    --temperature 1.5 \
    --text "这里可以加载一个大模型，多个adpter的技术，大大减少对显存的占用，并且将训练模型和状态价值模型合并，共用一套adapter参数，它同时有两个头，分别是LLM Head和 Value Head。这样就可以只加载一个大模型和两个Lora 参数的adapter来训练PPO。"