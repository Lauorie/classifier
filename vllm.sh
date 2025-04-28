export CUDA_VISIBLE_DEVICES=7
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model /root/app/finetune_bert/fine_tuned_m3_model_trainer/checkpoint-1600 \
    --served-model-name aigc \
    --task classify \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 1024 \
    --port 5022 > aigc.log 2>&1 &


curl -X POST http://localhost:5022/metrics \
  -H "Content-Type: application/json" \
  -d '{
    "model": "aigc",
    "text": "你好"
  }'