
export CUDA_VISIBLE_DEVICES=2

 python benchmarks/benchmark_latency.py \
 --model=/workspace/models_spec/llama2-70b-chat-awq/ -q=awq \
 --use-v2-block-manager --gpu_memory_utilization=0.6 \
 --num-iters-warmup=2 --num-iters=5 \
 --batch-size=8 \
 --input-len=256 \
 --output-len=10 \
 --speculative-model=/workspace/models_spec/llama2-1.1b-chat-gptq/ --num-speculative-tokens=2 \
 --profile --profile-result-dir=profile_cuda_graph_bs8 \