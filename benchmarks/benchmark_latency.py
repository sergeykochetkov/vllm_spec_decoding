"""Benchmark the latency of processing a single batch of requests."""
import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptType
from vllm.utils import FlexibleArgumentParser
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS


def main(args: argparse.Namespace):
    
    engine_args = EngineArgs.from_cli_args(args)

    # NOTE(woosuk): If the request cannot be processed in a single batch,
    # the engine will automatically process the request in multiple batches.
    llm = LLM(**dataclasses.asdict(engine_args))
    lora_kwargs={}
    if args.lora_path is not None:
        lora_kwargs['enable_lora']=True
        lora_kwargs['max_loras']=args.max_loras
        lora_kwargs['max_lora_rank']=args.max_lora_rank

    # NOTE(woosuk): If the request cannot be processed in a single batch,
    # the engine will automatically process the request in multiple batches.
    
    llm = LLM(
        model=args.model,
        speculative_model=args.speculative_model,
        num_speculative_tokens=args.num_speculative_tokens,
        speculative_draft_tensor_parallel_size=\
            args.speculative_draft_tensor_parallel_size,
        tokenizer=args.tokenizer,
        quantization=args.quantization,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        kv_cache_dtype=args.kv_cache_dtype,
        quantization_param_path=args.quantization_param_path,
        device=args.device,
        ray_workers_use_nsight=args.ray_workers_use_nsight,
        use_v2_block_manager=args.use_v2_block_manager,
        enable_chunked_prefill=args.enable_chunked_prefill,
        download_dir=args.download_dir,
        block_size=args.block_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        load_format=args.load_format,
        distributed_executor_backend=args.distributed_executor_backend,
        otlp_traces_endpoint=args.otlp_traces_endpoint,
        enable_prefix_caching=args.enable_prefix_caching,
        **lora_kwargs
    )

    sampling_params = SamplingParams(
        n=args.n,
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=args.output_len,
    )
    
    lora_request=LoRARequest(lora_name='lora', lora_int_id=0, lora_path=args.lora_path) if args.lora_path else None

    dummy_prompt_token_ids = np.random.randint(10000,
                                               size=(args.batch_size,
                                                     args.input_len))
    dummy_prompts: List[PromptType] = [{
        "prompt_token_ids": batch
    } for batch in dummy_prompt_token_ids.tolist()]

    def run_to_completion(profile_dir: Optional[str] = None):
        if profile_dir:
            with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(
                        str(profile_dir))) as p:
                llm.generate(dummy_prompts,
                             sampling_params=sampling_params,
                             use_tqdm=False,
                             lora_request=lora_request)
            print(p.key_averages())
        else:
            start_time = time.perf_counter()
            llm.generate(dummy_prompts,
                         sampling_params=sampling_params,
                         use_tqdm=False)
            end_time = time.perf_counter()
            latency = end_time - start_time
            return latency

    print("Warming up...")
    for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
        run_to_completion(profile_dir=None)

    if args.profile:
        profile_dir = args.profile_result_dir
        if not profile_dir:
            profile_dir = Path(
                "."
            ) / "vllm_benchmark_result" / f"latency_result_{time.time()}"
        print(f"Profiling (results will be saved to '{profile_dir}')...")
        run_to_completion(profile_dir=profile_dir)
        return

    # Benchmark.
    latencies = []
    for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
        latencies.append(run_to_completion(profile_dir=None))
    latencies = np.array(latencies)
    percentages = [10, 25, 50, 75, 90, 99]
    percentiles = np.percentile(latencies, percentages)
    print(f'Avg latency: {np.mean(latencies)} seconds')
    for percentage, percentile in zip(percentages, percentiles):
        print(f'{percentage}% percentile latency: {percentile} seconds')

    # Output JSON results if specified
    if args.output_json:
        results = {
            "avg_latency": np.mean(latencies),
            "latencies": latencies.tolist(),
            "percentiles": dict(zip(percentages, percentiles.tolist())),
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description='Benchmark the latency of processing a single batch of '
        'requests till completion.')
    parser.add_argument('--model', type=str, default='facebook/opt-125m')
    parser.add_argument('--lora-path', type=str, default=None)
    parser.add_argument('--max-loras', type=int, default=4)
    parser.add_argument('--max-lora-rank', type=int, default=64)
    parser.add_argument('--speculative-model', type=str, default=None)
    parser.add_argument('--num-speculative-tokens', type=int, default=None)
    parser.add_argument('--speculative-draft-tensor-parallel-size',
                        '-spec-draft-tp',
                        type=int,
                        default=None)
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=[*QUANTIZATION_METHODS, None],
                        default=None)
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    parser.add_argument('--input-len', type=int, default=32)
    parser.add_argument('--output-len', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--n',
                        type=int,
                        default=1,
                        help='Number of generated sequences per prompt.')
    parser.add_argument('--use-beam-search', action='store_true')
    parser.add_argument('--num-iters-warmup',
                        type=int,
                        default=10,
                        help='Number of iterations to run for warmup.')
    parser.add_argument('--num-iters',
                        type=int,
                        default=30,
                        help='Number of iterations to run.')
    parser.add_argument(
        '--profile',
        action='store_true',
        help='profile the generation process of a single batch')
    parser.add_argument(
        '--profile-result-dir',
        type=str,
        default=None,
        help=('path to save the pytorch profiler output. Can be visualized '
              'with ui.perfetto.dev or Tensorboard.'))
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Path to save the latency results in JSON format.')

    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
