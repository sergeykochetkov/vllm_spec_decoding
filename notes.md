
# Exp 1

scoring - 46

verification - 43

    verify_to_list - 42

    verify_prepare - 0 ms

    verify_sampler - 2


# Exp 2

scoring - 46

verification - 43

    verify_to_list - 0 ms

    verify_prepare - 42 ms

    verify_sampler - 1-2


----------
 gprof2dot -f pstats output.pstats -p /workspace/vllm_sergeykochetkov/vllm/spec_decode -o output_verify_tokens.dot

 pip install line_profiler

pip install -U tensorboard-plugin-profile

-----------------
batch / baseline
Avg latency: 10.002039948323121 seconds / with cuda graph
enforce-eager: Avg latency: 14.951877784371996 seconds

prefill / my
Avg latency: 9.059010420343839 seconds
enforce-eager: Avg latency: 13.397862712969072 seconds


-------------
batch
Avg latency: 16.612465888634325 seconds

prefill
Avg latency: 15.667380536394194 seconds
-------------------

FlashAttentionMetadata(num_prefills=1, num_prefill_tokens=3, num_decode_tokens=0, slot_mapping=tensor([256, 257, 258], device='cuda:0'), seq_lens=[259], seq_lens_tensor=tensor([259], device='cuda:0', dtype=torch.int32), max_query_len=3, max_prefill_seq_len=259, max_decode_seq_len=0, query_start_loc=tensor([0, 3], device='cuda:0', dtype=torch.int32), seq_start_loc=tensor([  0, 259], device='cuda:0', dtype=torch.int32), context_lens_tensor=tensor([256], device='cuda:0', dtype=torch.int32), block_tables=tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16]],
       device='cuda:0', dtype=torch.int32), use_cuda_graph=False, _cached_prefill_metadata=None, _cached_decode_metadata=None)

# kv_cache_shape

In FlashAttentionBackend.get_kv_cache_shape:
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)


----------------
model_input
ModelInputForGPUWithSamplingMetadata(input_tokens=tensor([16748,   701, 29875], device='cuda:0'), input_positions=tensor([256, 257, 258], device='cuda:0'), seq_lens=[259], query_lens=[3], lora_mapping=None, lora_requests=set(), attn_metadata=FlashAttentionMetadata(num_prefills=1, num_prefill_tokens=3, num_decode_tokens=0, slot_mapping=tensor([256, 257, 258], device='cuda:0'), seq_lens=[259], seq_lens_tensor=tensor([259], device='cuda:0', dtype=torch.int32), max_query_len=3, max_prefill_seq_len=259, max_decode_seq_len=0, query_start_loc=tensor([0, 3], device='cuda:0', dtype=torch.int32), seq_start_loc=tensor([  0, 259], device='cuda:0', dtype=torch.int32), context_lens_tensor=tensor([256], device='cuda:0', dtype=torch.int32), block_tables=tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16]],
       device='cuda:0', dtype=torch.int32), use_cuda_graph=False, _cached_prefill_metadata=FlashAttentionMetadata(num_prefills=1, num_prefill_tokens=3, num_decode_tokens=0, slot_mapping=tensor([256, 257, 258], device='cuda:0'), seq_lens=[259], seq_lens_tensor=tensor([259], device='cuda:0', dtype=torch.int32), max_query_len=3, max_prefill_seq_len=259, max_decode_seq_len=0, query_start_loc=tensor([0, 3], device='cuda:0', dtype=torch.int32), seq_start_loc=tensor([  0, 259], device='cuda:0', dtype=torch.int32), context_lens_tensor=tensor([256], device='cuda:0', dtype=torch.int32), block_tables=tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16]],
       device='cuda:0', dtype=torch.int32), use_cuda_graph=False, _cached_prefill_metadata=None, _cached_decode_metadata=None), _cached_decode_metadata=None), prompt_adapter_mapping=None, prompt_adapter_requests=set(), multi_modal_kwargs={}, request_ids_to_seq_ids={'0': [0]}, finished_requests_ids=[], virtual_engine=0, sampling_metadata=SamplingMetadata(seq_groups=[SequenceGroupToSample(seq_ids=[0], sampling_params=SamplingParams(n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0, seed=None, use_beam_search=False, length_penalty=1.0, early_stopping=False, stop=[], stop_token_ids=[], include_stop_str_in_output=False, ignore_eos=True, max_tokens=128, min_tokens=0, logprobs=None, prompt_logprobs=1, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None), seq_data={0: SequenceData(prompt_token_ids=array('l', [2732, 9845, 3264, 4859, 9225, 7891, 4373, 5874, 6744, 3468, 705, 2599, 2222, 7768, 2897, 9893, 537, 6216, 6921, 6036, 2163, 5072, 4851, 7877, 2046, 1871, 7599, 2496, 8291, 755, 797, 659, 3219, 8615, 7456, 3337, 2745, 4735, 8736, 6687, 714, 2292, 8343, 1207, 6172, 8994, 7221, 6021, 3622, 3560, 8948, 1641, 4984, 4353, 8622, 7250, 4187, 2659, 9781, 2956, 2251, 4420, 7108, 1071, 5251, 7012, 9396, 3918, 9359, 1684, 7098, 2957, 4469, 8752, 9797, 5795, 1472, 7263, 7365, 8448, 6001, 3762, 9008, 2435, 1634, 973, 4464, 8393, 2418, 3455, 6167, 5819, 6521, 6242, 7742, 9123, 6738, 2787, 7316, 4305, 2610, 5531, 6926, 7204, 6922, 4182, 307, 5302, 1152, 6950, 8467, 5294, 1208, 2492, 8829, 770, 8286, 5995, 2344, 3091, 3912, 1434, 6594, 5368, 8372, 7148, 7997, 3854, 8032, 8131, 4845, 5116, 3533, 2937, 9837, 4939, 9744, 3224, 5021, 1134, 25, 9680, 956, 1913, 2934, 9661, 2721, 928, 5627, 6265, 5446, 469, 8717, 1863, 1720, 5272, 591, 6185, 2322, 207, 4262, 3421, 5249, 8408, 8216, 5103, 7939, 2282, 1740, 6118, 5846, 3781, 2775, 2603, 7179, 6356, 1162, 623, 8962, 4051, 1241, 9013, 4403, 1198, 2997, 5661, 807, 2121, 8067, 3886, 8922, 6066, 9987, 1823, 199, 1447, 5181, 5208, 6177, 4863, 6180, 1792, 1483, 8389, 894, 5374, 136, 6273, 9584, 3419, 168, 6004, 2852, 9753, 4419, 8039, 8700, 3186, 5918, 5149, 1777, 3361, 8338, 5393, 4317, 4605, 2562, 6213, 9100, 4652, 6235, 423, 6854, 967, 4370, 9052, 6187, 5203, 433, 6237, 1429, 2546, 329, 3612, 8401, 6761, 3968, 8150, 1040, 6250, 8356, 8798, 7704, 6772, 5311, 9411, 16748, 701, 29875]), output_token_ids=array('l'), cumulative_logprob=0.0)}, seq_len=259, query_len=3, generator=None, is_prompt=True, prompt_logprob_indices=[0, 1, 2], sample_indices=[])], selected_token_indices=tensor([0, 1, 2], device='cuda:0'), categorized_sample_indices={<SamplingType.GREEDY: 0>: tensor([], device='cuda:0', size=(0, 2), dtype=torch.int32), <SamplingType.RANDOM: 1>: tensor([], device='cuda:0', size=(0, 2), dtype=torch.int32), <SamplingType.RANDOM_SEED: 2>: tensor([], device='cuda:0', size=(0, 2), dtype=torch.int32), <SamplingType.BEAM: 3>: tensor([], device='cuda:0', size=(0, 2), dtype=torch.int32)}), , is_prompt=True)
special variables:
function variables:
attn_metadata: FlashAttentionMetadata(num_prefills=1, num_prefill_tokens=3, num_decode_tokens=0, slot_mapping=tensor([256, 257, 258], device='cuda:0'), seq_lens=[259], seq_lens_tensor=tensor([259], device='cuda:0', dtype=torch.int32), max_query_len=3, max_prefill_seq_len=259, max_decode_seq_len=0, query_start_loc=tensor([0, 3], device='cuda:0', dtype=torch.int32), seq_start_loc=tensor([  0, 259], device='cuda:0', dtype=torch.int32), context_lens_tensor=tensor([256], device='cuda:0', dtype=torch.int32), block_tables=tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16]],
       device='cuda:0', dtype=torch.int32), use_cuda_graph=False, _cached_prefill_metadata=FlashAttentionMetadata(num_prefills=1, num_prefill_tokens=3, num_decode_tokens=0, slot_mapping=tensor([256, 257, 258], device='cuda:0'), seq_lens=[259], seq_lens_tensor=tensor([259], device='cuda:0', dtype=torch.int32), max_query_len=3, max_prefill_seq_len=259, max_decode_seq_len=0, query_start_loc=tensor([0, 3], device='cuda:0', dtype=torch.int32), seq_start_loc=tensor([  0, 259], device='cuda:0', dtype=torch.int32), context_lens_tensor=tensor([256], device='cuda:0', dtype=torch.int32), block_tables=tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16]],
       device='cuda:0', dtype=torch.int32), use_cuda_graph=False, _cached_prefill_metadata=None, _cached_decode_metadata=None), _cached_decode_metadata=None)
finished_requests_ids: []
input_positions: tensor([256, 257, 258], device='cuda:0')
input_tokens: tensor([16748,   701, 29875], device='cuda:0')
is_prompt: True
lora_mapping: None
lora_requests: {}
multi_modal_kwargs: {}
prompt_adapter_mapping: None
prompt_adapter_requests: {}
query_lens: [3]
request_ids_to_seq_ids: {'0': [0]}
sampling_metadata: SamplingMetadata(seq_groups=[SequenceGroupToSample(seq_ids=[0], sampling_params=SamplingParams(n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0, seed=None, use_beam_search=False, length_penalty=1.0, early_stopping=False, stop=[], stop_token_ids=[], include_stop_str_in_output=False, ignore_eos=True, max_tokens=128, min_tokens=0, logprobs=None, prompt_logprobs=1, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None), seq_data={0: SequenceData(prompt_token_ids=array('l', [2732, 9845, 3264, 4859, 9225, 7891, 4373, 5874, 6744, 3468, 705, 2599, 2222, 7768, 2897, 9893, 537, 6216, 6921, 6036, 2163, 5072, 4851, 7877, 2046, 1871, 7599, 2496, 8291, 755, 797, 659, 3219, 8615, 7456, 3337, 2745, 4735, 8736, 6687, 714, 2292, 8343, 1207, 6172, 8994, 7221, 6021, 3622, 3560, 8948, 1641, 4984, 4353, 8622, 7250, 4187, 2659, 9781, 2956, 2251, 4420, 7108, 1071, 5251, 7012, 9396, 3918, 9359, 1684, 7098, 2957, 4469, 8752, 9797, 5795, 1472, 7263, 7365, 8448, 6001, 3762, 9008, 2435, 1634, 973, 4464, 8393, 2418, 3455, 6167, 5819, 6521, 6242, 7742, 9123, 6738, 2787, 7316, 4305, 2610, 5531, 6926, 7204, 6922, 4182, 307, 5302, 1152, 6950, 8467, 5294, 1208, 2492, 8829, 770, 8286, 5995, 2344, 3091, 3912, 1434, 6594, 5368, 8372, 7148, 7997, 3854, 8032, 8131, 4845, 5116, 3533, 2937, 9837, 4939, 9744, 3224, 5021, 1134, 25, 9680, 956, 1913, 2934, 9661, 2721, 928, 5627, 6265, 5446, 469, 8717, 1863, 1720, 5272, 591, 6185, 2322, 207, 4262, 3421, 5249, 8408, 8216, 5103, 7939, 2282, 1740, 6118, 5846, 3781, 2775, 2603, 7179, 6356, 1162, 623, 8962, 4051, 1241, 9013, 4403, 1198, 2997, 5661, 807, 2121, 8067, 3886, 8922, 6066, 9987, 1823, 199, 1447, 5181, 5208, 6177, 4863, 6180, 1792, 1483, 8389, 894, 5374, 136, 6273, 9584, 3419, 168, 6004, 2852, 9753, 4419, 8039, 8700, 3186, 5918, 5149, 1777, 3361, 8338, 5393, 4317, 4605, 2562, 6213, 9100, 4652, 6235, 423, 6854, 967, 4370, 9052, 6187, 5203, 433, 6237, 1429, 2546, 329, 3612, 8401, 6761, 3968, 8150, 1040, 6250, 8356, 8798, 7704, 6772, 5311, 9411, 16748, 701, 29875]), output_token_ids=array('l'), cumulative_logprob=0.0)}, seq_len=259, query_len=3, generator=None, is_prompt=True, prompt_logprob_indices=[0, 1, 2], sample_indices=[])], selected_token_indices=tensor([0, 1, 2], device='cuda:0'), categorized_sample_indices={<SamplingType.GREEDY: 0>: tensor([], device='cuda:0', size=(0, 2), dtype=torch.int32), <SamplingType.RANDOM: 1>: tensor([], device='cuda:0', size=(0, 2), dtype=torch.int32), <SamplingType.RANDOM_SEED: 2>: tensor([], device='cuda:0', size=(0, 2), dtype=torch.int32), <SamplingType.BEAM: 3>: tensor([], device='cuda:0', size=(0, 2), dtype=torch.int32)}),
seq_lens: [259]
virtual_engine: 0
_abc_impl: <_abc._abc_data object at 0x7f42d4ae1700>


------------
prefill:
Avg latency: 21.096329744998364 seconds
10% percentile latency: 19.892291318951173 seconds
25% percentile latency: 20.094397073844448 seconds
50% percentile latency: 21.254516137065366 seconds
75% percentile latency: 22.14986324100755 seconds
90% percentile latency: 22.195136170834303 seconds
99% percentile latency: 22.222299928730354 seconds

SpecDecodeWorker stage times: average_time_per_proposal_tok_ms=3.57 scoring_time_ms=94.42 verification_time_ms=1.32
INFO 08-27 15:21:10 metrics.py:406] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 75.3 tokens/s, Running: 6 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 9.2%, CPU KV cache usage: 0.0%.
INFO 08-27 15:21:10 metrics.py:422] Speculative metrics: Draft acceptance rate: 0.343, System efficiency: 0.489, Number of speculative tokens: 2, Number of accepted tokens: 6455, Number of draft tokens: 18802, Number of emitted tokens: 13783.


without cuda graph:
Speculative metrics: Draft acceptance rate: 0.490, System efficiency: 0.584, Number of speculative tokens: 2, Number of accepted tokens: 7655, Number of draft tokens: 15626, Number of emitted tokens: 13700.
INFO 08-27 15:50:04 spec_decode_worker.py:773] SpecDecodeWorker stage times: average_time_per_proposal_tok_ms=3.17 scoring_time_ms=52.52 verification_time_ms=1.33

all_tokens
tensor([[  904,   510,  1446],
        [  903,   262,   504],
        [ 2879, 13200,   531],
        [  368,  4680, 29950],
        [29879, 29889,    13],
        [  466,  4012,   406],
        [ 3456,  4593,   262],
        [ 8675,  3276,  3639]], device='cuda:0')

batch:
Avg latency: 17.36865475117229 seconds
10% percentile latency: 17.05709356125444 seconds
25% percentile latency: 17.203141598962247 seconds
50% percentile latency: 17.41754260403104 seconds
75% percentile latency: 17.56962864403613 seconds
90% percentile latency: 17.643791081244125 seconds
99% percentile latency: 17.688288543568923 seconds

Draft acceptance rate: 0.412, System efficiency: 0.533, Number of speculative tokens: 2, Number of accepted tokens: 7015, Number of draft tokens: 17022, Number of emitted tokens: 13603.
INFO 08-27 15:12:21 spec_decode_worker.py:773] SpecDecodeWorker stage times: average_time_per_proposal_tok_ms=3.39 scoring_time_ms=58.08 verification_time_ms=1.27

----------------------------

with cuda_graph

input_tokens
tensor([16748,   701, 29875,   870,   608, 29874, 23077,   325, 30767,  1416,
          405, 29902, 10619,   931,   869, 15514,  2973,  1034,  6212, 25252,
         3012,  3964,   262, 14424], device='cuda:0')

input_positions
tensor([256, 257, 258, 256, 257, 258, 256, 257, 258, 256, 257, 258, 256, 257,
        258, 256, 257, 258, 256, 257, 258, 256, 257, 258], device='cuda:0')


FlashAttentionMetadata(num_prefills=8, num_prefill_tokens=24, num_decode_tokens=0, slot_mapping=tensor([2048, 2049, 2050, 2064, 2065, 2066, 2080, 2081, 2082, 2096, 2097, 2098,
        2112, 2113, 2114, 2128, 2129, 2130, 2144, 2145, 2146, 2160, 2161, 2162],
       device='cuda:0'), seq_lens=[259, 259, 259, 259, 259, 259, 259, 259], seq_lens_tensor=tensor([259, 259, 259, 259, 259, 259, 259, 259], device='cuda:0',
       dtype=torch.int32), max_query_len=3, max_prefill_seq_len=259, max_decode_seq_len=0, query_start_loc=tensor([ 0,  3,  6,  9, 12, 15, 18, 21, 24], device='cuda:0',
       dtype=torch.int32), seq_start_loc=tensor([   0,  259,  518,  777, 1036, 1295, 1554, 1813, 2072], device='cuda:0',
       dtype=torch.int32), context_lens_tensor=tensor([256, 256, 256, 256, 256, 256, 256, 256], device='cuda:0',
       dtype=torch.int32), block_tables=tensor([[  0,   1,   2,  ...,   0,   0,   0],
        [ 16,  17,  18,  ...,   0,   0,   0],
        [ 32,  33,  34,  ...,   0,   0,   0],
        ...,
        [ 80,  81,  82,  ...,   0,   0,   0],
        [ 96,  97,  98,  ...,   0,   0,   0],
        [112, 113, 114,  ...,   0,   0,   0]], device='cuda:0',
       dtype=torch.int32), use_cuda_graph=True, _cached_prefill_metadata=FlashAttentionMetadata(num_prefills=8, num_prefill_tokens=24, num_decode_tokens=0, slot_mapping=tensor([2048, 2049, 2050, 2064, 2065, 2066, 2080, 2081, 2082, 2096, 2097, 2098,
        2112, 2113, 2114, 2128, 2129, 2130, 2144, 2145, 2146, 2160, 2161, 2162],
       device='cuda:0'), seq_lens=[259, 259, 259, 259, 259, 259, 259, 259], seq_lens_tensor=tensor([259, 259, 259, 259, 259, 259, 259, 259], device='cuda:0',
       dtype=torch.int32), max_query_len=3, max_prefill_seq_len=259, max_decode_seq_len=0, query_start_loc=tensor([ 0,  3,  6,  9, 12, 15, 18, 21, 24], device='cuda:0',
       dtype=torch.int32), seq_start_loc=tensor([   0,  259,  518,  777, 1036, 1295, 1554, 1813, 2072], device='cuda:0',
       dtype=torch.int32), context_lens_tensor=tensor([256, 256, 256, 256, 256, 256, 256, 256], device='cuda:0',
       dtype=torch.int32), block_tables=tensor([[  0,   1,   2,  ...,   0,   0,   0],
        [ 16,  17,  18,  ...,   0,   0,   0],
        [ 32,  33,  34,  ...,   0,   0,   0],
        ...,
        [ 80,  81,  82,  ...,   0,   0,   0],
        [ 96,  97,  98,  ...,   0,   0,   0],
        [112, 113, 114,  ...,   0,   0,   0]], device='cuda:0',
       dtype=torch.int32), use_cuda_graph=True, _cached_prefill_metadata=None, _cached_decode_metadata=None), _cached_decode_metadata=None)

model_input.attn_metadata.block_tables.shape
torch.Size([8, 256])
SamplingMetadata(seq_groups=[SequenceGroupToSample(seq_ids=[0], sampling_params=SamplingParams(n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0, seed=None, use_beam_search=False, length_penalty=1.0, early_stopping=False, stop=[], stop_token_ids=[], include_stop_str_in_output=False, ignore_eos=True, max_tokens=128, min_tokens=0, logprobs=None, prompt_logprobs=1, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None), seq_data={0: SequenceData(prompt_token_ids=array('l', [2732, 9845, 3264, 4859, 9225, 7891, 4373, 5874, 6744, 3468, 705, 2599, 2222, 7768, 2897, 9893, 537, 6216, 6921, 6036, 2163, 5072, 4851, 7877, 2046, 1871, 7599, 2496, 8291, 755, 797, 659, 3219, 8615, 7456, 3337, 2745, 4735, 8736, 6687, 714, 2292, 8343, 1207, 6172, 8994, 7221, 6021, 3622, 3560, 8948, 1641, 4984, 4353, 8622, 7250, 4187, 2659, 9781, 2956, 2251, 4420, 7108, 1071, 5251, 7012, 9396, 3918, 9359, 1684, 7098, 2957, 4469, 8752, 9797, 5795, 1472, 7263, 7365, 8448, 6001, 3762, 9008, 2435, 1634, 973, 4464, 8393, 2418, 3455, 6167, 5819, 6521, 6242, 7742, 9123, 6738, 2787, 7316, 4305, 2610, 5531, 6926, 7204, 6922, 4182, 307, 5302, 1152, 6950, 8467, 5294, 1208, 2492, 8829, 770, 8286, 5995, 2344, 3091, 3912, 1434, 6594, 5368, 8372, 7148, 7997, 3854, 8032, 8131, 4845, 5116, 3533, 2937, 9837, 4939, 9744, 3224, 5021, 1134, 25, 9680, 956, 1913, 2934, 9661, 2721, 928, 5627, 6265, 5446, 469, 8717, 1863, 1720, 5272, 591, 6185, 2322, 207, 4262, 3421, 5249, 8408, 8216, 5103, 7939, 2282, 1740, 6118, 5846, 3781, 2775, 2603, 7179, 6356, 1162, 623, 8962, 4051, 1241, 9013, 4403, 1198, 2997, 5661, 807, 2121, 8067, 3886, 8922, 6066, 9987, 1823, 199, 1447, 5181, 5208, 6177, 4863, 6180, 1792, 1483, 8389, 894, 5374, 136, 6273, 9584, 3419, 168, 6004, 2852, 9753, 4419, 8039, 8700, 3186, 5918, 5149, 1777, 3361, 8338, 5393, 4317, 4605, 2562, 6213, 9100, 4652, 6235, 423, 6854, 967, 4370, 9052, 6187, 5203, 433, 6237, 1429, 2546, 329, 3612, 8401, 6761, 3968, 8150, 1040, 6250, 8356, 8798, 7704, 6772, 5311, 9411, 16748, 701, 29875]), output_token_ids=array('l'), cumulative_logprob=0.0)}, seq_len=259, query_len=3, generator=None, is_prompt=True, prompt_logprob_indices=[0, 1, 2], sample_indices=[]), SequenceGroupToSample(seq_ids=[2], sampling_params=SamplingParams(n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0, seed=None, use_beam_search=False, length_penalty=1.0, early_stopping=False, stop=[], stop_token_ids=[], include_stop_str_in_output=False, ignore_eos=True, max_tokens=128, min_tokens=0, logprobs=None, prompt_logprobs=1, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None), seq_data={2: SequenceData(prompt_token_ids=array('l', [9523, 9144, 6011, 2798, 8352, 2195, 4680, 6599, 9303, 3085, 5713, 5240, 732, 5028, 8473, 7594, 4566, 9500, 7444, 3396, 5347, 7034, 595, 647, 573, 6797, 5637, 8448, 5259, 9220, 6567, 4444, 2989, 586, 5102, 7601, 739, 4882, 5410, 437, 3898, 1847, 9724, 1020, 6930, 941, 8641, 5610, 9008, 2107, 9882, 4259, 945, 8393, 7468, 1805, 1862, 8742, 3751, 9864, 2040, 903, 8696, 8015, 5896, 7942, 7377, 9671, 5593, 3128, 7026, 3821, 2711, 8472, 1028, 2660, 2353, 5662, 7734, 8345, 7521, 1053, 2977, 5491, 3893, 2679, 4950, 2665, 3057, 6838, 3968, 851, 9592, 5028, 3793, 7316, 8053, 7152, 3331, 8318, 5930, 8769, 5652, 804, 5444, 3024, 112, 1967, 650, 4333, 1384, 63, 3999, 3988, 2502, 3516, 2671, 2387, 5394, 3441, 8010, 1963, 5763, 2956, 7396, 3898, 3969, 7296, 4903, 8890, 292, 9029, 4099, 5346, 7033, 4776, 7452, 6980, 4122, 736, 4461, 1971, 8389, 1671, 606, 2120, 6996, 9351, 1731, 7788, 3395, 6246, 8020, 8787, 5343, 2304, 3419, 1131, 2003, 7644, 1707, 9774, 8192, 7528, 691, 2547, 2683, 8535, 6995, 6862, 6176, 6598, 5985, 4524, 827, 6834, 3204, 93, 2467, 3778, 404, 5037, 9401, 375, 3945, 497, 7666, 7373, 9630, 8930, 4515, 6729, 3290, 1562, 8652, 3123, 1838, 9660, 6959, 4736, 3466, 4043, 6029, 4702, 5638, 7853, 5534, 6310, 2987, 4690, 3292, 2881, 5801, 7282, 8526, 8933, 9435, 8292, 2463, 7676, 8366, 8797, 7794, 3745, 4876, 3808, 9961, 9040, 9282, 5576, 2173, 9354, 4720, 6874, 1179, 8888, 7288, 2496, 2757, 7458, 4047, 2051, 6844, 3310, 7845, 1747, 7828, 9094, 3868, 4723, 4998, 4930, 604, 8156, 3686, 9061, 870, 608, 29874]), output_token_ids=array('l'), cumulative_logprob=0.0)}, seq_len=259, query_len=3, generator=None, is_prompt=True, prompt_logprob_indices=[3, 4, 5], sample_indices=[]), SequenceGroupToSample(seq_ids=[4], sampling_params=SamplingParams(n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0, seed=None, use_beam_search=False, length_penalty=1.0, early_stopping=False, stop=[], stop_token_ids=[], include_stop_str_in_output=False, ignore_eos=True, max_tokens=128, min_tokens=0, logprobs=None, prompt_logprobs=1, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None), seq_data={4: SequenceData(prompt_token_ids=array('l', [3451, 3781, 9545, 100, 4790, 9037, 6037, 5627, 8863, 3665, 3107, 8429, 6910, 9497, 21, 6573, 1253, 6102, 8592, 9198, 3191, 9893, 8063, 1734, 6540, 3418, 8778, 5046, 7246, 9022, 9800, 3205, 5290, 4547, 6282, 4850, 1337, 3547, 387, 5245, 3922, 1221, 1924, 7185, 8901, 8639, 350, 8856, 3715, 8616, 4260, 7738, 9393, 3511, 673, 1938, 8033, 8945, 7303, 1973, 8529, 5277, 7970, 115, 5719, 8043, 8169, 5696, 3404, 4242, 2820, 1799, 2691, 9264, 6437, 9709, 9776, 6253, 4078, 5405, 4611, 8266, 6634, 6007, 3604, 3280, 5162, 5618, 28, 1434, 2903, 3252, 6448, 9830, 8969, 7426, 9077, 612, 4186, 9284, 8809, 9738, 4108, 5736, 4263, 9120, 9594, 9114, 99, 7385, 2354, 5908, 1608, 5394, 9112, 7719, 284, 0, 4803, 5851, 2963, 9378, 2098, 4966, 2068, 6827, 5604, 6509, 7874, 8417, 7820, 1807, 3190, 1932, 3973, 266, 5382, 8806, 9557, 2418, 6834, 7500, 4765, 5641, 8819, 9160, 4485, 6772, 8373, 2283, 4289, 6732, 120, 8717, 222, 3676, 4332, 3632, 9768, 9114, 7326, 7557, 2544, 6432, 62, 6490, 2023, 8342, 8595, 2437, 2852, 127, 1508, 9989, 2508, 7322, 9299, 2022, 9351, 3252, 592, 6557, 1024, 3799, 4669, 2371, 9673, 7764, 116, 8933, 7935, 824, 7096, 6704, 5265, 8121, 3593, 5310, 8934, 7374, 8609, 2276, 1863, 7486, 8199, 6653, 8982, 1890, 9974, 1616, 9371, 4591, 626, 6103, 5966, 8990, 2138, 6819, 7239, 7021, 4517, 8629, 6245, 7008, 1745, 8292, 1409, 765, 5277, 4997, 8144, 4326, 6056, 2710, 3102, 6675, 9988, 4756, 7018, 2634, 123, 6460, 1629, 5672, 9425, 1320, 2538, 9694, 2589, 7774, 7406, 4478, 7011, 7335, 7837, 8700, 23077, 325, 30767]), output_token_ids=array('l'), cumulative_logprob=0.0)}, seq_len=259, query_len=3, generator=None, is_prompt=True, prompt_logprob_indices=[6, 7, 8], sample_indices=[]), SequenceGroupToSample(seq_ids=[6], sampling_params=SamplingParams(n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0, seed=None, use_beam_search=False, length_penalty=1.0, early_stopping=False, stop=[], stop_token_ids=[], include_stop_str_in_output=False, ignore_eos=True, max_tokens=128, min_tokens=0, logprobs=None, prompt_logprobs=1, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None), seq_data={6: SequenceData(prompt_token_ids=array('l', [3137, 1408, 1111, 3877, 5487, 4825, 1527, 9317, 1321, 3952, 3115, 1134, 3525, 4470, 2963, 6895, 8075, 8071, 4378, 8752, 2742, 4704, 4452, 6743, 9621, 5630, 2562, 7434, 3366, 7078, 8804, 4545, 9845, 571, 3748, 901, 2309, 6182, 6232, 5839, 6514, 9481, 7300, 3096, 9822, 3458, 8019, 3971, 3661, 1291, 8589, 2532, 2129, 9370, 1711, 5802, 8058, 2489, 1169, 4057, 6613, 183, 4207, 5346, 2827, 6241, 2286, 5131, 7137, 5217, 5471, 2349, 2822, 2598, 4147, 9232, 8087, 4867, 1882, 7802, 5533, 645, 5497, 4039, 6927, 6990, 9635, 6324, 5479, 6407, 761, 9139, 3997, 4791, 625, 8587, 5837, 3383, 2555, 4600, 5305, 6758, 7762, 3308, 165, 1415, 1821, 9294, 523, 7228, 8295, 699, 6273, 448, 8146, 3467, 6517, 3087, 131, 213, 3356, 1614, 2927, 9025, 4940, 1291, 9497, 1383, 9050, 4845, 5538, 3070, 656, 2593, 3105, 7340, 1510, 9000, 1130, 595, 4850, 1440, 5015, 324, 5535, 8854, 7232, 7909, 79, 8275, 6927, 5130, 5620, 9833, 9040, 1731, 1360, 5440, 9345, 363, 4562, 2386, 8889, 3478, 7951, 9103, 3399, 4409, 2874, 2317, 6034, 78, 8142, 5908, 5703, 7413, 3163, 6188, 7951, 1623, 6091, 1613, 8702, 7581, 5727, 7812, 5041, 2391, 7888, 7209, 9690, 1776, 9410, 7087, 2321, 1344, 236, 2127, 6218, 1186, 6822, 3062, 5781, 7970, 8565, 2474, 9599, 611, 6905, 3995, 2352, 5503, 1674, 9489, 6915, 7781, 5726, 1659, 7326, 5323, 3549, 7484, 6835, 6388, 243, 2581, 3215, 2800, 6410, 1681, 3047, 797, 7363, 4560, 2274, 8637, 9316, 1655, 2422, 1781, 6437, 1307, 5718, 1631, 3384, 6941, 2015, 5538, 8378, 6015, 3966, 4049, 5596, 7824, 5691, 1004, 1416, 405, 29902]), output_token_ids=array('l'), cumulative_logprob=0.0)}, seq_len=259, query_len=3, generator=None, is_prompt=True, prompt_logprob_indices=[9, 10, 11], sample_indices=[]), SequenceGroupToSample(seq_ids=[8], sampling_params=SamplingParams(n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0, seed=None, use_beam_search=False, length_penalty=1.0, early_stopping=False, stop=[], stop_token_ids=[], include_stop_str_in_output=False, ignore_eos=True, max_tokens=128, min_tokens=0, logprobs=None, prompt_logprobs=1, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None), seq_data={8: SequenceData(prompt_token_ids=array('l', [5127, 3796, 4843, 3293, 1064, 8404, 9325, 4199, 8772, 8686, 7116, 9079, 4080, 2596, 165, 8926, 373, 9710, 7459, 6064, 5681, 3257, 4361, 6194, 9392, 8177, 2547, 5156, 7454, 5746, 7315, 35, 4878, 6173, 60, 7505, 6979, 1057, 4531, 3958, 1435, 3300, 8248, 6421, 3818, 7498, 2589, 957, 9786, 5912, 4210, 6592, 3940, 769, 4354, 1858, 3834, 5580, 3742, 1575, 5709, 2993, 1306, 1164, 3313, 960, 3492, 3946, 6432, 9726, 7452, 2780, 1229, 6703, 9525, 9926, 4745, 3599, 6930, 1181, 3004, 2514, 6786, 2083, 3347, 3984, 3607, 7820, 4451, 7021, 9922, 6932, 2988, 2086, 7210, 7114, 362, 4648, 1903, 1459, 6947, 542, 1799, 7128, 239, 7315, 2893, 6119, 5944, 6117, 7623, 60, 8216, 3436, 7463, 367, 4212, 4112, 6362, 9006, 7392, 2654, 1876, 3863, 3727, 4268, 5239, 1591, 6445, 7138, 3799, 432, 5509, 8569, 4327, 881, 1820, 2116, 5884, 370, 7073, 7651, 9776, 5219, 8153, 1635, 129, 4295, 9316, 2637, 7788, 3250, 1435, 4192, 1579, 3859, 6498, 1325, 8292, 4432, 4985, 7083, 6720, 372, 3168, 1506, 6424, 9218, 6653, 3527, 2251, 243, 374, 4161, 2512, 6514, 5999, 6341, 8993, 5250, 4471, 3983, 4935, 2028, 6941, 1443, 2785, 945, 848, 319, 7640, 9658, 5370, 4669, 3178, 7441, 2093, 5088, 9790, 3094, 7782, 1356, 759, 4496, 3387, 1340, 5785, 5262, 877, 7997, 81, 2246, 4594, 4274, 3670, 6325, 9994, 9630, 5947, 9358, 4728, 1080, 4545, 1896, 8823, 3049, 6432, 2978, 5661, 6195, 7641, 8629, 5312, 3963, 2202, 9684, 2437, 7000, 2321, 2360, 337, 73, 740, 3248, 739, 5390, 4669, 8608, 8604, 4267, 178, 5960, 2469, 8977, 4239, 707, 10619, 931, 869]), output_token_ids=array('l'), cumulative_logprob=0.0)}, seq_len=259, query_len=3, generator=None, is_prompt=True, prompt_logprob_indices=[12, 13, 14], sample_indices=[]), SequenceGroupToSample(seq_ids=[10], sampling_params=SamplingParams(n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0, seed=None, use_beam_search=False, length_penalty=1.0, early_stopping=False, stop=[], stop_token_ids=[], include_stop_str_in_output=False, ignore_eos=True, max_tokens=128, min_tokens=0, logprobs=None, prompt_logprobs=1, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None), seq_data={10: SequenceData(prompt_token_ids=array('l', [9557, 5616, 3165, 5330, 7687, 9015, 5146, 8836, 8281, 3695, 5715, 4754, 2656, 9599, 5612, 8579, 1682, 8397, 5778, 2316, 6767, 8007, 6957, 3792, 2136, 6445, 6527, 6363, 165, 434, 1968, 1228, 9220, 5104, 5043, 1198, 7579, 3406, 4966, 5517, 3588, 4822, 5391, 7197, 9166, 7338, 1227, 7387, 2582, 7278, 1492, 3767, 1154, 7833, 9853, 1664, 3423, 184, 7278, 5074, 7726, 5209, 2765, 1676, 6462, 293, 4051, 4983, 8092, 9576, 7769, 3869, 9268, 3796, 5393, 8832, 1506, 8385, 7213, 1146, 2737, 5630, 1345, 1365, 2482, 9542, 7075, 8676, 2199, 2170, 2130, 5222, 89, 8942, 3185, 8347, 8021, 1537, 2313, 1873, 5184, 2295, 5955, 3577, 8246, 1316, 8211, 1007, 3781, 4820, 6781, 5289, 208, 2704, 2233, 2326, 1416, 4926, 1193, 8294, 9820, 2156, 907, 5026, 91, 1841, 9639, 1097, 1880, 2792, 9441, 4814, 9220, 1716, 3911, 7533, 3333, 8743, 8527, 662, 9433, 5011, 7998, 8989, 9993, 6243, 3794, 7752, 255, 8035, 6770, 6385, 9527, 9738, 5400, 5471, 1677, 5070, 4151, 6149, 8599, 6186, 5735, 1381, 9803, 1909, 7096, 4248, 3409, 6110, 3218, 5465, 1565, 7461, 2993, 5288, 4593, 7391, 4199, 3339, 8504, 8181, 7615, 6286, 9504, 6337, 5241, 373, 833, 6180, 2073, 6216, 2968, 6538, 6663, 948, 2504, 5347, 2257, 1175, 5749, 1205, 4620, 5549, 5551, 8871, 5779, 1258, 1901, 5042, 2987, 594, 1486, 101, 4280, 6834, 986, 8715, 2668, 3211, 6564, 3966, 3249, 4044, 6097, 5343, 4042, 5936, 4682, 5666, 3532, 2801, 7463, 9738, 8962, 9593, 5103, 75, 1966, 3123, 6644, 4090, 3715, 2320, 7949, 3878, 4356, 9107, 4689, 3216, 2430, 8856, 9483, 8342, 3644, 9435, 15514, 2973, 1034]), output_token_ids=array('l'), cumulative_logprob=0.0)}, seq_len=259, query_len=3, generator=None, is_prompt=True, prompt_logprob_indices=[15, 16, 17], sample_indices=[]), SequenceGroupToSample(seq_ids=[12], sampling_params=SamplingParams(n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0, seed=None, use_beam_search=False, length_penalty=1.0, early_stopping=False, stop=[], stop_token_ids=[], include_stop_str_in_output=False, ignore_eos=True, max_tokens=128, min_tokens=0, logprobs=None, prompt_logprobs=1, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None), seq_data={12: SequenceData(prompt_token_ids=array('l', [4738, 5000, 5192, 6438, 1588, 2392, 2463, 8947, 3734, 9514, 6331, 310, 6089, 5913, 5945, 7365, 6489, 301, 1339, 1964, 1178, 3704, 2677, 4729, 8608, 5774, 3399, 1725, 513, 3574, 7548, 2413, 7339, 2522, 9231, 8004, 1226, 6229, 106, 7659, 4352, 9809, 3258, 5339, 5046, 8623, 8982, 3396, 3962, 830, 3876, 8913, 8390, 8454, 3504, 7708, 4173, 2441, 469, 7160, 1188, 8476, 5620, 970, 9730, 2469, 7111, 7255, 1758, 4049, 9731, 111, 6992, 1779, 4690, 4914, 1981, 2538, 1827, 8501, 3441, 5479, 5674, 6684, 8319, 8363, 9299, 4158, 6793, 1243, 4835, 2912, 6175, 8086, 3454, 5945, 7189, 3901, 3881, 4971, 7831, 7421, 2792, 5940, 279, 9338, 3912, 3309, 815, 9871, 6081, 2654, 810, 7934, 512, 2950, 4829, 3139, 4383, 2845, 6160, 4093, 9478, 4329, 4363, 3409, 4989, 9515, 6467, 511, 2679, 4366, 270, 447, 4469, 43, 7526, 5012, 2583, 5410, 3360, 2881, 2837, 2628, 7981, 1000, 4888, 4020, 3720, 5345, 8442, 5898, 2622, 1565, 139, 8519, 3642, 8149, 6471, 8857, 492, 822, 5238, 339, 1401, 490, 7299, 884, 7124, 4081, 851, 2297, 2597, 6588, 876, 9735, 3004, 7162, 4450, 1512, 7335, 7713, 8033, 6345, 6231, 5444, 9222, 8033, 1394, 4863, 2109, 7156, 7355, 3575, 4893, 1263, 9208, 4124, 7320, 4562, 9877, 3334, 1301, 3703, 2970, 8153, 6949, 1400, 7627, 2699, 7478, 9733, 7930, 8061, 7758, 9449, 110, 2668, 1054, 3938, 9987, 128, 1885, 1260, 3967, 2302, 3072, 2461, 1627, 1010, 7155, 9444, 1211, 5854, 9829, 7206, 9935, 6870, 1773, 7697, 1260, 2674, 9922, 8459, 1929, 947, 8964, 3500, 6344, 492, 505, 1253, 362, 2205, 9371, 6801, 6212, 25252, 3012]), output_token_ids=array('l'), cumulative_logprob=0.0)}, seq_len=259, query_len=3, generator=None, is_prompt=True, prompt_logprob_indices=[18, 19, 20], sample_indices=[]), SequenceGroupToSample(seq_ids=[14], sampling_params=SamplingParams(n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0, seed=None, use_beam_search=False, length_penalty=1.0, early_stopping=False, stop=[], stop_token_ids=[], include_stop_str_in_output=False, ignore_eos=True, max_tokens=128, min_tokens=0, logprobs=None, prompt_logprobs=1, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None), seq_data={14: SequenceData(prompt_token_ids=array('l', [395, 3377, 4440, 9731, 1648, 3897, 3506, 131, 8054, 7140, 6076, 4253, 2997, 9784, 6347, 7543, 5199, 4698, 2127, 7393, 5131, 7671, 926, 1604, 8507, 8736, 3877, 7339, 8829, 6429, 250, 9371, 795, 8318, 6216, 5195, 3278, 3530, 1782, 8025, 6176, 9906, 7208, 9414, 6430, 1686, 9339, 3659, 4564, 3012, 9791, 4868, 4980, 9373, 5823, 603, 7379, 8552, 6234, 8777, 6066, 3934, 9927, 7992, 8393, 9948, 2096, 3015, 6773, 3707, 7879, 2366, 2341, 3077, 7777, 55, 7644, 2086, 2312, 9409, 5442, 5182, 5309, 6559, 3466, 4116, 9316, 5902, 3704, 52, 8647, 1925, 3342, 9842, 750, 819, 7200, 8599, 5881, 9051, 4870, 1099, 9047, 2556, 5830, 5156, 5027, 8254, 1477, 442, 4684, 1025, 717, 9818, 5096, 3139, 1897, 6545, 26, 9965, 4621, 8760, 8250, 6364, 8583, 6417, 6852, 5675, 6269, 1372, 3577, 8008, 3770, 7963, 903, 2741, 3707, 4811, 1117, 4529, 4573, 2935, 6584, 8125, 8343, 2745, 5769, 2688, 403, 3122, 3486, 3621, 6072, 1918, 4073, 8144, 5594, 4541, 966, 3275, 9873, 3103, 4767, 4180, 4186, 757, 2226, 1610, 6724, 4223, 13, 9379, 8678, 3467, 359, 4427, 6128, 5202, 2739, 6306, 6074, 80, 540, 8428, 9436, 9881, 3422, 101, 5819, 9595, 8158, 5113, 9402, 3346, 6668, 2945, 4618, 784, 2684, 3846, 6656, 6151, 6211, 1851, 3832, 6516, 6564, 3797, 1782, 3272, 3238, 1962, 3041, 7844, 6212, 8973, 328, 6790, 9566, 9356, 8802, 1923, 7696, 2309, 3624, 9655, 1717, 8470, 2583, 8443, 3962, 9124, 7915, 7110, 6534, 344, 9804, 5137, 9776, 3877, 3530, 6769, 7172, 7775, 9148, 1663, 6666, 4893, 5993, 1264, 1549, 4371, 41, 7596, 9323, 3576, 3964, 262, 14424]), output_token_ids=array('l'), cumulative_logprob=0.0)}, seq_len=259, query_len=3, generator=None, is_prompt=True, prompt_logprob_indices=[21, 22, 23], sample_indices=[])], selected_token_indices=tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23], device='cuda:0'), categorized_sample_indices={<SamplingType.GREEDY: 0>: tensor([], device='cuda:0', size=(0, 2), dtype=torch.int32), <SamplingType.RANDOM: 1>: tensor([], device='cuda:0', size=(0, 2), dtype=torch.int32), <SamplingType.RANDOM_SEED: 2>: tensor([], device='cuda:0', size=(0, 2), dtype=torch.int32), <SamplingType.BEAM: 3>: tensor([], device='cuda:0', size=(0, 2), dtype=torch.int32)}), 

hidden_or_intermediate_states.shape
torch.Size([24, 8192])
hidden_or_intermediate_states.min
<built-in method min of Tensor object at 0x7f9ef020cc20>
hidden_or_intermediate_states.min()
tensor(-33.0938, device='cuda:0', dtype=torch.float16)
hidden_or_intermediate_states.max()
tensor(21.5938, device='cuda:0', dtype=torch.float16)
hidden_or_intermediate_states.mean()
tensor(0.0033, device='cuda:0', dtype=torch.float16)
hidden_or_intermediate_states[:,0]
tensor([ 0.2847,  0.5605,  0.9561, -1.1221,  0.3691,  0.0518,  0.9775, -1.1875,
        -0.9722, -0.3582,  1.1602, -0.1100,  0.4370,  0.7607, -1.1934,  0.3115,
        -0.1843,  0.9131,  0.4333,  0.2791,  2.5039,  1.8047, -0.7188,  0.0719],
       device='cuda:0', dtype=torch.float16)

no cuda graph

       ------------------
       lashAttentionMetadata(num_prefills=8, num_prefill_tokens=24, num_decode_tokens=0, slot_mapping=tensor([2048, 2049, 2050, 2064, 2065, 2066, 2080, 2081, 2082, 2096, 2097, 2098,
        2112, 2113, 2114, 2128, 2129, 2130, 2144, 2145, 2146, 2160, 2161, 2162],
       device='cuda:0'), seq_lens=[259, 259, 259, 259, 259, 259, 259, 259], seq_lens_tensor=tensor([259, 259, 259, 259, 259, 259, 259, 259], device='cuda:0',
       dtype=torch.int32), max_query_len=3, max_prefill_seq_len=259, max_decode_seq_len=0, query_start_loc=tensor([ 0,  3,  6,  9, 12, 15, 18, 21, 24], device='cuda:0',
       dtype=torch.int32), seq_start_loc=tensor([   0,  259,  518,  777, 1036, 1295, 1554, 1813, 2072], device='cuda:0',
       dtype=torch.int32), context_lens_tensor=tensor([256, 256, 256, 256, 256, 256, 256, 256], device='cuda:0',
       dtype=torch.int32), block_tables=tensor([[  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
          14,  15, 128],
        [ 16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
          30,  31, 129],
        [ 32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,
          46,  47, 130],
        [ 48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,
          62,  63, 131],
        [ 64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
          78,  79, 132],
        [ 80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,
          94,  95, 133],
        [ 96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
         110, 111, 134],
        [112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
         126, 127, 135]], device='cuda:0', dtype=torch.int32), use_cuda_graph=False, _cached_prefill_metadata=FlashAttentionMetadata(num_prefills=8, num_prefill_tokens=24, num_decode_tokens=0, slot_mapping=tensor([2048, 2049, 2050, 2064, 2065, 2066, 2080, 2081, 2082, 2096, 2097, 2098,
        2112, 2113, 2114, 2128, 2129, 2130, 2144, 2145, 2146, 2160, 2161, 2162],
       device='cuda:0'), seq_lens=[259, 259, 259, 259, 259, 259, 259, 259], seq_lens_tensor=tensor([259, 259, 259, 259, 259, 259, 259, 259], device='cuda:0',
       dtype=torch.int32), max_query_len=3, max_prefill_seq_len=259, max_decode_seq_len=0, query_start_loc=tensor([ 0,  3,  6,  9, 12, 15, 18, 21, 24], device='cuda:0',
       dtype=torch.int32), seq_start_loc=tensor([   0,  259,  518,  777, 1036, 1295, 1554, 1813, 2072], device='cuda:0',
       dtype=torch.int32), context_lens_tensor=tensor([256, 256, 256, 256, 256, 256, 256, 256], device='cuda:0',
       dtype=torch.int32), block_tables=tensor([[  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
          14,  15, 128],
        [ 16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,
          30,  31, 129],
        [ 32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,
          46,  47, 130],
        [ 48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,
          62,  63, 131],
        [ 64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
          78,  79, 132],
        [ 80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,
          94,  95, 133],
        [ 96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
         110, 111, 134],
        [112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
         126, 127, 135]], device='cuda:0', dtype=torch.int32), use_cuda_graph=False, _cached_prefill_metadata=None, _cached_decode_metadata=None), _cached_decode_metadata=None)


         model_input.attn_metadata.block_tables.shape
        torch.Size([8, 17])


        hidden_or_intermediate_states[:,0]
tensor([ 0.2847,  0.5605,  0.9561,  0.7959,  0.8242,  0.5957, -0.8281, -1.7900,
        -1.0938,  0.2534, -0.0207,  0.7388, -1.1855, -0.1301, -0.5669, -0.4536,
         0.1995, -1.4121, -1.0781, -0.0108,  0.8335,  1.9668, -0.0807,  0.3955],
       device='cuda:0', dtype=torch.float16)