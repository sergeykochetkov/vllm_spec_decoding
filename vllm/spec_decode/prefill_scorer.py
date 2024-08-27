from copy import deepcopy
from itertools import chain, count
from typing import Iterator, List, Tuple

import torch

from vllm.sequence import (
    ExecuteModelRequest,
    SamplerOutput,
    SequenceData,
    SequenceGroupMetadata,
    SequenceGroupState,
    SequenceStage,
    get_all_seq_ids,
)
from vllm.spec_decode.interfaces import (
    SpeculativeProposals,
    SpeculativeScorer,
    SpeculativeScores,
)
from vllm.spec_decode.util import (
    Timer,
    nvtx_range,
    sampler_output_to_torch,
    split_batch_by_proposal_len,
)
from vllm.worker.worker_base import WorkerBase

SeqId = int
TargetSeqId = int
TokenId = int


class PrefillTop1Scorer(SpeculativeScorer):
    """Implements a speculative scorer that uses batch expansion to get
    probabilities of speculative tokens according to the scoring model.

    Batch expansion converts a list of sequences and multiple query positions
    to a new batch of sequences, each with a single query position. This allows
    for MQA-like scoring in speculative decoding without requiring an MQA
    kernel.

    It is strictly less efficient than MQA scoring.

    It only supports scoring the top1 proposal tokens of the proposer, instead
    of topk/tree.
    """

    def __init__(self, scorer_worker: WorkerBase, device: str, vocab_size: int):
        self._scorer_worker = scorer_worker
        self._device = device
        self._vocab_size = vocab_size
        
    @nvtx_range("PrefillTop1Scorer.score_proposals")
    def score_proposals(
        self,
        execute_model_req: ExecuteModelRequest,
        proposals: SpeculativeProposals,
    ) -> SpeculativeScores:
        """Score the proposed tokens via the scorer model.

        This converts each input sequence to a set of k+1 target sequences. The
        target sequences have the unique continuations to be scored and a
        unique sequence ID that is different from all input sequence ids.

        If a speculative sequence length would exceed the max model length, then
        no speculation is produced for that sequence.

        Args:
            execute_model_req: The execution request.
            proposals: The speculative proposals to score.
        Returns:
            SpeculativeScores: The scores of each speculative token, along with
                which sequences were ignored during scoring.
        """
        
        execute_model_req_prefill = self._prepare_inputs(execute_model_req, proposals)

        target_sampler_output = self._scorer_worker.execute_model(
            execute_model_req=execute_model_req_prefill
        )

        output=self._prepare_outputs(proposals, target_sampler_output)
    
        return output
    

    def _prepare_outputs(self,  proposals: SpeculativeProposals, target_sampler_output)->SpeculativeScores:
        batch_size, num_spec_tokens=proposals.proposal_token_ids.shape

        assert len(target_sampler_output) == 1, "expected single-step output"
        target_sampler_output = target_sampler_output[0]

        all_probs = target_sampler_output.sampled_token_probs.reshape(batch_size, 1+num_spec_tokens,-1)

        all_tokens = all_probs.argmax(dim=-1)

        #batch_indices = torch.arange(batch_size).unsqueeze(1).expand_as(all_tokens)
        
        #all_probs[batch_indices, torch.arange(all_tokens.size(1)), all_tokens] = 1.0
        for b in range(batch_size):
            for token_position in range(len(all_tokens[b])):
                for t in all_tokens:
                    all_probs[b,token_position,t]=1.0

        spec_logprobs = torch.log(all_probs)

        return SpeculativeScores(
            probs=all_probs,
            token_ids=all_tokens,
            logprobs=spec_logprobs,
            hidden_states=target_sampler_output.hidden_states,
        )


    def _prepare_inputs(self,execute_model_req: ExecuteModelRequest, proposals: SpeculativeProposals)->ExecuteModelRequest:

        # TODO(cade) perform this on GPU to remove blocking call.
        proposal_lens_list = proposals.proposal_lens.tolist()
        proposal_token_ids_list = proposals.proposal_token_ids.tolist()
        
        num_spec_tokens=proposal_lens_list[0]
        for l in proposal_lens_list:
            assert l==num_spec_tokens, "only equal num of spec tokens in batch is supported"

        # Filter the list to ignore -1 proposals.
        proposal_token_ids_list_without_skips = [
            proposals for proposals in proposal_token_ids_list if -1 not in proposals
        ]

        prefill_seq_group_metadata_list = []
        for i, seq_group_metadata in enumerate(
            execute_model_req.seq_group_metadata_list
        ):
            _seq_group_metadata = deepcopy(seq_group_metadata)
            _seq_data = {}
            max_input_seq_id=max(_seq_group_metadata.seq_data.keys())
            for k, v in _seq_group_metadata.seq_data.items():
                prompt_token_ids = list(v.prompt_token_ids)

                prompt_token_ids.extend(v.output_token_ids)

                # we need to compute the last output token ?
                num_computed_tokens = len(prompt_token_ids) - 1
                #num_computed_tokens = v.get_num_computed_tokens() # Why this does not work?

                prompt_token_ids.extend(proposal_token_ids_list_without_skips[i])
                v.prompt_token_ids = prompt_token_ids
                v.output_token_ids = []
                
                v.reset_state_for_recompute()
                v.update_num_computed_tokens(num_computed_tokens)

                _seq_data[k+max_input_seq_id] = v
                

            _seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group_metadata.request_id,
                is_prompt=True,
                seq_data=_seq_data,
                sampling_params=seq_group_metadata.sampling_params,
                block_tables={k+max_input_seq_id:v for k,v in seq_group_metadata.block_tables.items()},
                do_sample=False,
                pooling_params=seq_group_metadata.pooling_params,
                token_chunk_size=len(prompt_token_ids) - num_computed_tokens,
            )

            _seq_group_metadata.sampling_params.prompt_logprobs = 1

            prefill_seq_group_metadata_list.append(_seq_group_metadata)

        execute_model_req_ = execute_model_req.clone(
            seq_group_metadata_list=prefill_seq_group_metadata_list
        )

        return execute_model_req_