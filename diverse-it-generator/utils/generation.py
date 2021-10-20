'''
refer to https://github.com/huggingface/transformers/blob/
b020a736c374460af1b34267283f957988350630/src/transformers/generation_utils.py
https://huggingface.co/blog/how-to-generate
'''

import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import torch
import torch.nn.functional as F


from typing import Callable, List

from utils.generation_beam_search import BeamSearchScorer
from utils.generation_logits_process import (
    HammingDiversityLogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


def _get_logits_warper(
        top_k: int = None,
        top_p: float = None,
        temperature: float = None,
        num_beams: int = None) -> LogitsProcessorList:
    """
    This class returns a :obj:`~transformers.LogitsProcessorList` list object that contains all relevant
    :obj:`~transformers.LogitsWarper` instances used for multinomial sampling.
    """
    # instantiate warpers list
    warpers = LogitsProcessorList()

    # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
    # all samplers can be found in `generation_utils_samplers.py`
    if temperature is not None and temperature != 1.0:
        warpers.append(TemperatureLogitsWarper(temperature))
    if top_k is not None and top_k != 0:
        warpers.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=(2 if num_beams > 1 else 1)))
    if top_p is not None and top_p < 1.0:
        warpers.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=(2 if num_beams > 1 else 1)))
    return warpers


def _get_logits_processor(
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        bad_words_ids: List[List[int]],
        min_length: int,
        eos_token_id: int,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        num_beams: int,
        num_beam_groups: int,
        diversity_penalty: float,
) -> LogitsProcessorList:
    """
    This class returns a :obj:`~transformers.LogitsProcessorList` list object that contains all relevant
    :obj:`~transformers.LogitsProcessor` instances used to modify the scores of the language model head.
    """
    # instantiate processors list
    processors = LogitsProcessorList()

    # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
    # all samplers can be found in `generation_utils_samplers.py`
    if diversity_penalty is not None and diversity_penalty > 0.0:
        processors.append(
            HammingDiversityLogitsProcessor(
                diversity_penalty=diversity_penalty, num_beams=num_beams, num_beam_groups=num_beam_groups
            )
        )
    if repetition_penalty is not None and repetition_penalty != 1.0:
        processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
    if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
        processors.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
    if bad_words_ids is not None:
        processors.append(NoBadWordsLogitsProcessor(bad_words_ids, eos_token_id))
    if min_length is not None and eos_token_id is not None and min_length > -1:
        processors.append(MinLengthLogitsProcessor(min_length, eos_token_id))
    if prefix_allowed_tokens_fn is not None:
        processors.append(PrefixConstrainedLogitsProcessor(prefix_allowed_tokens_fn, num_beams))
    return processors


def expand_tensor(inputs, expand_size):
    '''
    can also refer to the implementation of _expand_inputs_for_generation
    https://github.com/huggingface/transformers/blob/b020a736c374460af1b34267283f957988350630/
    src/transformers/generation_utils.py#L424
    :param inputs:
    :param expand_size:
    :return:
    '''
    res = []
    for inp in inputs:
        if inp is None:
            res.append(inp)
        elif len(inp.shape) == 2:
            res.append(inp.unsqueeze(1).expand(-1, expand_size, -1).reshape(-1, inp.shape[-1]))
        elif len(inp.shape) == 3:
            res.append(inp.unsqueeze(1).expand(-1, expand_size, -1, -1)
                       .reshape(-1, inp.shape[-2], inp.shape[-1]))
    return res


def custom_generate(config, next_token_scores, input_ids=None,
                    generator=None, high=None, num_regions=None, cur_start=None,
                    grid_features=None, masked_token_ids=None, token_type_ids=None,
                    position_ids=None, attention_mask=None,
                    eos_token_id=None, pad_token_id=None, cluster_index=None):
    # set init values
    num_beams = config.num_beams
    num_beam_groups = config.num_beam_groups
    max_length = config.max_length
    do_sample = config.do_sample
    num_return_sequences = config.num_return_sequences
    repetition_penalty = config.repetition_penalty
    no_repeat_ngram_size = config.no_repeat_ngram_size
    bad_words_ids = config.bad_words_ids
    min_length = config.min_length
    prefix_allowed_tokens_fn = config.prefix_allowed_tokens_fn
    diversity_penalty = config.diversity_penalty

    # get distribution pre_processing samplers
    logits_processor = _get_logits_processor(
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        bad_words_ids=bad_words_ids,
        min_length=min_length,
        eos_token_id=eos_token_id,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        num_beams=num_beams,
        num_beam_groups=num_beam_groups,
        diversity_penalty=diversity_penalty,
    )

    # determine generation mode
    is_greedy_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is False
    is_sample_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is True
    is_beam_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is False
    is_beam_sample_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is True
    # Do not support is_group_beam_gen_mode
    # is_group_beam_gen_mode = (num_beams > 1) and (num_beam_groups > 1)
    # if num_beam_groups > num_beams:
    #     raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
    # if is_group_beam_gen_mode and do_sample is True:
    #     raise ValueError(
    #         "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
    #     )

    if is_greedy_gen_mode:
        if num_return_sequences > 1:
            raise ValueError(
                f"num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search."
            )

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_scores)

        # greedy search
        # next_tokens = torch.argmax(next_token_scores, dim=-1)
        probs = F.softmax(next_token_scores, dim=-1)
        pred_token_probs, pred_token_ids = probs.max(dim=-1)
        return pred_token_probs, pred_token_ids

    elif is_sample_gen_mode:
        # get probability distribution warper
        logits_warper = _get_logits_warper(
            top_k=config.top_k, top_p=config.top_p,
            temperature=config.temperature, num_beams=config.num_beams
        )

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_scores)
        next_token_scores = logits_warper(input_ids, next_token_scores)
        # sample
        probs = F.softmax(next_token_scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        pred_token_probs = probs[range(len(next_tokens)), next_tokens]
        return pred_token_probs, next_tokens

    elif is_beam_gen_mode:
        # Generates sequences for models with a language modeling head using beam search decoding.
        batch_size = grid_features.shape[0]

        length_penalty = config.length_penalty
        early_stopping = config.early_stopping

        if num_return_sequences > num_beams:
            raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            max_length=max_length,
            num_beams=num_beams,
            device=grid_features.device,
            length_penalty=length_penalty,
            do_early_stopping=early_stopping,
            num_beam_hyps_to_keep=num_return_sequences,
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        ori_masked_token_ids = masked_token_ids

        [grid_features, masked_token_ids, token_type_ids, position_ids, attention_mask] = expand_tensor(
            [grid_features, masked_token_ids, token_type_ids, position_ids, attention_mask],
            num_beams
        )

        batch_beam_size, cur_len = masked_token_ids.shape

        assert (
                num_beams * batch_size == batch_beam_size
        ), "Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=grid_features.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        for step in range(high):
            attention_mask[:, num_regions + cur_start + step] = 1  # attend to tokens before current step
            pred_scores = generator(
                grid_features, masked_token_ids, token_type_ids,
                position_ids, attention_mask,
                visual_mask=None, cluster_index=cluster_index)

            pred_scores = pred_scores[0]

            next_token_scores = pred_scores[:, num_regions + cur_start + step + 1, :]
            input_ids = masked_token_ids[:, cur_start:cur_start + step + 1]
            if config.consider_generated_sentence:
                logits_input_ids = masked_token_ids[:, :cur_start + step + 1]
            else:
                logits_input_ids = masked_token_ids[:, cur_start:cur_start + step + 1]

            next_token_scores = F.log_softmax(next_token_scores, dim=-1)  # (batch_size * num_beams, vocab_size)

            next_token_scores = logits_processor(logits_input_ids, next_token_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # ------------------------------------------------------------------
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )
            # ------------------------------------------------------------------

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            masked_token_ids[:, cur_start:cur_start + step + 2] = input_ids

            if beam_scorer.is_done:
                break

        sequence_outputs = beam_scorer.finalize(
            input_ids, beam_scores, next_tokens, next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id
        )
        if config.print_beams:
            from .tokenizer import get_tokenizer
            tokenizer, SEP, EOS, MASK, PAD, num_tokens = get_tokenizer(config)
            beam_scorer.print_beams(tokenizer, EOS)  # not exactly match with sequence_outputs
        res = sequence_outputs['sequences'][:, 1:high+1]
        ori_masked_token_ids[range(batch_size), cur_start:cur_start+res.shape[-1]] = res
        return ori_masked_token_ids

    elif is_beam_sample_gen_mode:
        # Generates sequences for models with a language modeling head using beam search with multinomial sampling.
        logits_warper = _get_logits_warper(
            top_k=config.top_k, top_p=config.top_p,
            temperature=config.temperature, num_beams=config.num_beams
        )

        batch_size = grid_features.shape[0] * num_return_sequences

        length_penalty = config.length_penalty
        early_stopping = config.early_stopping

        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            max_length=max_length,
            num_beams=num_beams,
            device=grid_features.device,
            length_penalty=length_penalty,
            do_early_stopping=early_stopping,
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        ori_masked_token_ids = masked_token_ids

        # interleave with `num_beams * num_return_sequences`
        [grid_features, masked_token_ids, token_type_ids, position_ids, attention_mask] = expand_tensor(
            [grid_features, masked_token_ids, token_type_ids, position_ids, attention_mask],
            num_beams * num_return_sequences
        )

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=grid_features.device)
        beam_scores = beam_scores.view((batch_size * num_beams,))

        for step in range(high):
            attention_mask[:, num_regions + cur_start + step] = 1  # attend to tokens before current step
            pred_scores = generator(
                grid_features, masked_token_ids, token_type_ids,
                position_ids, attention_mask,
                visual_mask=None, cluster_index=cluster_index)

            pred_scores = pred_scores[0]

            next_token_scores = pred_scores[:, num_regions + cur_start + step + 1, :]
            input_ids = masked_token_ids[:, cur_start:cur_start + step + 1]
            if config.consider_generated_sentence:
                logits_input_ids = masked_token_ids[:, :cur_start + step + 1]
            else:
                logits_input_ids = masked_token_ids[:, cur_start:cur_start + step + 1]

            next_token_scores = F.log_softmax(next_token_scores, dim=-1)  # (batch_size * num_beams, vocab_size)

            next_token_scores = logits_processor(logits_input_ids, next_token_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
            next_token_scores = logits_warper(input_ids, next_token_scores)  # only for sample method

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # ------------------------------------------------------------------
            probs = F.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
            next_token_scores = torch.gather(next_token_scores, -1, next_tokens)

            next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
            next_tokens = torch.gather(next_tokens, -1, _indices)
            # ------------------------------------------------------------------

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            masked_token_ids[:, cur_start:cur_start + step + 2] = input_ids

            if beam_scorer.is_done:
                break

        sequence_outputs = beam_scorer.finalize(
            input_ids, beam_scores, next_tokens, next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id
        )
        if config.print_beams:
            beam_scorer.print_beams()  # not exactly match with sequence_outputs
        res = sequence_outputs['sequences'][:, 1:high+1]
        ori_masked_token_ids[range(batch_size), cur_start:cur_start+res.shape[-1]] = res
        return ori_masked_token_ids

    # elif is_group_beam_gen_mode:
    #     # Generates sequences for models with a language modeling head using beam search decoding.
    #     batch_size = region_features.shape[0]
    #
    #     length_penalty = config.length_penalty
    #     early_stopping = config.early_stopping
    #
    #     if num_return_sequences > num_beams:
    #         raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")
    #
    #     if num_beams % num_beam_groups != 0:
    #         raise ValueError("`num_beams` should be divisible by `num_beam_groups` for group beam search.")
    #
    #     beam_scorer = BeamSearchScorer(
    #         batch_size=batch_size,
    #         max_length=max_length,
    #         num_beams=num_beams,
    #         device=region_features.device,
    #         length_penalty=length_penalty,
    #         do_early_stopping=early_stopping,
    #         num_beam_hyps_to_keep=num_return_sequences,
    #         num_beam_groups=num_beam_groups,
    #     )
    #
    #     batch_size = len(beam_scorer._beam_hyps)
    #     num_beams = beam_scorer.num_beams
    #     num_beam_groups = beam_scorer.num_beam_groups
    #     num_sub_beams = num_beams // num_beam_groups
    #
    #     ori_masked_token_ids = masked_token_ids
    #
    #     # interleave with `num_beams * num_return_sequences`
    #     [region_features, masked_token_ids, token_type_ids, position_ids, attention_mask] = expand_tensor(
    #         [region_features, masked_token_ids, token_type_ids, position_ids, attention_mask],
    #         num_beams * num_return_sequences
    #     )
    #
    #     batch_beam_size, cur_len = masked_token_ids.shape
    #
    #     assert (
    #         num_beams * batch_size == batch_beam_size
    #     ), f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
    #
    #     beam_scores = torch.full((batch_size, num_beams), -1e9, dtype=torch.float, device=region_features.device)
    #     # initialise score of first beam of each group with 0 and the rest with 1e-9. This ensures that the beams in
    #     # the same group don't produce same tokens everytime.
    #     beam_scores[:, ::num_sub_beams] = 0
    #     beam_scores = beam_scores.view((batch_size * num_beams,))
    #
    #     for step in range(high):
    #
    #         # predicted tokens in cur_len step
    #         current_tokens = torch.zeros(batch_size * num_beams,
    #                                      dtype=masked_token_ids.dtype, device=region_features.device)
    #
    #         # indices which will form the beams in the next time step
    #         reordering_indices = torch.zeros(batch_size * num_beams, dtype=torch.long, device=region_features.device)
    #
    #         attention_mask[:, num_regions + cur_start + step] = 1  # attend to tokens before current step
    #         pred_scores = generator(
    #             region_features, masked_token_ids, token_type_ids,
    #             position_ids, attention_mask,
    #             visual_mask=None, cluster_index=cluster_index)
    #
    #         pred_scores = pred_scores[0]
    #
    #         next_token_scores = pred_scores[:, num_regions + cur_start + step + 1, :]
    #         input_ids = masked_token_ids[:, cur_start:cur_start + step + 1]
    #
    #         ori_next_token_scores = next_token_scores.clone()
    #
    #         for beam_group_idx in range(num_beam_groups):
    #             group_start_idx = beam_group_idx * num_sub_beams
    #             group_end_idx = min(group_start_idx + num_sub_beams, num_beams)
    #             group_size = group_end_idx - group_start_idx
    #
    #             # indices of beams of current group among all sentences in batch
    #             batch_group_indices = []
    #
    #             for batch_idx in range(batch_size):
    #                 batch_group_indices.extend(
    #                     [batch_idx * num_beams + idx for idx in range(group_start_idx, group_end_idx)]
    #                 )
    #             group_input_ids = input_ids[batch_group_indices]
    #
    #             # select outputs of beams of current group only
    #             # next_token_logits = outputs.logits[batch_group_indices, -1, :]
    #             next_token_scores = ori_next_token_scores[batch_group_indices]
    #
    #             next_token_scores = F.log_softmax(next_token_scores, dim=-1)  # (batch_size * group_size, vocab_size)
    #             vocab_size = next_token_scores.shape[-1]
    #
    #             next_token_scores = logits_processor(
    #                 group_input_ids, next_token_scores, current_tokens=current_tokens, beam_group_idx=beam_group_idx
    #             )
    #             next_token_scores = next_token_scores + \
    #                                 beam_scores[batch_group_indices].unsqueeze(-1).expand_as(next_token_scores)
    #
    #             # reshape for beam search
    #             next_token_scores = next_token_scores.view(batch_size, group_size * vocab_size)
    #
    #             next_token_scores, next_tokens = torch.topk(
    #                 next_token_scores, 2 * group_size, dim=1, largest=True, sorted=True
    #             )
    #
    #             next_indices = next_tokens // vocab_size
    #             next_tokens = next_tokens % vocab_size
    #
    #             # stateless
    #             beam_outputs = beam_scorer.process(
    #                 group_input_ids,
    #                 next_token_scores,
    #                 next_tokens,
    #                 next_indices,
    #                 pad_token_id=pad_token_id,
    #                 eos_token_id=eos_token_id,
    #             )
    #             beam_scores[batch_group_indices] = beam_outputs["next_beam_scores"]
    #             beam_next_tokens = beam_outputs["next_beam_tokens"]
    #             beam_idx = beam_outputs["next_beam_indices"]
    #
    #             input_ids[batch_group_indices] = group_input_ids[beam_idx]
    #             group_input_ids = torch.cat([group_input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
    #             current_tokens[batch_group_indices] = group_input_ids[:, -1]
    #
    #             # (beam_idx // group_size) -> batch_idx
    #             # (beam_idx % group_size) -> offset of idx inside the group
    #             reordering_indices[batch_group_indices] = (
    #                     num_beams * (beam_idx // group_size) + group_start_idx + (beam_idx % group_size)
    #             )
    #
    #         # input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
    #         input_ids = torch.cat([input_ids, current_tokens.unsqueeze(-1)], dim=-1)
    #         masked_token_ids[:, cur_start:cur_start + step + 2] = input_ids
    #
    #         if beam_scorer.is_done:
    #             break
    #
    #     sequence_outputs = beam_scorer.finalize(
    #         input_ids, beam_scores, next_tokens, next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id
    #     )
    #     if config.print_beams:
    #         beam_scorer.print_beams()  # not exactly match with sequence_outputs
    #     res = sequence_outputs['sequences'][:, 1:high + 1]
    #     ori_masked_token_ids[range(batch_size), cur_start:cur_start + res.shape[-1]] = res
    #     return ori_masked_token_ids
