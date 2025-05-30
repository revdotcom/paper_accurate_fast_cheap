from typing import List, Tuple, Optional

import torch
from wenet.utils.common import log_add

from wenet.transformer.search import DecodeResult

class Sequence():

    __slots__ = {'hyp', 'score', 'cache'}

    def __init__(
        self,
        hyp: List[torch.Tensor],
        score,
        cache: List[torch.Tensor],
    ):
        self.hyp = hyp
        self.score = score
        self.cache = cache


class PrefixBeamSearch():

    def __init__(self, encoder, predictor, joint, ctc, blank):
        self.encoder = encoder
        self.predictor = predictor
        self.joint = joint
        self.ctc = ctc
        self.blank = blank

    def forward_decoder_one_step(
            self, encoder_x: torch.Tensor, pre_t: torch.Tensor,
            cache: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        padding = torch.zeros(pre_t.size(0), 1, device=encoder_x.device)
        pre_t, new_cache = self.predictor.forward_step(pre_t.unsqueeze(-1),
                                                       padding, cache)
        x = self.joint(encoder_x, pre_t)  # [beam, 1, 1, vocab]
        x = x.log_softmax(dim=-1)
        return x, new_cache

    def _forward_encoder(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
        cat_embs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Let's assume B = batch_size
        # 1. Encoder
        if simulate_streaming and decoding_chunk_size > 0:
            encoder_out, encoder_mask = self.encoder.forward_chunk_by_chunk(
                speech,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks
            )  # (B, maxlen, encoder_dim)
        else:
            encoder_out, encoder_mask = self.encoder(
                speech,
                speech_lengths,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks,
                cat_embs = cat_embs,
            )  # (B, maxlen, encoder_dim)
        return encoder_out, encoder_mask

    def prefix_beam_search(self,
                           speech: torch.Tensor,
                           speech_lengths: torch.Tensor,
                           decoding_chunk_size: int = -1,
                           beam_size: int = 5,
                           num_decoding_left_chunks: int = -1,
                           simulate_streaming: bool = False,
                           ctc_weight: float = 0.3,
                           transducer_weight: float = 0.7,
                           cat_embs: Optional[torch.Tensor] = None):
        """prefix beam search
           also see wenet.transducer.transducer.beam_search
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        device = speech.device
        batch_size = speech.shape[0]
        assert batch_size == 1

        # 1. Encoder
        encoder_out, _ = self.encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks, cat_embs)  # (B, maxlen, encoder_dim)
        maxlen = encoder_out.size(1)

        ctc_probs = self.ctc.log_softmax(encoder_out).squeeze(0)
        beam_init: List[Sequence] = []

        # 2. init beam using Sequence to save beam unit
        cache = self.predictor.init_state(1, method="zero", device=device)
        beam_init.append(Sequence(hyp=[self.blank], score=0.0, cache=cache))
        # 3. start decoding (notice: we use breathwise first searching)
        # !!!! In this decoding method: one frame do not output multi units. !!!!
        # !!!!    Experiments show that this strategy has little impact      !!!!
        for i in range(maxlen):
            # 3.1 building input
            # decoder taking the last token to predict the next token
            input_hyp = [s.hyp[-1] for s in beam_init]
            input_hyp_tensor = torch.tensor(input_hyp,
                                            dtype=torch.int,
                                            device=device)
            # building statement from beam
            cache_batch = self.predictor.cache_to_batch(
                [s.cache for s in beam_init])
            # build score tensor to do torch.add() function
            scores = torch.tensor([s.score for s in beam_init]).to(device)

            # 3.2 forward decoder
            logp, new_cache = self.forward_decoder_one_step(
                encoder_out[:, i, :].unsqueeze(1),
                input_hyp_tensor,
                cache_batch,
            )  # logp: (N, 1, 1, vocab_size)
            logp = logp.squeeze(1).squeeze(1)  # logp: (N, vocab_size)
            new_cache = self.predictor.batch_to_cache(new_cache)

            # 3.3 shallow fusion for transducer score
            #     and ctc score where we can also add the LM score
            logp = torch.log(
                torch.add(transducer_weight * torch.exp(logp),
                          ctc_weight * torch.exp(ctc_probs[i].unsqueeze(0))))

            # 3.4 first beam prune
            top_k_logp, top_k_index = logp.topk(beam_size)  # (N, N)
            scores = torch.add(scores.unsqueeze(1), top_k_logp)

            # 3.5 generate new beam (N*N)
            beam_A = []
            for j in range(len(beam_init)):
                # update seq
                base_seq = beam_init[j]
                for t in range(beam_size):
                    # blank: only update the score
                    if top_k_index[j, t] == self.blank:
                        new_seq = Sequence(hyp=base_seq.hyp.copy(),
                                           score=scores[j, t].item(),
                                           cache=base_seq.cache)

                        beam_A.append(new_seq)
                    # other unit: update hyp score statement and last
                    else:
                        hyp_new = base_seq.hyp.copy()
                        hyp_new.append(top_k_index[j, t].item())
                        new_seq = Sequence(hyp=hyp_new,
                                           score=scores[j, t].item(),
                                           cache=new_cache[j])
                        beam_A.append(new_seq)

            # 3.6 prefix fusion
            fusion_A = [beam_A[0]]
            for j in range(1, len(beam_A)):
                s1 = beam_A[j]
                if_do_append = True
                for t in range(len(fusion_A)):
                    # notice: A_ can not fusion with A
                    if s1.hyp == fusion_A[t].hyp:
                        fusion_A[t].score = log_add(
                            [fusion_A[t].score, s1.score])
                        if_do_append = False
                        break
                if if_do_append:
                    fusion_A.append(s1)

            # 4. second pruned
            fusion_A.sort(key=lambda x: x.score, reverse=True)
            beam_init = fusion_A[:beam_size]

        return beam_init, encoder_out

    def prefix_beam_search_decode(self,
        encoder_outs: torch.Tensor,
        encoder_lens: torch.Tensor,
        ctc_probs: torch.Tensor,
        decoding_chunk_size: int = -1,
        beam_size: int = 5,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
        ctc_weight: float = 0.3,
        transducer_weight: float = 0.7,
        cat_embs: Optional[torch.Tensor] = None):
        """prefix beam search
           also see wenet.transducer.transducer.beam_search
        """
        assert encoder_outs.shape[0] == encoder_lens.shape[0]
        assert encoder_outs.shape[0] == ctc_probs.shape[0]
        assert decoding_chunk_size != 0
        device = encoder_outs.device
        batch_size = encoder_outs.shape[0]

        # print(f"batch_size: {batch_size}")

        results = []
        if False and batch_size == 1:
            encoder_out = encoder_outs[0,:,:].unsqueeze(0)
            maxlen = encoder_lens[0]
            ctc_prob = ctc_probs[0,:,:]
            result = self.prefix_beam_search_decode_single(encoder_out, maxlen, ctc_prob, decoding_chunk_size, beam_size, num_decoding_left_chunks, simulate_streaming, ctc_weight, transducer_weight)
            results.append(result)
            return results

        if False and batch_size > 1:
            for i in range(batch_size):
                encoder_out = encoder_outs[i,:,:].unsqueeze(0)
                maxlen = encoder_lens[i]
                ctc_prob = ctc_probs[i,:,:]
                result = self.prefix_beam_search_decode_single(encoder_out, maxlen, ctc_prob, decoding_chunk_size, beam_size, num_decoding_left_chunks, simulate_streaming, ctc_weight, transducer_weight)
                results.append(result)
            return results  

        if True or batch_size > 1:
            return self.prefix_beam_search_decode_batch(encoder_outs, encoder_lens, ctc_probs, decoding_chunk_size, beam_size, num_decoding_left_chunks, simulate_streaming, ctc_weight, transducer_weight, cat_embs)



        for i in range(batch_size):
            encoder_out = encoder_outs[i,:,:].unsqueeze(0)
            maxlen = encoder_lens[i]
            ctc_prob = ctc_probs[i,:,:]

            beam_init: List[Sequence] = []

            # 2. init beam using Sequence to save beam unit
            cache = self.predictor.init_state(1, method="zero", device=device)
            beam_init.append(Sequence(hyp=[self.blank], score=0.0, cache=cache))
            # 3. start decoding (notice: we use breathwise first searching)
            # !!!! In this decoding method: one frame do not output multi units. !!!!
            # !!!!    Experiments show that this strategy has little impact      !!!!
            for i in range(maxlen):
                # 3.1 building input
                # decoder taking the last token to predict the next token
                input_hyp = [s.hyp[-1] for s in beam_init]
                input_hyp_tensor = torch.tensor(input_hyp,
                                                dtype=torch.int,
                                                device=device)
                # building statement from beam
                cache_batch = self.predictor.cache_to_batch(
                    [s.cache for s in beam_init])
                # build score tensor to do torch.add() function
                scores = torch.tensor([s.score for s in beam_init]).to(device)

                # 3.2 forward decoder
                logp, new_cache = self.forward_decoder_one_step(
                    encoder_out[:, i, :].unsqueeze(1),
                    input_hyp_tensor,
                    cache_batch,
                )  # logp: (N, 1, 1, vocab_size)
                logp = logp.squeeze(1).squeeze(1)  # logp: (N, vocab_size)
                new_cache = self.predictor.batch_to_cache(new_cache)

                # 3.3 shallow fusion for transducer score
                #     and ctc score where we can also add the LM score
                logp = torch.log(
                    torch.add(transducer_weight * torch.exp(logp),
                            ctc_weight * torch.exp(ctc_prob[i].unsqueeze(0))))

                # 3.4 first beam prune
                top_k_logp, top_k_index = logp.topk(beam_size)  # (N, N)
                scores = torch.add(scores.unsqueeze(1), top_k_logp)

                # 3.5 generate new beam (N*N)
                beam_A = []
                for j in range(len(beam_init)):
                    # update seq
                    base_seq = beam_init[j]
                    for t in range(beam_size):
                        # blank: only update the score
                        if top_k_index[j, t] == self.blank:
                            new_seq = Sequence(hyp=base_seq.hyp.copy(),
                                            score=scores[j, t].item(),
                                            cache=base_seq.cache)

                            beam_A.append(new_seq)
                        # other unit: update hyp score statement and last
                        else:
                            hyp_new = base_seq.hyp.copy()
                            hyp_new.append(top_k_index[j, t].item())
                            new_seq = Sequence(hyp=hyp_new,
                                            score=scores[j, t].item(),
                                            cache=new_cache[j])
                            beam_A.append(new_seq)

                # 3.6 prefix fusion
                fusion_A = [beam_A[0]]
                for j in range(1, len(beam_A)):
                    s1 = beam_A[j]
                    if_do_append = True
                    for t in range(len(fusion_A)):
                        # notice: A_ can not fusion with A
                        if s1.hyp == fusion_A[t].hyp:
                            fusion_A[t].score = log_add(
                                [fusion_A[t].score, s1.score])
                            if_do_append = False
                            break
                    if if_do_append:
                        fusion_A.append(s1)

                # 4. second pruned
                fusion_A.sort(key=lambda x: x.score, reverse=True)
                beam_init = fusion_A[:beam_size]

            # Add decoding results for each sample from batch
            nbest = [b.hyp[1:] for b in beam_init]
            nbest_scores = [b.score for b in beam_init]
            best = nbest[0]
            best_score = nbest_scores[0]

            decode_result = DecodeResult(tokens=best, score=best_score, nbest=nbest, nbest_scores=nbest_scores)
            results.append(decode_result)

        return results


    def prefix_beam_search_decode_single(self,
        encoder_out: torch.Tensor,  # Shape: [1, T, D]
        encoder_len: int,
        ctc_prob: torch.Tensor,    # Shape: [T, V]
        decoding_chunk_size: int = -1,
        beam_size: int = 5,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
        ctc_weight: float = 0.3,
        transducer_weight: float = 0.7):
        """Optimized prefix beam search for single sample case"""
        device = encoder_out.device
        
        # Initialize single beam
        cache = self.predictor.init_state(1, method="zero", device=device)
        beam_init = [Sequence(hyp=[self.blank], score=0.0, cache=cache)]

        # Process each time step
        for t in range(encoder_len):
            # Get current encoder state and CTC prob
            encoder_state = encoder_out[0, t, :].unsqueeze(0).unsqueeze(1)  # [1, 1, D]
            current_ctc_prob = ctc_prob[t].unsqueeze(0)  # [1, V]

            # Process all current beams in parallel
            num_beams = len(beam_init)
            input_hyps = torch.tensor([s.hyp[-1] for s in beam_init], device=device)
            cache_batch = self.predictor.cache_to_batch([s.cache for s in beam_init])
            scores = torch.tensor([s.score for s in beam_init], device=device)

            # Forward decoder (single forward pass for all beams)
            encoder_state = encoder_state.expand(num_beams, -1, -1)  # [num_beams, 1, D]
            logp, new_cache = self.forward_decoder_one_step(
                encoder_state,
                input_hyps,
                cache_batch,
            )
            logp = logp.squeeze(1).squeeze(1)  # [num_beams, vocab_size]
            new_cache = self.predictor.batch_to_cache(new_cache)

            # Shallow fusion
            logp = torch.log(
                torch.add(transducer_weight * torch.exp(logp),
                         ctc_weight * torch.exp(current_ctc_prob)))

            # Beam pruning
            top_k_logp, top_k_index = logp.topk(beam_size)  # [num_beams, beam_size]
            scores = scores.unsqueeze(1) + top_k_logp  # [num_beams, beam_size]

            # Generate new beams efficiently
            beam_A = []
            scores_flat = scores.flatten()
            indices_flat = top_k_index.flatten()
            beam_indices = torch.div(torch.arange(len(scores_flat), device=device), beam_size, rounding_mode='floor')
            
            # Sort by score for early pruning
            sorted_indices = torch.argsort(scores_flat, descending=True)[:num_beams * beam_size]
            
            # Process top candidates
            seen_hyps = set()
            for idx in sorted_indices:
                beam_idx = beam_indices[idx].item()
                token_idx = indices_flat[idx].item()
                score = scores_flat[idx].item()
                base_seq = beam_init[beam_idx]
                
                if token_idx == self.blank:
                    new_hyp = base_seq.hyp.copy()
                else:
                    new_hyp = base_seq.hyp.copy() + [token_idx]
                
                # Convert hypothesis to tuple for set membership test
                hyp_tuple = tuple(new_hyp)
                if hyp_tuple in seen_hyps:
                    # Find and update existing beam
                    for existing in beam_A:
                        if existing.hyp == new_hyp:
                            existing.score = log_add([existing.score, score])
                            break
                else:
                    seen_hyps.add(hyp_tuple)
                    new_seq = Sequence(
                        hyp=new_hyp,
                        score=score,
                        cache=base_seq.cache if token_idx == self.blank else new_cache[beam_idx]
                    )
                    beam_A.append(new_seq)
                    
                    # Early stopping if we have enough beams
                    if len(beam_A) >= beam_size:
                        break

            # Update beams for next iteration
            beam_A.sort(key=lambda x: x.score, reverse=True)
            beam_init = beam_A[:beam_size]

        # Get n-best results
        nbest = [b.hyp[1:] for b in beam_init]
        nbest_scores = [b.score for b in beam_init]
        
        return DecodeResult(
            tokens=nbest[0],
            score=nbest_scores[0],
            nbest=nbest,
            nbest_scores=nbest_scores
        )

    def prefix_beam_search_decode_batch(self,
        encoder_outs: torch.Tensor,  # Shape: [B, T, D]
        encoder_lens: torch.Tensor,  # Shape: [B]
        ctc_probs: torch.Tensor,     # Shape: [B, T, V]
        decoding_chunk_size: int = -1,
        beam_size: int = 5,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
        ctc_weight: float = 0.3,
        transducer_weight: float = 0.7,
        cat_embs: Optional[torch.Tensor] = None):
        """Batched version of prefix beam search decoder"""
        device = encoder_outs.device
        batch_size = encoder_outs.shape[0]
        max_len = encoder_lens.max().item()
        
        # Initialize beams for each sample in batch
        batch_beams = []
        for i in range(batch_size):
            cache = self.predictor.init_state(1, method="zero", device=device)
            batch_beams.append([Sequence(hyp=[self.blank], score=0.0, cache=cache)])
        
        # Process each time step
        for t in range(max_len):
            # Process each sample in batch that hasn't finished
            active_samples = [i for i in range(batch_size) if t < encoder_lens[i]]
            if not active_samples:
                break
            
            # Collect inputs for active samples
            total_beams = sum(len(batch_beams[i]) for i in active_samples)
            encoder_states = []
            input_hyps = []
            beam_caches = []
            beam_scores = []
            beam_map = []  # Maps beam index to (sample_idx, beam_idx)
            
            current_idx = 0
            for sample_idx in active_samples:
                beams = batch_beams[sample_idx]
                num_beams = len(beams)
                
                # Get encoder state for this sample at time t
                encoder_state = encoder_outs[sample_idx, t, :].unsqueeze(0).unsqueeze(1)
                encoder_states.append(encoder_state.expand(num_beams, -1, -1))
                
                # Collect beam info
                input_hyps.extend([s.hyp[-1] for s in beams])
                beam_caches.extend([s.cache for s in beams])
                beam_scores.extend([s.score for s in beams])
                beam_map.extend([(sample_idx, j) for j in range(num_beams)])
                current_idx += num_beams
            
            # Convert to tensors
            encoder_states = torch.cat(encoder_states, dim=0)  # [total_beams, 1, D]
            input_hyps = torch.tensor(input_hyps, device=device)
            beam_scores = torch.tensor(beam_scores, device=device)
            
            # Forward decoder for all beams at once
            cache_batch = self.predictor.cache_to_batch(beam_caches)
            logp, new_cache = self.forward_decoder_one_step(
                encoder_states,
                input_hyps,
                cache_batch,
            )
            logp = logp.squeeze(1).squeeze(1)  # [total_beams, vocab_size]
            new_cache = self.predictor.batch_to_cache(new_cache)
            
            # Process each active sample separately
            current_idx = 0
            for sample_idx in active_samples:
                beams = batch_beams[sample_idx]
                num_beams = len(beams)
                
                # Get scores for this sample
                sample_logp = logp[current_idx:current_idx + num_beams]
                sample_scores = beam_scores[current_idx:current_idx + num_beams]
                sample_new_cache = new_cache[current_idx:current_idx + num_beams]
                
                # Shallow fusion with CTC
                sample_logp = torch.log(
                    torch.add(transducer_weight * torch.exp(sample_logp),
                             ctc_weight * torch.exp(ctc_probs[sample_idx, t].unsqueeze(0))))
                
                # Beam pruning
                top_k_logp, top_k_index = sample_logp.topk(beam_size)
                scores = sample_scores.unsqueeze(1) + top_k_logp
                
                # Generate new beams
                beam_A = []
                scores_flat = scores.flatten()
                indices_flat = top_k_index.flatten()
                beam_indices = torch.div(torch.arange(len(scores_flat), device=device), 
                                       beam_size, rounding_mode='floor')
                
                # Sort by score for early pruning
                sorted_indices = torch.argsort(scores_flat, descending=True)[:num_beams * beam_size]
                
                # Process top candidates
                seen_hyps = set()
                for idx in sorted_indices:
                    beam_idx = beam_indices[idx].item()
                    token_idx = indices_flat[idx].item()
                    score = scores_flat[idx].item()
                    base_seq = beams[beam_idx]
                    
                    if token_idx == self.blank:
                        new_hyp = base_seq.hyp.copy()
                    else:
                        new_hyp = base_seq.hyp.copy() + [token_idx]
                    
                    hyp_tuple = tuple(new_hyp)
                    if hyp_tuple in seen_hyps:
                        for existing in beam_A:
                            if existing.hyp == new_hyp:
                                existing.score = log_add([existing.score, score])
                                break
                    else:
                        seen_hyps.add(hyp_tuple)
                        new_seq = Sequence(
                            hyp=new_hyp,
                            score=score,
                            cache=base_seq.cache if token_idx == self.blank else sample_new_cache[beam_idx]
                        )
                        beam_A.append(new_seq)
                        
                        if len(beam_A) >= beam_size:
                            break
                
                # Update beams for this sample
                beam_A.sort(key=lambda x: x.score, reverse=True)
                batch_beams[sample_idx] = beam_A[:beam_size]
                current_idx += num_beams
            
        # Collect results for all samples
        results = []
        for beams in batch_beams:
            nbest = [b.hyp[1:] for b in beams]
            nbest_scores = [b.score for b in beams]
            results.append(DecodeResult(
                tokens=nbest[0],
                score=nbest_scores[0],
                nbest=nbest,
                nbest_scores=nbest_scores
            ))
        
        return results
