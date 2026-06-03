"""Batched single-stream decode for stream-qwen3-8b (G1).

The bundled `stream_inference.generate` is single-sequence (per-row Python loop,
per-sequence masks). At ~0.38 s/row that makes 164 HumanEval problems ~4+ hours.
The g6 microbench proved decode is weight-read bound and *batch-starved*, and the
g6 batched sweep proved B=16 lifts aggregate throughput ~6x. This module brings
that lever to REAL prompt-conditioned generation: it advances B independent
task-streams in lockstep, each with its own prompt prefill, greedy (T=0) Output,
per-element silence penalty + skip_silence masking, mirroring the reference
generate() semantics exactly but in a batch.

Single-stream G1 use: we only read the Output channel (channel 1). The other 9
channels still decode (the substrate always emits all 10) but are ignored —
identical to what the reference single-stream path does.

Correctness is validated against the reference `stream_generate` in
scripts/g1_batched_validate.py (greedy outputs must match for B=1).
"""
from __future__ import annotations

import torch

# Channel layout (from stream_inference.CHANNELS): index 0 = User, 1 = Output, ...
USER_CH = 0
OUTPUT_CH = 1


def _tokenize_user(tokenizer, text: str) -> list[int]:
    """Match stream_inference._tokenize_user: per-whitespace-chunk, leading space."""
    tokens = []
    for chunk in text.split():
        ids = tokenizer.encode(" " + chunk, add_special_tokens=False)
        tokens.extend(ids)
    return tokens


def detect_silence_token(tokenizer) -> int:
    _sp = tokenizer.convert_ids_to_tokens(
        tokenizer.encode("test", add_special_tokens=False))[0].startswith("▁")
    return tokenizer.encode("-" if _sp else " -", add_special_tokens=False)[0]


@torch.no_grad()
def batched_stream_generate(
    model,
    tokenizer,
    prompts: list[str],
    *,
    max_rows: int = 320,
    silence_token: int | None = None,
    silence_penalty: float = 10.0,
    skip_silence: bool = True,
    eos_token_id: int | None = None,
    output_eos_rows: int = 6,
) -> list[str]:
    """Greedy (T=0) batched decode. Returns the decoded Output channel per prompt.

    Mirrors stream_inference.generate's non-interactive, non-warm-start path:
      - prompt tokens injected on channel 0 (User), one token per row, then
        silence after the prompt ends;
      - Output channel (1) gets silence_penalty (ramped) once its prompt ends;
      - skip_silence masks silence-token columns in attention;
      - greedy argmax per channel (top_k=1 / T->0 equivalent).

    Batching: prompts are LEFT-padded with silence on channel 0 to equal length
    so all elements share the row index / position ids. Output is read from the
    valid (post-padding) region per element. An element stops contributing once
    it has emitted `output_eos_rows` consecutive Output-silence rows after its
    prompt ended (cheap analog of the reference all-silent stop); decode runs to
    max_rows or until ALL elements have stopped.
    """
    device = model.get_input_embeddings().weight.device
    C = getattr(model.config, "num_channels", 10)
    B = len(prompts)
    if silence_token is None:
        silence_token = detect_silence_token(tokenizer)

    # Tokenize prompts on channel 0.
    user_tok = [_tokenize_user(tokenizer, p) for p in prompts]
    plens = [len(u) for u in user_tok]
    Lmax = max(plens)
    # Left-pad each prompt to Lmax with silence (so user input ends at the same
    # row for all elements -> shared silence-penalty ramp timing).
    pad = [Lmax - l for l in plens]
    user_rows = []  # [Lmax][B] channel-0 token id
    for r in range(Lmax):
        row = []
        for b in range(B):
            idx = r - pad[b]
            row.append(user_tok[b][idx] if 0 <= idx < plens[b] else silence_token)
        user_rows.append(row)
    user_end_row = Lmax  # after this row, channel 0 is silence and penalty ramps

    # KV cache via DynamicCache (batched).
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()

    # Track collected Output tokens per element, and stop bookkeeping.
    out_tokens: list[list[int]] = [[] for _ in range(B)]
    out_sil_streak = [0] * B
    stopped = [False] * B
    all_cached_ids = torch.empty((B, 0), device=device, dtype=torch.long)

    # First row: seed words (channel 0 gets first user token; others seed).
    # The reference seeds non-interactive from SEED_WORDS, but for batched greedy
    # prompt-conditioned gen we follow the prefill-free path: row 0 is the first
    # user row, decode proceeds row by row. We build the per-row input then call
    # forward to get logits for the NEXT row.
    row = [[silence_token] * C for _ in range(B)]
    for b in range(B):
        row[b][USER_CH] = user_rows[0][b] if Lmax > 0 else silence_token

    silence_pen_tensor = None

    for row_idx in range(0, max_rows + Lmax):
        input_ids = torch.tensor(row, device=device, dtype=torch.long)  # [B,C]
        position_ids = torch.full((B, C), row_idx, device=device, dtype=torch.long)
        channel_ids = torch.arange(C, device=device).unsqueeze(0).expand(B, -1).contiguous()

        cached_len = cache.get_seq_length()
        total = cached_len + C
        # Cross-stream block mask: each new channel attends to all past + only
        # its own new column.
        m = torch.zeros(C, total, device=device, dtype=torch.bfloat16)
        block = torch.full((C, C), -1e4, device=device, dtype=torch.bfloat16)
        block.fill_diagonal_(0.0)
        m[:, cached_len:] = block
        mask = m.view(1, 1, C, total).expand(B, 1, -1, -1).contiguous()

        if skip_silence:
            # Mask silence-token KV columns per element (past + current row).
            cur_sil = (input_ids == silence_token)  # [B,C]
            if all_cached_ids.shape[1] > 0:
                past_sil = (all_cached_ids == silence_token)  # [B, cached_len]
            else:
                past_sil = torch.empty((B, 0), device=device, dtype=torch.bool)
            sil_cols = torch.cat([past_sil, cur_sil], dim=1)  # [B,total]
            mask = mask.masked_fill(sil_cols.view(B, 1, 1, total), -1e4)
            # restore own-diagonal for the C new tokens (channel i attends to its
            # own new token even if it is silence) — matches reference.
            for i in range(C):
                mask[:, 0, i, cached_len + i] = 0.0

        out = model(
            input_ids=input_ids,
            attention_mask={"full_attention": mask, "sliding_attention": mask},
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            channel_ids=channel_ids,
        )
        cache = out.past_key_values
        all_cached_ids = torch.cat([all_cached_ids, input_ids], dim=1)
        logits = out.logits  # [B, C, V]

        # Silence penalty on Output channel once user input has ended (ramped).
        if row_idx >= user_end_row and silence_penalty > 0 and C >= 2:
            ramp = min(1.0, (row_idx - user_end_row + 1) / 5.0)
            logits[:, OUTPUT_CH, silence_token] -= silence_penalty * ramp

        # Greedy next row.
        nxt = logits.argmax(dim=-1)  # [B, C]
        next_row = nxt.tolist()

        # Channel 0: stay on the prompt schedule, else silence.
        for b in range(B):
            if row_idx + 1 < Lmax:
                next_row[b][USER_CH] = user_rows[row_idx + 1][b]
            else:
                next_row[b][USER_CH] = silence_token

        # Collect Output tokens emitted on THIS row (the row we just forwarded
        # is `row`; its Output token was decided last step). Actually we collect
        # from next_row's Output below after recording. To keep it simple and
        # aligned with the reference (which records the row it yields), we record
        # the Output token of `next_row` for rows past the prompt.
        for b in range(B):
            if stopped[b]:
                continue
            otok = next_row[b][OUTPUT_CH]
            if row_idx + 1 >= Lmax:  # only past the (left-padded) prompt region
                if otok == silence_token:
                    out_sil_streak[b] += 1
                    if out_sil_streak[b] >= output_eos_rows and out_tokens[b]:
                        stopped[b] = True
                else:
                    out_sil_streak[b] = 0
                    out_tokens[b].append(otok)
                    if eos_token_id is not None and otok == eos_token_id:
                        stopped[b] = True

        row = next_row
        if all(stopped):
            break

    # Decode each element's Output tokens.
    results = []
    for b in range(B):
        toks = [t for t in out_tokens[b] if t != silence_token]
        if eos_token_id is not None:
            toks = [t for t in toks if t != eos_token_id]
        results.append(tokenizer.decode(toks).strip() if toks else "")
    return results
