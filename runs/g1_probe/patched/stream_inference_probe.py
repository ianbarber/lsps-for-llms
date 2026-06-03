"""Multi-stream inference helper for the stream-qwen3.5-27b model.

This file ships inside the HF repo so users can either:

    # Option 1 — snapshot the repo and import directly
    from huggingface_hub import snapshot_download
    import sys
    sys.path.insert(0, snapshot_download("JonasGeiping/stream-qwen3.5-27b"))
    from stream_inference import StreamModel

    sm = StreamModel("JonasGeiping/stream-qwen3.5-27b", device="cuda")
    result = sm.generate("Hello, what's up?")
    print(result.output)
    print(result.channel_texts["Analytical"])

    # Option 2 — load model + tokenizer yourself, then drive the generator
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "JonasGeiping/stream-qwen3.5-27b",
        trust_remote_code=True,
        torch_dtype="bfloat16",
        device_map="auto",
    )
    tok = AutoTokenizer.from_pretrained("JonasGeiping/stream-qwen3.5-27b")
    silence_token = detect_silence_token(tok)
    for row_idx, row, is_prefill in generate(model, tok, "Hello", silence_token):
        print(row_idx, row)

Architecture summary: ten channels are generated in parallel per timestep using a
block-causal attention mask (tokens see all prior rows + self, but NOT same-row
peers). One forward pass produces all ten next-row tokens; the inference loop
below recycles the past-KV cache between rows.
"""

from collections.abc import Generator
from dataclasses import dataclass

import torch

CHANNELS = [
    "User",
    "Output",
    "Analytical",
    "Skeptical",
    "Intuitive",
    "Between",
    "Curious",
    "Void",
    "Instinct",
    "Synthesis",
]
C = len(CHANNELS)

# Seed words for row 0 — User/Output stay silent, channels 2-9 get a single
# voice-priming BPE token.
SEED_WORDS = [
    "-", "-",
    " thinking", " checking", " feeling", " relating",
    " asking", " drifting", " watching", " integrating",
]

# 11-row warm-start header that primes each thinking channel's voice before any
# user input arrives. Used when `warm_start=True`.
# fmt: off
SYSTEM_PROMPT_WORDS = [
    # User Output  Analytical  Skeptical    Intuitive  Between    Curious      Void       Instinct    Synthesis
    ["-", "-",   " idle",    " quiet",    " resting"," still",  " what",     " from",   " waiting", " settling"],
    ["-", "-",   " ready",   " clear",    " calm",   " space",  " comes",    " the",    " alert",   " into"],
    ["-", "-",   " for",     " so",       " breath", " open",   " next",     " form",   " steady",  " readiness"],
    ["-", "-",   " anything"," far",      " easy",   " for",    " wonder",   "less",    " patient", " all"],
    ["-", "-",   " think",   " check",    " present"," whoever"," who",      " void",   " careful", " voices"],
    ["-", "-",   " through", " every",    " here",   " arrives"," needs",    " gaping", " grounded"," finding"],
    ["-", " OK", " each",    " angle",    " letting"," welcome"," something"," m",      " before",  " their"],
    ["-", " I",  " problem", " first",    " it",     " them",   " fresh",    "aw",      " acting",  " rhythm"],
    ["-", "'m",  " carefully"," honestly"," happen", " openly", " perhaps",  " springs"," measured"," listening"],
    ["-", " ready"," consider"," then",   " now",    " ready",  " always",   " an",     " aware",   " together"],
    ["-", "-",   " now",     " always",   " open",   "-",       " always",   " entity", " now",     " here"],
]
# fmt: on


@dataclass
class StreamResult:
    """Result of a multi-stream generation."""

    tokens: list[list[int]]  # [R, C] raw token IDs
    channel_texts: dict[str, str]
    num_rows: int
    silence_token: int

    @property
    def output(self) -> str:
        return self.channel_texts.get("Output", "")

    @property
    def user(self) -> str:
        return self.channel_texts.get("User", "")

    def stream(self, name: str) -> str:
        return self.channel_texts[name]

    def silence_ratio(self, channel: int | str) -> float:
        if isinstance(channel, str):
            channel = CHANNELS.index(channel)
        sil = sum(1 for row in self.tokens if row[channel] == self.silence_token)
        return sil / max(len(self.tokens), 1)


def detect_silence_token(tokenizer) -> int:
    """Detect the '-' token ID. Handles BPE vs SentencePiece tokenizers."""
    _sp = tokenizer.convert_ids_to_tokens(tokenizer.encode("test", add_special_tokens=False))[0].startswith("▁")
    return tokenizer.encode("-" if _sp else " -", add_special_tokens=False)[0]


def _tokenize_user(tokenizer, text: str) -> list[int]:
    """Tokenize user text matching the training data pipeline (per-chunk with leading space)."""
    tokens = []
    for chunk in text.split():
        ids = tokenizer.encode(" " + chunk, add_special_tokens=False)
        tokens.extend(ids)
    return tokens


def build_system_prompt_prefill(tokenizer, silence_token: int, num_channels: int = C) -> list[list[int]]:
    """Build the 11-row warm-start header as token-id rows."""
    rows = []
    for word_row in SYSTEM_PROMPT_WORDS:
        token_row = []
        for word in word_row[:num_channels]:
            if word == "-":
                token_row.append(silence_token)
            else:
                toks = tokenizer.encode(word, add_special_tokens=False)
                token_row.append(toks[0])
        rows.append(token_row)
    return rows


def sample_top_p(logits, temperature=0.8, top_p=0.95, top_k=20):
    """Sample from logits with temperature, top-k, and nucleus filtering."""
    if temperature <= 0:
        return logits.argmax(dim=-1).item()
    logits = logits / temperature
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        top_vals, _ = torch.topk(logits, k)
        logits = logits.where(logits >= top_vals[-1], torch.tensor(float("-inf"), device=logits.device))
    probs = torch.softmax(logits.float(), dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask = cumsum - sorted_probs > top_p
    sorted_probs[mask] = 0.0
    sorted_probs /= sorted_probs.sum()
    token = sorted_idx[torch.multinomial(sorted_probs, 1)]
    return token.item()


@torch.no_grad()
def generate(
    model,
    tokenizer,
    user_text: str,
    silence_token: int,
    *,
    max_rows: int = 200,
    pre_think: int = 1,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 20,
    silence_penalty: float = 10.0,
    think_silence_penalty: float = 0.0,
    ablate_channels: list[int] | None = None,
    prefill_rows: list[list[int]] | None = None,
    skip_silence: bool = False,
    warm_start: bool = False,
) -> Generator[tuple, int | None, None]:
    """Generate a multi-stream rollout, yielding (row_idx, row_tokens, is_prefill) per row.

    Supports `.send(token_id)` for interactive user-input injection when
    `user_text` is empty/None.

    Key knobs:
        max_rows: stop after this many rows (one row = ten tokens).
        pre_think: rows of silence before user input begins.
        warm_start: prepend the 11-row voice-priming header.
        silence_penalty: penalize silence on Output once user input ends.
        skip_silence: mask silence tokens in attention keys (cleaner for long rollouts).
    """
    C = getattr(model.config, "num_channels", 10)
    _seed_words = SEED_WORDS[:C] if C <= len(SEED_WORDS) else SEED_WORDS + ["-"] * (C - len(SEED_WORDS))
    device = model.get_input_embeddings().weight.device

    if warm_start:
        prefill_rows = build_system_prompt_prefill(tokenizer, silence_token, num_channels=C)
        user_tokens = _tokenize_user(tokenizer, user_text) if user_text else []
        pre_think = len(prefill_rows)
    else:
        user_tokens = _tokenize_user(tokenizer, user_text) if user_text else []

    interactive = not user_tokens
    user_end_row = pre_think + len(user_tokens)
    got_user_input = False
    silence_streak = 0
    all_silent_streak = 0
    # PROBE PATCH: track whether the Output channel (idx 1) has ever spoken.
    # The original early-stop (`all_silent_streak >= 1`) halts generation the
    # instant every channel is silent for one row — which for non-trivial
    # prompts fires BEFORE Output has emitted a single token (the "silent
    # output" failure mode). We loosen it: the early-stop may only fire once
    # Output has produced at least one non-silence token, and then only after a
    # long all-silent run (so we don't truncate mid-answer on a brief pause).
    output_has_spoken = False
    EARLY_STOP_THRESHOLD = 40
    ablate = set(ablate_channels or [])

    if prefill_rows is not None:
        n_prefill = len(prefill_rows)
        flat = [t for row in prefill_rows for t in row]
        input_ids = torch.tensor([flat], device=device, dtype=torch.long)
        position_ids = torch.tensor(
            [[r for r in range(n_prefill) for _ in range(C)]],
            device=device, dtype=torch.long,
        )
        channel_ids = torch.tensor(
            [[c for _ in range(n_prefill) for c in range(C)]],
            device=device, dtype=torch.long,
        )

        N = n_prefill * C
        # Block-causal mask, vectorized — see comment in the 27B copy. Python
        # double-loop was 1-9M iterations for typical prefills (~20-60s).
        rows_idx = torch.arange(N, device=device) // C
        allowed = (rows_idx.unsqueeze(0) < rows_idx.unsqueeze(1)) | torch.eye(
            N, dtype=torch.bool, device=device
        )
        mask = torch.where(
            allowed, torch.tensor(0.0, device=device), torch.tensor(-1e4, device=device)
        ).to(torch.bfloat16).view(1, 1, N, N)
        if skip_silence:
            sil_cols = input_ids[0] == silence_token
            mask[0, 0, :, :].masked_fill_(sil_cols.unsqueeze(0), -1e4)
            mask[0, 0].diagonal().clamp_(min=0.0)

        outputs = model(
            input_ids=input_ids,
            attention_mask={"full_attention": mask, "sliding_attention": mask},
            position_ids=position_ids,
            use_cache=True,
            channel_ids=channel_ids,
        )
        past_kv = outputs.past_key_values
        logits = outputs.logits[0]

        for r in range(n_prefill):
            yield (r, prefill_rows[r], True)

        last_logits = logits[(n_prefill - 1) * C : n_prefill * C]
        row = [sample_top_p(last_logits[c], temperature, top_p, top_k) for c in range(C)]
        for c in ablate:
            row[c] = silence_token
        if user_tokens and n_prefill >= pre_think and n_prefill - pre_think < len(user_tokens):
            row[0] = user_tokens[n_prefill - pre_think]
        elif not interactive:
            row[0] = silence_token
        sent_value = yield (n_prefill, row, False)
        if interactive and sent_value is not None:
            row[0] = sent_value
            if sent_value != silence_token:
                got_user_input = True
        start_row = n_prefill + 1
        _prefill_ids = input_ids[0]
    else:
        row = []
        for c, word in enumerate(_seed_words):
            if c in ablate:
                row.append(silence_token)
            elif word == "-":
                row.append(silence_token)
            else:
                toks = tokenizer.encode(word, add_special_tokens=False)
                row.append(toks[0])
        sent_value = yield (0, row, False)
        if interactive and sent_value is not None:
            row[0] = sent_value
            if sent_value != silence_token:
                got_user_input = True
        past_kv = None
        start_row = 1
        _prefill_ids = None

    if _prefill_ids is not None:
        all_cached_ids = _prefill_ids.clone()
    else:
        all_cached_ids = torch.tensor([], device=device, dtype=torch.long)

    for row_idx in range(start_row, start_row + max_rows):
        input_ids = torch.tensor([row], device=device, dtype=torch.long)
        position_ids = torch.full((1, C), row_idx - 1, device=device, dtype=torch.long)
        channel_ids = torch.arange(C, device=device, dtype=torch.long).unsqueeze(0)

        if past_kv is None:
            mask = torch.full((1, 1, C, C), -1e4, device=device, dtype=torch.bfloat16)
            for i in range(C):
                mask[0, 0, i, i] = 0.0
        else:
            cached_len = past_kv.get_seq_length()
            total = cached_len + C
            mask = torch.zeros(1, 1, C, total, device=device, dtype=torch.bfloat16)
            for i in range(C):
                for j in range(C):
                    if i != j:
                        mask[0, 0, i, cached_len + j] = -1e4

        if skip_silence:
            row_ids = input_ids[0]
            sil_cols = torch.cat([all_cached_ids == silence_token, row_ids == silence_token])
            mask[0, 0].masked_fill_(sil_cols.unsqueeze(0), -1e4)
            peer_offset = len(all_cached_ids)
            for i in range(C):
                mask[0, 0, i, peer_offset + i] = 0.0

        outputs = model(
            input_ids=input_ids,
            attention_mask={"full_attention": mask, "sliding_attention": mask},
            position_ids=position_ids,
            past_key_values=past_kv,
            use_cache=True,
            channel_ids=channel_ids,
        )
        past_kv = outputs.past_key_values
        all_cached_ids = torch.cat([all_cached_ids, input_ids[0]])
        logits = outputs.logits[0]

        if interactive:
            apply_post_user_silence = got_user_input and row[0] == silence_token
            if apply_post_user_silence:
                silence_streak += 1
                ramp = min(1.0, silence_streak / 5.0)
                if silence_penalty > 0 and C >= 2:
                    logits[1, silence_token] -= silence_penalty * ramp
                if think_silence_penalty > 0:
                    for c in range(2, C):
                        logits[c, silence_token] -= think_silence_penalty * ramp
            else:
                silence_streak = 0
        elif row_idx >= user_end_row:
            ramp = min(1.0, (row_idx - user_end_row + 1) / 5.0)
            if silence_penalty > 0 and C >= 2:
                logits[1, silence_token] -= silence_penalty * ramp
            if think_silence_penalty > 0:
                for c in range(2, C):
                    logits[c, silence_token] -= think_silence_penalty * ramp

        if interactive:
            user_tok = silence_token
        elif row_idx < pre_think or row_idx - pre_think >= len(user_tokens):
            user_tok = silence_token
        else:
            user_tok = user_tokens[row_idx - pre_think]

        next_row = [user_tok] + [sample_top_p(logits[c], temperature, top_p, top_k) for c in range(1, C)]
        for c in ablate:
            next_row[c] = silence_token
        row = next_row

        if C >= 2 and row[1] != silence_token:
            output_has_spoken = True
        if all(t == silence_token for t in row):
            all_silent_streak += 1
        else:
            all_silent_streak = 0
        # PROBE PATCH: only allow the all-silent early-stop after Output has
        # spoken, and only after a sustained all-silent run.
        if (not interactive and output_has_spoken
                and all_silent_streak >= EARLY_STOP_THRESHOLD):
            return

        sent_value = yield (row_idx, row, False)
        if interactive and sent_value is not None:
            row[0] = sent_value
            if sent_value != silence_token:
                got_user_input = True


def collect_result(tokenizer, silence_token: int, rows) -> StreamResult:
    """Collect generator output into a StreamResult."""
    all_rows = [row for _, row, _ in rows]
    nc = len(all_rows[0]) if all_rows else C
    names = CHANNELS[:nc]
    channel_texts = {}
    for c, name in enumerate(names):
        col_tokens = [r[c] for r in all_rows]
        non_silence = [t for t in col_tokens if t != silence_token]
        channel_texts[name] = tokenizer.decode(non_silence).strip() if non_silence else ""
    return StreamResult(
        tokens=all_rows,
        channel_texts=channel_texts,
        num_rows=len(all_rows),
        silence_token=silence_token,
    )


class StreamDataCollator:
    """Collator for fine-tuning on the `JonasGeiping/stream-data` (processed) parquet.

    Each input feature is a dict with the 10 channel columns (User, Output,
    Analytical, …, Synthesis), each a list of token ids of length `num_rows`.
    This collator flattens row-by-row, builds the block-causal additive mask,
    shifts labels by `num_channels` (next-row same-channel prediction), and
    returns the batch dict the bundled modeling code expects.

    Use it with HuggingFace's `transformers.Trainer`:

        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
        from stream_inference import StreamDataCollator

        ds = load_dataset("JonasGeiping/stream-data", "processed", split="train")
        tok = AutoTokenizer.from_pretrained(REPO)
        model = AutoModelForCausalLM.from_pretrained(REPO, trust_remote_code=True, torch_dtype=torch.bfloat16)

        collator = StreamDataCollator(pad_token_id=tok.pad_token_id, max_seq_length=4096)
        Trainer(model=model, args=TrainingArguments(...), train_dataset=ds, data_collator=collator).train()

    The flattened sequence layout (same as the training pipeline) is
    `[r0_c0, r0_c1, ..., r0_c9, r1_c0, r1_c1, ...]`. The block-causal mask
    says token `i` attends to token `j` iff `row(j) < row(i) OR j == i` —
    i.e. all prior rows plus the diagonal, never same-row peers.
    """

    NEG_INF = -1e4

    def __init__(
        self,
        pad_token_id: int,
        num_channels: int = C,
        max_seq_length: int | None = None,
        truncation: str = "tail",
    ):
        """Args:
            pad_token_id: Token id used to pad rows below `max_seq_length`.
            num_channels: Number of channels (10 for the released models).
            max_seq_length: If set, truncate flattened sequences to this length
                (rounded down to a multiple of `num_channels`).
            truncation: "tail" (drop trailing rows, keep sample prefix) or
                "head" (drop leading rows, keep sample suffix). Default "tail".
        """
        import torch as _torch

        self._torch = _torch
        self.pad_token_id = pad_token_id
        self.num_channels = num_channels
        self.max_seq_length = max_seq_length
        if truncation not in ("tail", "head"):
            raise ValueError(f"truncation must be 'tail' or 'head', got {truncation!r}")
        self.truncation = truncation

    def _flatten_example(self, ex: dict) -> list[int]:
        """Row-by-row flatten the 10 channel columns into a 1D list."""
        cols = [ex[name] for name in CHANNELS[: self.num_channels]]
        num_rows = len(cols[0])
        for col in cols:
            if len(col) != num_rows:
                raise ValueError(f"channel length mismatch: {len(col)} vs {num_rows}")
        flat = []
        for r in range(num_rows):
            for c in range(self.num_channels):
                flat.append(cols[c][r])
        return flat

    def __call__(self, features: list[dict]) -> dict:
        torch = self._torch
        C = self.num_channels

        flats = [self._flatten_example(f) for f in features]
        max_rows_cap = (self.max_seq_length // C) if self.max_seq_length else None
        if max_rows_cap is not None:
            for i, flat in enumerate(flats):
                if len(flat) // C > max_rows_cap:
                    keep_tokens = max_rows_cap * C
                    if self.truncation == "tail":
                        flats[i] = flat[:keep_tokens]
                    else:  # head
                        flats[i] = flat[-keep_tokens:]

        B = len(flats)
        S_max = max(len(f) for f in flats)
        # Round S_max up to a multiple of C so labels stay row-aligned.
        if S_max % C != 0:
            S_max += C - (S_max % C)

        input_ids = torch.full((B, S_max), self.pad_token_id, dtype=torch.long)
        labels = torch.full((B, S_max), -100, dtype=torch.long)
        position_ids = torch.zeros((B, S_max), dtype=torch.long)
        channel_ids = torch.zeros((B, S_max), dtype=torch.long)
        mask = torch.full((B, 1, S_max, S_max), self.NEG_INF, dtype=torch.float32)

        for b, flat in enumerate(flats):
            S = len(flat)
            ids = torch.tensor(flat, dtype=torch.long)
            input_ids[b, :S] = ids

            pos = torch.arange(S)
            position_ids[b, :S] = pos // C
            channel_ids[b, :S] = pos % C

            rows = pos // C
            can_attend = (rows.unsqueeze(0) < rows.unsqueeze(1)) | torch.eye(S, dtype=torch.bool)
            mask[b, 0, :S, :S] = torch.where(can_attend, 0.0, self.NEG_INF)

            if S > C:
                labels[b, : S - C] = ids[C:]

        # Restore the diagonal everywhere (including padding) so all-masked rows
        # don't produce NaN in softmax.
        diag = torch.arange(S_max)
        mask[:, 0, diag, diag] = 0.0

        return {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
            "channel_ids": channel_ids,
            "attention_mask": {"full_attention": mask, "sliding_attention": mask},
        }


class StreamModel:
    """Stateful wrapper: loads the model + tokenizer and provides .generate()."""

    def __init__(self, model_path: str, device: str = "cuda", device_map: str | None = "auto"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        load_kwargs = dict(trust_remote_code=True, torch_dtype=torch.bfloat16)
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        if device_map is None:
            self.model = self.model.to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.silence_token = detect_silence_token(self.tokenizer)
        self.device = device

    def generate(self, prompt: str, **kwargs) -> StreamResult:
        """Run multi-stream generation and return a StreamResult."""
        rows = list(generate(self.model, self.tokenizer, prompt, self.silence_token, **kwargs))
        return collect_result(self.tokenizer, self.silence_token, rows)

    def generate_stream(self, prompt: str, **kwargs):
        """Generator interface — yields (row_idx, token_ids, is_prefill) per row."""
        yield from generate(self.model, self.tokenizer, prompt, self.silence_token, **kwargs)
