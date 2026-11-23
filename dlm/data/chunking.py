"""Reasoning-step chunking utilities for Phase 2: C1 (Step-Aligned) vs. Baseline (Fixed)."""
from __future__ import annotations

from typing import List, Tuple


def fixed_chunk_tokens(token_ids: List[int], c: int) -> List[List[int]]:
    """Vanilla CoLaR chunking: fixed c tokens per chunk, crossing step boundaries."""
    if not token_ids:
        return []
    return [token_ids[i : i + c] for i in range(0, len(token_ids), c)]


def step_aligned_chunk_tokens(step_token_lists: List[List[int]], c_prime: int) -> List[List[int]]:
    """Component C1 chunking: chunks never cross step boundaries, capped at c_prime tokens."""
    chunks = []
    for step_tokens in step_token_lists:
        if not step_tokens:
            continue
        for i in range(0, len(step_tokens), c_prime):
            chunks.append(step_tokens[i : i + c_prime])
    return chunks


def build_chunk_indices(
    step_spans: List[Tuple[int, int]],
    total_tokens: int,
    strategy: str = "fixed",
    chunk_size: int = 5,
) -> List[Tuple[int, int]]:
    """
    Returns a list of (start_idx, end_idx) token spans for each chunk.
    
    strategy:
      - 'fixed': chunks of size chunk_size, completely agnostic to step boundaries.
      - 'step_aligned': chunks constrained within each (start, end) step span, capped at chunk_size.
    """
    if strategy == "fixed":
        spans = []
        for i in range(0, total_tokens, chunk_size):
            spans.append((i, min(i + chunk_size, total_tokens)))
        return spans
    elif strategy == "step_aligned":
        spans = []
        for s_start, s_end in step_spans:
            step_len = s_end - s_start
            if step_len <= 0:
                continue
            for i in range(s_start, s_end, chunk_size):
                spans.append((i, min(i + chunk_size, s_end)))
        return spans
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy!r}")
