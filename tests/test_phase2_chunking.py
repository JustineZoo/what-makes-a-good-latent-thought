"""Tests for Phase 2 chunking strategies: Fixed (CoLaR baseline) vs. Step-Aligned (C1)."""
import pytest
from typing import List


def fixed_chunk_tokens(token_ids: List[int], c: int) -> List[List[int]]:
    """Vanilla CoLaR chunking: fixed c tokens per chunk, crossing step boundaries."""
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


def test_chunking_properties():
    # Simulate 3 reasoning steps with lengths 8, 14, 6 tokens
    step1 = list(range(100, 108))  # 8 tokens
    step2 = list(range(200, 214))  # 14 tokens
    step3 = list(range(300, 306))  # 6 tokens
    all_tokens = step1 + step2 + step3  # 28 tokens total

    # Fixed chunking at c=5:
    fixed_chunks = fixed_chunk_tokens(all_tokens, c=5)
    assert len(fixed_chunks) == 6  # 5, 5, 5, 5, 5, 3
    # Check that chunk 1 crosses boundary: 3 tokens from step1 (105..107) + 2 from step2 (200..201)
    assert fixed_chunks[1] == [105, 106, 107, 200, 201]

    # Step-aligned chunking at c'=10:
    # step1 (8 tokens) -> 1 chunk of 8
    # step2 (14 tokens) -> 2 chunks of 10 and 4
    # step3 (6 tokens) -> 1 chunk of 6
    aligned_chunks = step_aligned_chunk_tokens([step1, step2, step3], c_prime=10)
    assert len(aligned_chunks) == 4  # 1 + 2 + 1 = 4 chunks
    assert aligned_chunks[0] == step1
    assert aligned_chunks[1] == step2[:10]
    assert aligned_chunks[2] == step2[10:]
    assert aligned_chunks[3] == step3

    # Verify no chunk contains tokens from two different steps
    for chunk in aligned_chunks:
        prefixes = set(t // 100 for t in chunk)
        assert len(prefixes) == 1, f"Chunk crosses steps: {chunk}"
