"""Exact step-level and final-answer evaluator for GSM8K-Aug (torch-free)."""
from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

_STEP_RE = re.compile(r"(?:<<|«)\s*([^<>«»=]+?)\s*=\s*([^<>«»]+?)\s*(?:>>|»)")
_FINAL_HASH_RE = re.compile(r"####\s*([-+]?\$?\d[\d,]*(?:\.\d+)?)")
_NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def parse_number(text: str) -> Optional[float]:
    """Parse a numeric string (handling commas and optional leading $)."""
    if text is None:
        return None
    cleaned = text.strip().replace(",", "").replace("$", "")
    if not cleaned:
        return None
    try:
        val = float(cleaned)
        return val if math.isfinite(val) else None
    except ValueError:
        return None


def safe_eval_expr(expr: str) -> Optional[float]:
    """Safely evaluate an arithmetic expression string using Python AST."""
    if not expr or not expr.strip():
        return None
    cleaned = expr.strip().replace(",", "").replace("$", "")
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError:
        return None

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise OverflowError("exponent too large")
            return float(_BIN_OPS[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return float(_UNARY_OPS[type(node.op)](_eval(node.operand)))
        raise ValueError(f"Unsupported AST node: {type(node).__name__}")

    try:
        val = _eval(tree)
        return val if math.isfinite(val) else None
    except (ZeroDivisionError, ValueError, TypeError, OverflowError):
        return None


def nums_equal(a: Optional[float], b: Optional[float], tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return False
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def extract_steps(text_or_steps: str | Sequence[str]) -> List[Tuple[str, str, Optional[float]]]:
    """Extract (expr, raw_val, parsed_val) tuples from a trace string or step list."""
    joined = text_or_steps if isinstance(text_or_steps, str) else " ".join(text_or_steps)
    out: List[Tuple[str, str, Optional[float]]] = []
    for expr, raw_val in _STEP_RE.findall(joined):
        out.append((expr.strip(), raw_val.strip(), parse_number(raw_val)))
    return out


def extract_final_answer(text: str) -> Optional[float]:
    """Extract final numeric answer after '####', falling back to last step/number."""
    m = _FINAL_HASH_RE.search(text)
    if m:
        return parse_number(m.group(1))
    steps = extract_steps(text)
    if steps and steps[-1][2] is not None:
        return steps[-1][2]
    nums = _NUM_RE.findall(text)
    return parse_number(nums[-1]) if nums else None


@dataclass
class TraceEvalResult:
    answer_correct: bool
    step_exact_acc: float
    arithmetic_consistency: float
    n_gold_steps: int
    n_pred_steps: int
    first_error_pos: int      # 0-indexed first failing step, or -1 if all gold steps match
    error_type: str           # 'none' | 'arithmetic_error' | 'wrong_operand' | 'malformed_step'


def score_prediction(
    pred_text: str,
    gold_steps: Sequence[str],
    gold_answer: str,
) -> TraceEvalResult:
    """Evaluate one generated trace against gold <<expr=value>> steps and final answer."""
    gold_parsed = extract_steps(gold_steps)
    pred_parsed = extract_steps(pred_text)

    gold_ans_val = parse_number(gold_answer)
    pred_ans_val = extract_final_answer(pred_text)
    answer_correct = nums_equal(pred_ans_val, gold_ans_val)

    # Arithmetic consistency across all emitted steps: does expr evaluate to value?
    if pred_parsed:
        consistent = sum(
            1 for expr, _, val in pred_parsed if nums_equal(safe_eval_expr(expr), val)
        )
        arith_consistency = consistent / len(pred_parsed)
    else:
        arith_consistency = 0.0 if gold_parsed else 1.0

    n_gold = len(gold_parsed)
    n_pred = len(pred_parsed)
    if n_gold == 0:
        err = (
            "none"
            if answer_correct
            else ("malformed_step" if pred_ans_val is None else "arithmetic_error")
        )
        return TraceEvalResult(
            answer_correct=answer_correct,
            step_exact_acc=1.0 if (n_pred == 0 or answer_correct) else 0.0,
            arithmetic_consistency=arith_consistency,
            n_gold_steps=0,
            n_pred_steps=n_pred,
            first_error_pos=-1 if answer_correct else 0,
            error_type=err,
        )

    matched = 0
    first_error_pos = -1
    error_type = "none"

    for i, (_, _, g_val) in enumerate(gold_parsed):
        if i >= n_pred:
            if first_error_pos == -1:
                first_error_pos = i
                error_type = "malformed_step"
            continue

        p_expr, _, p_val = pred_parsed[i]
        if nums_equal(p_val, g_val):
            matched += 1
        elif first_error_pos == -1:
            first_error_pos = i
            if p_val is None:
                error_type = "malformed_step"
            elif nums_equal(safe_eval_expr(p_expr), p_val):
                # Equation is self-consistent, so the mistake is wrong operand/setup
                error_type = "wrong_operand"
            else:
                error_type = "arithmetic_error"

    return TraceEvalResult(
        answer_correct=answer_correct,
        step_exact_acc=matched / n_gold,
        arithmetic_consistency=arith_consistency,
        n_gold_steps=n_gold,
        n_pred_steps=n_pred,
        first_error_pos=first_error_pos,
        error_type=error_type,
    )


def summarize_results(results: Sequence[TraceEvalResult]) -> Dict[str, object]:
    """Aggregate TraceEvalResult objects into summary metrics."""
    n = len(results)
    if n == 0:
        return {}
    err_counts = {"none": 0, "arithmetic_error": 0, "wrong_operand": 0, "malformed_step": 0}
    err_positions: List[int] = []
    for r in results:
        err_counts[r.error_type] = err_counts.get(r.error_type, 0) + 1
        if r.first_error_pos >= 0:
            err_positions.append(r.first_error_pos)
    return {
        "n": n,
        "answer_accuracy": sum(r.answer_correct for r in results) / n,
        "step_exact_accuracy": sum(r.step_exact_acc for r in results) / n,
        "arithmetic_consistency": sum(r.arithmetic_consistency for r in results) / n,
        "mean_first_error_pos": (sum(err_positions) / len(err_positions)) if err_positions else -1.0,
        "error_type_counts": err_counts,
    }
