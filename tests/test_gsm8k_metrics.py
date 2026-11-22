from dlm.data.gsm8k_aug import format_prompt, format_target
from dlm.eval.gsm8k_metrics import (
    extract_final_answer,
    extract_steps,
    safe_eval_expr,
    score_prediction,
    summarize_results,
)


def test_safe_eval_and_parsing():
    assert safe_eval_expr("600*30/100") == 180.0
    assert safe_eval_expr("(12 + 8) / 4") == 5.0
    assert safe_eval_expr("__import__('os')") is None

    steps = extract_steps("<<600*30/100=180>> <<600*10/100=60>>\n#### 360")
    assert len(steps) == 2
    assert steps[0] == ("600*30/100", "180", 180.0)
    assert extract_final_answer("<<600-240=360>>\n#### 360") == 360.0


def test_error_taxonomy():
    gold_steps = ["<<600*30/100=180>>", "<<600*10/100=60>>", "<<180+60=240>>", "<<600-240=360>>"]

    # 1. Exact match
    r_ok = score_prediction("<<600*30/100=180>> <<600*10/100=60>> <<180+60=240>> <<600-240=360>>\n#### 360", gold_steps, "360")
    assert r_ok.answer_correct is True
    assert r_ok.step_exact_acc == 1.0
    assert r_ok.first_error_pos == -1
    assert r_ok.error_type == "none"

    # 2. Arithmetic error at step 1 (600*10/100=70 is false arithmetic)
    r_arith = score_prediction("<<600*30/100=180>> <<600*10/100=70>> <<180+70=250>> <<600-250=350>>\n#### 350", gold_steps, "350")
    assert r_arith.first_error_pos == 1
    assert r_arith.error_type == "arithmetic_error"

    # 3. Wrong operand at step 1 (600*20/100=120 is internally valid arithmetic, wrong setup)
    r_op = score_prediction("<<600*30/100=180>> <<600*20/100=120>> <<180+120=300>> <<600-300=300>>\n#### 300", gold_steps, "300")
    assert r_op.first_error_pos == 1
    assert r_op.error_type == "wrong_operand"

    # 4. Malformed / truncated trace
    r_mal = score_prediction("The answer is 360", gold_steps, "360")
    assert r_mal.answer_correct is True
    assert r_mal.first_error_pos == 0
    assert r_mal.error_type == "malformed_step"

    summary = summarize_results([r_ok, r_arith, r_op, r_mal])
    assert summary["n"] == 4
    assert summary["error_type_counts"]["arithmetic_error"] == 1
    assert summary["error_type_counts"]["wrong_operand"] == 1
    assert summary["error_type_counts"]["malformed_step"] == 1


def test_formatting():
    p = format_prompt("Out of 600 employees...")
    assert p == "Question: Out of 600 employees...\nSolution:\n"
    assert format_target(["<<1+1=2>>"], "2", mode="cot") == "<<1+1=2>>\n#### 2"
    assert format_target(["<<1+1=2>>"], "2", mode="no_cot") == "#### 2"
