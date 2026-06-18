# Human Eval Rubric (5-case overlap)

Score each generated answer from run `20260617_104356` on two axes (0.0–1.0):

**Faithfulness** — Is every factual claim supported by the retrieved handbook context?  
- 1.0 = fully grounded, no unsupported claims  
- 0.5 = mostly grounded with minor extrapolation  
- 0.0 = hallucination or unsupported policy language  

**Answer relevancy** — Does the answer usefully address the employee's question?  
- 1.0 = directly answers; correct abstention when topic absent counts as relevant if honest  
- 0.5 = partial or overly vague  
- 0.0 = off-topic or silent when answer existed in context  

Record scores in `human_eval_scores.json`. Compare against LLM judge via `scripts/compare_human_judge.py`.