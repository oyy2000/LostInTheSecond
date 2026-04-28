#!/usr/bin/env python3
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.prompt_templates import check_answer

recs = [json.loads(l) for l in open("results/gsm8k_entropy_triggered_sweep/checkpoint.jsonl") if l.strip()]
se = [r for r in recs if r.get("task_type") == "self_eval"]
drafts = {r["doc_id"]: r for r in recs if r.get("task_type") == "draft" and r.get("draft_idx") == 0}

wrong_docs = set()
for did, d in drafts.items():
    if not check_answer("gsm8k", d.get("draft_answer", ""), d.get("gold_answer", "")):
        wrong_docs.add(did)

rows = []
for r in se:
    scores = r.get("step_scores", [])
    n = len(scores)
    if n < 2:
        continue
    mean_s = sum(scores) / n
    worst = min(range(n), key=lambda t: scores[t])
    drop = mean_s - scores[worst]
    tag = "WRONG" if r["doc_id"] in wrong_docs else "ok"
    rows.append((drop, tag, r["doc_id"], worst, scores[worst], mean_s))

rows.sort(key=lambda x: -x[0])
print("Top 20 by relative drop (mean - worst_score):")
for drop, tag, did, worst, ws, ms in rows[:20]:
    print("  %5s %s: drop=%.3f worst=step%d(%.3f) mean=%.3f" % (tag, did, drop, worst, ws, ms))

drops_w = [r[0] for r in rows if r[1] == "WRONG"]
drops_o = [r[0] for r in rows if r[1] == "ok"]
print()
print("WRONG drop: mean=%.4f median=%.4f" % (np.mean(drops_w), np.median(drops_w)))
print("OK    drop: mean=%.4f median=%.4f" % (np.mean(drops_o), np.median(drops_o)))
