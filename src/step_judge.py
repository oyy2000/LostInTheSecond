"""
GPT-based first-error locator for multi-step math trajectories.

Given a question, gold answer, and a list of reasoning steps, asks GPT to
identify the first incorrect step (1-indexed) and propose a minimal correction.

Returns structured JSON: {first_error_step, total_steps, reason, correction}.
"""

import json
import os
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI


def load_env_file(env_path: Path):
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        val = v.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def build_first_error_prompt(
    question: str,
    gold_answer: str,
    steps: List[str],
) -> str:
    numbered = "\n".join(f"[{i+1}] {s}" for i, s in enumerate(steps))
    return f"""You are a rigorous math reasoning verifier.

Question:
{question}

Correct final answer: {gold_answer}

The model produced the following step-by-step solution (which arrives at a WRONG final answer):

{numbered}

Judgment criteria (follow strictly):
- Focus ONLY on mathematical operations. For each step, check whether the
  arithmetic (addition, subtraction, multiplication, percentages, etc.) is
  performed correctly on the numbers it uses.
- A step is CORRECT if: (a) its arithmetic is right, AND (b) the numbers it
  uses come from the problem statement or from a preceding step's result.
  It does not matter what the step calls or labels the result.
- A step is INCORRECT only if: (a) it computes an arithmetic operation wrong
  (e.g. 2+3=6), OR (b) it applies a formula/percentage to a number that
  neither the problem statement nor any preceding step provided as the
  appropriate base for that operation (e.g. the problem says "150% of the
  house price" but the step computes 150% of a different quantity).
- Steps that merely restate the problem, outline a plan, set up notation,
  or compute simple sums of previously stated numbers are correct even if
  the label/name they give the result is imprecise.
- Do NOT mark a step wrong for imprecise wording or naming.
- Do NOT mark a step wrong just because it propagates an earlier mistake.

Task:
1. Walk through every step. For each, decide correct/incorrect by the criteria above.
2. Report the FIRST step that is itself incorrect.
3. Explain in one sentence what error that step introduces.
4. Propose the MINIMAL correction to that step only (keep everything else unchanged).
   The correction must contain ONLY the step content, without the [N] prefix.

Output STRICT JSON (no markdown fences):
{{
  "first_error_step": <int, 1-indexed>,
  "total_steps": {len(steps)},
  "reason": "<one sentence explaining the error this step introduces>",
  "correction": "<corrected step content, no [N] prefix>"
}}
"""


def parse_gpt_json(raw: str) -> Optional[Dict]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                return None
    return None


def call_gpt_first_error(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    max_output_tokens: int = 500,
    retries: int = 5,
    sleep_base: float = 1.0,
) -> Tuple[Optional[Dict], str]:
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.responses.create(
                model=model,
                input=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            raw = (resp.output_text or "").strip()
            if raw:
                parsed = parse_gpt_json(raw)
                return parsed, raw
        except AttributeError:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_output_tokens,
                )
                raw = (resp.choices[0].message.content or "").strip()
                if raw:
                    parsed = parse_gpt_json(raw)
                    return parsed, raw
            except Exception as e:
                last_err = e
        except Exception as e:
            last_err = e
        time.sleep(sleep_base * (2 ** attempt))

    print(f"  [WARN] API failed after {retries} retries: {last_err}", file=sys.stderr)
    return None, ""


def make_client() -> OpenAI:
    kwargs = {}
    if os.environ.get("BASE_URL"):
        kwargs["base_url"] = os.environ["BASE_URL"]
    return OpenAI(**kwargs)
