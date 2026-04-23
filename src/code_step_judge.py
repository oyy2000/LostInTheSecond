"""
GPT-based first-error locator for multi-line code solutions.

Given a problem description, test cases, and a wrong code solution,
asks GPT to identify the first incorrect logical block and propose
a minimal correction.

Returns structured JSON:
{first_error_block, total_blocks, reason, correction}
"""

import json
import os
import sys
import time
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


# Placeholder -- will be filled in below
PROMPT_TEMPLATE = ""


def build_first_error_prompt(
    problem_description: str,
    sample_input: str,
    expected_output: str,
    code_blocks: List[str],
) -> str:
    numbered = "\n".join(
        f"[Block {i+1}]\n{b}" for i, b in enumerate(code_blocks)
    )
    return f"""You are a rigorous competitive-programming code verifier.

Problem Description:
{problem_description}

Sample Input:
{sample_input}

Expected Output:
{expected_output}

The programmer produced the following solution (which gives a WRONG answer).
The code has been split into logical blocks:

{numbered}

Judgment criteria (follow strictly):
- Focus on LOGICAL correctness. For each block, check whether the logic
  (algorithm, data structures, control flow, arithmetic) is correct given
  the problem statement and preceding blocks.
- A block is CORRECT if: (a) its logic correctly implements the intended
  algorithm step, AND (b) it correctly uses variables/values from preceding
  blocks or the problem input.
- A block is INCORRECT if: (a) it has a logical bug (wrong formula, off-by-one,
  wrong condition, wrong data structure usage), OR (b) it misinterprets the
  problem constraints, OR (c) it uses a variable incorrectly from a preceding
  block.
- Blocks that read input, import modules, or define correct helper functions
  are correct unless they contain a bug.
- Do NOT mark a block wrong just because it propagates an earlier mistake.
- Do NOT mark a block wrong for style issues.

Task:
1. Walk through every block. For each, decide correct/incorrect.
2. Report the FIRST block that is itself incorrect.
3. Explain in one sentence what bug that block introduces.
4. Propose the MINIMAL correction to that block only (keep everything else
   unchanged). The correction must contain ONLY the block content.

Output STRICT JSON (no markdown fences):
{{
  "first_error_block": <int, 1-indexed>,
  "total_blocks": {len(code_blocks)},
  "reason": "<one sentence explaining the bug>",
  "correction": "<corrected block content>"
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
    max_output_tokens: int = 800,
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

    print(f"  [WARN] API failed after {retries} retries: {last_err}",
          file=sys.stderr)
    return None, ""


def make_client() -> OpenAI:
    kwargs = {}
    if os.environ.get("BASE_URL"):
        kwargs["base_url"] = os.environ["BASE_URL"]
    return OpenAI(**kwargs)
