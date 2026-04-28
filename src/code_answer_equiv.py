"""
Code solution correctness checker for CodeContests.

Uses test-case execution to determine if a generated Python solution
is correct. Falls back to string matching when execution is unavailable.

Public API
----------
- run_solution(code, input_str, timeout) -> (stdout, stderr, ok)
- check_solution(code, public_tests, timeout) -> (pass_rate, details)
- split_code_blocks(code) -> List[str]
- extract_python_code(text) -> str
"""

import os
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def run_solution(
    code: str,
    input_str: str,
    timeout: float = 10.0,
) -> Tuple[str, str, bool]:
    """Execute *code* with *input_str* on stdin; return (stdout, stderr, ok)."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(code)
        f.flush()
        tmp_path = f.name

    proc = None
    try:
        proc = subprocess.Popen(
            ["python3", tmp_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        stdout, stderr = proc.communicate(
            input=input_str, timeout=timeout)
        ok = proc.returncode == 0
        return stdout, stderr, ok
    except subprocess.TimeoutExpired:
        if proc is not None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            proc.wait()
        return "", "TIMEOUT", False
    except Exception as e:
        if proc is not None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            proc.wait()
        return "", str(e), False
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _normalize_output(s: str) -> str:
    return s.strip().replace("\r\n", "\n")


def check_solution(
    code: str,
    inputs: List[str],
    outputs: List[str],
    timeout: float = 10.0,
    early_exit: bool = True,
) -> Tuple[float, List[Dict]]:
    """Run *code* against paired (inputs, outputs) test cases.

    Returns (pass_rate, details) where details is a list of per-test dicts.
    When *early_exit* is True, stops at the first failing test case.
    """
    if not inputs or len(inputs) != len(outputs):
        return 0.0, []

    details = []
    n_pass = 0
    for i, (inp, expected) in enumerate(zip(inputs, outputs)):
        stdout, stderr, ok = run_solution(code, inp, timeout=timeout)
        match = ok and _normalize_output(stdout) == _normalize_output(expected)
        if match:
            n_pass += 1
        details.append({
            "test_idx": i,
            "passed": match,
            "expected": _normalize_output(expected),
            "actual": _normalize_output(stdout) if ok else f"ERROR: {stderr[:200]}",
        })
        if early_exit and not match:
            break

    pass_rate = n_pass / len(inputs) if inputs else 0.0
    return pass_rate, details


def split_code_lines(code: str) -> List[str]:
    """Split code into non-empty lines, preserving line numbers."""
    return code.splitlines()


def split_code_blocks(code: str) -> List[str]:
    """Split code into logical blocks (separated by blank lines).

    Each block is a contiguous group of non-empty lines.
    Returns blocks as strings (with internal newlines preserved).
    """
    lines = code.splitlines()
    blocks, current = [], []
    for line in lines:
        if line.strip() == "":
            if current:
                blocks.append("\n".join(current))
                current = []
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def extract_python_code(text: str) -> str:
    """Extract code from markdown fences if present."""
    if "```python" in text:
        start = text.index("```python") + len("```python")
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()
    if "```" in text:
        start = text.index("```") + 3
        if text[start:start+1] == "\n":
            start += 1
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()
    return text.strip()


def check_humaneval(
    code: str,
    test_code: str,
    entry_point: str,
    timeout: float = 5.0,
) -> bool:
    """Check a HumanEval solution by running code + test harness.

    Returns True if all assertions pass.
    """
    full = code + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(full)
        f.flush()
        tmp_path = f.name

    proc = None
    try:
        proc = subprocess.Popen(
            ["python3", tmp_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        _, stderr = proc.communicate(timeout=timeout)
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        if proc is not None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            proc.wait()
        return False
    except Exception:
        if proc is not None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            proc.wait()
        return False
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def check_mbpp(
    code: str,
    test_list: List[str],
    test_imports: Optional[List[str]] = None,
    timeout: float = 5.0,
) -> bool:
    """Check an MBPP solution by running code + assert statements."""
    imports = "\n".join(test_imports) + "\n" if test_imports else ""
    tests = "\n".join(test_list)
    full = imports + code + "\n\n" + tests + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(full)
        f.flush()
        tmp_path = f.name

    proc = None
    try:
        proc = subprocess.Popen(
            ["python3", tmp_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        _, _ = proc.communicate(timeout=timeout)
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        if proc is not None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            proc.wait()
        return False
    except Exception:
        if proc is not None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            proc.wait()
        return False
    finally:
        Path(tmp_path).unlink(missing_ok=True)
