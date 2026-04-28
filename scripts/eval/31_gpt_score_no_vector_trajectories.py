#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT 5.1 Step-Level Scoring for runs/no_vector baselines.

Reuses the GPT step-verification logic from 28_gpt_step_verification.py,
adapted for lm-eval JSONL trajectories. Scores each generation step with
GPT 5.1 (CORRECT/INCORRECT), caches results, and plots per-step correctness
curves (correct vs wrong vs all) for every (task, model) combination.

Usage:
    # Full run: score + plot
    python scripts/eval/31_gpt_score_no_vector_trajectories.py

    # Specific models only
    python scripts/eval/31_gpt_score_no_vector_trajectories.py \
        --models Llama-3.2-1B-Instruct Llama-3.2-3B-Instruct Llama-3.1-8B-Instruct

    # Plot only from cached scores
    python scripts/eval/31_gpt_score_no_vector_trajectories.py --plot-only

    # Dry run (print first prompt, no API calls)
    python scripts/eval/31_gpt_score_no_vector_trajectories.py --dry-run
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Stub: rest filled via StrReplace
