#!/bin/bash
OUT=/tmp/lora_gpt4o_eval_status.txt
{
    echo "=== $(date) ==="
    echo "--- squeue ---"
    squeue -u swang47 2>&1
    echo ""
    echo "--- sacct for 38138805 ---"
    sacct -j 38138805 --format=JobID,State,Start,End,Elapsed,ExitCode 2>&1
    echo ""
    echo "--- log file check ---"
    ls -la /ocean/projects/cis250050p/swang47/yang/LostInTheSecond/logs/lemma_gsm8k_38138805.out 2>&1
    if [ -f /ocean/projects/cis250050p/swang47/yang/LostInTheSecond/logs/lemma_gsm8k_38138805.out ]; then
        echo "--- log tail ---"
        tail -50 /ocean/projects/cis250050p/swang47/yang/LostInTheSecond/logs/lemma_gsm8k_38138805.out 2>&1
    fi
} > "$OUT" 2>&1
echo "Written to $OUT"
