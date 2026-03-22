#!/bin/bash
#
# Monitor SLURM jobs, detect failures, and log status.
# Usage: bash scripts/slurm/monitor_jobs.sh
#

LOGDIR="/ocean/projects/cis250050p/swang47/yang/LostInTheSecond/logs"
MONITOR_LOG="${LOGDIR}/monitor.log"
SLURM_DIR="/jet/home/swang47/yang/projects/LostInTheSecond/scripts/slurm"

JOBS=(
    "37995896:lora_q3b_pf:23a_lora_qwen3b_prefill.sh"
    "37995897:lora_q3b_wr:23b_lora_qwen3b_wr.sh"
    "37995903:lora_l8b_pf:23c_lora_llama8b_prefill.sh"
    "37995904:lora_l8b_wr:23d_lora_llama8b_wr.sh"
    "37995905:ft_q3b_pf:24a_ft_qwen3b_prefill.sh"
    "37995906:ft_q3b_wr:24b_ft_qwen3b_wr.sh"
    "37995907:ft_l8b_pf:24c_ft_llama8b_prefill.sh"
    "37995908:ft_l8b_wr:24d_ft_llama8b_wr.sh"
)

declare -A RESUBMITTED
declare -A COMPLETED
declare -A FINAL_STATUS

mkdir -p "${LOGDIR}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${MONITOR_LOG}"
}

check_job_state() {
    local jid="$1"
    scontrol show job "$jid" 2>/dev/null | grep -oP 'JobState=\K\S+'
}

check_log_errors() {
    local jid="$1"
    local name="$2"
    local pattern="${LOGDIR}/${name//_q3b_pf/}_*"
    
    for errfile in "${LOGDIR}"/*"${jid}".err; do
        [ -f "$errfile" ] || continue
        if [ -s "$errfile" ]; then
            local last_lines
            last_lines=$(tail -20 "$errfile" 2>/dev/null)
            if echo "$last_lines" | grep -qiE "Error|Traceback|FAILED|OOM|CUDA|killed|exception"; then
                echo "$last_lines"
                return 1
            fi
        fi
    done
    return 0
}

get_out_file() {
    local jid="$1"
    for f in "${LOGDIR}"/*"${jid}".out; do
        [ -f "$f" ] && echo "$f" && return
    done
}

get_err_file() {
    local jid="$1"
    for f in "${LOGDIR}"/*"${jid}".err; do
        [ -f "$f" ] && echo "$f" && return
    done
}

log "=========================================="
log "MONITOR STARTED - Tracking ${#JOBS[@]} jobs"
log "=========================================="

POLL_INTERVAL=120  # 2 minutes

while true; do
    all_done=true
    running_count=0
    pending_count=0
    completed_count=0
    failed_count=0

    for entry in "${JOBS[@]}"; do
        IFS=':' read -r jid name script <<< "$entry"

        if [ "${COMPLETED[$jid]:-}" = "1" ]; then
            ((completed_count++))
            continue
        fi

        state=$(check_job_state "$jid")

        case "$state" in
            PENDING)
                all_done=false
                ((pending_count++))
                ;;
            RUNNING)
                all_done=false
                ((running_count++))
                outfile=$(get_out_file "$jid")
                if [ -n "$outfile" ] && [ -f "$outfile" ]; then
                    progress=$(tail -5 "$outfile" 2>/dev/null | grep -oP '\d+%' | tail -1)
                    if [ -n "$progress" ]; then
                        log "  [RUNNING] $name ($jid) - progress: $progress"
                    else
                        last_line=$(tail -1 "$outfile" 2>/dev/null)
                        log "  [RUNNING] $name ($jid) - last: ${last_line:0:100}"
                    fi
                else
                    log "  [RUNNING] $name ($jid) - no output yet"
                fi
                ;;
            COMPLETED)
                errfile=$(get_err_file "$jid")
                outfile=$(get_out_file "$jid")
                
                has_error=false
                if [ -n "$errfile" ] && [ -f "$errfile" ] && [ -s "$errfile" ]; then
                    err_content=$(tail -20 "$errfile" 2>/dev/null)
                    if echo "$err_content" | grep -qiE "Traceback|Error.*:.*|FAILED|OOM|killed"; then
                        has_error=true
                    fi
                fi
                
                if [ -n "$outfile" ] && [ -f "$outfile" ]; then
                    if ! grep -q "Completed at" "$outfile" 2>/dev/null; then
                        has_error=true
                    fi
                fi

                if $has_error; then
                    log "  [FAILED!] $name ($jid) completed with errors"
                    if [ -n "$errfile" ] && [ -f "$errfile" ]; then
                        log "  === Last 15 lines of stderr ==="
                        tail -15 "$errfile" 2>/dev/null | while IFS= read -r line; do
                            log "    $line"
                        done
                    fi
                    
                    if [ "${RESUBMITTED[$jid]:-}" != "1" ]; then
                        log "  -> Resubmitting: sbatch ${SLURM_DIR}/${script}"
                        new_jid=$(sbatch --parsable "${SLURM_DIR}/${script}" 2>&1)
                        if [[ "$new_jid" =~ ^[0-9]+$ ]]; then
                            log "  -> Resubmitted as new job $new_jid"
                            RESUBMITTED[$jid]="1"
                            COMPLETED[$jid]="1"
                            FINAL_STATUS[$jid]="FAILED->resubmitted:$new_jid"
                            JOBS+=("${new_jid}:${name}:${script}")
                            ((failed_count++))
                        else
                            log "  -> Resubmit FAILED: $new_jid"
                            COMPLETED[$jid]="1"
                            FINAL_STATUS[$jid]="FAILED"
                            ((failed_count++))
                        fi
                    fi
                else
                    log "  [SUCCESS] $name ($jid) completed successfully"
                    COMPLETED[$jid]="1"
                    FINAL_STATUS[$jid]="SUCCESS"
                    ((completed_count++))
                fi
                ;;
            FAILED|TIMEOUT|CANCELLED*|NODE_FAIL|OUT_OF_MEMORY)
                log "  [FAILED!] $name ($jid) - state: $state"
                errfile=$(get_err_file "$jid")
                if [ -n "$errfile" ] && [ -f "$errfile" ]; then
                    log "  === Last 15 lines of stderr ==="
                    tail -15 "$errfile" 2>/dev/null | while IFS= read -r line; do
                        log "    $line"
                    done
                fi

                if [ "${RESUBMITTED[$jid]:-}" != "1" ]; then
                    log "  -> Resubmitting: sbatch ${SLURM_DIR}/${script}"
                    new_jid=$(sbatch --parsable "${SLURM_DIR}/${script}" 2>&1)
                    if [[ "$new_jid" =~ ^[0-9]+$ ]]; then
                        log "  -> Resubmitted as new job $new_jid"
                        RESUBMITTED[$jid]="1"
                        COMPLETED[$jid]="1"
                        FINAL_STATUS[$jid]="${state}->resubmitted:$new_jid"
                        JOBS+=("${new_jid}:${name}:${script}")
                    else
                        log "  -> Resubmit FAILED: $new_jid"
                        COMPLETED[$jid]="1"
                        FINAL_STATUS[$jid]="$state"
                    fi
                    ((failed_count++))
                fi
                ;;
            "")
                log "  [UNKNOWN] $name ($jid) - job not found in scheduler (may have completed)"
                outfile=$(get_out_file "$jid")
                if [ -n "$outfile" ] && [ -f "$outfile" ] && grep -q "Completed at" "$outfile" 2>/dev/null; then
                    log "  -> Output confirms successful completion"
                    COMPLETED[$jid]="1"
                    FINAL_STATUS[$jid]="SUCCESS"
                    ((completed_count++))
                else
                    COMPLETED[$jid]="1"
                    FINAL_STATUS[$jid]="UNKNOWN"
                    ((failed_count++))
                fi
                ;;
        esac
    done

    log "--- Summary: ${pending_count} pending, ${running_count} running, ${completed_count} completed, ${failed_count} failed ---"

    if $all_done; then
        log "=========================================="
        log "ALL JOBS FINISHED"
        log "=========================================="
        for entry in "${JOBS[@]}"; do
            IFS=':' read -r jid name script <<< "$entry"
            log "  $name ($jid): ${FINAL_STATUS[$jid]:-UNKNOWN}"
        done
        log "=========================================="
        break
    fi

    sleep $POLL_INTERVAL
done
