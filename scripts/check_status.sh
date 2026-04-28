#!/bin/bash
OUT=/tmp/check_status_out.txt
echo "=== squeue ===" > $OUT
squeue -u swang47 >> $OUT 2>&1
echo "" >> $OUT
echo "=== recent logs ===" >> $OUT
ls -lt /jet/home/swang47/yang/projects/LostInTheSecond/logs/ 2>&1 | head -20 >> $OUT
echo "" >> $OUT
echo "=== job 38128274 ===" >> $OUT
ls -la /jet/home/swang47/yang/projects/LostInTheSecond/logs/fp_train_38128274* >> $OUT 2>&1
echo "" >> $OUT
echo "=== job 38128275 ===" >> $OUT
ls -la /jet/home/swang47/yang/projects/LostInTheSecond/logs/fp_train_38128275* >> $OUT 2>&1
echo "" >> $OUT
echo "=== sacct ===" >> $OUT
sacct -u swang47 --starttime=2026-03-21 --format=JobID,JobName,State,ExitCode,Start,End -n | tail -20 >> $OUT 2>&1
echo "DONE" >> $OUT
