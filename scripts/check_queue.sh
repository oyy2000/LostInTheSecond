#!/bin/bash
squeue -u swang47 --format="%.18i %.30j %.8T %.10M %.20R" > /jet/home/swang47/yang/projects/LostInTheSecond/logs/queue_status.txt 2>&1
