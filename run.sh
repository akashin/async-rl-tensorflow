#!/bin/bash

join() { local IFS="$1"; shift; echo "$*"; }
echo_and_run() { echo "$@"; $@; }

ps_ports=()
for ((i=0;i<ps_num;i++)); do
  ps_ports+=(0.0.0.0:$(($start_port+$i)))
done

worker_ports=()
for ((i=0;i<worker_num;i++)); do
  worker_ports+=(0.0.0.0:$(($start_port+ps_num+$i)))
done

ps_hosts=`join , "${ps_ports[@]}"`
worker_hosts=`join , "${worker_ports[@]}"`

for ((i=0;i<$ps_num;i++)); do
  echo_and_run python3 main.py \
      --ps_hosts=$ps_hosts \
      --worker_hosts=$worker_hosts \
      --job_name=ps --task_index=$i "$@" &
done

for ((i=0;i<$worker_num;i++)); do
  echo_and_run python3 main.py \
      --ps_hosts=$ps_hosts \
      --worker_hosts=$worker_hosts \
      --job_name=worker --task_index=$i "$@" &
done
