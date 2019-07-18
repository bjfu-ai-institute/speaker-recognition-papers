#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$PWD:../../../
export PATH=$PATH:$PWD
export TF_CPP_MIN_LOG_LEVEL=2

save_dir=`cat config.json | grep save_path | awk -F'"' '{print $4}'`

mkdir -p $save_dir/log $save_dir/model $save_dir/backup

if [ -e $save_dir/data ];then
    echo ""
else
    mkdir -p $save_dir/data
    python3 create_record.py
fi


n=`ls $save_dir | wc -l`
if [ n -eq 0 ]; then
    time=`data | awk '{print $4}'`
    mkdir -p $save_dir/backup/$time
    if [ -e $save_dir/log ]; then
        mv $save_dir/log $save_dir/backup/$time/
    fi
    if [ -e $save_dir/model ]; then
        mv $save_dir/model $save_dir/backup/$time/
    fi
fi
    
python3 train_deepspeaker.py
