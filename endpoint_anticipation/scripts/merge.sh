#!/bin/bash

datasets="spokenwoz"
# datasets="spokenwoz switchboard "
groups="group4 "
# groups="group8 group9"
# groups="group1 "
print_for_fc="400 480 560"
# groups="group3 "

python3 /mnt/matylda4/udupa/exps/endpointing/NAC-LD-Endpointer/scripts/merge.py --datasets $datasets --groups $groups --print_for_fc $print_for_fc
