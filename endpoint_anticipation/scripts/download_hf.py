#!/bin/bash

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="CASPER-SSSD/CASPER",
    repo_type="dataset",
    local_dir="/mnt/matylda4/udupa/data/",
    local_dir_use_symlinks=False
)
