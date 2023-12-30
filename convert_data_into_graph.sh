#!/bin/bash

# Convert data to disk
python convert_data_into_graph.py \
--groundtruth_dir=/data/zhangjj/SNAPE-Dock/data_init/PDBbind_v2020_all \
--pdbbind_dir=/data/zhangjj/SNAPE-Dock/data_init/data_dock6_pose \
--input_list=/data/zhangjj/SNAPE-Dock/data_init/pdb_list_ \
--label_list_file=data_graph \
--output_file=pdbbind_rmsd_srand_coor2 \
--pdb_version=2018 \
--dataset=coor2 \
--use_new_data \
--bond_th=6 \
--pocket_th=12 \
--thread_num=50 \
--cv=0 

# Create raw data directory
mkdir data_graph/pdbbind_rmsd_srand_coor2/raw

# Move test data to raw directory
mv data_graph/pdbbind_rmsd_srand_coor2/test/ data_graph/pdbbind_rmsd_srand_coor2/raw/

# Move train data to raw directory
mv data_graph/pdbbind_rmsd_srand_coor2/train/ data_graph/pdbbind_rmsd_srand_coor2/raw/
