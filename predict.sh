#!/bin/bash

python predict.py \
--gpu_id=5 \
--data_dir=/data/zhangjj/SNAPE-Dock/data_graph/pdbbind_rmsd_srand_coor2 \
--prediction_model=/data/zhangjj/SNAPE-Dock/best_model_plt/model_25.pt \
--batch_size=1 \
# --output_csv_path=/data/zhangjj/MedusaGraph3.0/pred_output/pred_coreset_by_test550_GATv2_lr-5_nlayer3_batch128_heads1_dlayer128_ln+fc_notresidue_model45.csv