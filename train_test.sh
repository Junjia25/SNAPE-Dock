#!/bin/bash

python train_test.py \
--gpu_id=7 \
--data_path=/data/zhangjj/SNAPE-Dock/data_graph/pdbbind_rmsd_srand_coor2 \
--output=/data/zhangjj/SNAPE-Dock/output/output_GATv2_lr-5_nlayer3_batch128_heads1_dlayer128_ln+fc_attention \
--model_dir=/data/zhangjj/SNAPE-Dock/output/model_output_GATv2_lr-5_nlayer3_batch128_heads1_dlayer128_ln+fc_attention \
--loss_curve=/data/zhangjj/SNAPE-Dock/output/losscurve_output_GATv2_lr-5_nlayer3_batch128_heads1_dlayer128_ln+fc_attention \
--batch_size=128 \
--epoch=100 \
--n_graph_layer=3 \
--d_graph_layer=128 \
--lr=0.00001 \
--dropout_rate=0.3 \
--start_epoch=1 \
--flexible \
--heads=1 \
--model_type=Net_coor \
--edge_dim=3 \
--loss=MSELoss \
--loss_reduction=mean \
--hinge=0 \
--tot_seed=1 \
--residue 