import argparse
import numpy as np
import csv
import os
import sys

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

from dataset import PDBBindCoor, PDBBindNextStep2
from model import Net_coor, Net_screen
from molecular_optimization import get_refined_pose_file
import math

from time import time 
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", help="id of gpu", type=int, default = 1)
parser.add_argument("--data_dir", help="the directory containing the input data", type=str, default='/data/zhangjj/MedusaGraph/Graph_input_coreset_rest/pdbbind_rmsd_srand_coor2')
parser.add_argument("--prediction_model", help="path to the trained prediction model file", type=str, default='/data/zhangjj/MedusaGraph3.0/train_output/coreset_models_4_256_atom_hinge0/model_39.pt')
parser.add_argument("--batch_size", help="batch_size", type=int, default = 1)
parser.add_argument("--iterative", help="if we iteratively calculate the pose", type=int, default = 0)
parser.add_argument("--tot_seed", help="num of seeds in the dataset", type=int, default = 1)
# parser.add_argument("--output_csv_path", help="output csv data path", type=str, default="/data/zhangjj/MedusaGraph3.0/pred_output/output_xyz.csv")
args = parser.parse_args()
print(args)

gpu_id = str(args.gpu_id)
device_str = 'cuda:' + gpu_id if torch.cuda.is_available() else 'cpu'
device = torch.device(device_str)

# device = torch.device('cpu')

data_dir = args.data_dir
prediction_model = args.prediction_model

def generate_pose(data_dir, prediction_model, device):
	# load test dataset
	test_dataset=PDBBindCoor(root=data_dir, split='test', data_type='autodock')
	test_loader=DataLoader(test_dataset, batch_size=args.batch_size)

	test_dataset_size = len(test_dataset)
	test_loader_size = len(test_loader.dataset)
	print(f"test_dataset_size: {test_dataset_size}, test_loader_size: {test_loader_size}")
	
	t = time()

	model = torch.load(prediction_model).to(device)

	diff_complex = 0
	rmsd_per_pdb = []
	num_pose_per_pdb = []
	rmsd_per_pdb_in = []
	pdb = ''
	pdbs = []

	all_rmsds = []

	out_atom_list = []
	node_weights_list = []

	pbar = tqdm(total=test_loader_size)
	pbar.set_description('Testing poses...')

	for data in test_loader:
		pbar.update(1)
		num_atoms = data.x.size()[0]
		num_flexible_atoms = data.x[data.flexible_idx.bool()].size()[0]
	
		if data.pdb != pdb:
			diff_complex += 1
			rmsd_per_pdb.append(0.0)
			rmsd_per_pdb_in.append(0.0)
			num_pose_per_pdb.append(0)
			pdb = data.pdb
			pdbs.append(pdb[0])

		if data.x.size()[0] != num_atoms:
			print(f"num_flexible_atoms: {num_flexible_atoms}, data.x.size: {data.x.size()[0]}, data.y.size: {num_atoms}")
		
		# 获取模型的输出1
		# out = model(data.x.to(device), data.edge_index.to(device), data.dist.to(device))[data.flexible_idx.bool()]

		# 获取模型的输出2
		out, node_weights = model(data.x.to(device), data.edge_index.to(device), data.dist.to(device))
		# 使用data.flexible_idx.bool()对out和node_weights进行索引
		out = out[data.flexible_idx.bool()]
		# node_weights = node_weights[data.flexible_idx.bool()]  # 输出为tensor格式
		node_weights = node_weights.cpu().detach().numpy()[data.flexible_idx.bool()]  # 输出为numpy格式

		# 计算RMSD
		rmsds = math.sqrt(F.mse_loss(data.y.to(device), out, reduction='sum').cpu().item() / num_flexible_atoms)
		all_rmsds.append(rmsds)
		num_pose_per_pdb[-1] += 1
		rmsd_per_pdb[-1] += rmsds

		# 存储out_atom和node_weights
		out_atom_list.append(out)
		node_weights_list.append(node_weights)
	
	pbar.close()
	tt = time() - t
	print(f"Spend {tt}s")

	# 数量统计
	print(f'diff_complex {diff_complex}')
	assert diff_complex % args.tot_seed == 0
	diff_complex = diff_complex // args.tot_seed
	print(f'diff_complex {diff_complex}')
	for ii in range(1, args.tot_seed):
		for jj in range(diff_complex):
			num_pose_per_pdb[jj] += num_pose_per_pdb[ii * diff_complex + jj]
			rmsd_per_pdb[jj] += rmsd_per_pdb[ii * diff_complex + jj]
			rmsd_per_pdb_in[jj] += rmsd_per_pdb_in[ii * diff_complex + jj]

	return all_rmsds, rmsd_per_pdb, num_pose_per_pdb, pdbs, out_atom_list, node_weights_list

all_rmsds, rmsd_per_pdb, num_pose_per_pdb, pdbs, out_atom_list, node_weights_list = generate_pose(data_dir, prediction_model, device)

# # 存储output坐标向量组
# with open(args.output_csv_path, 'w') as f:
#     writer = csv.writer(f)  
#     writer.writerow(['Prediction'])
    
#     for pred in out_atom_list:
#         writer.writerow([pred])

# 输出模型预测结果
print(f"all_rmsds: {all_rmsds}")
print(f"rmsd_per_pdb: {rmsd_per_pdb}")
print(f"num_pose_per_pdb: {num_pose_per_pdb}")
print(f"pdbs: {pdbs}")
# print(f"out_atom_list: {out_atom_list}")
print(f"node_weights_list: {node_weights_list}")
