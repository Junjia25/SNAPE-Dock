# SNAPE-Dock
## Prerequisites
```
python = 3.70
pytorch = 1.13.0      
pytorch-cuda = 11.7
torch-geometric = 2.3.1
numpy = 1.21.5
pandas = 1.3.5
matplotlib = 3.5.3
rdkit = 2020.09.5
```
## Run
### First, convert the data into graph
```
bash convert_data_into_graph.sh
```
### Predicting Protein-Ligand Docking Structures Using the Provided Model
```
bash predict.sh >predict_output.log
```
### Training your own model
```
bash train_test.sh >train_output.log
```


