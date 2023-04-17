import pandas as pd
from sklearn import datasets


# iris_datas = datasets.load_iris()
# print(iris_datas.target)

class chemprop_model:
    def __init__(self, data_path, target_columns, dataset_type,
                 save_dir, metric, epochs, model_name='DMPNN',
                 batch_size=256, gpu=0, dropout=0.1, init_lr=0.001,
                 max_lr=0.01, final_lr=0.001):
        self.data_path = data_path
        self.target_columns = target_columns
        self.dataset_type = dataset_type
        self.save_dir = save_dir
        self.metric = metric
        self.epochs = epochs
        self.model_name = model_name
        self.batch_size = batch_size
        self.gpu = gpu
        self.dropout = dropout
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr

    def set_train_args(self):
        for key, value in vars(self).items():
            if hasattr(self.train_args, key):
                setattr(self.train_args, key, value)

a = chemprop_model(data_path='a', target_columns='a', dataset_type='a',
                   save_dir='a', metric='a', epochs=200, model_name='DMPNN',
                   batch_size=256, gpu=0, dropout=0.1, init_lr=0.001,
                   max_lr=0.01, final_lr=0.001)
a.set_train_args()
a.train_args.batch_size


import os
import shutil
filenames = os.listdir('/mnt/home/linjie/projects/solubility/data/classification/modelling_data_1')
filenames = list(set(filenames)-set(['modelling_data_1.csv', 'F_data', 'MDCK_ER', 'PAMPA_permeability']))
os.chdir('/mnt/home/linjie/projects/molprop/molprop/datasets/data/multiclass')
for name in filenames:
    shutil.copytree(os.path.join('/mnt/home/linjie/projects/solubility/data/classification/modelling_data_1', name),
                    name)
os.chdir('/mnt/home/linjie/projects/molprop/molprop/datasets/data/classification')
for name in ['F_data', 'MDCK_ER', 'PAMPA_permeability']:
    shutil.copytree(os.path.join('/mnt/home/linjie/projects/solubility/data/classification/modelling_data_1', name),
                    name)

filenames = os.listdir('/mnt/home/linjie/projects/solubility/data/classification/CYP_hERG_clean_data')
filenames = list(set(filenames)-set(['CYP_hERG_clean_data.csv']))
os.chdir('/mnt/home/linjie/projects/molprop/molprop/datasets/data/classification')
for name in filenames:
    shutil.copytree(os.path.join('/mnt/home/linjie/projects/solubility/data/classification/CYP_hERG_clean_data', name),
                    name)


import os
import shutil

filenames = os.listdir('/mnt/home/linjie/projects/solubility/result/classification/modelling_data_1')
filenames = list(set(filenames)-set(['run_all.py', 'F_data', 'MDCK_ER', 'PAMPA_permeability']))

os.chdir('/mnt/home/linjie/projects/molprop/molprop/models/model_ckpt/classification')
for name in ['F_data', 'MDCK_ER', 'PAMPA_permeability']:
    shutil.copytree(os.path.join('/mnt/home/linjie/projects/solubility/result/classification/modelling_data_1', name, 'AutoGluonModel'),
                    os.path.join(name, 'AutoGluonModel'))


df = pd.read_excel('/mnt/home/linjie/projects/molprop/molprop/datasets/data/raw/classification/modelling_data_1/HHep_T_half.xlsx')
df.dropna(subset=['sub_smiles'], axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)
for idx, smiles in enumerate(df['sub_smiles']):
    smiles_list = smiles.split('.')
    smi = max(smiles_list, key=len, default='')
    df.loc[idx, 'smiles'] = smi
df['name'] = ['name'+str(i) for i in range(len(df))]
df_ = df.loc[:, ['name', 'smiles']]
df_.to_csv('/mnt/home/linjie/projects/molprop/molprop/tests/example.csv', index=False)
df = pd.read_csv('/mnt/home/linjie/projects/molprop/molprop/tests/example.csv')
mol_in = {}
for idx, row in df.iterrows():
    mol_in[row['name']] = row['smiles']

df = pd.read_csv('/mnt/home/linjie/projects/molprop/molprop/tests/example.csv')
mol_in = {}
for idx, row in df.iterrows():
    mol_in[row['name']] = row['smiles']
import json
json_str = json.dumps(mol_in)
with open('/mnt/home/linjie/projects/molprop/molprop/tests/example.json', 'w') as json_file:
    json_file.write(json_str)


df = pd.read_csv('/mnt/home/linjie/projects/molprop/molprop/datasets/data/classification/CYP1A2_data_clean/test_CYP1A2_data_clean.csv')
df['name'] = ['name'+str(i) for i in range(len(df))]
mol_in = {}
for idx, row in df.iterrows():
    mol_in[row['name']] = row['smiles']
from molprop.models.predict_model import predict
res_out = predict(mol_in=mol_in, model='CMPNN', prop='CYP1A2_Class')
print(res_out)

filenames = os.listdir('/mnt/home/linjie/projects/solubility/data/classification/CYP_hERG_clean_data')
filenames = list(set(filenames)-set(['CYP_hERG_clean_data.csv']))
for name in filenames:
    shutil.copytree(os.path.join('/mnt/home/linjie/projects/molprop/molprop/models/model_ckpt/classification', name, 'DMPNN'),
                    os.path.join('/mnt/home/linjie/projects/solubility/result/classification/CYP_hERG_clean_data', name, 'DMPNN'))



