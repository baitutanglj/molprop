import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import cDataStructs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import column_or_1d
from tqdm import tqdm

def valid_smiles(df):
    for idx, smiles in enumerate(df['smiles']):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            df.loc[idx, 'smiles'] = Chem.MolToSmiles(mol)
        else:
            print(idx)
            df.loc[idx, 'smiles'] = np.nan
    df_valid = df.dropna(subset='smiles')
    # df_valid.reset_index(drop=True, inplace=True)
    df_invalid = df.loc[df['smiles'].isnull(),:]
    return df_valid, df_invalid


def smiles_to_ecfp(smiles, size=1024):
    """Converts a single SMILES into an ECFP4

    Parameters:
        smiles (str): The SMILES string .
        size (int): Size (dimensions) of the ECFP4 vector to be calculated.

    Returns:
        ecfp4 (arr): An n dimensional ECFP4 vector of the molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=size)
    arr = np.zeros((0,), dtype=np.int8)
    cDataStructs.ConvertToNumpyArray(ecfp, arr)
    return arr

def get_df_ecfp(df, target_column, dataset_type):
    print('get_df_ecfp')
    arr_list = []
    for idx, el in df.iterrows():
        arr = smiles_to_ecfp(el['smiles'])
        arr_list.append(arr)
    new_df = pd.DataFrame(arr_list)
    new_df.columns = [str(i) for i in range(1024)]
    new_df['smiles'] = df['smiles']
    new_df[target_column] = df[target_column]
    if dataset_type != 'regression':
        new_df['target_label'] = df['target_label']

    return new_df

def get_smiles_df_ecfp(df):
    print('get_df_ecfp')
    arr_list = []
    for idx, el in df.iterrows():
        arr = smiles_to_ecfp(el['smiles'])
        arr_list.append(arr)
    new_df = pd.DataFrame(arr_list)
    new_df.columns = [str(i) for i in range(1024)]
    new_df['smiles'] = df['smiles']

    return new_df

def remove_salt(data_path, smiles_column='sub_smiles', target_column=None):
    if os.path.splitext(data_path)[-1] == '.xlsx':
        df = pd.read_excel(data_path)
    elif os.path.splitext(data_path)[-1] == '.csv':
        df = pd.read_csv(data_path)
    if target_column:
        df.dropna(subset=[target_column], axis=0, inplace=True)
    df.dropna(subset=[smiles_column], axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)

    for idx, smiles in tqdm(enumerate(df[smiles_column]), total=len(df)):
        smiles_list = smiles.split('.')
        smi = max(smiles_list, key=len, default='')
        mol = Chem.MolFromSmiles(smi)
        if mol:
            df.loc[idx, 'smiles'] = Chem.MolToSmiles(mol, canonical=True)
    df.dropna(subset=['smiles'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def df_drop_duplicated(df, target_column, dataset_type, save_path):
    base_dir = os.path.join(os.path.abspath(os.path.join(save_path, "../..")))
    base_dir_name = Path(base_dir).stem

    data_summary = {base_dir_name: Path(save_path).stem, 'raw': len(df)}
    print('before len(df)=', len(df))
    df_duplicated = df.loc[df.duplicated(subset='smiles', keep=False), :]
    smiles_key = list(set(df_duplicated['smiles']))
    print(f'df_duplicated: {len(df_duplicated)}| smiles_key:{len(smiles_key)}')
    if dataset_type == 'regression':
        # fun = lambda x: x.fill(df.groupby('smiles').target_column.mean()[x.target_column])
        # df = df.apply(lambda x: fun(x), axis=1)
        for key in smiles_key:
            target_value = np.mean(df.loc[df['smiles'] == key, target_column])
            df.loc[df['smiles'] == key, target_column] = target_value
        df.drop_duplicates(subset='smiles', keep='first', inplace=True, ignore_index=True)
        data_summary.update({'processed': len(df), 'target_column': target_column})
    else:
        invalid_target_value = []
        print(f'data[{target_column}] classification level: {pd.unique(df[target_column]).tolist()}')
        df_groups = df.groupby('smiles')
        for smiles, group in df_groups:
            if len(set(group[target_column])) > 1:
                invalid_target_value.append(smiles)
        df = df.loc[~df['smiles'].isin(invalid_target_value), :]
        df.reset_index(drop=True, inplace=True)
        df = df.drop_duplicates(subset='smiles', keep='first', ignore_index=True)
        data_summary.update({'processed': len(df), 'target_column': target_column, 'level': str(pd.unique(df[target_column]))})
    print('after len(df)=', len(df))
    print('after df_duplicated', len(df.loc[df.duplicated(subset='smiles', keep=False), :]))

    df.to_csv(save_path, index=False)
    if os.path.exists(os.path.join(base_dir, base_dir_name+'.csv')):
        header=False
    else:
        header=True
    pd.DataFrame(data_summary, index=[0]).to_csv(os.path.join(base_dir, base_dir_name+'.csv'), mode='a', header=header, index=False)

    return df

class MyLabelEncoder(LabelEncoder):

    def fit(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_ = pd.Series(y).unique()
        return self

def label_encoder(df, target_column, label_encoder_path):
    label = np.unique(df[target_column])
    if 'High' in label:
        label = label[::-1]
    else:
        label = label
    le = LabelEncoder()
    le.fit(label)
    df['target_label'] = le.transform(df[target_column])
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(le, f)

    # with open(label_encoder_path, 'rb') as f:
    #     le_departure = pickle.load(f)
    #     df['target_label'] = le.transform(df[target_column])
    return df



