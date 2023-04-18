import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import cDataStructs


def valid_smiles(df, prop):
    for idx, smiles in enumerate(df['smiles']):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            df.loc[idx, 'smiles'] = Chem.MolToSmiles(mol)
        else:
            print(idx)
            df.loc[idx, 'smiles'] = np.nan
    df_valid = df.dropna(subset='smiles')
    # df_valid.reset_index(drop=True, inplace=True)
    df_invalid = df.loc[df['smiles'].isnull(),:].copy()
    df_invalid['smiles'] = 'NULL'
    df_invalid.rename(columns={'smiles':prop}, inplace=True)

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


def get_df_ecfp(df):
    print('get_df_ecfp')
    arr_list = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        arr = smiles_to_ecfp(row['smiles'])
        arr_list.append(arr)
    new_df = pd.DataFrame(arr_list)
    new_df.columns = [str(i) for i in range(1024)]
    new_df = pd.merge(new_df, df, left_index=True, right_index=True)

    return new_df
