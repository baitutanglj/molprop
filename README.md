# molprop
This repository contains `AutoGluonModel`, `DMPNN` and `CMPNN` model to predict molecular property.
- The models are trained separately from the data: `['ELD_activity_value', 'ESL_pLogS', 'CYP1A2_data_clean', 'CYP2C9_data_clean', 'CYP2C19_data_clean', 'CYP2D6_data_clean', 'CYP3A4_data_clean', 'HERG-220720_clean', 'HHep_T_half', 'HLM_T_half', 'MLM_T_half', 'RLM_T_half', 'HHepClint', 'HLMClint_ml_min_kg', 'HLMClint_ul_min_mg', 'RHepClint', 'RLMClint_ml_min_kg', 'RLMClint_ul_min_mg', 'F_data', 'MDCK_ER', 'PAMPA_permeability']`
- molecular property list: `['activity_value', 'pLogS', 'CYP1A2_Class', 'CYP2C9_Class', 'CYP2C19_Class', 'CYP2D6_Class', 'CYP3A4_Class', 'hERG_Class', 't1/2_Class', 'HHepClint_Class', 'HLM_Class', 'RHepClint_Class', 'RLM_Class', 'LogF_Class', 'ER_Class', 'CNS_Perm_Class']`
## Dependencies
- cudatoolkit >= 10.2
- rdkit
- torch >= 1.12.0
- scikit-learn >= 1.0.2
- autogluon >= 0.5.1
- chemprop >=1.5.1

  - The DMPNN and CMPNN model are build based on [chemprop](https://github.com/chemprop/chemprop).
  - The AutoGluonModel is build based on [autogluon](https://github.com/awslabs/autogluon).
  - But the chemprop autogluon code is modified, please follow the steps below to install the Conda virtual environment.
## Installation
```
1. conda env create -f environment.yml
2. conda activate molprop
3. cd autogluon
   pip install .
   cd ../chemprop
   pip install .
4. cd ..
   pip install .
``` 

## Use example:
- The data file must be be a dict. For example:
```
mol_in = {'name0': '[H]OC[C@@H](O[H])CON([H])C(=O)C1=C(N([H])C2=CC=C(I)C=C2F)C(F)=C(F)C=C1',
           'name1': 'C[C@H](CN1C=NC2=C1N=CN=C2N)OCP(O)(=O)OP(O)(=O)OP(O)(O)=O', 
           ...}
```
- predict
```
from molprop.models.predict_model import predict
res_out = predict(mol_in=mol_in, model='CMPNN', prop='t1/2_Class', return_pred_prod=True)
```




