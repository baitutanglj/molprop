import os

import numpy as np
import pandas as pd


from chemprop.args import PredictArgs
from chemprop.train import make_predictions
from molprop.datasets.load_dataset import default_dataname
from molprop.models.AutoGluon_model import predict_AutoGluon_model
from molprop.models.utils import valid_smiles, get_df_ecfp

default_dataname = default_dataname
MODEL_MODULE = os.path.split(os.path.realpath(__file__))[0]
DATA_MODULE = os.path.join(MODEL_MODULE, '..', 'datasets')

Gostar_LogD_Solubility = {'activity_value': 'ELD_activity_value', 'pLogS': 'ESL_pLogS'}
CYP_hERG_clean_data = {'CYP1A2_Class': 'CYP1A2_data_clean', 'CYP2C9_Class': 'CYP2C9_data_clean',
                       'CYP2C19_Class': 'CYP2C19_data_clean', 'CYP2D6_Class': 'CYP2D6_data_clean',
                       'CYP3A4_Class': 'CYP3A4_data_clean', 'hERG_Class': 'HERG-220720_clean'}

modelling_data_1 = {
    't1/2_Class': ['HHep_T_half', 'HLM_T_half', 'MLM_T_half', 'RLM_T_half'],
    'HHepClint_Class': 'HHepClint', 'HLM_Class': ['HLMClint_ml_min_kg', 'HLMClint_ul_min_mg'],
    'RHepClint_Class': 'RHepClint', 'RLM_Class': ['RLMClint_ml_min_kg', 'RLMClint_ul_min_mg']}

modelling_data_1_cls = {'LogF_Class': 'F_data', 'ER_Class': 'MDCK_ER', 'CNS_Perm_Class': 'PAMPA_permeability'}

default_dataname = list(Gostar_LogD_Solubility.values()) \
                   + list(CYP_hERG_clean_data.values()) \
                   + list(modelling_data_1.values()) \
                   + list(modelling_data_1_cls.values())
temp = []
for i in default_dataname:
    if type(i) == list:
        temp += i
    else:
        temp.append(i)
default_dataname = temp

best_model_dict = {'activity_value': 'ELD_activity_value_to_CMPNN',
                   'pLogS': 'ESL_pLogS_to_AutoGluonModel',
                   'CYP1A2_Class': 'CYP1A2_data_clean_to_AutoGluonModel',
                   'CYP2C9_Class': 'CYP2C9_data_clean_to_AutoGluonModel',
                   'CYP2C19_Class': 'CYP2C19_data_clean_to_AutoGluonModel',
                   'CYP2D6_Class': 'CYP2D6_data_clean_to_AutoGluonModel',
                   'CYP3A4_Class': 'CYP3A4_data_clean_to_AutoGluonModel',
                   'hERG_Class': 'HERG-220720_clean_to_AutoGluonModel',
                   't1/2_Class': 'HLM_T_half_to_AutoGluonModel',
                   'HHepClint_Class': 'HHepClint_to_AutoGluonModel',
                   'HLM_Class': 'HLMClint_ul_min_mg_to_AutoGluonModel',
                   'RHepClint_Class': 'RHepClint_to_AutoGluonModel',
                   'RLM_Class': 'RLMClint_ul_min_mg_to_AutoGluonModel',
                   'LogF_Class': 'F_data_to_AutoGluonModel',
                   'ER_Class': 'MDCK_ER_to_AutoGluonModel',
                   'CNS_Perm_Class': 'PAMPA_permeability_to_AutoGluonModel'
                   }

data_type = {'Gostar_LogD_Solubility': 'regression',
             'CYP_hERG_clean_data': 'classification',
             'modelling_data_1': 'multiclass',
             'modelling_data_1_cls': 'classification'}

# prop_list = ['activity_value', 'pLogS', 'CYP1A2_Class', 'CYP2C9_Class',
#              'CYP2C19_Class', 'CYP2D6_Class', 'CYP3A4_Class','hERG_Class',
#              't1/2_Class', 'HHepClint_Class', 'HLM_Class', 'RHepClint_Class',
#              'RLM_Class', 'LogF_Class', 'ER_Class', 'CNS_Perm_Class']

prop_type_dict = {'activity_value': 'regression', 'pLogS': 'regression',
                  'CYP1A2_Class': 'classification', 'CYP2C9_Class': 'classification',
                  'CYP2C19_Class': 'classification', 'CYP2D6_Class': 'classification',
                  'CYP3A4_Class': 'classification', 'hERG_Class': 'classification',
                  't1/2_Class': 'multiclass', 'HHepClint_Class': 'multiclass',
                  'HLM_Class': 'multiclass', 'RHepClint_Class': 'multiclass',
                  'RLM_Class': 'multiclass', 'LogF_Class': 'classification',
                  'ER_Class': 'classification', 'CNS_Perm_Class': 'classification'}
prop_list = list(prop_type_dict.keys())

model_type = ['AutoGluonModel', 'CMPNN', 'DMPNN', 'default']


def predict(mol_in: dict, model: str, prop: str, return_pred_prob: bool=False):
    """
    :param mol_in: dict as {name_1: smiles_1, name_2: smiles_2, ...} to predict model
    :param model: choose one of ['AutoGluonModel', 'CMPNN', 'DMPNN', 'default'],
                if model=default, it will automatically select the best model to predict the property
    :param prop: predict the property in the model, choose one of prop_list:
                ['activity_value', 'pLogS', 'CYP1A2_Class', 'CYP2C9_Class',
                'CYP2C19_Class', 'CYP2D6_Class', 'CYP3A4_Class','hERG_Class',
                't1/2_Class', 'HHepClint_Class', 'HLM_Class', 'RHepClint_Class',
                'RLM_Class', 'LogF_Class', 'ER_Class', 'CNS_Perm_Class']
    """
    assert model in model_type, f"model={model} is not in {model_type}"
    assert prop in prop_list, f"prop={prop} is not in {prop_list}"
    mol_df = pd.DataFrame({'name':mol_in.keys(), 'smiles': mol_in.values()})
    df_valid, df_invalid = valid_smiles(mol_df, prop)
    dataset_type = prop_type_dict[prop]
    dataset_name, best_model = best_model_dict[prop].split('_to_')
    model_name = model if model !='default' else best_model
    checkpoint_dir = os.path.join(MODEL_MODULE, 'model_ckpt', dataset_type, dataset_name, model_name)
    label_encoder_path = os.path.join(DATA_MODULE, 'data', dataset_type, dataset_name,
                                      dataset_name + '.pkl') if dataset_type != 'regression' else None
    return_pred_prob = False if dataset_type=='regression' else return_pred_prob

    if model_name == 'AutoGluonModel':
        df_valid_ecfp = get_df_ecfp(df_valid)
        print('predict model')
        predict_df = predict_AutoGluon_model(test_data=df_valid_ecfp, prop=prop,
                                             checkpoint_dir=checkpoint_dir,
                                             label_encoder_path=label_encoder_path,
                                             return_pred_prob=return_pred_prob)
    else:
        if return_pred_prob:
            predict_args = PredictArgs().parse_args(
                ["--checkpoint_dir", checkpoint_dir, "--return_pred_prob"])
        else:
            predict_args = PredictArgs().parse_args(
                ["--checkpoint_dir", checkpoint_dir])

        smiles_list = list(df_valid['smiles'].apply(lambda x: [x]))
        _, _, predict_df = make_predictions(args=predict_args, smiles=smiles_list)
        pred_prop = list(predict_df.columns)
        pred_prop.remove('smiles')
        prop = [prop] if return_pred_prob==False else pred_prop
        predict_df.columns = ['smiles']+prop
        predict_df.set_index(df_valid.index, inplace=True)
        predict_df['name'] = df_valid['name']
        predict_df = predict_df.loc[:, ['name']+prop]

    prop_names = list(predict_df.columns)
    prop_names.remove('name')
    predict_df = pd.concat([predict_df, df_invalid], axis=0)
    predict_df = predict_df[['name']+prop_names].copy()
    predict_df.sort_index(axis=0, ascending=True, inplace=True)
    if return_pred_prob:
        res_out = predict_df.set_index(['name'])[prop_names].T.to_dict()
        for name, value_dict in res_out.items():
            if np.isnan(value_dict[prop_names[0]]):
                res_out[name] = 'NULL'

    else:
        res_out = predict_df.set_index(['name'])[prop_names[0]].to_dict()
    return res_out


if __name__ == '__main__':
    import json
    with open('/mnt/home/linjie/projects/molprop/molprop/tests/example.json', 'rb') as f:
        mol_in = json.load(f)

    res_out = predict(mol_in=mol_in, model='CMPNN', prop='t1/2_Class')
    print(res_out.head(10))
