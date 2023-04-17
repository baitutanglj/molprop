"""
Base IO code for all datasets
"""
from molprop.datasets.utils import *

DATA_MODULE = os.path.split(os.path.realpath(__file__))[0]
# DATA_MODULE = "molprop.datasets.data"
Gostar_LogD_Solubility = {'ELD_activity_value': 'activity_value', 'ESL_pLogS': 'pLogS'}
CYP_hERG_clean_data = {'CYP1A2_data_clean': 'CYP1A2_Class', 'CYP2C9_data_clean': 'CYP2C9_Class',
                       'CYP2C19_data_clean': 'CYP2C19_Class', 'CYP2D6_data_clean': 'CYP2D6_Class',
                       'CYP3A4_data_clean': 'CYP3A4_Class', 'HERG-220720_clean': 'hERG_Class'}
modelling_data_1 = {'HHep_T_half': 't1/2_Class', 'HHepClint': 'HHepClint_Class',
                    'HLM_T_half': 't1/2_Class', 'HLMClint_ml_min_kg': 'HLM_Class',
                    'HLMClint_ul_min_mg': 'HLM_Class', 'MHep_T_half': 't1/2_Class',
                    'MLM_T_half': 't1/2_Class', 'RHep_T_half': 't1/2_Class',
                    'RHepClint': 'RHepClint_Class', 'RLM_T_half': 't1/2_Class',
                    'RLMClint_ml_min_kg': 'RLM_Class','RLMClint_ul_min_mg': 'RLM_Class'}
modelling_data_1_cls = {'F_data': 'LogF_Class', 'MDCK_ER': 'ER_Class', 'PAMPA_permeability': 'CNS_Perm_Class'}

default_dataname = list(Gostar_LogD_Solubility.keys()) \
                   + list(CYP_hERG_clean_data.keys()) \
                   + list(modelling_data_1.keys()) \
                   + list(modelling_data_1_cls.keys())

data_type = {'Gostar_LogD_Solubility': 'regression',
             'CYP_hERG_clean_data': 'classification',
             'modelling_data_1': 'multiclass',
             'modelling_data_1_cls': 'classification'}


def get_dataset_type(dataname: str=None):
    """
    :param dataname: default dataset file name in default_dataname
    :return: dataset_type: str in ['regression', 'classification']
    """
    assert dataname in default_dataname, 'dataname is not in the default data'
    if dataname in Gostar_LogD_Solubility.keys():
        return 'regression'
    elif dataname in modelling_data_1:
        return 'multiclass'
    else:
        return 'classification'


# def load_default_data(dataname: str=None,
#                       return_ecfp: bool=True):
#     """
#     :param dataname: default dataset file name in default_dataname
#     :param return_ecfp: if True, calculate ECFP
#     :return:
#         data: pandas.DataFrame
#         ECFP_data: pandas.DataFrame, only present when `return_ecfp=True`.
#     """
#     assert dataname in default_dataname, 'dataname is not in the default data'
#     dataset_type = get_dataset_type(dataname)
#     dataset_module = f"{DATA_MODULE}.{dataset_type}.{dataname}"
#     csv_file_name = 'smiles_target_'+dataname+'.csv'
#     with resources.open_text(dataset_module, csv_file_name) as csv_file:
#         data_file = csv.reader(csv_file)
#         temp = next(data_file)
#         df_smiles_target = pd.DataFrame(columns=temp)
#         for i, irow in enumerate(data_file):
#             df_smiles_target.loc[i, :] = irow
#
#     if return_ecfp:
#         ecfp_file_name = 'ECFP_' + dataname + '.csv'
#         with resources.open_text(dataset_module, ecfp_file_name) as csv_file:
#             data_file = csv.reader(csv_file)
#             temp = next(data_file)
#             ecfp_df = pd.DataFrame(columns=temp)
#             for i, irow in enumerate(data_file):
#                 ecfp_df.loc[i, :] = irow
#         return df_smiles_target, ecfp_df
#     else:
#         return df_smiles_target

    # data = pkgutil.get_data(__package__ + '.data.regression.ELD_activity_value', 'test_ELD_activity_value.csv')

def load_default_data(dataname: str=None):
    """
    :param dataname: default dataset file name in default_dataname
    :return:
        data: pandas.DataFrame
        ECFP_data: pandas.DataFrame, only present.
    """
    csv_file_path, ecfp_file_path, label_encoder_path = get_default_data_path(dataname)
    df_smiles_target = pd.read_csv(os.path.join(csv_file_path))

    ecfp_df = pd.read_csv(ecfp_file_path)
    return df_smiles_target, ecfp_df


def get_default_data_path(dataname: str):
    """
    :param dataname: default dataset file name in default_dataname
    """
    assert dataname in default_dataname, 'dataname is not in the default data'
    dataset_type = get_dataset_type(dataname)
    dataset_dir = os.path.join(DATA_MODULE, 'data', dataset_type, dataname)
    csv_file_path = os.path.join(os.path.join(dataset_dir, 'smiles_target_'+dataname+'.csv'))
    ecfp_file_path = os.path.join(os.path.join(dataset_dir, 'ECFP_' + dataname + '.csv'))
    pkl_path = os.path.join(dataset_dir, dataname + '.pkl')
    label_encoder_path = pkl_path if os.path.exists(pkl_path) else None

    return csv_file_path, ecfp_file_path, label_encoder_path


def load_train_data(datapath: str = None,
                    target_column: str = None,
                    dataset_type: str = 'regression',
                    save_path: str = None):
    """
    :param datapath: dataset file path to load
    :param target_column: Name of the columns containing target values.
    :param dataset_type: type of  dataset, in ['regression', 'classification', 'multiclass']
    :param return_ecfp: if True, calculate ECFP
    :param save_path: csv path to save the processed data
    :return:
        data: pandas.DataFrame
        ECFP_data: pandas.DataFrame, only present when `return_ecfp=True`.
    """

    assert save_path, 'save_path must be provided'
    assert not os.path.exists(save_path), f'save_path:{save_path} already exists'
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    smiles_target_path = '/smiles_target_'.join(save_path.rsplit('/', 1))
    ecfp_path = '/ECFP_'.join(save_path.rsplit('/', 1))
    # pkl_path = save_path.replace('.csv', '.pkl')
    label_encoder_path = None

    # raw data processed
    df = remove_salt(datapath, target_column=target_column)
    df = df_drop_duplicated(df, target_column, dataset_type, save_path)

    # data only retains the [smiles, target] column
    df_smiles_target = df.loc[:, ['smiles', target_column]]
    if dataset_type != 'regression':
        label_encoder_path = save_path.replace('.csv', '.pkl')
        df_smiles_target = label_encoder(df=df_smiles_target, target_column=target_column,
                                         label_encoder_path=label_encoder_path)

    df_smiles_target.to_csv(smiles_target_path, index=False)

    ecfp_df = get_df_ecfp(df_smiles_target, target_column, dataset_type)
    ecfp_df.to_csv(ecfp_path, index=False)

    return df_smiles_target, ecfp_df, smiles_target_path, ecfp_path, label_encoder_path


def load_test_data(datapath: str = None,
                   save_path: str = None,
                   smiles_column: str = 'sub_smiles'):
    """
        :param datapath: dataset file path to load
        :param save_path: csv path to save the processed data
        :param smiles_column: smiles column name in dataset file
        :return:
            data: pandas.DataFrame
            ECFP_data: pandas.DataFrame
        """
    assert save_path, 'save_path must be provided'
    assert not os.path.exists(save_path), f'save_path:{save_path} already exists'
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    label_encoder_path = save_path.replace('csv', 'pkl')

    # raw data processed
    df = remove_salt(datapath, smiles_column=smiles_column)
    df.drop_duplicates(subset=['smiles'], keep='first', inplace=True, ignore_index=True)
    df.reset_index(drop=True, inplace=True)

    # data only retains the [smiles] column
    df_smiles = pd.DataFrame(df.loc[:, 'smiles'], columns=['smiles'])
    df_smiles.to_csv(save_path, index=False)

    # get ecfp
    ecfp_df = get_smiles_df_ecfp(df_smiles)
    ecfp_df.to_csv('/ECFP_'.join(save_path.rsplit('/', 1)), index=False)

    return df_smiles, ecfp_df



