import pickle
import pandas as pd
from collections import OrderedDict
from autogluon.tabular import TabularDataset, TabularPredictor

metric_list = ['accuracy', 'balanced_accuracy', 'f1', 'f1_macro', 'f1_micro', 'f1_weighted',
               'roc_auc', 'roc_auc_ovo_macro', 'average_precision', 'precision', 'precision_macro',
               'precision_micro', 'precision_weighted', 'recall', 'recall_macro', 'recall_micro',
               'recall_weighted', 'log_loss', 'pac_score', 'root_mean_squared_error', 'mean_squared_error',
               'mean_absolute_error', 'median_absolute_error','mean_absolute_percentage_error',
               'r2', 'quadratic_kappa']

presets_list = ['best_quality', 'high_quality', 'good_quality', 'medium_quality',
                'optimize_for_deployment', 'interpretable', 'ignore_text']


def train_AutoGluon_model(dataset_type, target_column, metric_type,
                          save_dir, time_limit=3600, presets='best_quality',
                          train_data=None, train_path=None):
    """
    :param train_data: data to train a AutoGluon model.
    :param dataset_type: type of prediction problem this Predictor has been trained for.
    :param target_column: Name of the columns containing target values.
    :param metric_type: metric is used to evaluate predictive performance. choose one in the metric_list.
    :param save_dir: specifies folder to store trained models
    :param time_limit: how long model train run (wallclock time in seconds). default:3600.
    :param presets: significantly impact predictive accuracy. choose one in the presets_list, default: best_quality
    """
    assert metric_type in metric_list, f"metric_type: {metric_type} is not in {metric_list}"
    assert all([train_data, train_path]) == False, "train_data and train_path can only provide one!"
    assert all(
        [train_data == None, train_path == None]) == False, 'train_data and train_path must be provide one!'
    print('train model')
    if train_path:
        train_data = pd.read_csv(train_path)
    train_data = TabularDataset(data=train_data)
    train_data.drop(columns=['smiles'], inplace=True)
    if dataset_type != 'regression':
        train_data.drop(columns=[target_column], inplace=True)
    dataset_type = 'binary' if dataset_type == 'classification' else dataset_type
    label = target_column if dataset_type == 'regression' else 'target_label'

    predictor = TabularPredictor(label=label, problem_type=dataset_type, path=save_dir,
                                 eval_metric=metric_type).fit(train_data, time_limit=time_limit,
                                                              presets=presets)
    results = predictor.fit_summary(show_plot=True)
    print('------------------train model fiinshed------------------')

    return predictor, results



def predict_AutoGluon_model(test_data, prop, checkpoint_dir, label_encoder_path=None, return_pred_prob=False):
    """
    :param test_path: test data file path to predict AutoGluon model.
    :param preds_path: .csv file path to save predict result.
    :param checkpoint_dir: The path to directory in which this Predictor was previously saved. If model==None, checkpoint_dir must be provided.
    """
    model = TabularPredictor.load(checkpoint_dir)
    test_data = TabularDataset(data=test_data)
    y_pred = model.predict(test_data.iloc[:, :1024], return_pred_prob=return_pred_prob)

    class_map = OrderedDict()
    le = None
    if label_encoder_path:
        with open(label_encoder_path, 'rb') as f:
            le = pickle.load(f)
        for cl in le.classes_:
            class_map.update({le.transform([cl])[0]: cl})

    if return_pred_prob and class_map:
        prop_names = class_map[1] if len(class_map)<3 else list(class_map.values())
        predict_df = pd.DataFrame({'name': test_data['name']}, index=y_pred.index)
        predict_df[prop_names] = y_pred
    else:
        predict_df = pd.DataFrame({'name': test_data['name'], prop: y_pred})
        if le:
            predict_df[prop] = le.inverse_transform(predict_df[prop])
    return predict_df
