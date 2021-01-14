import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ipdb
import shap
from data_preproc import get_labels
import re
import json
from datetime import datetime
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, average_precision_score
from sklearn.metrics import precision_score, recall_score, roc_curve, precision_recall_curve
from sklearn.model_selection import cross_validate, GridSearchCV
from scipy.stats import sem
from imblearn.over_sampling import ADASYN
import lightgbm as lgb


WINDOWS_PATH = Path('/mnt/e/')           # running this inside WSL
RESULTS_PATH = Path('../results/')


def clean_name(string) -> str:
    return re.sub('/', '-', string)


def fit_lgbm(data: pd.DataFrame, verbose=True, lr=0.1, num_leaves=10,
             num_iterations=10000, min_data_in_leaf=1000,
             early_stopping_rounds=200, clf=None):
    if clf is None:
        clf = lgb.LGBMClassifier(class_weight='balanced', learning_rate=lr,
                                 num_leaves=num_leaves,
                                 num_iterations=num_iterations,
                                 min_data_in_leaf=min_data_in_leaf)
    else:
        if verbose:
            print('Fine-tuning using provided classifier')
    # Set up the data
    X_train, y_train = prepare_data(data, split='train')
    X_validation, y_validation = prepare_data(data, split='validation')
    feature_names = X_train.columns.values

    clf.fit(X_train, y_train, eval_set=[(X_validation, y_validation)],
            early_stopping_rounds=early_stopping_rounds, verbose=False)

    return clf, feature_names


def fit_LR(data: pd.DataFrame, verbose: bool = False, clf=None, final: bool = False):
    print(f'Fitting logistic regression model!')
    if clf is None:
        clf = LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=5000, verbose=True)
    else:
        if verbose:
            print('Fine-tuning provided classifier')
    X_train, y_train = prepare_data(data, split='train', final=final)
    feature_names = X_train.columns.values
    clf.fit(X_train, y_train)
    return clf, feature_names


def fit_model(data: pd.DataFrame, model: str = 'LGBM',
              upsample: bool = False, verbose=True,
              clf=None, final: bool = False):
    if model == 'LR':
        return fit_LR(data, verbose, clf=clf, final=final)
    elif model == 'lasso':
        raise NotImplementedError
        clf = LogisticRegression(class_weight='balanced', penalty='l1', solver='liblinear', max_iter=5000)
    elif model == 'RF':
        raise NotImplementedError
        clf = RandomForestClassifier(class_weight='balanced', max_depth=310)
    elif model == 'LGBM':
        return fit_lgbm(data, verbose=verbose, clf=clf)
    elif model == 'GBT':
        raise NotImplementedError
        # mostly vestigial (for regression testing LGBM)
        clf = GradientBoostingClassifier(n_estimators=500, subsample=0.8)
    elif model == 'EBM':
        raise NotImplementedError
        clf = ExplainableBoostingClassifier()
    else:
        raise ValueError(model)

    X_train, y_train = prepare_data(data, split='train')
    feature_names = X_train.columns.values

    # optional
    if upsample:
        print('Upsampling rare examples!')
        sm = ADASYN(sampling_strategy=0.3)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print(f'New training set prevalence: {y_train.mean()}')

    if verbose:
        print(f'Fitting a model on {X_train.shape[0]} samples!')
    clf.fit(X_train, y_train)

    return clf, feature_names


def prepare_data(data, split='train', final: bool = False):
    if split is None:
        split_data = data.drop(columns='split')
    else:
        if final:
            print('WARNING: Preparing data with FINAL = TRUE')
            splits = data['split'].unique()
            assert 'validation' in splits
            assert 'train' in splits
            split_data = data[data['split'].isin(['validation', 'train'])].drop(columns='split')
        else:
            split_data = data[data['split'] == split].drop(columns='split')
    X = split_data.drop(columns='test_outcome')
    if 'test_month' in X.columns:
        X.drop(columns='test_month', inplace=True)
    if 'person_id' in X.columns:
        X.drop(columns='person_id', inplace=True)
    y = split_data['test_outcome']

    return X, y


def compute_metrics(clf, X, y):
    metrics = dict()
    prevalence = y.mean()
    metrics['prevalence'] = prevalence
    y_pred_prob = clf.predict_proba(X)[:, 1]
    y_pred = clf.predict(X)

    accuracy = accuracy_score(y, y_pred)
    recall = recall_score(y, y_pred)
    roc_fpr, roc_tpr, roc_thresholds = roc_curve(y, y_pred_prob)
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y, y_pred_prob)
    try:
        prob_true, prob_pred = calibration_curve(y, y_pred_prob, n_bins=100)
    except ValueError:
        ipdb.set_trace()
    brier_score = brier_score_loss(y, y_pred_prob)
    try:
        precision = precision_score(y, y_pred)
        AUC = roc_auc_score(y, y_pred_prob)
        AUPRC = average_precision_score(y, y_pred_prob)
    except ValueError:
        print(f'ERROR computing metrics, probably an issue with prevalence: {prevalence}')
        precision = np.nan
        AUC = np.nan
        AUPRC = np.nan

    metrics['timestamp'] = str(datetime.now())
    metrics['accuracy'] = accuracy
    metrics['recall'] = recall
    metrics['precision'] = precision
    metrics['AUC'] = AUC
    metrics['AUPRC'] = AUPRC
    metrics['brier'] = brier_score
    metrics['ROC_FPR'] = roc_fpr.tolist()
    metrics['ROC_TPR'] = roc_tpr.tolist()
    metrics['ROC_thresholds'] = roc_thresholds.tolist()
    metrics['PR_precision'] = pr_precision.tolist()
    metrics['PR_recall'] = pr_recall.tolist()
    metrics['PR_thresholds'] = pr_thresholds.tolist()

    return metrics


def evaluate(clf, data, model: str = 'LGBM', plot: bool = False, save:bool = True,
             split: str = 'train', verbose: bool = True, force_rerun: bool = False,
             identifier: str = '', n_boots: int = 0) -> dict:
    out_path = RESULTS_PATH / f'metrics_{model}_{identifier}_{split}.json'
    if out_path.exists() and not force_rerun:
        print(f'Metrics have already been computed at {out_path} - rerun with "force_rerun" to recmopute')
        metrics = json.load(open(out_path))
        return metrics
    if verbose:
        print(f'Evaluating on {split} data using {n_boots} bootstrap replicates...')
    X, y = prepare_data(data, split=split)
    if verbose:
        print(f'Evaluating on {X.shape[0]} examples')

    if type(clf) in [list, tuple]:
        # CV has occurred
        if verbose:
            print('Evaluating multiple models, assumed from CV!')
    else:
        clf = [clf]

    metrics = dict()
    metrics['model'] = model
    metrics['split'] = split
    for i, c in enumerate(clf):
        if n_boots > 0:
            # bootstrap
            N = X.shape[0]
            for j in range(n_boots):
                sample = np.random.choice(N, N, replace=True)
                X_s = X.iloc[sample]
                y_s = y.iloc[sample]
                m_s = compute_metrics(c, X_s, y_s)
                metrics[f'boot_{j}'] = m_s
            if verbose:
                print('Finished bootstrapping')
                print('Aggregating curves')
            aggregated_curves = aggregate_curves(metrics, n_boots=n_boots)
            metrics.update(aggregated_curves)
            if verbose:
                print('Aggregating point estimates')
            aggregated_points = aggregate_point_estimates_from_bootstraps(metrics, n_boots=n_boots)
            metrics.update(aggregated_points)
            # now delete the boots
            for j in range(n_boots):
                del metrics[f'boot_{j}']
        else:
            m = compute_metrics(c, X, y)
            metrics[i] = m
            if verbose:
                print(m['AUC'])
    if save:
        if verbose:
            print(f'Saving metrics to {out_path}')
        with open(out_path, 'w') as fp:
            json.dump(metrics, fp, indent=4)

    return metrics


def aggregate_point_estimates_from_bootstraps(metrics, n_boots=None) -> dict:
    estimates = dict()
    if n_boots is None:
        n_boots = len(metrics.keys()) - 2
    AUC_samples = [np.nan]*n_boots
    AUPRC_samples = [np.nan]*n_boots
    prevalence_samples = [np.nan]*n_boots
    for b in range(n_boots):
        if f'boot_{b}' in metrics.keys():
            m_boot = metrics[f'boot_{b}']
        else:
            raise ValueError(b)
        AUC_samples[b] = m_boot['AUC']
        AUPRC_samples[b] = m_boot['AUPRC']
        prevalence_samples[b] = m_boot['prevalence']
    AUC_samples = np.array(AUC_samples)
    AUPRC_samples = np.array(AUPRC_samples)
    prevalence_samples = np.array(prevalence_samples)
    # Mean
    estimates['AUC_mean'] = np.mean(AUC_samples)
    estimates['AUPRC_mean'] = np.mean(AUPRC_samples)
    estimates['prevalence_mean'] = np.mean(prevalence_samples)
    # CI lower and upper
    estimates['AUC_ci_lower'] = np.percentile(AUC_samples, q=2.5)
    estimates['AUPRC_ci_lower'] = np.percentile(AUPRC_samples, q=2.5)
    estimates['prevalence_ci_lower'] = np.percentile(prevalence_samples, q=2.5)
    estimates['AUC_ci_upper'] = np.percentile(AUC_samples, q=97.5)
    estimates['AUPRC_ci_upper'] = np.percentile(AUPRC_samples, q=97.5)
    estimates['prevalence_ci_upper'] = np.percentile(prevalence_samples, q=97.5)
    # sem
    estimates['AUC_sem'] = sem(AUC_samples)
    estimates['AUPRC_sem'] = sem(AUPRC_samples)
    estimates['prevalence_sem'] = sem(prevalence_samples)
    return estimates


def aggregate_curves(metrics, n_boots=None) -> dict:
    # ROC, PR
    curves_aggregated = dict()
    if n_boots is None:
        n_boots = len(metrics.keys()) - 2
    for curve in ['ROC', 'PR']:
        if curve == 'ROC':
            x = 'ROC_FPR'
            y = 'ROC_TPR'
        elif curve == 'PR':
            x = 'PR_recall'
            y = 'PR_precision'
        else:
            raise NotImplementedError
        series_list = []
        canonical_x = np.linspace(0, 1, 250)
        canonical = pd.Series(np.nan, index=canonical_x)
        for b in range(n_boots):
            if f'boot_{b}' in metrics.keys():
                m_boot = metrics[f'boot_{b}']
            else:
                raise ValueError(b)
            # collect all the curves
            x_vals = m_boot[x]
            y_vals = m_boot[y]

            s = pd.Series(y_vals, index=x_vals)
            s = s.groupby(s.index).mean()
            s = s.append(canonical.loc[[x for x in canonical.index if not x in s.index]])
            series_list.append(s)
        all_series = pd.concat(series_list, axis=1)
        all_series.sort_index(inplace=True)
        all_series.interpolate(inplace=True)
        # now extract the canonical values
        interpolated_series = all_series.loc[canonical.index]
        # now compute means and such
        curve_mean = interpolated_series.mean(axis=1)
        curve_mean.index.name = x
        curve_mean.name = y
        # 95% ci?
        curve_ci_lower = np.percentile(interpolated_series.values, q=2.5, axis=1)
        curve_ci_upper = np.percentile(interpolated_series.values, q=97.5, axis=1)
        # SEM
        curve_sem = sem(interpolated_series.values, axis=1)
        # Save it in a nice JSONable format
        curves_aggregated[f'{x}_mean'] = curve_mean.index.tolist()
        curves_aggregated[f'{y}_mean'] = curve_mean.values.tolist()
        curves_aggregated[f'{curve}_ci_lower'] = curve_ci_lower.tolist()
        curves_aggregated[f'{curve}_ci_upper'] = curve_ci_upper.tolist()
        curves_aggregated[f'{curve}_sem'] = curve_sem.tolist()
    return curves_aggregated


def compute_SHAP(clf, feature_labels, df, model: str = 'LGBM',
                 identifier: str = '', split: str = 'validation',
                 setting: str = 'inpatient') -> None:
    """ Compute SHAP values for all patients across data """
    print('Computing SHAP values...')
    X, _ = prepare_data(df, split=split)
    if model in ['RF', 'GBT', 'LGBM']:
        explainer = shap.TreeExplainer(clf)
    elif model in ['LR', 'lasso']:
        explainer = shap.LinearExplainer(clf, data=X)
    else:
        raise NotImplementedError

    shap_values = explainer.shap_values(X)
    if model == 'LGBM':
        # for lightGBM SHAP returns two sets of shap values
        shap_values = shap_values[1]

    # attach the patient information and then save as csv
    shap_df = pd.DataFrame(shap_values)
    shap_df.index = X.index
    shap_df.columns = feature_labels
    shap_df['visit_type'] = X['visit_type']
    # now save the shap values
    shap_df.to_csv(RESULTS_PATH / f'SHAP_{identifier}_{model}_{setting}_{split}.csv')
    return


def interaction_plot(feature, interaction, clf, X, model='GBT', shap_values=None):
    """ For exploration """
    if shap_values is None:
        assert clf is not None
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X)
        if model == 'LGBM':
            shap_values = shap_values[1]
    shap.dependence_plot(ind=feature, shap_values=shap_values, features=X,
                         interaction_index=interaction, alpha=0.7, x_jitter=0.1)
    plt.axhline(y=0, ls='--', alpha=0.5, color='black')
    identifier = f'{model}_dependence_{feature}-{interaction}'
    plt.tight_layout()
    plt.savefig(WINDOWS_PATH / f'SHAP_{identifier}.png')
    plt.clf()
    plt.close()
    return shap_values


def load_SHAP(identifier: str = 'test', model: str = 'LGBM', by:str = 'visit_type',
              setting: str = 'inpatient', split: str = 'validation', n_boots: int = 0):
    path = f'SHAP_aggregated_{identifier}_{model}_{setting}_{split}_{n_boots}.npy'
    try:
        df_shaps = np.load(RESULTS_PATH / path, allow_pickle=True).item()
    except FileNotFoundError:
        print(f'Couldn\'t find {path} - generating anew')
        shap_path = f'SHAP_{identifier}_{model}_{setting}_{split}.csv'
        df = pd.read_csv(RESULTS_PATH / shap_path)
        df_shaps = dict()
        if by == 'visit_type':
            options = ['overall'] + df['visit_type'].unique().tolist()
        elif by == 'test_outcome':
            labels = get_labels()
            options = ['overall', 0, 1]
        for o in options:
            if o == 'overall':
                df_vt = df.drop(columns=['visit_type', 'person_id'])
            else:
                if by == 'visit_type':
                    df_vt = df[df['visit_type'] == o].drop(columns=['visit_type', 'person_id'])
                elif by == 'test_outcome':
                    pids = labels[labels['test_outcome'] == o]['person_id']
                    df_vt = df[df['person_id'].isin(pids)].drop(columns=['visit_type', 'person_id'])
            abs_df_vt = np.abs(df_vt)
            if n_boots == 0:
                abs_shap_mean = abs_df_vt.mean(axis=0)
                abs_shap_median = np.median(abs_df_vt, axis=0)
                abs_shap_std = abs_df_vt.std(axis=0)
                abs_shap_sem = sem(abs_df_vt, axis=0)
                # confidence intervals
                abs_shap_lower = np.nan
                abs_shap_upper = np.nan
            else:
                print(f'Bootstrapping SHAP values with {n_boots} samples!')
                # we are using bootstrapping to compute the mean abs shap value
                # so we are collecting samples of the mean
                means = np.zeros(shape=(n_boots, abs_df_vt.shape[1]))
                N = abs_df_vt.shape[0]
                for b in range(n_boots):
                    boot_idx = np.random.choice(N, N, replace=True)
                    boot_sample = abs_df_vt.iloc[boot_idx, :]
                    means[b, :] = boot_sample.mean(axis=0)
                abs_shap_mean = pd.Series(means.mean(axis=0), index=abs_df_vt.columns)
                abs_shap_median = np.nan
                abs_shap_std = np.nan
                abs_shap_sem = np.nan
                abs_shap_lower = pd.Series(np.percentile(means, q=2.5, axis=0),
                                           index=abs_df_vt.columns)
                abs_shap_upper = pd.Series(np.percentile(means, q=97.5, axis=0),
                                           index=abs_df_vt.columns)
            df_shap = pd.DataFrame({'abs_shap_mean': abs_shap_mean,
                                    'abs_shap_median': abs_shap_median,
                                    'abs_shap_std': abs_shap_std,
                                    'abs_shap_sem': abs_shap_sem,
                                    'abs_shap_lower': abs_shap_lower,
                                    'abs_shap_upper': abs_shap_upper})
            df_shap.sort_values(by='abs_shap_mean', inplace=True, ascending=False)
            df_shaps[o] = df_shap
        np.save(RESULTS_PATH / path, df_shaps)
    return df_shaps


def get_top_features(visit_types, top_n=10, shap=True, verbose=False) -> set:
    top_features_all = set()
    if shap:
        val = 'abs_shap_mean'
    else:
        val = 'abs_coef'
    for visit_type in visit_types:
        visit_type.sort_values(by=val, inplace=True, ascending=False)
        visit_type['rank'] = np.arange(visit_type.shape[0]) + 1
        visit_type_top_features = set(visit_type.head(top_n).index)
        if verbose:
            print(f'Top features from visit_type: {visit_type_top_features}')
        top_features_all = top_features_all.union(visit_type_top_features)

    print(f'Identified {len(top_features_all)} shared top features from {top_n} features across all {len(visit_types)} visit_types')
    return top_features_all


def tidy_feature_names(feature_names) -> list:
    """ Tidy up feature names for plotting purposes """
    new_feature_names = []
    for f in feature_names:
        if f == 'pdens18':
            f = 'population density'
        nf = re.sub('vitals_', '', f)
        nf = re.sub('measurements_', '', nf)
        nf = re.sub('static_', '', nf)
        nf = re.sub('drug_', '', nf)
        nf = re.sub('-7D-mean', ' (7D mean)', nf)
        nf = re.sub('-14D-mean', ' (14D mean)', nf)
        nf = re.sub('chronic_', '(chronic) ', nf)
        nf = re.sub('symptom_', '(symptom) ', nf)
        nf = re.sub('_last', ' (last)', nf)
        nf = re.sub('_timedelta', ' (time since)', nf)
        nf = re.sub('_', ' ', nf)
        if nf == 'zip recent disease':
            nf = 'recent disease in zip'
        if nf == 'median household income dollars':
            nf = 'median household income'
        print(f'Mapping {f} to {nf}')
        new_feature_names.append(nf.capitalize())
    return new_feature_names


def find_best_HPs(data: pd.DataFrame, model='LGBM') -> dict:
    if model == 'LGBM':
        clf = lgb.LGBMClassifier(class_weight='balanced')
        fixed_grid = {'learning_rate': [0.1],
                      'num_leaves': [5],
                      'min_data_in_leaf': [500],
                      'max_depth': [5],
                      'num_iterations': [150]}
        param_grid = {'learning_rate': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0],
                      'num_leaves': [2, 3, 5, 10, 25, 50, 200, 1000, 2500, 5000],
                      'min_data_in_leaf': [10, 50, 100, 200, 500, 1000, 10000],
                      'num_iterations': [10, 50, 100, 150, 200, 1000, 2000]}
    else:
        raise NotImplementedError
    X, y = prepare_data(data, split='train')
    grid = GridSearchCV(clf, param_grid, verbose=2, scoring='roc_auc')
    grid.fit(X, y)
    cv_results = pd.DataFrame(grid.cv_results_)
    cv_results.sort_values(by='rank_test_score', inplace=True)
    return cv_results

