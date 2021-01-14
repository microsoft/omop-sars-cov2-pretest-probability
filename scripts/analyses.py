import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from shap import summary_plot
import json
from data_preproc import load_category_mappings
from pathlib import Path
from scipy.stats import sem
from omop_metadata import define_feature_categories
import train_eval
from train_eval import clean_name, get_top_features, prepare_data, tidy_feature_names

plt.style.use('seaborn-paper')
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

WINDOWS_PATH = Path('/mnt/e/')  # running this inside WSL
RESULTS_PATH = Path('../results/')

overall_colour = '#1c1c1c'
inpatient_colour = '#0a8181'
outpatient_colour = '#b80e56'
telehealth_colour = '#ad7606'

visit_colours = {
    'overall': overall_colour,
    'inpatient/ICU/ED': inpatient_colour,
    'outpatient/rehab': outpatient_colour,
    'telehealth/telephone': telehealth_colour
}

setting_colours = {
    'inpatient': overall_colour,
    'clinic': '#56B4E9',
    'community': '#cc79a7'
}

setting_name_remap = {
    'inpatient': 'All features',
    'clinic': 'Only clinic features',
    'community': 'Only community features'
}

visit_type_name_remap = {
    'overall': 'All patients',
    'inpatient/ICU/ED': 'Inpatients',
    'outpatient/rehab': 'Outpatients',
    'telehealth/telephone': 'Telehealth patients'
}


def plot_wrapper(df: pd.DataFrame, identifier: str = '1010',
                 back=0.817) -> None:
    for wc in ['ROC', 'PR']:
        plot_visit_comparison_curves(identifier=identifier,
                                     setting='inpatient',
                                     model='LGBM',
                                     which_curve=wc,
                                     no_overall=False,
                                     only_overall=False,
                                     split='heldout_test',
                                     calibrate=False)
        plot_model_comparison_curves(identifier=identifier,
                                     visit_type='overall',
                                     setting='inpatient',
                                     which_curve=wc,
                                     split='heldout_test')
    visit_types = visit_colours.keys()
    for vt in visit_types:
        plot_SHAP_values(df,
                         identifier=identifier,
                         visit_type=vt,
                         split='heldout_test')

    vis_feature_importance_by_visit_type(identifier=identifier,
                                         top_n=10,
                                         sort_by_visit_type='overall',
                                         model='LGBM',
                                         final=True)

    bar_chart_comparison(compare_y=False,
                         identifier=identifier,
                         plot_name='ablation',
                         split='heldout_test',
                         visit_types=['overall'],
                         ylim=(None, 0.850))

    compare_restricted_featuresets(identifier, split='heldout_test')

    res, _ = analysis_individual_features(None,
                                          n_boots=100,
                                          identifier=identifier,
                                          model='LGBM',
                                          split='heldout_test')
    plot_individual_feature_analysis(res,
                                     back,
                                     identifier=identifier,
                                     model='LGBM',
                                     top_n=20,
                                     include_background=False)

    return


def run_analyses(data: pd.DataFrame,
                 setting: str = 'inpatient',
                 run_on_visit_types: bool = True,
                 refit: bool = False,
                 finetune: bool = False,
                 final: bool = False,
                 identifier: str = 'test',
                 model='LGBM',
                 n_boots: int = 1000,
                 force_rerun: bool = False,
                 calibrate: bool = False) -> None:
    """
    Just do the whole thing
    """
    # First, subset to the available data
    feature_categories, nec = define_feature_categories(data)
    relevant_features = list(set(feature_categories[setting] + nec))
    df = data[relevant_features]
    # Now train the model
    # TODO allow for other models
    print(f'Fitting model of type {model}')
    clf, feature_names = train_eval.fit_model(df,
                                              model=model,
                                              calibrate=calibrate,
                                              final=final)
    print('Model fit, now evaluating...')
    # Now evaluate on the four patient classes
    if final:
        print('FINAL MODE!')
        split = 'heldout_test'
    else:
        split = 'validation'
    # overall
    print(f'\t...using split {split}')
    _ = train_eval.evaluate(
        clf,
        df,
        model=model,
        split=split,
        identifier=f'{identifier}_{setting}_overall{calibrate*"_calibrated"}',
        n_boots=n_boots,
        force_rerun=force_rerun)
    if calibrate:
        print('Not computing SHAP values because calibrate is True')
    else:
        print('Computing SHAP values...')
        train_eval.compute_SHAP(clf,
                                feature_names,
                                df,
                                model=model,
                                identifier=identifier,
                                split=split,
                                setting=setting)
    if run_on_visit_types is False:
        return
    visit_types = df['visit_type'].unique()
    v_map = load_category_mappings()['visit_type']
    for vt in visit_types:
        print(f'Evaluating on visit type {v_map[str(vt)]}')
        df_vt = df[df['visit_type'] == vt]
        ident = f'{identifier}_{setting}_{clean_name(v_map[str(vt)])}'
        if calibrate:
            ident = f'{ident}_calibrated'
        if refit:
            clf, feature_names = train_eval.fit_model(df_vt, model=model)
        elif finetune:
            clf, feature_names = train_eval.fit_model(df_vt,
                                                      model=model,
                                                      clf=clf)
        _ = train_eval.evaluate(clf,
                                df_vt,
                                model=model,
                                split=split,
                                identifier=ident,
                                n_boots=n_boots,
                                force_rerun=force_rerun)
    return


def report_results(identifier: str = 'test',
                   model='LGBM',
                   final: bool = False,
                   calibrate: bool = False):
    v_map = load_category_mappings()['visit_type']
    if final:
        split = 'heldout_test'
    else:
        split = 'validation'
    for setting in ['community', 'clinic', 'inpatient']:
        print(f'Setting: {setting}')
        for vt in ['overall'] + list(v_map.values()):
            print(f'\tvisit type: {vt}')
            try:
                metrics = json.load(
                    open(
                        RESULTS_PATH /
                        f'metrics_{model}_{identifier}_{setting}_{clean_name(vt)}{calibrate*"_calibrated"}_{split}.json'
                    ))
            except FileNotFoundError:
                print(f'Visit type {vt} not available')
                continue
            # DEBUG
            print('\t\tAUC:')
            print(
                f'\t\t{metrics["AUC_ci_lower"]} : {metrics["AUC_mean"]} : {metrics["AUC_ci_upper"]}'
            )

    return


def report_missingness(df: pd.DataFrame) -> None:
    feature_categories, nec = define_feature_categories(df)
    vm = load_category_mappings()['visit_type']
    feature_classes = ['community', 'clinic', 'inpatient']
    features_accounted = []
    for fc in feature_classes:
        print(fc)
        print(
            f'New features are... {[x for x in feature_categories[fc] if not x in features_accounted]}'
        )
        features_accounted = features_accounted + feature_categories[fc]
        df_sub = df.loc[:, feature_categories[fc]]
        overall_missingness = 100 * df_sub.isna().mean().mean()
        print(f'\toverall: \t\t{np.round(overall_missingness, 1)}')
        for v, k in vm.items():
            dfv = df.loc[df['visit_type'] == int(v), :]
            dfv_sub = dfv.loc[:, feature_categories[fc]]
            missingness = 100 * dfv_sub.isna().mean().mean()
            print(f'\t{k}: \t{np.round(missingness, 1)}')


def analysis_individual_features(df: pd.DataFrame,
                                 n_boots: int = 5,
                                 identifier: str = 'test',
                                 model: str = 'LGBM',
                                 visit_type: str = 'overall',
                                 split: str = 'validation',
                                 force_rerun: bool = False):
    res_path = f'individal_features_{identifier}_{model}_{clean_name(visit_type)}_{split}.csv'
    try:
        results = pd.read_csv(RESULTS_PATH / res_path)
        results.set_index('Unnamed: 0', inplace=True)
        return results, np.nan
    except FileNotFoundError:
        pass
    clf, feature_names = train_eval.fit_model(df, model=model)
    considered_features = []
    conditional_performance = []
    conditional_performance_lower = []
    conditional_performance_upper = []
    conditional_dropped_performance = []
    conditional_dropped_performance_lower = []
    conditional_dropped_performance_upper = []
    n_patients_array = []
    df = df.copy()
    if visit_type == 'overall':
        pass
    else:
        print(f'Restricting to patients with visit type {visit_type}')
        vm = load_category_mappings()['visit_type']
        # sorry world
        vt = int(list(vm.keys())[list(vm.values()).index(visit_type)])
        df = df.copy()
        df = df[df['visit_type'] == vt]
    m = train_eval.evaluate(clf,
                            df,
                            model=model,
                            split=split,
                            identifier=f'{identifier}_features',
                            save=False,
                            n_boots=n_boots,
                            force_rerun=force_rerun)
    background_performance = m['AUC_mean']
    for f in feature_names:
        miss = df[f].isna().mean()
        if miss == 0:
            print(f'Skipping {f} because it is always observed')
            continue
        if miss == 1:
            print(f'Skipping {f} because it is NEVER observed')
            continue
        df_f = df.loc[df[f].notna(), :]
        print(df_f['split'].value_counts()['validation'])
        if df_f['split'].value_counts()['validation'] == 0:
            print(f'Skipping {f} because there are no validation samples')
            continue
        considered_features.append(f)
        n_patients = df_f.shape[0]
        n_patients_array.append(n_patients)
        print(f'Evaluating on the {n_patients} patients who have {f}')
        # Just get the general performance
        m = train_eval.evaluate(clf,
                                df_f,
                                model=model,
                                split=split,
                                identifier=f'{identifier}_{f}',
                                save=False,
                                n_boots=n_boots,
                                force_rerun=force_rerun)
        conditional_performance.append(m['AUC_mean'])
        conditional_performance_lower.append(m['AUC_ci_lower'])
        conditional_performance_upper.append(m['AUC_ci_upper'])
        # Now make an imputed version where we destroy information in f
        df_f[f] = df_f[f].mean()
        m = train_eval.evaluate(clf,
                                df_f,
                                model=model,
                                split=split,
                                identifier=f'{identifier}_{f}_dropped',
                                save=False,
                                n_boots=n_boots,
                                force_rerun=force_rerun)
        conditional_dropped_performance.append(m['AUC_mean'])
        conditional_dropped_performance_lower.append(m['AUC_ci_lower'])
        conditional_dropped_performance_upper.append(m['AUC_ci_upper'])
    results = pd.DataFrame(index=considered_features,
                           data={
                               'AUC': conditional_performance,
                               'AUC_lower': conditional_performance_lower,
                               'AUC_upper': conditional_performance_upper,
                               'AUC_drop': conditional_dropped_performance,
                               'AUC_drop_lower':
                               conditional_dropped_performance_lower,
                               'AUC_drop_upper':
                               conditional_dropped_performance_upper,
                               'N': n_patients_array
                           })

    results.sort_values(by='AUC', inplace=True, ascending=False)
    results.to_csv(RESULTS_PATH / res_path)
    plot_individual_feature_analysis(results,
                                     background_performance,
                                     identifier=identifier,
                                     model=model,
                                     top_n=25,
                                     visit_type=visit_type)
    return results, background_performance


def plot_individual_feature_analysis(results,
                                     background,
                                     identifier='test',
                                     model='LGBM',
                                     top_n=20,
                                     visit_type='overall',
                                     include_background: bool = True):
    if top_n is not None:
        res = results.iloc[:top_n, :]

    n_feat = res.shape[0]
    fig, axarr = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))

    y_points = np.arange(n_feat)[::-1]
    colors = np.where(res['AUC'] > res['AUC_drop'], 'blue', 'red')
    lower_CI = np.where(res['AUC'] > res['AUC_drop'], res['AUC_drop_lower'],
                        res['AUC_lower'])
    upper_CI = np.where(res['AUC'] > res['AUC_drop'], res['AUC_drop_upper'],
                        res['AUC_upper'])

    offset = 0.0
    linewidth = 2.0
    in_color = '#872068'
    masked_color = 'grey'
    axarr.hlines(y_points,
                 xmin=res['AUC_lower'],
                 xmax=res['AUC_upper'],
                 colors=in_color,
                 linewidth=linewidth,
                 alpha=0.25)
    axarr.hlines(y_points - offset,
                 xmin=res['AUC_drop_lower'],
                 xmax=res['AUC_drop_upper'],
                 colors=masked_color,
                 linewidth=linewidth,
                 alpha=0.25)
    axarr.scatter(y=y_points,
                  x=res['AUC'],
                  c=in_color,
                  marker='d',
                  label='Feature present')
    axarr.scatter(y=y_points - offset,
                  x=res['AUC_drop'],
                  c=masked_color,
                  marker='d',
                  label='Feature masked')
    #    axarr.hlines(y_points, xmin=res['AUC_drop'], xmax=res['AUC'], colors=colors, linewidth=4)
    #    axarr.hlines(y_points, xmin=lower_CI, xmax=upper_CI, colors=colors, alpha=0.3)
    if include_background:
        axarr.axvline(x=background,
                      ls='--',
                      color='black',
                      label='Overall performance',
                      alpha=0.5)
    # put in Ns
    for i, y in enumerate(y_points):
        n = res.iloc[i]['N']
        axarr.text(0.962,
                   y,
                   str(int(n)),
                   verticalalignment='center',
                   color='gray',
                   fontsize=9)
        #axarr.text(0.85, y, str(int(n)), verticalalignment='center', color='gray', fontsize=9)

    axarr.set_yticks(y_points)
    features = res.index
    axarr.set_yticklabels(tidy_feature_names(features), rotation=0, ha='right')
    axarr.set_xlabel('AUC')
    axarr.legend()

    plt.tight_layout()
    plt.savefig(
        WINDOWS_PATH /
        f'feature_performance_{identifier}_{model}_{clean_name(visit_type)}.png'
    )
    plt.savefig(
        WINDOWS_PATH /
        f'feature_performance_{identifier}_{model}_{clean_name(visit_type)}.pdf'
    )
    plt.clf()
    plt.close()
    return


def vis_feature_importance_by_visit_type(identifier: str = 'test',
                                         model='LGBM',
                                         top_n=5,
                                         sort_by_visit_type='overall',
                                         by='visit_type',
                                         setting='inpatient',
                                         final: bool = False,
                                         shap=True,
                                         xmax=None,
                                         n_boots: int = 1000) -> None:
    visit_types = [
        'overall', 'inpatient/ICU/ED', 'outpatient/rehab',
        'telehealth/telephone'
    ]
    if final:
        split = 'heldout_test'
    else:
        split = 'validation'
    if shap:
        shap_dict = train_eval.load_SHAP(identifier=identifier,
                                         model=model,
                                         setting=setting,
                                         split=split,
                              by=by,
                              n_boots=n_boots)
        val = 'abs_shap_mean'
        if n_boots == 0:
            err = 'abs_shap_sem'
        else:
            lower = 'abs_shap_lower'
            upper = 'abs_shap_upper'
    else:
        raise NotImplementedError
    if by == 'visit_type':
        vm = load_category_mappings()['visit_type']
        vm['overall'] = 'overall'
        # remapping the kerys from the shap dict to be strings
        shap_dict_string = dict()
        for k, v in shap_dict.items():
            shap_dict_string[vm[str(k)]] = v
        shap_dict = shap_dict_string
    try:
        assert sort_by_visit_type in visit_types
        assert sort_by_visit_type in shap_dict.keys()
    except AssertionError:
        print(visit_types)
        print(shap_dict.keys())

    top_features = get_top_features(shap_dict.values(), top_n=top_n, shap=shap)
    top_features_sorted = shap_dict[sort_by_visit_type].loc[
        top_features].sort_values(by=val, inplace=False, ascending=False).index
    shap_list = [shap_dict[vt].loc[top_features_sorted] for vt in visit_types]
    #shap_list = [p.loc[top_features] for p in shap_list]
    #    shap_list = [p.loc[top_features_sorted] for p in shap_list]

    n_features = len(top_features)
    fig, axarr = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4.5))
    spacing = 1.05
    y = np.arange(n_features)[::-1] * spacing
    n_visit_types = len(visit_types)
    width = spacing * 0.75 / n_visit_types
    visit_type_offsets = np.arange(-(n_visit_types - 1) / 2,
                                   (n_visit_types - 1) / 2 + 1e-5, 1)[::-1]
    assert len(visit_type_offsets) == n_visit_types

    #visit_type_colours = plt.cm.viridis(np.linspace(0, 1, n_visit_types)).tolist()
    visit_type_colours = [visit_colours[x] for x in visit_types]
    assert len(visit_type_colours) == n_visit_types
    for i, (visit_type, df) in enumerate(zip(visit_types, shap_list)):
        if n_boots == 0:
            axarr.barh(y + visit_type_offsets[i] * width,
                       df[val],
                       width,
                       label=visit_type_name_remap[visit_type],
                       xerr=df[err],
                       color=visit_type_colours[i],
                       alpha=0.8)
        else:
            lower_error = df[val] - df[lower]
            upper_error = df[upper] - df[val]
            errors = [lower_error, upper_error]
            axarr.barh(y + visit_type_offsets[i] * width,
                       df[val],
                       width,
                       label=visit_type_name_remap[visit_type],
                       xerr=errors,
                       color=visit_type_colours[i],
                       alpha=1.0)
    # xticks and such
    axarr.set_yticks(y)
    axarr.set_yticklabels(tidy_feature_names(top_features_sorted))
    axarr.legend()
    if shap:
        axarr.set_xlabel('Feature importance\n(Mean absolute SHAP value)')
    else:
        axarr.set_xlabel('absolute coefficient')

    if xmax is not None:
        xmin, _ = axarr.get_xlim()
        axarr.set_xlim(xmin, xmax)
    plt.tight_layout()
    plot_label = f'{identifier}_{model}_{setting}_{by}'
    if shap:
        plt.savefig(WINDOWS_PATH / f'feature_importance_SHAP_{plot_label}.png')
        plt.savefig(WINDOWS_PATH / f'feature_importance_SHAP_{plot_label}.pdf')
    else:
        plt.savefig(WINDOWS_PATH /
                    f'feature_importance_coefficients_{plot_label}.png')
        plt.savefig(WINDOWS_PATH /
                    f'feature_importance_coefficients_{plot_label}.pdf')
    plt.clf()
    plt.close()
    return


def bar_chart_comparison(identifier: str = '1010',
                         model: str = 'LGBM',
                         settings=[
                             'inpatient', 'sans_symptoms',
                             'sans_chronic_conditions', 'sans_labs',
                             'sans_vitals', 'sans_geography', 'sans_insurance',
                             'sans_demographics', 'sans_drugs'
                         ],
                         split='validation',
                         plot_name='ablation',
                         visit_types=['overall'],
                         compare_y: bool = True,
                         ylim=None) -> None:

    setting_colours = cm.get_cmap('tab20')(np.linspace(0, 1, len(settings)))
    setting_colours = ['#6b6b6b'] * len(settings)
    if visit_types is None:
        visit_types = [
            'overall', 'inpatient/ICU/ED', 'outpatient/rehab',
            'telehealth/telephone'
        ]
        fig, axarr = plt.subplots(nrows=len(visit_types),
                                  ncols=1,
                                  figsize=(4.5, 4.5),
                                  sharex=True)
        color_by_visit_type = True
    else:
        fig, axarr = plt.subplots(nrows=len(visit_types),
                                  ncols=1,
                                  figsize=(4, 3),
                                  sharex=True)
    x = np.arange(len(settings))
    width = 0.75
    if len(visit_types) == 1:
        axarr = [axarr]
    for i, visit_type in enumerate(visit_types):
        setting_AUC_mean = []
        err_lower = []
        err_upper = []
        for setting in settings:
            msv = json.load(
                open(
                    RESULTS_PATH /
                    f'metrics_{model}_{identifier}_{setting}_{clean_name(visit_type)}_{split}.json'
                ))
            setting_AUC_mean.append(msv['AUC_mean'])
            err_lower.append(msv['AUC_mean'] - msv['AUC_ci_lower'])
            err_upper.append(msv['AUC_ci_upper'] - msv['AUC_mean'])
            del msv
        df = pd.DataFrame(
            {
                'mean': setting_AUC_mean,
                'lower': err_lower,
                'upper': err_upper
            },
            index=settings)
        df.sort_values(by='mean', ascending=False, inplace=True)
        if i == 0:
            labels = df.index
        else:
            df = df.loc[labels]
        errors = np.array(df['lower'], df['upper'])
        if color_by_visit_type:
            color = visit_colours[visit_type]
        else:
            color = setting_colours
        axarr[i].bar(x,
                     df['mean'],
                     width,
                     yerr=errors,
                     color=color,
                     ecolor='black')
        #axarr[i].set_ylabel(visit_type, rotation=0, ha='right')
        if len(visit_types) > 1:
            axarr[i].set_ylabel(f'AUC\n{visit_type_name_remap[visit_type]}',
                                rotation=0,
                                horizontalalignment='right')
        else:
            axarr[i].set_ylabel('AUC')
        background = df.loc['inpatient', :]
        bg_lower = background['mean'] - background['lower']
        bg_upper = background['mean'] + background['upper']

        if color_by_visit_type:
            axarr[i].axhspan(bg_lower,
                             bg_upper,
                             color=color,
                             alpha=0.2,
                             hatch='XX')
        else:
            axarr[i].axhspan(bg_lower,
                             bg_upper,
                             color=setting_colours[0],
                             alpha=0.2,
                             hatch='XX')
        #        axarr[i].axhline(y=setting_AUC_mean[0], ls='--', color=setting_colours[0])

        max_AUC = max(setting_AUC_mean) + max(err_upper)
        min_AUC = min(setting_AUC_mean) - max(err_lower)
        axarr[i].set_ylim(0.95 * min_AUC, 1.05 * max_AUC)

    axarr[-1].set_xticks(x)
    # tidy up the labels
    nl = []
    for l in labels:
        if 'sans' in l:
            what = ' '.join(l.split('_')[1:]).capitalize()
            nl.append(f'–{what}')
        else:
            if l == 'inpatient':
                nl.append('All features')
            else:
                nl.append(l)
    axarr[-1].set_xticklabels(nl, rotation=45, ha='right')

    ymax = 0
    if compare_y:
        plot_name = f'{plot_name}_fix_yaxis'
        for ax in axarr:
            _, ymax_ax = ax.get_ylim()
            if ymax_ax > ymax:
                ymax = ymax_ax
        for ax in axarr:
            ax.set_ylim(0.5, ymax)
    if not ylim is None:
        for ax in axarr:
            ax.set_ylim(ylim)
    plt.tight_layout()
    plt.savefig(WINDOWS_PATH /
                f'ablation_{identifier}_{model}_{plot_name}.png')
    plt.savefig(WINDOWS_PATH /
                f'ablation_{identifier}_{model}_{plot_name}.pdf')
    plt.clf()
    plt.close()
    return


def compare_restricted_featuresets(identifier: str = '1010',
                                   model: str = 'LGBM',
                                   split='validation',
                                   plot_name='bars',
                                   compare_y: bool = True,
                                   ylim=None) -> None:

    visit_types = ['overall', 'inpatient/ICU/ED', 'outpatient/rehab', 'telehealth/telephone']
    colours = [visit_colours[x] for x in visit_types]
    fig, axarr = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 3))
    setting_hatches = ['\\\\', '///', None]
    #setting_alphas = [1.0, 0.6, 0.3]
    setting_alphas = ['77', '99', 'FF']
    settings = ['community', 'clinic', 'inpatient']
    setting_labels = ['Minimal', 'Intermediate', 'Full']
    n_settings = len(settings)
    setting_offsets = np.arange(-(n_settings - 1) / 2,
                                   (n_settings - 1) / 2 + 1e-5, 1)
    spacing = 1.05
    width = spacing * 0.75 / n_settings
    x = np.arange(len(visit_types))
    axarr.set_ylabel('AUC')
    for i, setting in enumerate(settings):
        visit_type_AUC_mean = []
        err_lower = []
        err_upper = []
        for visit_type in visit_types:
            msv = json.load(open(RESULTS_PATH / f'metrics_{model}_{identifier}_{setting}_{clean_name(visit_type)}_{split}.json'))
            visit_type_AUC_mean.append(msv['AUC_mean'])
            err_lower.append(msv['AUC_mean'] - msv['AUC_ci_lower'])
            err_upper.append(msv['AUC_ci_upper'] - msv['AUC_mean'])
            if setting == 'community':
                print(f'Community upper CI: {visit_type}, {msv["AUC_ci_upper"]}')
            elif setting == 'inpatient':
                print(f'Inpatient lower CI: {visit_type}, {msv["AUC_ci_lower"]}')
            del msv
        df = pd.DataFrame({'mean': visit_type_AUC_mean, 'lower': err_lower, 'upper': err_upper},
                          index=visit_types)
        df.loc[visit_types]
        errors = np.array(df['lower'], df['upper'])
        cols = [x + setting_alphas[i] for x in colours]
        axarr.bar(x + setting_offsets[i] * width,
                  df['mean'],
                  width,
                  yerr=errors,
                  color=cols,
                  ecolor='black',
                  hatch=None)
        axarr.bar(x + setting_offsets[i] * width,
                  df['mean'],
                  width,
                  fill=False,
                  edgecolor=cols,
                  alpha=1,
                  hatch=setting_hatches[i],
                  label=setting_labels[i])


    axarr.legend(title='Features available', ncol=len(settings), loc='upper center', bbox_to_anchor=(0.5, 1.2))
    axarr.set_xticks(x)
    # tidy up the labels
    labels = [visit_type_name_remap[x] for x in visit_types]
    labels[-1] = 'Telehealth\nPatients'
    axarr.set_xticklabels(labels, rotation=0, ha='center')

    if ylim is None:
        axarr.set_ylim(0.5, None)
    else:
        axarr.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(WINDOWS_PATH / f'featuresets_{identifier}_{model}_{plot_name}.png')
    plt.savefig(WINDOWS_PATH / f'featuresets_{identifier}_{model}_{plot_name}.pdf')
    plt.clf()
    plt.close()
    return

def model_perf_v_dataset_size(data: pd.DataFrame, model: str = 'LGBM'):
    N = data.shape[0]
    reps = 5
    for N_sub in np.int32([N / 250, N / 100, N / 75, N / 10, N / 5, N / 2, N]):
        auc = []
        for r in range(reps):
            data_sub = data.iloc[np.random.choice(N, N_sub, replace=False)]
            while data_sub.loc[data_sub['split'] ==
                               'test', 'test_outcome'].sum() == 0:
                print('Reselecting to get non-zero positive labels in test...')
                data_sub = data.iloc[np.random.choice(N, N_sub, replace=False)]

            clf, feature_labels = train_eval.fit_model(data_sub,
                                                       model=model,
                                                       upsample=False,
                                                       verbose=False)
            metrics = train_eval.evaluate(clf,
                                          data_sub,
                                          split='validation',
                                          model=model,
                                          verbose=False)
            auc.append(metrics['AUC'])
        auc = np.array(auc)
        print(N_sub, np.mean(auc), sem(auc))
    return


def plot_visit_comparison_curves(identifier='2209',
                                 model='LGBM',
                                 which_curve='ROC',
                                 setting: str = 'inpatient',
                                 only_overall: bool = False,
                                 no_overall: bool = False,
                                 split: str = 'validation',
                                 calibrate: bool = False):
    fig, axarr = plt.subplots(nrows=1, ncols=1, figsize=(3.5, 3.5))
    s = 3
    if which_curve == 'ROC':
        x = 'ROC_FPR'
        xlabel = '1 - Specificity (FPR)'
        y = 'ROC_TPR'
        ylabel = 'Sensitivity (TPR)'
        label = 'AUC_mean'
        axarr.plot([0, 1], [0, 1],
                   alpha=0.4,
                   color='grey',
                   ls='--',
                   label='Random: 0.5')
    elif which_curve == 'PR':
        x = 'PR_recall'
        xlabel = 'Recall (Sensitivity)'
        y = 'PR_precision'
        ylabel = 'Precision (PPV)'
        label = 'AUPRC_mean'
    elif which_curve == 'reliability':
        x = 'reliability_prob_pred'
        xlabel = 'Predicted probability'
        y = 'reliability_prob_true'
        ylabel = 'Fraction with event'
        axarr.plot([0, 1], [0, 1], alpha=0.4, color='grey', ls=':')
        label = 'brier_mean'

    v_map = load_category_mappings()['visit_type']
    v_map['all'] = 'overall'
    visit_types = v_map.values()
    xmax = 0
    for visit_type in visit_types:
        if only_overall:
            if not visit_type == 'overall':
                continue
        if no_overall:
            if visit_type == 'overall':
                continue
        try:
            metrics = json.load(
                open(
                    RESULTS_PATH /
                    f'metrics_{model}_{identifier}_{setting}_{clean_name(visit_type)}{calibrate*"_calibrated"}_{split}.json'
                ))
            # DEBUG
            if not 'brier_mean' in metrics.keys():
                metrics['brier_mean'] = np.nan
        except FileNotFoundError:
            print(
                f'WARNING: Could not find results for visit type {visit_type}')
            continue
        axarr.scatter(metrics[f'{x}_mean'],
                      metrics[f'{y}_mean'],
                      s=s,
                      color=visit_colours[visit_type])
        curve = axarr.plot(
            metrics[f'{x}_mean'],
            metrics[f'{y}_mean'],
            alpha=0.5,
            label=
            f'{visit_type_name_remap[visit_type]}: {np.round(metrics[label], 2)}',
            color=visit_colours[visit_type])
        axarr.fill_between(metrics[f'{x}_mean'],
                           metrics[f'{which_curve}_ci_lower'],
                           metrics[f'{which_curve}_ci_upper'],
                           alpha=0.2,
                           color=visit_colours[visit_type])
        candidate_xmax = max(metrics[f'{x}_mean'])
        if candidate_xmax > xmax:
            xmax = candidate_xmax
        if which_curve == 'PR':
            axarr.axhline(y=metrics['prevalence_mean'],
                          color=curve[-1].get_color(),
                          ls='--',
                          alpha=0.4)
            # , label=f'random: {np.round(metrics["prevalence_mean"], 2)}')

    if which_curve == 'reliability':
        axarr.set_xlim(0, xmax * 1.05)
        axarr.set_ylim(0, xmax * 1.05)
    axarr.set_ylabel(ylabel)
    axarr.set_ylabel(ylabel)
    axarr.set_xlabel(xlabel)
    axarr.set_aspect(1)
    axarr.legend()
    plt.tight_layout()
    plt.savefig(
        WINDOWS_PATH /
        f'{which_curve}_{identifier}{calibrate*"_calibrated"}_{model}_visits_{setting}.png'
    )
    plt.savefig(
        WINDOWS_PATH /
        f'{which_curve}_{identifier}{calibrate*"_calibrated"}_{model}_visits_{setting}.pdf'
    )
    plt.clf()
    plt.close()
    return


def plot_setting_comparison_curves(identifier='2209',
                                   model='LGBM',
                                   which_curve='ROC',
                                   visit_type: str = 'overall',
                                   split: str = 'validation',
                                   calibrate: bool = False):
    fig, axarr = plt.subplots(nrows=1, ncols=1, figsize=(3.5, 3.5))
    s = 3
    if which_curve == 'ROC':
        x = 'ROC_FPR'
        xlabel = '1 - Specificity (FPR)'
        y = 'ROC_TPR'
        ylabel = 'Sensitivity (TPR)'
        label = 'AUC_mean'
        axarr.plot([0, 1], [0, 1],
                   alpha=0.4,
                   color='grey',
                   ls='--',
                   label='Random: 0.5')
    elif which_curve == 'PR':
        x = 'PR_recall'
        xlabel = 'Recall (Sensitivity)'
        y = 'PR_precision'
        ylabel = 'Precision (PPV)'
        label = 'AUPRC_mean'
    elif which_curve == 'reliability':
        x = 'reliability_prob_pred'
        xlabel = 'Predicted probability'
        y = 'reliability_prob_true'
        ylabel = 'Fraction with event'
        axarr.plot([0, 1], [0, 1], alpha=0.4, color='grey', ls=':')
        label = 'brier_mean'

    v_map = load_category_mappings()['visit_type']
    v_map['all'] = 'overall'
    settings = ['inpatient', 'clinic', 'community']
    xmax = 0
    for setting in settings:
        try:
            metrics = json.load(
                open(
                    RESULTS_PATH /
                    f'metrics_{model}_{identifier}_{setting}_{clean_name(visit_type)}{calibrate*"_calibrated"}_{split}.json'
                ))
            # DEBUG
            if not 'brier_mean' in metrics.keys():
                metrics['brier_mean'] = np.nan
        except FileNotFoundError:
            print(f'WARNING: Could not find results for setting {setting}')
            continue
        axarr.scatter(metrics[f'{x}_mean'],
                      metrics[f'{y}_mean'],
                      s=s,
                      color=setting_colours[setting])
        curve = axarr.plot(
            metrics[f'{x}_mean'],
            metrics[f'{y}_mean'],
            alpha=0.5,
            label=
            f'{setting_name_remap[setting]}: {np.round(metrics[label], 2)}',
            color=setting_colours[setting])
        axarr.fill_between(metrics[f'{x}_mean'],
                           metrics[f'{which_curve}_ci_lower'],
                           metrics[f'{which_curve}_ci_upper'],
                           alpha=0.2,
                           color=setting_colours[setting])
        candidate_xmax = max(metrics[f'{x}_mean'])
        if candidate_xmax > xmax:
            xmax = candidate_xmax
        if which_curve == 'PR':
            axarr.axhline(y=metrics['prevalence_mean'],
                          color=curve[-1].get_color(),
                          ls='--',
                          alpha=0.4)
            # , label=f'random: {np.round(metrics["prevalence_mean"], 2)}')

    if which_curve == 'reliability':
        axarr.set_xlim(0, xmax * 1.05)
        axarr.set_ylim(0, xmax * 1.05)
    axarr.set_ylabel(ylabel)
    axarr.set_ylabel(ylabel)
    axarr.set_xlabel(xlabel)
    axarr.set_aspect(1)
    axarr.legend()
    plt.tight_layout()
    plt.savefig(
        WINDOWS_PATH /
        f'{which_curve}_{identifier}{calibrate*"_calibrated"}_{model}_settings_{clean_name(visit_type)}.png'
    )
    plt.savefig(
        WINDOWS_PATH /
        f'{which_curve}_{identifier}{calibrate*"_calibrated"}_{model}_settings_{clean_name(visit_type)}.pdf'
    )
    plt.clf()
    plt.close()
    return


def old_plot_setting_comparison_curves(identifier='2209',
                                       model='LGBM',
                                       fold=2,
                                       which_curve='ROC',
                                       aggregate: bool = True,
                                       visit_type: str = 'overall'):
    fig, axarr = plt.subplots(nrows=1, ncols=1)
    s = 3
    if which_curve == 'ROC':
        x = 'ROC_FPR'
        xlabel = '1 - Specificity (FPR)'
        y = 'ROC_TPR'
        ylabel = 'Sensitivity (TPR)'
        label = 'AUC'
        axarr.plot([0, 1], [0, 1], alpha=0.5, color='black', ls='--')
    elif which_curve == 'PR':
        x = 'PR_recall'
        xlabel = 'Recall (Sensitivity)'
        y = 'PR_precision'
        ylabel = 'Precision (PPV)'
        label = 'AUPRC'
    elif which_curve == 'reliability':
        x = 'reliability_prob_pred'
        xlabel = 'Predicted probability'
        y = 'reliability_prob_true'
        ylabel = 'Fraction with event'
        axarr.plot([0, 1], [0, 1], alpha=0.5, color='black', ls='--')
        label = 'brier'
    if aggregate:
        label = f'{label}_mean'

    v_map = load_category_mappings()['visit_type']
    v_map['all'] = 'overall'
    settings = ['community', 'clinic', 'inpatient']
    xmax = 0
    for setting in settings:
        try:
            metrics = json.load(
                open(
                    RESULTS_PATH /
                    f'metrics_{model}_{identifier}_{setting}_{clean_name(visit_type)}_validation.json'
                ))
        except FileNotFoundError:
            print(f'WARNING: Could not find results for setting {setting}')
            continue
        # FOR NOW pick just one fold
        if aggregate:
            axarr.scatter(metrics[f'{x}_mean'], metrics[f'{y}_mean'], s=s)
            curve = axarr.plot(
                metrics[f'{x}_mean'],
                metrics[f'{y}_mean'],
                alpha=0.5,
                label=f'{setting}: {np.round(metrics[label], 3)}')
            axarr.fill_between(metrics[f'{x}_mean'],
                               metrics[f'{which_curve}_ci_lower'],
                               metrics[f'{which_curve}_ci_upper'],
                               alpha=0.2,
                               color=curve[-1].get_color())
            # TODO add aggregated AUC
            candidate_xmax = max(metrics[f'{x}_mean'])
            if candidate_xmax > xmax:
                xmax = candidate_xmax
        else:
            m = metrics[str(fold)]
            axarr.scatter(m[x], m[y], s=s)
            curve = axarr.plot(m[x],
                               m[y],
                               alpha=0.5,
                               label=f'{visit_type}: {np.round(m[label], 3)}')
        if which_curve == 'PR':
            axarr.axhline(y=m['prevalence'],
                          color=curve[-1].get_color(),
                          ls='--',
                          alpha=0.5)

    if which_curve == 'reliability':
        axarr.set_xlim(0, xmax * 1.05)
        axarr.set_ylim(0, ymax * 1.05)
    axarr.set_ylabel(ylabel)
    axarr.set_xlabel(xlabel)
    axarr.set_aspect(1)
    axarr.legend()
    plt.savefig(
        WINDOWS_PATH /
        f'{which_curve}_{identifier}_{model}_settings_{clean_name(visit_type)}.png'
    )
    plt.savefig(
        WINDOWS_PATH /
        f'{which_curve}_{identifier}_{model}_settings_{clean_name(visit_type)}.pdf'
    )
    plt.clf()
    plt.close()
    return


def plot_model_comparison_curves(identifier='2209',
                                 models=['LGBM', 'LR'],
                                 fold=2,
                                 which_curve='ROC',
                                 aggregate: bool = True,
                                 visit_type: str = 'overall',
                                 setting: str = 'inpatient',
                                 split: str = 'validation'):
    fig, axarr = plt.subplots(nrows=1, ncols=1, figsize=(3.5, 3.5))
    s = 3
    if which_curve == 'ROC':
        x = 'ROC_FPR'
        xlabel = '1 - Specificity (FPR)'
        y = 'ROC_TPR'
        ylabel = 'Sensitivity (TPR)'
        label = 'AUC'
        axarr.plot([0, 1], [0, 1], alpha=0.5, color='black', ls='--')
    elif which_curve == 'PR':
        x = 'PR_recall'
        xlabel = 'Recall (Sensitivity)'
        y = 'PR_precision'
        ylabel = 'Precision (PPV)'
        label = 'AUPRC'
    elif which_curve == 'reliability':
        x = 'reliability_prob_pred'
        xlabel = 'Predicted probability'
        y = 'reliability_prob_true'
        ylabel = 'Fraction with event'
        axarr.plot([0, 1], [0, 1], alpha=0.5, color='black', ls='--')
        label = 'brier'
    if aggregate:
        label = f'{label}_mean'

    v_map = load_category_mappings()['visit_type']
    v_map['all'] = 'overall'
    xmax = 0
    model_colours = {'LGBM': visit_colours[visit_type], 'LR': '#1f9c32'}
    for model in models:
        try:
            metrics = json.load(
                open(
                    RESULTS_PATH /
                    f'metrics_{model}_{identifier}_{setting}_{clean_name(visit_type)}_{split}.json'
                ))
        except FileNotFoundError:
            print(f'WARNING: Could not find results for model {model}')
            continue
        if aggregate:
            axarr.scatter(metrics[f'{x}_mean'],
                          metrics[f'{y}_mean'],
                          s=s,
                          color=model_colours[model])
            curve = axarr.plot(metrics[f'{x}_mean'],
                               metrics[f'{y}_mean'],
                               alpha=0.5,
                               label=f'{model}: {np.round(metrics[label], 2)}',
                               color=model_colours[model])
            axarr.fill_between(metrics[f'{x}_mean'],
                               metrics[f'{which_curve}_ci_lower'],
                               metrics[f'{which_curve}_ci_upper'],
                               alpha=0.2,
                               color=curve[-1].get_color())
            candidate_xmax = max(metrics[f'{x}_mean'])
            if candidate_xmax > xmax:
                xmax = candidate_xmax
        else:
            m = metrics[str(fold)]
            axarr.scatter(m[x], m[y], s=s)
            curve = axarr.plot(m[x],
                               m[y],
                               alpha=0.5,
                               label=f'{model}: {np.round(m[label], 2)}')
        if which_curve == 'PR':
            axarr.axhline(y=metrics['prevalence_mean'],
                          color=curve[-1].get_color(),
                          ls='--',
                          alpha=0.5)

    if which_curve == 'reliability':
        axarr.set_xlim(0, xmax * 1.05)
        axarr.set_ylim(0, xmax * 1.05)
    axarr.set_ylabel(ylabel)
    axarr.set_xlabel(xlabel)
    axarr.set_ylim(0, 1)
    axarr.set_aspect(1)
    axarr.legend()
    plt.tight_layout()
    plt.savefig(
        WINDOWS_PATH /
        f'{which_curve}_{identifier}_{setting}_{clean_name(visit_type)}_model_comparison.png'
    )
    plt.savefig(
        WINDOWS_PATH /
        f'{which_curve}_{identifier}_{setting}_{clean_name(visit_type)}_model_comparison.pdf'
    )
    plt.clf()
    plt.close()
    return


def plot_pooling(identifier='2209',
                 model='LGBM',
                 fold=2,
                 strategy='ranked',
                 setting='inpatient',
                 xmax=400):
    fig, axarr = plt.subplots(nrows=1, ncols=1)
    v_map = load_category_mappings()['visit_type']
    v_map['all'] = 'overall'
    visit_types = v_map.values()
    for visit_type in visit_types:
        if visit_type == 'unknown':
            continue
        metrics = json.load(
            open(
                RESULTS_PATH /
                f'metrics_{model}_{identifier}_{setting}_{clean_name(visit_type)}_validation.json'
            ))
        print(metrics.keys())
        m = metrics[str(fold)]
        if strategy == 'ranked':
            curve = axarr.plot(m['POOL_pool_sizes'],
                               m['POOL_frac_pos_model'],
                               alpha=0.8,
                               label=f'by model, {clean_name(visit_type)}')
            axarr.plot(m['POOL_pool_sizes'],
                       m['POOL_frac_pos_random'],
                       alpha=0.8,
                       label=f'random, {clean_name(visit_type)}',
                       color=curve[-1].get_color(),
                       ls='--')
        else:
            curve = axarr.plot(m['POOL_thresholds'],
                               m['POOL_frac_pos_model'],
                               alpha=0.8,
                               label=f'by model, {clean_name(visit_type)}')
            random_background = np.nanmedian(m['POOL_frac_pos_random'])
            axarr.axhline(y=random_background,
                          alpha=0.8,
                          label=f'random, {clean_name(visit_type)}',
                          color=curve[-1].get_color(),
                          ls='--')
    if strategy == 'ranked':
        axarr.set_xlabel('Pool size')
        if xmax is not None:
            axarr.set_xlim(1.5, xmax)
        axarr.set_xscale('log')
    else:
        axarr.set_xlabel('Decision threshold')
        axarr.set_title(f'Pool size: 10')  # TODO update if changed
    axarr.set_ylabel('Fraction of tests with a positive case')
    #    axarr.set_ylim(0, 1)
    axarr.legend()
    plt.savefig(WINDOWS_PATH / f'pooling_{identifier}.png')
    plt.savefig(WINDOWS_PATH / f'pooling_{identifier}.pdf')
    plt.clf()
    plt.close()
    return


def plot_SHAP_values(df,
                     identifier='1010',
                     model='LGBM',
                     visit_type='overall',
                     setting='inpatient',
                     split='validation') -> list:
    shap_all = pd.read_csv(RESULTS_PATH /
                           f'SHAP_{identifier}_{model}_{setting}_{split}.csv')
    v_map = load_category_mappings()['visit_type']
    df = df.copy()
    shap_all['visit_type'] = shap_all['visit_type'].astype('str').map(v_map)
    df['visit_type'] = df['visit_type'].astype('str').map(v_map)
    if visit_type == 'overall':
        shap_df = shap_all
        df_sub = df
    else:
        assert visit_type in shap_all['visit_type'].unique()
        shap_df = shap_all[shap_all['visit_type'] == visit_type]
        df_sub = df.loc[df['visit_type'] == visit_type, :]
    # tidy it up
    shap_df.drop(columns=['visit_type', 'person_id'], inplace=True)
    # get top features, and such (this is a bit overkill)
    abs_shap_mean = np.abs(shap_df).mean(axis=0)
    abs_shap_mean = np.abs(shap_df).mean(axis=0)
    abs_shap_median = np.median(np.abs(shap_df), axis=0)
    abs_shap_std = np.abs(shap_df).std(axis=0)
    abs_shap_sem = sem(np.abs(shap_df), axis=0)
    agg_df = pd.DataFrame({
        'abs_shap_mean': abs_shap_mean,
        'abs_shap_median': abs_shap_median,
        'abs_shap_std': abs_shap_std,
        'abs_shap_sem': abs_shap_sem
    })
    agg_df.sort_values(by='abs_shap_mean', inplace=True, ascending=False)

    top_features = agg_df.iloc[:20].index.values
    print(f'Visualising SHAP values for features {top_features}')
    X, _ = prepare_data(df_sub, split=split)
    assert X.shape[0] == shap_df.shape[0]
    # now plot
    summary_plot(shap_df.loc[:, top_features].values,
                 X.loc[:, top_features],
                 feature_names=tidy_feature_names(top_features),
                 show=False,
                 sort=False,
                 plot_type='dot')
    fig = plt.gcf()
    my_cmap = plt.get_cmap('winter')
    viridisBig = cm.get_cmap('viridis', 512)
    my_cmap = ListedColormap(viridisBig(np.linspace(0, 0.75, 256)))
    # Change the colormap of the artists
    for fc in fig.get_children():
        for fcc in fc.get_children():
            if hasattr(fcc, "set_cmap"):
                fcc.set_cmap(my_cmap)
    fig.set_size_inches(10, 7.5)
    axes = fig.axes
    #    axes[0].spines['left'].set_visible(True)
    axes[0].grid(which='major', axis='x', linestyle='--', alpha=0.8)
    # axes[0].set_facecolor('#e3e3e3')
    plt.tight_layout()
    plt.savefig(
        WINDOWS_PATH /
        f'SHAP_{identifier}_{model}_{setting}_{clean_name(visit_type)}_{split}.png'
    )
    plt.savefig(
        WINDOWS_PATH /
        f'SHAP_{identifier}_{model}_{setting}_{clean_name(visit_type)}_{split}.pdf'
    )
    plt.clf()
    plt.close()
    return top_features
