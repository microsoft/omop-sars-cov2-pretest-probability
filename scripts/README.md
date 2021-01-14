## Description of files

- `analyses.py`: Performing high-level analyses and generating figures for the publication.
- `data_preproc.py`: Data preprocessing functions and pipeline. Important functions are `collect_data` (assembles data from disparate tables), `get_labels` (defines the prediction label for each patient), `build_dataset` (uses collected data to build training dataset)
- `misc_utils.py`: Miscellaneous functions for working with OMOP data. Not used in rest of repo but useful for data exploration.
- `omop_metadata.py`: Information about OMOP table schemas, mapping dictionaries from OMOP concept IDs to human-readable variable names, thresholds on variables, and classifications into feature categories.
- `train_eval.py`: Functions for training and evaluating model (full analyses for paper are in analyses.py), including hyperparameter optimisation, computing and reporting SHAP values.

## Typical workflow

A lot of the functionality is contained in wrapper functions, so I would recommend starting there. Most of the logic is in the data preprocessing - after that, we are just training standard models on different slices of the data, and pulling results into figures.

### Perform data preprocessing (build a dataset):
This uses `data_preproc.py`

```
df = build_dataset(load=False, version='your_version')
```
Some notes on the use of `build_dataset`:

- The `version` is the dataset identifier, so it's important to specify it.
- After you've generated it the first time, you will use `load = True` to avoid recomputation and ensuring you're not inadvertently re-computing test/train splits.

`build_dataset` has other options, e.g.:
- `categorical_features`: specify how to encode them (e.g. `no encode` if you want to explore the dataset and not train on it, `integer` for training with LightGBM). Note that this option is applied after the data is loaded, so you can always change the encoding on an otherwise pre-computed dataset.
- `row/column_missingness_threshold`: Specify the thresholds of missingness filtering for rows (patients) and columns (features) respectively.
- `do_scaling`: Do (robust) scaling of data? (Not required for tree-based models)
- `do_imputation`': Do imputation? (Not required for tree-based models)

If you want to dig into `build_dataset`, you will notice that it itself is basically a glorified wrapper around `collect_data`, which is where the dataset construction really gets going.`, which is where the dataset construction really gets going.`, which is where the dataset construction really gets going.`, which is where the dataset construction really gets going.

### Run analyses

Using the dataframe produced by `build_dataset`, training/evaluating the model in different feature/patient classes happens in `analyses.py`

```
run_analyses(df, identifier='your_analysis_identifier', final=False, setting='your_setting_choice')
```
This is a wrapper which will train a model using a set of features specified by `setting`, then evaluate it on the three different visit types. The `identiifer` here does not _have_ to be the same as your data version, but if you want to remember which data version you used for the analysis, you could put it in here.

If you set `final = True`, the evalution will use the final fully held-out test set, otherwise it will use the validation set (do not set final = True until you are finished with model development).

The values that `setting` can take are the keys of the dictionary output by `define_feature_categories` from `omop_metadata.py` - basically it's the feature set. This allows us to easily train/evaluate variant models with different subsets of features.

`run_analyses` has other options like:
- `model`: do you want to train a logistic regression (`LR`) or LightGBM (`LGBM`) model, or others? The list of implemented models can be seen in the options to `fit_model` in `train_eval.py`, and this is where you would add more, if you want.
- `n_boots` is how many bootstrap replicates to use in evaluation. The default is `1000` but this can be slow, so if you are testing you can set it to some small number.


### Generate figures

The output of `run_analyses` is essentially a bunch of aggregated results files (AUC of model X in setting Y, etc). If you want to reproduce the figures from the paper, I am concerned about how you have access to the dataset. However, the `analyses.py` file has a function for generating all the plots:

```
plot_wrapper(df, identifier='your_analysis_identifier')
```
This once again takes the data output from `build_dataset` - this is mostly not requird, except that colouring SHAP plots requires the underlying feature values.

In this case, the `identifier` _must_ match whatever you used in `run_analyses`, because this function loads the output of that function.

This function is predominantly helpful for documenting the list of functions used to generate the full set of figures (+ others that weren't included).
