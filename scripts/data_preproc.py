import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from omop_metadata import tables, covid_result_mapping, ethnicity_mapping
from omop_metadata import visit_type, thresholds
from omop_metadata import chronic_conditions, signs_and_symptoms
from omop_metadata import measurements, vitals, static, drugs
from omop_metadata import gender_mapping, race_mapping, occupation_mapping
import json


DATA_DIR = Path('../data_dir/')
PROCESSED_DATA_DIR = Path('../processed_data/')
DATA_VERSION = '20200817'
FIXED_TEST_PATH = Path('../metadata/patients_test.csv')

categorical_columns = ['race', 'county', 'zip',
                       'occupation_status', 'occupation_category',
                       'reason_for_test', 'visit_type']


def generate_feature_report(df: pd.DataFrame, identifier: str = '') -> None:
    """ For internal data exploration """
    binary_variables = []
    non_numeric_variables = []
    for col in df.columns:
        if set(df[col].unique()) == {0, 1}:
            binary_variables.append(col)
        try:
            _ = pd.to_numeric(df[col])
        except ValueError:
            non_numeric_variables.append(col)
    print(f'Warning: Found {len(non_numeric_variables)} non-numeric variables')
    numeric_variables = [x for x in df.columns if x not in binary_variables and x not in non_numeric_variables]

    missingness = df.isna().mean(axis=0)
    missingness.name = 'missingness'
    mean = df[numeric_variables + binary_variables].mean(axis=0)
    mean.name = 'mean'
    std = df[numeric_variables + binary_variables].std(axis=0)
    std.name = 'std'
    stats_list = [missingness, mean, std]
    if 'visit_type' in df.columns:
        visit_types = df['visit_type'].unique()
        for vt in visit_types:
            vt_missingness = df[df['visit_type'] == vt].isna().mean(axis=0)
            vt_missingness.name = f'missingness_{vt}'
            stats_list.append(vt_missingness)
    stats = pd.concat(stats_list, axis=1)
    stats.to_csv(PROCESSED_DATA_DIR / f'variable_stats_{identifier}.csv')
    return


def filter_by_time(df: pd.DataFrame,
                   max_time_before_test=pd.Timedelta('28D'),
                   data_dir: Path = DATA_DIR,
                   timestamp_start: str = 'datetime',
                   timestamp_end: str = None) -> pd.DataFrame:
    if timestamp_end is None:
        timestamp_end = timestamp_start
    assert timestamp_start in df.columns
    assert timestamp_end in df.columns
    assert 'measurement_datetime' not in df.columns

    # The time filtering here is with respect to the labels
    labels = get_labels(data_dir)[['person_id', 'measurement_datetime', 'test_outcome']]
    labels['measurement_datetime'] = pd.to_datetime(labels['measurement_datetime'])
    labels.set_index('person_id', inplace=True)

    # First: remove data after the label
    df = df.join(labels['measurement_datetime'], on='person_id')
    n = df.shape[0]
    df = df[df[timestamp_start] <= df['measurement_datetime']]
    print(f'Removed {n - df.shape[0]} rows from table for occurring after the test')

    # Now, remove data from too far before the test
    if max_time_before_test is not None:
        n = df.shape[0]
        time_before_test = df['measurement_datetime'] - df[timestamp_end]
        if timestamp_start == timestamp_end:
            # we should have already filtered these cases
            assert np.all(time_before_test >= pd.Timedelta(0))
        df = df[time_before_test <= max_time_before_test]
        print(f'Removed {n - df.shape[0]} rows for occurring more than {max_time_before_test} before test')

    assert np.all(df[timestamp_start] <= df['measurement_datetime'])
    df.drop(columns='measurement_datetime', inplace=True)
    return df


def add_drugs(df: pd.DataFrame, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    patients = df.index.unique().values
    drug_tablename = 'drug_exposure'
    processed_path = PROCESSED_DATA_DIR / f'{drug_tablename}_filtered.csv'
    try:
        drugs_df = pd.read_csv(processed_path)
        print(f'Loaded drugs from {processed_path}')
        pids_in_drugs_df = set(drugs_df.index)
        pids_in_df = set(drugs_df['person_id'])
        missing = pids_in_df.difference(pids_in_drugs_df)
        if len(missing) > 0:
            print(f'WARNING: {len(missing)} patients were not included i {processed_path}')
    except FileNotFoundError:
        drugs_df = collect_drugs(data_dir=data_dir,
                                 patients=patients,
                                 tablename=drug_tablename,
                                 max_time_before_test=pd.Timedelta('30D'))
        drugs_df = drugs_df[['person_id', 'variable_name']]
        drugs_df.to_csv(processed_path, index=False)
    # copied from condition
    one_hot_drugs = pd.get_dummies(drugs_df['variable_name'],
                                   drop_first=False,
                                   dummy_na=False,
                                   prefix='drug')
    drugs_df = drugs_df.join(one_hot_drugs).drop(columns='variable_name')
    drugs_df = drugs_df.groupby('person_id').sum()
    # We don't care how many times they had this drug
    drugs_df = (drugs_df > 0).astype('int')
    # Expand index so we can set unobserved drugs to 0
    drugs_df = drugs_df.reindex(df.index)
    drugs_df.fillna(0, inplace=True)
    # Now join it on
    df = df.join(drugs_df)
    return df


def collect_drugs(patients: np.array,
                  data_dir: Path = DATA_DIR,
                  tablename: str = 'drug_era',
                  max_time_before_test=pd.Timedelta('14D')) -> pd.DataFrame:
    table_metadata = tables[tablename]
    drugs_df = pd.read_csv(DATA_DIR / f'{tablename}_{DATA_VERSION}.csv',
                           usecols=['person_id', table_metadata.concept_id,
                                    table_metadata.timestamp,
                                    table_metadata.timestamp_end])
    print(f'[drugs] Loaded {drugs_df.shape[0]} rows from table {tablename}')
    # filter to variables
    drugs_df = drugs_df[drugs_df[table_metadata.concept_id].isin(drugs)]
    print(f'[drugs] Filtered to {drugs_df.shape[0]} rows from {len(drugs)} drug concepts')

    # Filter to patients
    drugs_df = drugs_df[drugs_df['person_id'].isin(patients)]
    print(f'[drugs] Filtered to drugs data on {len(drugs_df["person_id"].unique())} patients')

    drugs_df[table_metadata.timestamp] = pd.to_datetime(drugs_df[table_metadata.timestamp])
    drugs_df[table_metadata.timestamp_end] = pd.to_datetime(drugs_df[table_metadata.timestamp_end])
    if tablename == 'drug_era':
        drugs_df[table_metadata.timestamp] = drugs_df[table_metadata.timestamp] + pd.Timedelta('23:59:59')
        drugs_df[table_metadata.timestamp_end] = drugs_df[table_metadata.timestamp_end] + pd.Timedelta('23:59:59')
    elif tablename == 'drug_exposure':
        # put it to the end of the day, unless that would make it after the end timestamp
        drugs_df[table_metadata.timestamp] = drugs_df[table_metadata.timestamp] + pd.Timedelta('23:59:59')
        timestamp_violations = drugs_df[table_metadata.timestamp] > drugs_df[table_metadata.timestamp_end]
        drugs_df.loc[timestamp_violations, table_metadata.timestamp] = drugs_df.loc[timestamp_violations, table_metadata.timestamp_end] - pd.Timedelta('00:01:00')
    else:
        raise ValueError(tablename)

    assert np.all(drugs_df[table_metadata.timestamp_end] >= drugs_df[table_metadata.timestamp])
    print(f'[drugs] Note: {(drugs_df[table_metadata.timestamp_end] == drugs_df[table_metadata.timestamp]).sum()} rows have same start and end time')
    # remove drug periods with START time AFTER the test
    # drug during "relevant period" IF drug end time: greater than (test time - window size)
    # that is, the drug was active at any point during the relevant period
    drugs_df = filter_by_time(drugs_df, max_time_before_test=max_time_before_test,
                              timestamp_start=table_metadata.timestamp,
                              timestamp_end=table_metadata.timestamp_end)
    drugs_df['variable_name'] = drugs_df[table_metadata.concept_id].map(drugs)

    drugs_df.to_csv(PROCESSED_DATA_DIR / f'{tablename}_{max_time_before_test}_filtered.csv', index=False)
    return drugs_df


def collect_measurements_or_observations(data_dir: Path,
                                         patients: np.array,
                                         tablename: str = 'measurement') -> pd.DataFrame:
    processed_path = PROCESSED_DATA_DIR / f'{tablename}_filtered.csv'
    try:
        df = pd.read_csv(processed_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        print(f'Loaded {tablename} from {processed_path}')
        return df
    except FileNotFoundError:
        print(f'Could not read {processed_path}!')
    table_metadata = tables[tablename]
    df = pd.read_csv(DATA_DIR / f'{tablename}_{DATA_VERSION}.csv',
                     usecols=['person_id', table_metadata.concept_id,
                              table_metadata.timestamp,
                              'value_as_number'])
    # Filter to those in measurements and vitals
    concept_ids = set()
    for s in [measurements, vitals, static]:
        concept_ids = concept_ids.union(set(s))
    print(f'Filtering to {len(concept_ids)} concepts')
    df = df[df[table_metadata.concept_id].isin(concept_ids)]

    # Filter to patients
    print(f'Collecting data on {len(patients)} patients!')
    df = df[df['person_id'].isin(patients)]

    N = df.shape[0]
    print(f'Collected {N} rows from table {tablename}')

    # Remove those with missing values
    df = df[df['value_as_number'].notna()]
    print(f'Removed {N - df.shape[0]} rows for having a non-numeric value')

    # Prepare timestamps
    df['datetime'] = pd.to_datetime(df[table_metadata.timestamp])
    df.drop(columns=table_metadata.timestamp, inplace=True)

    # Label the variables by their "type"
    concept_ID_mapping = {}
    for feature_class, feature_class_name in [(vitals, 'vitals'),
                                              (measurements, 'measurements'),
                                              (static, 'static')]:
        dict_with_names = feature_class.copy()
        for k in dict_with_names:
            dict_with_names[k] = feature_class_name + '_' + dict_with_names[k]
        concept_ID_mapping.update(dict_with_names)
    # Rename concepts to English
    df['variable_name'] = df[table_metadata.concept_id].map(concept_ID_mapping)

    df.to_csv(PROCESSED_DATA_DIR / f'{tablename}_filtered.csv', index=False)

    return df


def aggregate_timeseries(df: pd.DataFrame, functions=None) -> pd.DataFrame:
    """
    Aggregate time series measurements
    Currently just takes the mean, but you could use other statistics
    """
    assert 'datetime' in df.columns
    assert 'variable_name' in df.columns
    assert 'value_as_number' in df.columns

    if functions is None:
        functions = [np.nanmean]
    df_pivoted = pd.pivot_table(df, values='value_as_number',
                                index=['person_id'],
                                columns='variable_name',
                                aggfunc=functions)
    # this next line adds the name of the aggregation function (sans "nan") to the feature name
    flattened_columns = [f'{x[1]}-{re.sub("nan", "", x[0])}' for x in df_pivoted.columns]
    df_pivoted.columns = flattened_columns
    return df_pivoted


def keep_most_recent_value(df: pd.DataFrame, include_timestamp=False,
                           test_time: pd.Series = None) -> pd.DataFrame:
    assert 'datetime' in df.columns
    assert 'variable_name' in df.columns
    assert 'person_id' in df.columns

    df.sort_values(by='datetime', inplace=True)
    most_recent = df.groupby(['person_id', 'variable_name']).tail(1)
    most_recent_values = most_recent.pivot(index='person_id',
                                           columns='variable_name',
                                           values='value_as_number')
    most_recent_values.columns = [x + '_last' for x in most_recent_values.columns]
    if include_timestamp:
        most_recent_times = most_recent.pivot(index='person_id',
                                              columns='variable_name',
                                              values='datetime')
        most_recent_times.columns = ['timestamp_' + x for x in most_recent_times.columns]
        df_pivoted = pd.concat([most_recent_values, most_recent_times], axis=1)
        if test_time is not None:
            df_pivoted = add_time_since_value(test_time, df_pivoted)
    else:
        df_pivoted = most_recent_values
    return df_pivoted


def add_time_since_value(timestamp: pd.Series, df: pd.DataFrame) -> pd.DataFrame:
    ts = timestamp[df.index]
    for col_with_last in df.columns:
        if 'timestamp_' not in col_with_last:
            assert col_with_last[-5:] == '_last'
            col = col_with_last[:-5]
            df[f'{col}_timedelta'] = (ts - df[f'timestamp_{col}'])/np.timedelta64(60, 'm')
    timestamp_cols = [x for x in df.columns if 'timestamp' in x]
    df.drop(columns=timestamp_cols, inplace=True)
    return df


def add_measurements_and_observations(df: pd.DataFrame,
                                      data_dir: Path = DATA_DIR,
                                      just_keep_last: bool = True) -> pd.DataFrame:
    """
    """
    patients = df.index.unique().values
    # This is a bit messy
    measurements_df = collect_measurements_or_observations(data_dir, patients, tablename='measurement')
    observation_df = collect_measurements_or_observations(data_dir, patients, tablename='observation')

    # Threshold values
    measurements_df = threshold_values(measurements_df, long_form=True)
    observation_df = threshold_values(observation_df, long_form=True)

    df_both = pd.concat([measurements_df, observation_df])
    # Remove all data after test
    df_both = filter_by_time(df_both, max_time_before_test=None)

    # Deal with each feature class
    df_static = df_both[df_both['variable_name'].str.contains('static_')]
    df_measurements = df_both[df_both['variable_name'].str.contains('measurements_')]
    df_vitals = df_both[df_both['variable_name'].str.contains('vitals_')]

    # We keep the last in static
    df_static = keep_most_recent_value(df_static, include_timestamp=False)

    # For measurements, we keep the last 14D *and* the last measurement
    print(f'Filtering measurements to data from last 14D and aggregating')
    df_measurements_T = filter_by_time(df_measurements.copy(),
                                       max_time_before_test=pd.Timedelta('14D'))
    df_measurements_T['variable_name'] = df_measurements_T['variable_name'] + '-' + '14D'
    df_measurements_T = aggregate_timeseries(df_measurements_T)
    df_measurements_ever = keep_most_recent_value(df_measurements,
                                                  test_time=df['measurement_datetime'],
                                                  include_timestamp=True)

    # For vitals, we keep last 7D and the last measurement
    df_vitals_ever = keep_most_recent_value(df_vitals.copy(),
                                            test_time=df['measurement_datetime'],
                                            include_timestamp=True)
    df_vitals_T_list = []
    for d in [7]:
        print(f'Filtering vitals to data from last {d}D and aggregating')
        df_vitals_T = filter_by_time(df_vitals.copy(),
                                     max_time_before_test=pd.Timedelta(f'{d}D'))
        df_vitals_T['variable_name'] = df_vitals_T['variable_name'] + '-' + f'{d}D'
        df_vitals_T = aggregate_timeseries(df_vitals_T)
        df_vitals_T_list.append(df_vitals_T)

    # Now join it all together
    df = df.join(df_static)
    df = df.join(df_measurements_T)
    df = df.join(df_measurements_ever)
    df = df.join(df_vitals_ever)
    for df_T in df_vitals_T_list:
        df = df.join(df_T)
    return df


def add_zcta_metadata(df: pd.DataFrame,
                      data_dir: Path = DATA_DIR) -> pd.DataFrame:
    assert 'zip' in df.columns
    zcta_cols = ['ZCTA5CE10', 'pdens18', 'median_household_income_dollars', 'ACS2018_total_pop']
    metadata = pd.read_csv(DATA_DIR / 'zcta_population_metrics.csv', usecols=zcta_cols)
    df_with_metadata = pd.merge(df.reset_index(), metadata,
                                left_on='zip',
                                right_on='ZCTA5CE10', how='left').drop(columns='ZCTA5CE10')
    df_with_metadata.set_index('person_id', inplace=True)
    df_with_metadata['zip'] = remap_rare_categories(df_with_metadata['zip'], freq_threshold=0.01)
    df_with_metadata['county'] = remap_rare_categories(df_with_metadata['county'], freq_threshold=0.01)
    return df_with_metadata


def add_recent_zip_activity(df: pd.DataFrame, max_time=pd.Timedelta('14D')) -> pd.DataFrame:
    zips = df['zip'].unique()
    df['measurement_datetime'] = pd.to_datetime(df['measurement_datetime'])
    # we need to impute the total population because we are scaling the rate of testing
    mean_pop = df['ACS2018_total_pop'].mean()
    df['ACS2018_total_pop'].fillna(mean_pop, inplace=True)
    # we will drop it at the end
    pids = []
    disease_activity_array = []
    test_activity_array = []
    for z in zips:
        df_z = df.loc[df['zip'] == z, ['measurement_datetime', 'test_outcome', 'ACS2018_total_pop']]
        df_z.dropna(subset=['measurement_datetime'], inplace=True)
        for idx, row in df_z.iterrows():
            time_distances = row['measurement_datetime'] - df_z['measurement_datetime']
            # row's measurement datetime must be larger than the other row's
            relevant_rows = (time_distances > pd.Timedelta('1D')) & (time_distances <= max_time)
            disease_activity = df_z.loc[relevant_rows, 'test_outcome'].mean()
            test_activity = 100*relevant_rows.sum()/row['ACS2018_total_pop']
            pids.append(idx)
            disease_activity_array.append(disease_activity)
            test_activity_array.append(test_activity)
    da = pd.DataFrame(index=pids,
                      data={'zip_recent_disease': disease_activity_array,
                            'zip_recent_tests': test_activity_array})
    df['zip_recent_disease'] = da['zip_recent_disease']
    df['zip_recent_tests'] = da['zip_recent_tests']
    # we only needed population for this
    df.drop(columns='ACS2018_total_pop', inplace=True)
    # actually let's not include the recent test activity
    df.drop(columns='zip_recent_tests', inplace=True)
    return df


def add_zip_and_county(df: pd.DataFrame,
                       data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Join with location table
    """
    assert 'location_id' in df.columns
    location_df = pd.read_csv(DATA_DIR / f'location_{DATA_VERSION}.csv')
    # There is probably a more elegant way to do this in pandas
    # Non-int and missing zips get cast to 0
    location_df['zip'] = pd.to_numeric(location_df['zip'], errors='coerce').fillna(0).astype('int')
    loc_id_to_zip = location_df[['location_id', 'zip'
                                 ]].set_index('location_id')['zip'].to_dict()
    loc_id_to_county = location_df[[
        'location_id', 'county'
    ]].set_index('location_id')['county'].to_dict()
    df['zip'] = df['location_id'].map(loc_id_to_zip)
    df['county'] = df['location_id'].map(loc_id_to_county)
    return df


def add_visit_type(df: pd.DataFrame, data_dir: Path = DATA_DIR,
                   unknown_as_telehealth: bool = True) -> pd.DataFrame:
    visit_df = pd.read_csv(DATA_DIR / f'visit_occurrence_{DATA_VERSION}.csv')
    visit_df['visit_type'] = visit_df['visit_concept_id'].map(visit_type)
    visit_df = visit_df[['visit_type', 'visit_occurrence_id']]
    visit_df.set_index('visit_occurrence_id', inplace=True)
    n = df.shape[0]
    df = df.join(visit_df, on='visit_occurrence_id')
    print(f'{df["visit_type"].isna().mean()*100}% of patients have unknown visit type')
    print(f'This is {df["visit_type"].isna().sum()} patients')
    if unknown_as_telehealth:
        assert 'telehealth/telephone' in visit_type.values()
        assert 'telehealth/telephone' in df['visit_type'].unique()
        df['visit_type'].fillna('telehealth/telephone', inplace=True)
        print(f'Filled unknown visit type with "telehealth/telephone"')
    else:
        df['visit_type'].fillna('unknown', inplace=True)
        print(f'Filled unknown visit type with "unknown"')
    assert df.shape[0] == n
    print('Added visit type...')
    return df


def add_insurance_from_binary(df: pd.DataFrame,
                              data_dir: Path = DATA_DIR) -> pd.DataFrame:
    processed_path = PROCESSED_DATA_DIR / f'insurance_binary_filtered.csv'
    try:
        insurance_by_visit = pd.read_csv(processed_path)
        print(f'Loaded insurance (binary) form {processed_path}')
        if 'person_id' in insurance_by_visit.columns:
            insurance_by_visit.set_index('visit_occurrence_id', inplace=True)
    except FileNotFoundError:
        print(f'Couldn\'t find {processed_path}, loading insurance_binary table')
        insurance_df = pd.read_csv(DATA_DIR / f'insurance_binary_{DATA_VERSION}.csv')
        column_renaming = {'hasMedicaid': 'insurance_medicaid',
                           'hasMedicare': 'insurance_medicare',
                           'hasMilitaryCare': 'insurance_military',
                           'hasCommercial': 'insurance_commercial',
                           'hasOtherForms': 'insurance_other',
                           'hasNullInfo': 'insurance_null',
                           'OnlySelfPay': 'insurance_selfpay'}
        insurance_df.rename(columns=column_renaming, inplace=True)
        # The same visit occurrence can happen with multiple insurance types
        # But these are receiving separte rows, so we need to merge them again
        # Create a mapping of vID -> pID to check we didn't mess anything up
        pre_visit_to_pid_dict = insurance_df[['visit_occurrence_id', 'person_id']].set_index('visit_occurrence_id')['person_id'].to_dict()
        insurance_by_visit = insurance_df.groupby('visit_occurrence_id').mean()
        insurance_by_visit['person_id'] = insurance_by_visit['person_id'].astype('int')
        post_visit_to_pid_dict = insurance_by_visit['person_id'].to_dict()
        assert pre_visit_to_pid_dict == post_visit_to_pid_dict
        insurance_by_visit.to_csv(processed_path)
        print(f'Saved to {processed_path}')
    n = df.shape[0]
    insurance_by_visit.drop(columns='person_id', inplace=True)
    df = df.join(insurance_by_visit, on='visit_occurrence_id')
    assert df.shape[0] == n
    return df


def add_insurance(df: pd.DataFrame,
                  data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Join with insurance status
    Patient can have different insurance status on each visit, so we need
    to x-ref the visit time
    """
    assert 'visit_occurrence_id' in df.columns      # from the test
    processed_path = PROCESSED_DATA_DIR / f'insurance_filtered.csv'
    try:
        insurance_df = pd.read_csv(processed_path)
        print(f'Loaded insurance from {processed_path}')
    except FileNotFoundError:
        insurance_df = pd.read_csv(DATA_DIR / f'insurance_{DATA_VERSION}.csv')
        print(f'Loaded {insurance_df.shape[0]} rows of insurance data')
        print(f'{df["visit_occurrence_id"].isin(insurance_df["visit_occurrence_id"]).mean()*100}% of visits are represented in the insurance table')
        insurance_df.drop_duplicates(inplace=True)
        print(f'Dropped duplicates, now have {insurance_df.shape[0]} rows')
        insurance_df.dropna(inplace=True)
        print(f'Dropped missing data, now have {insurance_df.shape[0]} rows')
        insurance_df.drop_duplicates(subset='visit_occurrence_id',
                                     inplace=True, keep='first')
        print(f'Keeping just first insurance info per visit, now have {insurance_df.shape[0]} rows')
        print(f'After this processing, {df["visit_occurrence_id"].isin(insurance_df["visit_occurrence_id"]).mean()*100}% of visits are represented in the insurance table')
        insurance_df.to_csv(processed_path, index=False)
    visit_id_to_insurance = insurance_df[['visit_occurrence_id', 'Insurance_type']].set_index('visit_occurrence_id')['Insurance_type'].to_dict()
    df['insurance'] = df['visit_occurrence_id'].map(visit_id_to_insurance)
    # WARNING: this is missing in most cases
    return df


def add_occupation(df: pd.DataFrame,
                   data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    Join with occupation table
    """
    occupation_df = pd.read_csv(DATA_DIR /
                                f'occupation_{DATA_VERSION}.csv')
    person_id_to_occupation_status = occupation_df[[
        'person_id', 'EMPY_STATUS_C_DESCR'
    ]].set_index('person_id')['EMPY_STATUS_C_DESCR'].to_dict()
    df['occupation_status'] = df.index.map(person_id_to_occupation_status)
    df['occupation_status'] = df['occupation_status'].replace(occupation_mapping)
    person_id_to_occupation_category = occupation_df[[
        'person_id', 'OCCUPATION_CATEGORIES'
    ]].set_index('person_id')['OCCUPATION_CATEGORIES'].to_dict()
    df['occupation_category'] = df.index.map(person_id_to_occupation_category)
    return df


def add_conditions(df: pd.DataFrame,
                   data_dir: Path = DATA_DIR) -> pd.DataFrame:
    processed_path = PROCESSED_DATA_DIR / 'conditions_filtered.csv'
    try:
        conditions = pd.read_csv(processed_path)
        conditions.set_index('person_id', inplace=True)
        print(f'Loaded conditions from {processed_path}')
    except FileNotFoundError:
        print(f'Couldn\'t find {processed_path}, loading conditions table...')
        conditions = pd.read_csv(DATA_DIR / f'condition_occurrence_{DATA_VERSION}.csv',
                                 usecols=['person_id', 'condition_concept_id',
                                          'condition_start_datetime'])
        conditions.rename(columns={'condition_start_datetime': 'datetime'}, inplace=True)
        # Note that there is an end time as well, but it seems to be the same as the start time in this table
        conditions['datetime'] = pd.to_datetime(conditions['datetime'])
        # Filter to the same set of patients
        cond_subset_joined = None
        for condition_type, condition_mapping in [('chronic', chronic_conditions), ('symptom', signs_and_symptoms)]:
            print(f'Processing conditions of type {condition_type}')
            cond_subset = conditions[conditions['condition_concept_id'].isin(condition_mapping.keys())]
            # Filter to conditions recorded before test result
            if 'chronic' in condition_type:
                max_time_before_test = None
            else:
                max_time_before_test = pd.Timedelta('14D')
            cond_subset = filter_by_time(cond_subset, max_time_before_test=max_time_before_test)

            # Map to one-hot
            cond_subset['condition_concept_id'] = cond_subset['condition_concept_id'].map(condition_mapping)
            one_hot_conditions = pd.get_dummies(cond_subset['condition_concept_id'],
                                                drop_first=False,
                                                dummy_na=False,
                                                prefix=condition_type)
            cond_subset = cond_subset.join(one_hot_conditions)
            cond_subset.drop(columns=['condition_concept_id', 'datetime'], inplace=True)
            cond_subset = cond_subset.groupby('person_id').sum()
            # We don't count how many times they had this diagnosis
            cond_subset = (cond_subset > 0)*1
            # Expand index so we can set unobserved conditions to 0
            # (assuming lack of diagnosis = no condition)
            # Not necessarily true but our model can't handle uncertainty in input data
            cond_subset = cond_subset.reindex(df.index)
            cond_subset.fillna(0, inplace=True)
            # Now join it on
            if cond_subset_joined is None:
                cond_subset_joined = cond_subset
            else:
                cond_subset_joined = cond_subset_joined.join(cond_subset)
        conditions = cond_subset_joined
        cond_subset_joined.to_csv(processed_path)
    df = df.join(conditions)
    return df


def add_chronic_medical_conditions(df: pd.DataFrame,
                                   data_dir: Path = DATA_DIR) -> pd.DataFrame:
    conditions = pd.read_csv(DATA_DIR / f'condition_occurrence_{DATA_VERSION}.csv',
                             usecols=['person_id', 'condition_concept_id',
                                      'condition_start_datetime'])
    # Filter to the same set of patients
    conditions = conditions[conditions['person_id'].isin(df.index)]
    # Chronic conditions and their identifiers - see omop_metadata file
    # Filter the table again to just those conditions
    conditions = conditions[conditions['condition_concept_id'].isin(chronic_conditions.keys())]
    # Filter to conditions recorded before test result
    # Note that there is an end time as well, but it seems to be the same as the start time in this table
    conditions['condition_start_datetime'] = pd.to_datetime(conditions['condition_start_datetime'])
    conditions = conditions.join(df['measurement_datetime'], on='person_id')
    n = conditions.shape[0]
    conditions = conditions[conditions['condition_start_datetime'] <= conditions['measurement_datetime']]
    print(f'Removed {n - conditions.shape[0]} rows from condition table for occurring after the test')
    # Map to one-hot
    conditions['condition_concept_id'] = conditions['condition_concept_id'].map(chronic_conditions)
    chronic_medical_conditions = pd.get_dummies(conditions['condition_concept_id'],
                                                drop_first=False,
                                                dummy_na=False)
    conditions = conditions.join(chronic_medical_conditions)
    conditions.drop(columns=['condition_concept_id', 'condition_start_datetime', 'measurement_datetime'], inplace=True)
    conditions = conditions.groupby('person_id').sum()
    # We don't count how many times they had this diagnosis
    conditions = (conditions > 0)*1
    # Expand index so we can set unobserved conditions to 0 (assuming lack of diagnosis = no condition)
    conditions = conditions.reindex(df.index)
    conditions.fillna(0, inplace=True)
    # Now join it on
    df = df.join(conditions)
    return df


def add_label(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    labels = get_labels(data_dir)
    labels = labels[['person_id', 'measurement_datetime',
                     'test_outcome', 'test_month',
                     'reason_for_test',
                     'visit_occurrence_id']]
    labels['measurement_datetime'] = pd.to_datetime(labels['measurement_datetime'])
    labels.set_index('person_id', inplace=True)
    df_len = df.shape[0]
    df = df.join(labels)
    assert df.shape[0] == df_len
    return df


def remap_rare_categories(df, freq_threshold=0.05):
    category_counts = df.value_counts()
    category_fracs = category_counts/(df.dropna().shape[0])
    infrequent_categories = category_fracs[category_fracs < freq_threshold].index
    print(f'Replacing {len(infrequent_categories)} categories with "other"')
    remap = dict(zip(category_counts.index.values, category_counts.index.values))
    for cat in infrequent_categories:
        remap[cat] = 'Other'
    df_mapped = df.map(remap)
    return df_mapped


def integer_encode(df: pd.DataFrame) -> pd.DataFrame:
    """ LightGBM takes integer-encoded categorical variables """
    mappings = dict()
    for col in categorical_columns:
        if col in df.columns:
            print(f'Integer encoding column {col}')
            # map rare categories
            if col not in ['zip', 'county']:
                # already remapped these
                freq_threshold = 0.01
                df[col] = remap_rare_categories(df[col], freq_threshold=freq_threshold).astype('category')
            col_mapping = dict(enumerate(df[col].cat.categories))
            mappings[col] = col_mapping
            df[col] = df[col].cat.codes
    with open(PROCESSED_DATA_DIR / 'category_mappings.json', 'w') as fp:
        json.dump(mappings, fp)
    return df


def load_category_mappings():
    mappings = json.load(open(PROCESSED_DATA_DIR / 'category_mappings.json'))
    return mappings


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    for col in categorical_columns:
        if col in df.columns:
            print(f'One-hot encoding column {col}')
            # map rare categories
            if col == 'zip':
                freq_threshold = 0.01
            else:
                freq_threshold = 0.05
            df[col] = remap_rare_categories(df[col], freq_threshold=freq_threshold)
            one_hot = pd.get_dummies(df[col],
                                     drop_first=True,
                                     dummy_na=True,
                                     prefix=col)
            df = df.join(one_hot)
        else:
            print(f'Tried to map column {col} but it\'s not in this dataframe')
    df.drop(columns=categorical_columns, inplace=True, errors='ignore')
    return df


def threshold_values(df, verbose=False, long_form: bool = True) -> pd.DataFrame:
    """ Apply variable-specific thresholds """
    if long_form:
        variables_present = df['variable_name'].unique()
        for variable_name, (lower, upper) in thresholds.items():
            relevant_variables = [x for x in variables_present if variable_name in x]
            # We don\'t threshold timedeltas
            relevant_variables = [x for x in relevant_variables if not 'timedelta' in x]
            if len(relevant_variables) == 0:
                continue
            relevant_rows = df['variable_name'].isin(relevant_variables)
            var_values = df.loc[relevant_rows, 'value_as_number']
            in_range = var_values.between(lower, upper)
            var_values[~in_range] = np.nan
            df.loc[relevant_rows, 'value_as_number'] = var_values
    else:
        for variable_name, (lower, upper) in thresholds.items():
            column_names = [x for x in df.columns if variable_name in x]
            for column_name in column_names:
                if 'timedelta' in column_name:
                    # We don\'t threshold timedeltas
                    continue
                try:
                    column_values = df[column_name]
                    in_range = column_values.between(lower, upper)
                    column_values[~in_range] = np.nan
                    if verbose:
                        n_to_replace = (~in_range).sum()
                        if n_to_replace > 100:
                            print(f'Replacing {n_to_replace} values in column {column_name} outside the range [{lower}, {upper}] with NAN')
                    df[column_name] = column_values
                except KeyError:
                    print(f'WARNING: Column {column_name} is not in the data?')
                    continue
                except TypeError:
                    print(f'Issue threshold column {column_name}!')
                    break
    return df


def collect_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """
    - collect the columns from person
    - join with location to get zip and county
    - join with occupation to get occupation status
    - add conditions info (chronic and signs/symptoms)
    - add labels
    """
    try:
        df = pd.read_csv(PROCESSED_DATA_DIR / f'data.csv')
        df.set_index('person_id', inplace=True)
    except FileNotFoundError:
        print('Didn\'t find data file - generating anew')
        columns = [
            'person_id', 'age_years', 'location_id', 'gender_source_value',
            'race_source_value', 'ethnicity_source_value'
        ]
        df = pd.read_csv(DATA_DIR / f'person_{DATA_VERSION}.csv', usecols=columns)
        print(f'Collected data on {len(df["person_id"].unique())} patients!')
        # Rename some columns for simplicity
        df.rename(columns={
            'race_source_value': 'race',
            'age_years': 'static_age'
            },
            inplace=True)
        # Tidy up gender
        df['female'] = df['gender_source_value'].map(gender_mapping).astype('int')
        df.drop(columns='gender_source_value', inplace=True)
        # Tidy up ethnicity
        df['hispanic'] = df['ethnicity_source_value'].map(ethnicity_mapping)
        df.drop(columns='ethnicity_source_value', inplace=True)
        # Tidy up race
        df['race'] = df['race'].replace(race_mapping)
        df.set_index('person_id', inplace=True)
        # Join with other files and such
        df = add_label(df, data_dir)
        df = add_visit_type(df, data_dir)
        df = add_measurements_and_observations(df, data_dir)
        df = add_zip_and_county(df, data_dir)
        df = add_zcta_metadata(df, data_dir)
        df = add_recent_zip_activity(df)
        df = add_occupation(df, data_dir)
        df = add_insurance_from_binary(df, data_dir)
        df = add_conditions(df, data_dir)
        df = add_drugs(df, data_dir)
        df = threshold_values(df, long_form=False)
        # make categorical
        df.to_csv(PROCESSED_DATA_DIR / f'data.csv')
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype('category')
    return df


def impute_visit_ID(df: pd.DataFrame) -> pd.DataFrame:
    print('Imputing visit ID ... Loading measurement details table (this is quite large)')
    # We are using a specific version of this table because its version is different to the rest
    meas_df = pd.read_csv(DATA_DIR / 'measurement_details_20201002.csv')
    # filter to rows pertaining to the labels
    meas_df = meas_df[meas_df['measurement_id'].isin(df['measurement_id'])]
    meas_df['measurement_datetime'] = pd.to_datetime(meas_df['measurement_datetime'])
    # filter to imputed visit occurrence IDs no more than 2 days before the measurement
    meas_df = meas_df[meas_df['nearest_datetime'] <= 2]
    # filter to columns of relevant
    meas_df = meas_df[['person_id', 'measurement_id', 'visit_occurrence_id_imputed', 'visit_start_datetime', 'visit_end_datetime', 'measurement_datetime']]
    # left join onto the labels
    df_imp = pd.merge(df, meas_df, how='left', on=['measurement_id', 'person_id'], suffixes=('', '_R'))
    # where the visit_occurrence_id is missing, use the imputed value
    visit_missing = df_imp['visit_occurrence_id'].isna()
    df_imp.loc[visit_missing, 'visit_occurrence_id'] = df_imp.loc[visit_missing, 'visit_occurrence_id_imputed']
    # record what we did and drop other columns
    df_imp['visit_occurrence_id_was_imputed'] = visit_missing
    df_imp.drop(columns=['visit_occurrence_id_imputed', 'visit_start_datetime', 'visit_end_datetime', 'measurement_datetime_R'])
    return df_imp


def get_labels(data_dir: Path = DATA_DIR,
               keep_just_first: bool = True) -> pd.DataFrame:
    labels_path = PROCESSED_DATA_DIR / 'labels.csv'
    try:
        df = pd.read_csv(labels_path)
        print(f'Loaded labels from {labels_path}')
        df['measurement_datetime'] = pd.to_datetime(df['measurement_datetime'])
    except FileNotFoundError:
        print(f'Couldn\'t find labels at {labels_path} - computing fresh')
        cIDs = get_COVID_IDs(data_dir=data_dir)
        # i THINK these are all in measurements
        table = 'measurement'
        df = pd.read_csv(DATA_DIR / f'{table}_{DATA_VERSION}.csv')
        print(f'Loaded table {table}')
        table_metadata = tables[table]
        # Filter to those concept IDs
        df = df[df[table_metadata.concept_id].isin(cIDs)]
        present_cIDs = df[table_metadata.concept_id].unique()
        print(f'There are {df.shape[0]} measurements from {len(present_cIDs)} concepts')
        print(f'The concept IDs present are: {present_cIDs}')
        print(f'There are now {df.shape[0]} measurements from {len(df["person_id"].unique())} patients')
        # Tidy up the dataframe
        df = df[[
            'person_id',
            'measurement_id',
            table_metadata.concept_id, table_metadata.timestamp,
            'value_source_value',
            'visit_occurrence_id'
        ]]
        df[table_metadata.timestamp] = pd.to_datetime(
            df[table_metadata.timestamp])
        # Get the FIRST for each patient
        # Fancier version of this would check to see if there's another test shortly afterwards
        # ... which would indicate that the first test wasn't trusted
        # But in our data these repeat tests are often the same as the first one
        df.sort_values(by=['person_id', table_metadata.timestamp],
                       inplace=True)
        if keep_just_first:
            df.drop_duplicates(subset='person_id', keep='first', inplace=True)
        num_measurements = df.shape[0]
        num_patients = len(df['person_id'].unique())
        if keep_just_first:
            assert num_measurements == num_patients
            print(f'After dropping duplicates, there are now {num_measurements} measurements from {num_patients} patients')
        # Convert into binary yes/no/unknown
        df['test_outcome'] = df['value_source_value'].map(covid_result_mapping)
        # assign month
        df['test_month'] = df[table_metadata.timestamp].dt.month
        # impute the visit ID
        df = impute_visit_ID(df)
        # assign reason for test
        df = add_reason_for_test(df)
        print(f'Saving labels to {labels_path}')
        df.to_csv(labels_path, index=False)
    return df


def add_reason_for_test(df: pd.DataFrame,
                        data_dir: Path = DATA_DIR) -> pd.DataFrame:
    test_detail = pd.read_csv(data_dir / f'test_details_{DATA_VERSION}.csv')
    test_detail = test_detail[['person_id', 'measurement_id', 'Indications']]
    test_detail.drop_duplicates(inplace=True, subset=['person_id', 'measurement_id'], keep='first')
    n = df.shape[0]
    df_with_indication = pd.merge(df, test_detail, on=['person_id', 'measurement_id'], how='left')
    assert df_with_indication.shape[0] == n
    df_with_indication['reason_for_test'] = df_with_indication['Indications']
    df_with_indication.drop(columns='Indications', inplace=True)
    df_with_indication['reason_for_test'].fillna('unknown', inplace=True)
    df_with_indication['symptomatic'] = df_with_indication['reason_for_test'].str.lower().str.contains('asymptomatic').apply(lambda x: not x if type(x) == bool else x)
    return df_with_indication


def require_variables(df: pd.DataFrame, variables=['vitals_temperature']):
    print(f'Filtering to patients with any of {variables} and derived quantities')
    relevant_columns = [x for x in df.columns for y in variables if y in x]
    print(f'Checking for any values in {relevant_columns}')
    has_any = np.any(df[relevant_columns].notna(), axis=1)
    print(f'Keeping {has_any.sum()} patients - filtering out {sum(has_any == False)} patients')
    df = df[has_any]
    return df


def missingness_filtering(df: pd.DataFrame, column_threshold: float,
                          row_threshold: float) -> None:
    """ Remove rows and columns with too much missing data """
    relevant_columns = ['vitals_temperature_last', 'race', 'female', 'hispanic',
                        'occupation_status', 'static_age', 'zip', 'county', 'visit_type']
    print(f'Filtering patients based on missingness in variables: {relevant_columns} (removed symptoms)')
    missingness = df[relevant_columns].isna().mean(axis=1)
    bad_rows = missingness[missingness > row_threshold].index
    print(f'Dropping {len(bad_rows)} rows (patients) for having missingness over {row_threshold*100}%')
    df.drop(bad_rows, inplace=True)
    # Now remove columns with generally high missingness
    missingness = df.isna().mean(axis=0)
    bad_columns = missingness[missingness > column_threshold].index
    print(f'Dropping {len(bad_columns)} columns for having missingness over {column_threshold*100}%')
    print(f'These columns are: {bad_columns}')
    df.drop(columns=bad_columns, inplace=True)

    return


def imputation(df: pd.DataFrame) -> None:
    assert 'split' in df.columns
    train = df[df['split'] == 'train']
    # categorical values - not sure why normal procedure doesn't work
    categorical_columns = train.columns[train.dtypes == 'category']
    for col in categorical_columns:
        mode = train[col].mode().values[0]
        df[col].fillna(mode, inplace=True)
    #  integer values
    integer_columns = train.columns[train.dtypes == 'int'].tolist() + ['hispanic', 'female']
    integer_columns = list(set(integer_columns))
    mode_vals = train.loc[:, integer_columns].mode(axis=0)
    df.fillna(mode_vals, inplace=True)
    # continuous values
    mean_vals = train.loc[:, train.dtypes == 'float'].mean(axis=0)
    df.fillna(mean_vals, inplace=True)
    return


def scaling(df: pd.DataFrame) -> pd.DataFrame:
    """ Do robust scaling on continuous columns """
    df_train = df[df['split'] == 'train']
    continuous_columns = [x for x in df.columns if 'measurements_' in x or 'vitals_' in x] + ['static_age', 'pdens18', 'static_height_last', 'static_weight_last', 'median_household_income_dollars']

    # robust scaling
    df_continuous = df_train[continuous_columns]
    df_continuous_mean = df_continuous.mean(axis=0)
    df_continuous_std = np.nanstd(df_continuous, axis=0)

    # apply this to all (numpy handles broadcasting)
    df_continuous_all = df[continuous_columns]
    df_continuous_all_scaled = (df_continuous_all - df_continuous_mean)/df_continuous_std

    for col in continuous_columns:
        df[col] = df_continuous_all_scaled[col]

    return df


def remove_multicollinearity(df: pd.DataFrame, threshold=0.8, ignore_classes=None) -> None:
    """
    Identify columns which are multicollinear
    And remove them
    """
    print(f'Removing multicollinear features!')
    cols_to_drop = ['split'] + df.columns[df.dtypes == 'category'].tolist()
    cols_to_drop = [x for x in cols_to_drop if x in df.columns]
    df_num = df.drop(columns=cols_to_drop)
    features = df_num.columns
    df_vals = df_num.values
    feature_corr = np.corrcoef(df_vals.T)
    # Zero out the diagonal
    feature_corr_offdiag = feature_corr - np.diag(np.diag(feature_corr))
    # Zero out below the diagonal since it's symmetric
    for i in range(len(features)):
        for j in range(0, i):
            feature_corr_offdiag[i, j] = 0
    above_threshold = feature_corr_offdiag > threshold
    print(f'Found {np.nansum(above_threshold)} entries with correlation above {threshold}')
    rows, columns = np.where(above_threshold)
    indices = list(zip(rows, columns))
    if len(indices) > 0:
        to_remove = []
        print('Identified the following highly-correlated features:')
        for i, j in indices:
            print(features[i], features[j], feature_corr[i, j])
            to_remove.append(features[i])
        if ignore_classes is not None:
            print(f'NOT removing features from class {ignore_classes}')
            to_remove_filtered = [x for x in to_remove if ignore_classes not in x]
            to_remove = to_remove_filtered
        print(f'Dropping columns {set(to_remove)}')
        df.drop(columns=to_remove, inplace=True)
        if ignore_classes is None:
            print(f'Testing multicollinearity again...')
            remove_multicollinearity(df, threshold=threshold)
    return


def build_dataset(data_dir: Path = DATA_DIR,
                  version: str = 'dev2',
                  require_variables_present: bool = False,
                  load: bool = False,
                  categorical_features: str = 'no encode',
                  row_missingness_threshold: float = 0.5,
                  column_missingness_threshold: float = 0.9,
                  do_scaling: bool = True,
                  do_imputation: bool = False,
                  filter_missingness: bool = True) -> pd.DataFrame:
    """
    - Pull time series data
    -- process time series into features
    - Pull static data
    -- add labels
    - Split into train/test
    """
    dataset_name = f'data_{version}_filtered'
    if do_scaling:
        dataset_name = f'{dataset_name}_scaled'
    if do_imputation:
        dataset_name = f'{dataset_name}_imputed'
    dataset_path = PROCESSED_DATA_DIR / f'{dataset_name}.csv'
    if load:
        print(f'Loading data from {dataset_path}!')
        df = pd.read_csv(dataset_path)
        df.set_index('person_id', inplace=True)
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype('category')
    else:
        # for now, only static - simple
        df = collect_data(data_dir)
        # Remove rows missing label data
        n = df.shape[0]
        df = df[df['test_outcome'].notna()]
        print(f'Removed {n - df.shape[0]} rows with missing labels')
        # Train test split
        df = train_test_split(df)
        # Remove nonsensical columns
        df.drop(columns=['location_id', 'measurement_datetime', 'visit_occurrence_id'], inplace=True)
        # Save
        df.to_csv(PROCESSED_DATA_DIR / f'data_{version}.csv')
        # Filter out patients missing key variables
        if require_variables_present:
            df = require_variables(df)
        # Filter by missinginess
        if filter_missingness:
            missingness_filtering(df, row_threshold=row_missingness_threshold,
                                  column_threshold=column_missingness_threshold)
            df.to_csv(PROCESSED_DATA_DIR / f'data_{version}_filtered.csv')
        # Now scale
        if do_scaling:
            df = scaling(df)
            df.to_csv(PROCESSED_DATA_DIR / f'data_{version}_filtered_scaled.csv')
        # Imputation
        if do_imputation:
            imputation(df)
            # Remove multicollinearity - only works if we have imputed
            remove_multicollinearity(df, ignore_classes='vitals')
        # And NOW save
        df.to_csv(dataset_path)
        #  Visualise if we want
        # vis_feature_values(df, identifier='scaled')
    if categorical_features == 'integer':
        df = integer_encode(df)
    elif categorical_features == 'one hot':
        df = one_hot_encode(df)
    elif categorical_features == 'no encode':
        pass
    else:
        raise ValueError(categorical_features)
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype('category')
    return df


def get_COVID_IDs(data_dir: Path = DATA_DIR, check: bool = False) -> list:
    # from Jimmy's code
    concept_IDs = [
        706154, 706155, 706156, 706157, 706158, 706159, 706160, 706161, 706162,
        706163, 706165, 706166, 706167, 706168, 706169, 706170, 706171, 706172,
        706173, 706174, 706175
    ]
    if check:
        concept_df = pd.read_csv(DATA_DIR / f'concept_{DATA_VERSION}.csv')
        # run through the IDs
        print('Collecting the following IDs for COVID test result...')
        for cID in concept_IDs:
            print(concept_df[concept_df['concept_id'] == cID])
    return concept_IDs


def train_test_split(df, test_frac=0.2):
    """
    Create a split column (train/test)
    """
    N = df.shape[0]
    df['split'] = 'train'
    # Always have super held-out
    try:
        test_patients = pd.read_csv(FIXED_TEST_PATH)['person_id']
    except FileNotFoundError:
        print(f'WARNING: Could not find {FIXED_TEST_PATH}, creating anew!')
        n_test = int(test_frac*N)
        test_patients = np.random.choice(df.index, n_test, replace=False)
        test_patients = pd.DataFrame({'person_id': test_patients})
        test_patients.to_csv(FIXED_TEST_PATH, index=False)
        test_patients = test_patients['person_id']
    present_test_patients = [x for x in test_patients if x in df.index]
    df.loc[present_test_patients, 'split'] = 'heldout_test'

    # Now we split up the remaining data
    remaining_patients = df.loc[df['split'] != 'heldout_test', :].index
    n_validation = int(test_frac*len(remaining_patients))
    validation_patients = np.random.choice(remaining_patients, n_validation, replace=False)
    df.loc[validation_patients, 'split'] = 'validation'

    print('Split counts:')
    counts = df['split'].value_counts()
    print(counts)
    print('Split fractions:')
    print(counts/df.shape[0])

    return df


def vis_feature_values(data, features=None, identifier: str = '', include_test=False) -> dict:
    """
    Just visualise the distribution of features
    """
    outpath = Path('./')
    if type(data) == pd.DataFrame:
        if 'split' in data.columns:
            data_train = data[data['split'] == 'train'].drop(columns='split')
            data_test = data[data['split'] == 'validation'].drop(columns='split')
        else:
            data_train = data
            data_test = None

        features = data_train.columns
    elif type(data) == np.ndarray:
        assert features is not None
        assert len(features) == data.shape[1]
        data_train = data
        data_test = None
    else:
        raise ValueError(data)

    if include_test:
        assert data_test is not None

    n_features = len(features)
    if n_features > 100:
        print(f'Warning: There are {n_features} features, lots of plots will be generated!')

    binary_features = {}

    for i, feature in enumerate(features):
        if set(data_train[feature]) == {0, 1}:
            # this is a binary feature
            print(f'We think {feature} is binary, not plotting')
            perc_pos = np.nanmean(data_train[feature])
            binary_features[feature] = perc_pos
            continue
        try:
            _ = pd.to_numeric(data_train[feature])
        except ValueError:
            print(f'Feature {feature} contains non-numeric data - skipping')
            continue
        fig, axarr = plt.subplots(nrows=1, ncols=3)
        axarr = np.array([axarr])
        sns.distplot(data_train[feature], ax=axarr[0, 0], label='train')
        sns.distplot(data_train.loc[data_train['test_outcome'] == 0, feature],
                     ax=axarr[0, 1], label='train')
        sns.distplot(data_train.loc[data_train['test_outcome'] == 1, feature],
                     ax=axarr[0, 2], label='train')
        if data_test is not None:
            sns.distplot(data_test[feature], ax=axarr[0, 0], label='test')
            sns.distplot(data_test.loc[data_test['test_outcome'] == 0, feature],
                         ax=axarr[0, 1], label='test', color='grey')
            sns.distplot(data_test.loc[data_test['test_outcome'] == 1, feature],
                         ax=axarr[0, 2], label='test', color='grey')
        axarr[-1, 0].set_xlabel('all')
        axarr[-1, 1].set_xlabel('neg')
        axarr[-1, 2].set_xlabel('pos')
        axarr[-1, 0].legend()
        axarr[0, 1].set_title(f'{feature}')
        plt.savefig(outpath / 'feature_plots' / f'distplot_{identifier}_{re.sub("/", "-", feature)}.png')
        plt.tight_layout()
        plt.clf()
        plt.close()
    return binary_features
