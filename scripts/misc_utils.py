import pandas as pd
from pathlib import Path
from omop_metadata import tables

DATA_DIR = Path('../data_dir/')
PROCESSED_DATA_DIR = Path('../processed_data/')
DATA_VERSION = '20200817'


def filter_concepts_to_those_observed(data_dir: Path = DATA_DIR,
                                      table_name: str = 'condition_occurrence') -> None:
    """
    just report all the concept ids that actually appear in the conditions table
    - make a subset of the concept table
    """
    table = tables[table_name]
    print(f'Loading {table_name} table')
    df = pd.read_csv(DATA_DIR / f'{table_name}_{DATA_VERSION}.csv',
                     usecols=['person_id', table.concept_id])
    grouped_df = df.groupby(table.concept_id)
    print('Computing number of observations and number of patients per concept')
    num_observations = grouped_df.apply(lambda x: x.shape[0])
    num_patients = grouped_df.apply(lambda x: len(x['person_id'].unique()))
    print('Combining these into a data frame')
    df = num_patients.to_frame(name='num_patients').join(num_observations.to_frame('num_observations'))

    # Now join on the concept info
    print('Loading concepts table...')
    concepts = pd.read_csv(DATA_DIR / f'concept_{DATA_VERSION}.csv',
                           usecols=['concept_id', 'concept_name'])
    concepts.set_index('concept_id', inplace=True)

    # Now join
    print('Joining concept names')
    df_with_names = df.join(concepts)
    print('Saving...')
    df_with_names.to_csv(PROCESSED_DATA_DIR / f'concepts_in_{table_name}_with_counts.csv')
    print('Done!')

    return


def filter_concepts_by_observed_measurements(data_dir: Path = DATA_DIR) -> None:
    """
    just report all the concept ids that actually appear in the table
    - make a subset of the concept table
    """
    print('Loading conditions table')
    conditions = pd.read_csv(DATA_DIR / f'condition_occurrence_{DATA_VERSION}.csv',
                             usecols=['person_id', 'condition_concept_id'])
    grouped_conditions = conditions.groupby('condition_concept_id')
    print('Computing number of observations and number of patients per condition concept')
    num_observations = grouped_conditions.apply(lambda x: x.shape[0])
    num_patients = grouped_conditions.apply(lambda x: len(x['person_id'].unique()))
    print('Combining these into a data frame')
    conditions = num_patients.to_frame(name='num_patients').join(num_observations.to_frame('num_observations'))

    # Now join on the concept info
    print('Loading concepts table...')
    concepts = pd.read_csv(DATA_DIR / f'concept_{DATA_VERSION}.csv',
                           usecols=['concept_id', 'concept_name'])
    concepts.set_index('concept_id', inplace=True)

    # Now join
    print('Joining concept names')
    conditions_with_names = conditions.join(concepts)
    print('Saving...')
    conditions_with_names.to_csv(PROCESSED_DATA_DIR / 'conditions_with_concept_names.csv')
    print('Done!')

    return
