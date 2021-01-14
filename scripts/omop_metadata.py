import numpy as np
import pandas as pd


class TableMetadata(object):
    def __init__(self, table_name, concept_id, timestamp, timestamp_end,
                 variable_name, numeric_value, string_value):
        self.table_name = table_name
        self.concept_id = concept_id
        self.timestamp = timestamp
        self.timestamp_end = timestamp_end
        self.variable_name = variable_name
        self.numeric_value = numeric_value
        self.string_value = string_value


measurement = TableMetadata(table_name='measurement',
                            concept_id='measurement_concept_id',
                            timestamp='measurement_datetime',
                            timestamp_end=None,
                            variable_name='measurement_source_value',
                            numeric_value='value_as_number',
                            string_value=None)

observation = TableMetadata(table_name='observation',
                            concept_id='observation_concept_id',
                            timestamp='observation_datetime',
                            timestamp_end=None,
                            variable_name='observation_source_value',
                            numeric_value='value_as_number',
                            string_value='value_as_string')

condition_occurrence = TableMetadata(table_name='condition_occurrence',
                                     concept_id='condition_concept_id',
                                     timestamp='condition_start_datetime',
                                     timestamp_end='condition_end_datetime',
                                     variable_name='condition_source_value',
                                     numeric_value=None,
                                     string_value=None)

device_exposure = TableMetadata(table_name='device_exposure',
                                concept_id='device_concept_id',
                                timestamp='device_exposure_start_datetime',
                                timestamp_end='device_exposure_end_datetime',
                                variable_name='device_source_value',
                                numeric_value='quantity',
                                string_value=None)

drug_era = TableMetadata(table_name='drug_era',
                         concept_id='drug_concept_id',
                         timestamp='drug_era_start_date',
                         timestamp_end='drug_era_end_date',
                         variable_name=None,
                         numeric_value='drug_exposure_count',
                         # Note that for drugs there is also gap days
                         string_value=None)

drug_exposure = TableMetadata(table_name='drug_exposure',
                              concept_id='drug_concept_id',
                              timestamp='drug_exposure_start_datetime',
                              timestamp_end='drug_exposure_end_datetime',
                              variable_name=None,         # no human-readable variable name available it seems
                              numeric_value=None,         # quantity is empty
                              string_value='sig')          # there is a lot going on in here

procedure_occurrence = TableMetadata(table_name='procedure_occurrence',
                                     concept_id='procedure_concept_id',
                                     timestamp='procedure_datetime',
                                     timestamp_end=None,
                                     variable_name=None,
                                     numeric_value=None,
                                     string_value=None)

visit_occurrence = TableMetadata(table_name='visit_occurrence',
                                 concept_id='visit_concept_id',
                                 timestamp='visit_start_datetime',
                                 timestamp_end='visit_end_datetime',
                                 variable_name='visit_source_value',
                                 # Note there is more information in this table,
                                 # e.g. where patient was admitted to/discharged to
                                 numeric_value=None,
                                 string_value=None)


tables = {visit_occurrence.table_name: visit_occurrence,
          procedure_occurrence.table_name: procedure_occurrence,
          drug_exposure.table_name: drug_exposure,
          drug_era.table_name: drug_era,
          device_exposure.table_name: device_exposure,
          measurement.table_name: measurement,
          observation.table_name: observation,
          condition_occurrence.table_name: condition_occurrence}

# --- to do with individual fields --- #
covid_result_mapping = {'Detected (qualifier value)': 1,
                        'Not detected (qualifier value)': 0,
                        'Inconclusive.': np.nan,
                        'Positive (qualifier value)': 1}


ethnicity_mapping = {'Hispanic or Latino': 1,
                     'Not Hispanic or Latino': 0,
                     'Unavailable or Unknown': np.nan,
                     'Declined to Answer': np.nan}

race_mapping = {'Unavailable or Unknown': np.nan,
                'Declined to Answer': np.nan,
                'Patient not Present': np.nan}

gender_mapping = {'F': 1,
                  'M': 0,
                  'U': 0,
                  'N': 0,
                  'X': 0,
                  np.nan: 0}

occupation_mapping = {'Unknown': np.nan,
                      'UNKNOWN': np.nan}

# map from concept IDs to concepts of interset
visit_type = {9201: 'inpatient/ICU/ED',
              32037: 'inpatient/ICU/ED',
              9203: 'inpatient/ICU/ED',
              9202: 'outpatient/rehab',
              581479: 'outpatient/rehab',
              5083: 'telehealth/telephone',
              44790889: 'telehealth/telephone'}


chronic_conditions = {
                      433753: 'alcohol abuse',
                      435243: 'alcohol abuse',
                      442077: 'anxiety',
                      434613: 'anxiety',
                      313217: 'atrial fibrillation',
                      4154290: 'atrial fibrillation',
                      45877009: 'asthma',
                      317009: 'asthma',
                      45768910: 'asthma',           # "uncomplicated"
                      1415665: 'asthma',
                      45877009: 'asthma',
                      764123: 'atherosclerosis',
                      317576: 'atherosclerosis',
                      436659: 'anemia',
                      439777: 'anemia',
                      44784217: 'cardiac arrhythmia',
                      319835: 'congestive heart failure',
                      4229440: 'congestive heart failure',
                      45877605: 'copd',               # empty
                      255573: 'copd',
                      45611133: 'copd',
                      45906202: 'copd',
                      45436802: 'copd',
                      4282096: 'depression',
                      4077577: 'depression',
                      4282316: 'depression',
                      435220: 'depression',
                      4049623: 'depression',
                      4141454: 'depression',
                      201820: 'diabetes',
                      201826: 'diabetes',
                      4193704: 'diabetes',
                      37018196: 'prediabetes',
                      45952005: 'hypertension',       # empty
                      320128: 'hypertension',
                      320128: 'hypertension',
                      4322024: 'hypertension',
                      319826: 'hypertension',
                      196680: 'hypertension',
                      45952005: 'hypertension',
                      45526011: 'hypertension',
                      45614297: 'hypertension',
                      42538697: 'hypertension',
                      1415649: 'hypertension',
                      432867: 'hyperlipidemia',
                      437833: 'hypokalemia',
                      317002: 'hypotension',
                      319041: 'hypotension',             # orthostatic
                      140673: 'hypothyroidism',
                      40483287: 'kidney disease',
                      46271022: 'kidney disease',
                      443597: 'kidney disease',
                      44782429: 'kidney disease',   # due to hypertension
                      443612: 'kidney disease',
                      443601: 'kidney disease',
                      443611: 'kidney disease',
                      433736: 'obesity',
                      434005: 'obesity',
                      4100857: 'obesity',
                      4144111: 'gastroesophageal reflux',
                      197320: 'renal disease',
                      194686: 'renal disease',
                      193782: 'renal disease',
                      40437042: 'smoking',
                      436070: 'vitamin D deficiency'
                     }


signs_and_symptoms = {
                      200219: 'abdominal pain',
                      77670: 'chest pain',
                      254761: 'cough',
                      75860: 'constipation',
                      435796: 'dehydration',
                      196523: 'diarrhea',
                      433316: 'dizziness',
                      312437: 'dyspnea',
                      4223659: 'fatigue',
                      437663: 'fever',
                      378253: 'headache',
                      437390: 'hypoxemia',
                      436962: 'insomnia',
                      257011: 'respiratory infection',
                      318736: 'migraine',
                      442752: 'muscle pain',
                      24134: 'neck pain',
                      31967: 'nausea and vomiting',
                      27674: 'nausea and vomiting',
                      255848: 'pneumonia',
                      374034: 'visual disturbances'
        }

measurements = {
                3016723: 'creatinine',
                3004501: 'glucose',
                3023103: 'potassium',
                3019550: 'sodium',
                3014576: 'chloride',
                3013682: 'urea nitrogen',
                3045716: 'anion gap',
                3049187: 'eGFR',
                3023314: 'hematocrit',
                3000963: 'hemoglobin',
                3024929: 'platelets',
                3000905: 'leukocytes',
                3019897: 'erythrocyte',
                3004327: 'lymphocytes',
                3013650: 'neutrophils',
                3018405: 'lactate',
                3006923: 'ALT',
                3013721: 'AST',
                3024561: 'albumin',
                3024128: 'bilirubin (total)',
                3022192: 'triglyceride',
                3027114: 'cholesterol',
                3033891: 'prothrombin time',
                3010156: 'CRP',
                3018677: 'aPTT',
                3015242: 'ferritin',
                3021337: 'troponin-I',
                3003694: 'blood type',
                3003396: 'base excess',
                3016502: 'O2 sat',
                3027801: 'pO2',
                3019977: 'pH (arterial)',
                3012544: 'pH (venous)',
                3008152: 'bicarbonate',
                3027946: 'pCO2'
                }

vitals = {
          3027018: 'heart rate',
          3004249: 'SBP',
          3012888: 'DBP',
          3024171: 'respiratory rate',
          3020891: 'temperature',
          4196147: 'O2 sat (peripheral)'
          }

# these are "vitals" which are static
# we get the most recently-recorded value regardless of how long ago
static = {
          3025315: 'weight',
          3036277: 'height',
          }

# drugs are drugs
drugs = {
        1125315: 'acetaminophen',
        40162515: 'acetaminophen 325 mg / hydrocodone bitartrate 5 mg oral tablet',
        19049105: 'potassium chloride',
        1000560: 'ondansetron',
        1154343: 'albuterol',
        19131109: 'albuterol',
        1734104: 'azithromycin',
        19133860: 'azithromycin',
        1124957: 'oxycodone',
        40232756: 'oxycodone',
        938268: 'sennosides',
        797399: 'gabapentin',
        19077548: 'gabapentin',
        923645: 'omeprazole',
        19019418: 'omeprazole',
        1177480: 'ibuprofen',
        989878: 'lidocaine',
        36221099: 'lidocaine',
        967823: 'sodium chloride',
        1771162: 'cefazolin',
        1551099: 'prednisone',
        1551170: 'prednisone',
        1713332: 'amoxicillin',
        19115197: 'amoxicillin',
        1301025: 'enoxaparin',
        1112807: 'aspirin',
        791967: 'lorazepam',
        1154029: 'fentanyl',
        903963: 'triamcinolone',
        1308216: 'lisinopril',
        948078: 'pantoprazole',
        1545958: 'atorvastatin',
        1129625: 'diphenhydramine',
        44348871: 'diphenhydramine',
        44345074: 'diphenhydramine',
        1136980: 'ketorolac',
        1707687: 'vancomycin',
        1777806: 'ceftriaxone',
        19093848: 'magnesium sulfate',
        993631: 'magnesium oxide',
        1307046: 'metoprolol',
        941258: 'docusate',
        1759842: 'clavulanate',
        956874: 'furosemide',
        1518254: 'dexamethasone',
        19095164: 'cholecalciferol',
        1705674: 'trimethoprim',
        1332418: 'amlodipine',
        939506: 'sodium bicarbonate',
        19055137: 'heparin',
        986417: 'polyethylene glycol 3350',
        40169268: 'fluticasone',
        1550023: 'insulin lispro',
        1126658: 'hydromorphone',
        1836434: 'sulfamethoxazole 800 mg / trimethoprim 160 mg oral tablet',
        40175289: 'polyethylene glycol 3350 236000 mg / potassium chloride 2970 mg / sodium bicarbonate 6740 mg / sodium chloride 5860 mg / sodium sulfate 22740 mg powder for oral solution',
        19021041: 'doxycycline',
        19075034: 'cephalexin',
        1110410: 'morphine',
        708298: 'midazolam',
        1742253: 'levofloxacin',
        19075380: 'ciprofloxacin'
        }

thresholds = {
              'static_age': [0, 110],
              'static_weight': [40, 150],
              'static_height': [130, 210],
              'measurements_ALT': [0, 250],
              'measurements_AST': [0, 250],
              'measurements_aPTT': [0, 50],
              'measurements_bilirubin (total)': [0, 20],
              'measurements_glucose': [0, 350],
              'measurements_platelets': [0, 1000],
              'measurements_triglyceride': [0, 400],
              'measurements_albumin': [3, 5],
              'measurements_anion gap': [0, 20],
              'measurements_base excess': [2.5, 7.5],
              'measurements_bicarbonate': [20, 25],
              'measurements_chloride': [0, 300],
              'measurements_cholesterol': [120, 170],
              'measurements_creatinine': [0, 10],
              'measurements_pO2': [100, 300],
              'measurements_pCO2': [25, 75],
              'measurements_CRP': [0, 150],
              'measurements_urea nitrogen': [0, 100],
              'measurements_hematocrit': [20, 55],
              'measurements_leukocytes': [0, 50],
              'measurements_neutrophils': [0, 50],
              'measurements_lymphocytes': [0, 50],
              'measurements_O2 sat': [60, 100],
              'measurements_prothrombin time': [0, 5],
              'vitals_O2 sat (peripheral)': [60, 100],
              'vitals_temperature': [34, 45],
              'vitals_heart rate': [25, 200],
              'vitals_respiratory rate': [10, 40]}


def define_feature_categories(data):
    if type(data) == pd.DataFrame:
        columns = data.columns
    elif type(data) == list:
        columns = data
    else:
        raise ValueError(data)
    necessary_columns = ['test_outcome', 'split', 'test_month', 'visit_type']
    necessary_columns = [x for x in necessary_columns if x in columns]
    all_cols = [x for x in columns if x not in necessary_columns]
    # cut up the columns
    chronic_conditions = [x for x in columns if 'chronic_' in x]
    symptoms = [x for x in columns if 'symptom_' in x]
    labs = [x for x in columns if 'measurements_' in x]
    vitals = [x for x in columns if 'vitals_' in x] + ['static_height_last', 'static_weight_last']
    geography = ['zip', 'county', 'pdens18', 'median_household_income_dollars', 'zip_recent_disease']
    demographics = ['race', 'female', 'hispanic', 'static_age', 'occupation_status', 'occupation_category']
    insurance = [x for x in columns if 'insurance_' in x]
    drugs = [x for x in columns if 'drug_' in x]
    # by test setting
    community = symptoms + ['vitals_temperature_last'] + demographics + geography + ['visit_type']
    clinic = community + chronic_conditions + insurance + drugs + vitals
    inpatient = all_cols
    # community ablations
    community_sans_symptoms = ['vitals_temperature_last'] + demographics + geography + ['visit_type']
    community_sans_temp = symptoms + demographics + geography + ['visit_type']
    community_sans_demographics = symptoms + ['vitals_temperature_last'] + geography + ['visit_type']
    community_sans_geography = symptoms + ['vitals_temperature_last'] + demographics + ['visit_type']
    community_sans_visit_type = symptoms + ['vitals_temperature_last'] + demographics + geography
    # general ablations
    sans_symptoms = [x for x in inpatient if not x in symptoms]
    sans_chronic_conditions = [x for x in inpatient if not x in chronic_conditions]
    sans_labs = [x for x in inpatient if not x in labs]
    sans_vitals = [x for x in inpatient if not x in vitals]
    sans_geography = [x for x in inpatient if not x in geography]
    sans_insurance = [x for x in inpatient if not x in insurance]
    sans_demographics = [x for x in inpatient if not x in demographics]
    sans_drugs = [x for x in inpatient if not x in drugs]

    feature_categories = {'all': all_cols,
                          'chronic': chronic_conditions,
                          'symptoms': symptoms,
                          'labs': labs,
                          'vital': vitals,
                          'geography': geography,
                          'demographic': demographics,
                          'drugs': drugs,
                          'community': community,
                          'clinic': clinic,
                          'inpatient': inpatient,
                          'community_sans_symptoms': community_sans_symptoms,
                          'community_sans_temp': community_sans_temp,
                          'community_sans_demographics': community_sans_demographics,
                          'community_sans_geography': community_sans_geography,
                          'community_sans_visit_type': community_sans_visit_type,
                          'sans_symptoms': sans_symptoms,
                          'sans_chronic_conditions': sans_chronic_conditions,
                          'sans_labs': sans_labs,
                          'sans_vitals': sans_vitals,
                          'sans_geography': sans_geography,
                          'sans_insurance': sans_insurance,
                          'sans_demographics': sans_demographics,
                          'sans_drugs': sans_drugs}
    return feature_categories, necessary_columns
