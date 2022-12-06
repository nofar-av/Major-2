import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def prepare_blood_type(data): 
  data['blood_type_A'] = data["blood_type"].isin(["A+", "A-"])
  data['blood_type_B'] = data["blood_type"].isin(["AB+", "AB-", "B-", "B+"])
  data['blood_type_O'] = data["blood_type"].isin(["O+", "O-"])
  data.drop("blood_type", axis=1, inplace=True)
  return data

def prepare_symptoms(data):
  symptoms = ['cough', 'fever', 'low_appetite', 'shortness_of_breath', 'sore_throat']
  for symptom in symptoms:
    data[symptom] = data["symptoms"].str.contains(symptom)
  data = data.fillna({"fever" : False,"sore_throat" : False,"low_appetite" : False, "shortness_of_breath" : False,"cough" : False}) 
  data.drop("symptoms", inplace=True, axis=1)
  return data

def prepare_location(data):
  data = pd.concat([data.drop('current_location', axis = 1, inplace=False), 
            (data.current_location.str.split(",").str[:2].apply(pd.Series)
            .rename(columns={0:'location_a', 1:'location_b'}))], axis = 1)
  data['location_a'] = data['location_a'].str[2:-1:].astype(float)
  data['location_b'] = data['location_b'].str[2:-2:].astype(float)
  return data




#splitting : blood_type, pcr_date, current_location, symptoms, converting booleans to :[-1,1], changes data
def prepare_df(df):
  data = df.copy()
  data = prepare_blood_type(data)
  data = prepare_symptoms(data)
  data = prepare_location(data)
  data.replace({'sex':{'M':True, 'F':False}}, inplace=True)
  #converting bools and strings to ints
  bool_variables =data.select_dtypes(include=['bool']).columns
  data[bool_variables] = data[bool_variables].astype(int).replace([0, 1], [-1, 1])
  data.drop(["patient_id", "pcr_date", "happiness_score"], axis=1, inplace=True)
  return data

def prepare_data(training_data, new_data):
  prepared_train = prepare_df(training_data)
  prepared_new_data= prepare_df(new_data)
  #normalization:
  minmax_scaler = MinMaxScaler(feature_range=(-1,1))
  standard_scaler = StandardScaler()

  minmaxed_cols = ['sport_activity', 'PCR_01', 'PCR_02', 'PCR_03', 'PCR_04', 'PCR_05', 'PCR_06', 'PCR_07', 'PCR_09', 'PCR_10', 'location_a', 'location_b']
  standardized_cols = ['age', 'weight', 'num_of_siblings','household_income', 'conversations_per_day', 'sugar_levels','PCR_08']
  #fitting the scalers to training data
  minmax_scaler.fit(prepared_train[minmaxed_cols])
  standard_scaler.fit(prepared_train[standardized_cols])
  prepared_train[minmaxed_cols] = minmax_scaler.transform(prepared_train[minmaxed_cols])
  prepared_train[standardized_cols] =  standard_scaler.transform(prepared_train[standardized_cols])

  #transforming new data
  prepared_new_data[minmaxed_cols] = minmax_scaler.transform(prepared_new_data[minmaxed_cols])
  prepared_new_data[standardized_cols] =  standard_scaler.transform(prepared_new_data[standardized_cols])

  return prepared_new_data
