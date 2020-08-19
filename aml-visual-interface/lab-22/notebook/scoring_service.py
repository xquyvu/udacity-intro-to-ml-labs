
import json
import numpy as np
import pandas as pd
import azureml.core
import azureml.train.automl
from azureml.core.model import Model
import joblib

columns = ['vendorID', 'passengerCount', 'tripDistance', 'hour_of_day', 'day_of_week', 'day_of_month', 
           'month_num', 'normalizeHolidayName', 'isPaidTimeOff', 'snowDepth', 'precipTime', 
           'precipDepth', 'temperature']

def init():
    # One-time initialization of predictive model and scaler
    global model
    
    print("Azure ML SDK version:", azureml.core.VERSION)
    model_name = 'nyc-taxi-automl-predictor'
    print('Looking for model path for model: ', model_name)
    model_path = Model.get_model_path(model_name=model_name)
    print('Looking for model in: ', model_path)
    model = joblib.load(model_path)
    print('Model loaded...')

def run(input_json):     
    try:
        inputs = json.loads(input_json)
        data_df = pd.DataFrame(np.array(inputs).reshape(-1, len(columns)), columns = columns)
        # Get the predictions...
        prediction = model.predict(data_df)
        prediction = json.dumps(prediction.tolist())
    except Exception as e:
        prediction = str(e)
    return prediction
