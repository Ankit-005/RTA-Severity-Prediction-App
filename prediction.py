import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
#encoder=joblib.load('label encoder.pkl')

def label_encoder(input_val,feat):
    le=LabelEncoder()
    le.fit(feat)
    value=le.transform(np.array([input_val]))
    return value[0]

def get_prediction(data,model):
    return model.predict(data)

def get_severity(val):
    res=""
    if(val==0):
        res="Serious Injury"
    elif(val==1):
        res="Slight Injury"
    else:
        res="Fatal Injury"
    return res