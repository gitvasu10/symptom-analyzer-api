from cmath import log
from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

@app.post("/diabetes")
def postanitem(
    glucose: int,
    insulin: int,
    bmi: int,
    age: int
    ):
    inp = np.array([glucose, insulin, bmi, age]).reshape(1,-1)
    #logging.debug("input array created {}".format(inp))
    filename = 'pickle_modle.pkl'
    #logging.debug("file found: {}".format(str(path.exists('model_1.pkl'))))
    loaded_model = pickle.load(open(filename, 'rb'))
    out = loaded_model.predict(inp)
    print(float(out))
    if int(out):
        return {'Diagnosis': 'Positive'}
    else: 
        return {'Diagnosis': 'Negative'}

@app.post("/covid")
def postanitem(
    Breathing_problem: int,
    Fever: int,
    Dry_cough: int,
    Sore_throat: int,
    Running_nose: int,
    Athma: int,
    Chronic_lung_Disease: int,
    Headache: int,
    Heart_disease: int,
    Diabetes: int,
    Fatigue: int,
    Gastroestinal: int,
    Abroad_travel: int,
    Contact_with_covid_patient: int,
    Attended_large_gatherings: int,
    Visited_public_exposed_places: int,
    Family_caught_covid: int,
    Wearing_masks: int,
    Sanitizatoin_from_market: int,
    Hypertension: int
    ):
    inp = np.array([Breathing_problem,
    Fever,
    Dry_cough,
    Sore_throat,
    Running_nose,
    Athma,
    Chronic_lung_Disease,
    Headache,
    Heart_disease,
    Diabetes,
    Fatigue,
    Gastroestinal,
    Abroad_travel,
    Contact_with_covid_patient,
    Attended_large_gatherings,
    Visited_public_exposed_places,
    Family_caught_covid,
    Wearing_masks,
    Sanitizatoin_from_market,
    Hypertension]).reshape(1,-1)
    #logging.debug("input array created {}".format(inp))
    filename = 'pickle_model_2.pkl'
    #logging.debug("file found: {}".format(str(path.exists('model_1.pkl'))))
    loaded_model = pickle.load(open(filename, 'rb'))
    out = loaded_model.predict(inp)
    print(float(out))
    if int(out):
        return {'Diagnosis': 'There are chances to be affected with covid. Contact the nearest doctor.'}
    else: 
        return {'Diagnosis': 'Negative. Precautions are still necessary!'}


