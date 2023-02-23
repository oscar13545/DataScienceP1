import joblib

def predict(data):
    clf = joblib.load("final_modelDTR.pkl")
    pipe = joblib.load("full_pipeline.pkl")
    data1 = pipe.tranform(data)
    return clf.predict(data1)