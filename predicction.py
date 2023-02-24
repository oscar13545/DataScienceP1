import joblib

def predict(data):
    model = joblib.load("final_modelDTR.pkl")
    pipeline = joblib.load('full_pipeline.pkl')
    data = pipeline.transform(data)
    
    return model.predict(data)
