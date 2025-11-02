from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Load the pipeline
with open('pipeline_v1.bin', 'rb') as f:
    pipeline = pickle.load(f)

app = FastAPI()

class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

@app.post("/predict")
def predict(lead: Lead):
    client = lead.dict()
    probability = pipeline.predict_proba([client])[0, 1]
    return {"probability": float(probability)}

@app.get("/")
def root():
    return {"message": "Lead scoring API is running"}