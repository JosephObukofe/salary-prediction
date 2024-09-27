import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from salary_prediction.config.config import TRAINED_LINEAR_REGRESSOR

app = FastAPI()

model = joblib.load(TRAINED_LINEAR_REGRESSOR)


class InputData(BaseModel):
    years_of_experience: List[float]


@app.post("/predict")
def predict(input_data: InputData):
    try:
        if not input_data.years_of_experience:
            raise HTTPException(status_code=400, detail="Input data cannot be empty.")
        input_features = [[years] for years in input_data.years_of_experience]
        predicted_salaries = model.predict(input_features)
        return {"predicted salaries": predicted_salaries.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
