import os
from dotenv import load_dotenv

load_dotenv()

PROCESSED_DATA = os.getenv("PROCESSED_DATA")
TRAINED_LINEAR_REGRESSOR = os.getenv("TRAINED_LINEAR_REGRESSOR")
TRAINED_RIDGE_REGRESSOR = os.getenv("TRAINED_RIDGE_REGRESSOR")
VALIDATION_TRAINING_DATA = os.getenv("VALIDATION_TRAINING_DATA")
EVALUATION_TEST_DATA = os.getenv("EVALUATION_TEST_DATA")
