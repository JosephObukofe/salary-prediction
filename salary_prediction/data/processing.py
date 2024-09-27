import numpy as np
import pandas as pd
from typing import Any, Tuple
from sklearn.model_selection import train_test_split


def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads raw data from the specified file path

    Parameters:
    - file_path: Path to the raw (converted) data

    Returns:
    - A DataFrame of the converted raw data from the file.
    """

    raw_data = pd.read_csv(file_path)

    return raw_data


def split_data(
    X, y, test_size: float = 0.2, random_state: int = 42
) -> Tuple[Any, Any, Any, Any]:
    """
    Splits the data into training and test sets.

    Parameters:
    - X: Features (independent variables)
    - y: Target (dependent variable)
    - test_size: Proportion of the dataset to include in the test split (default is 0.2)
    - random_state: Controls the shuffling applied to the data before applying the split (default is 42 for reproducibility)

    Returns:
    - Tuple containing X_train, X_test, y_train, and y_test
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test
