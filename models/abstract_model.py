from abc import ABC, abstractmethod
import pandas as pd


class AbstractModel(ABC):
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    @abstractmethod
    def fit(self, train_data: pd.DataFrame):
        return

    @abstractmethod
    def predict_df(self, test_data: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame()
