from models.abstract_model import AbstractModel
from utils.utils import bucket_outcomes, bucketed_outcomes
import pandas as pd
import numpy as np


class NaivestModel(AbstractModel):
    def fit(self, train_data):
        self.train_data = self._preprocess(train_data.copy())
        events_onehot = pd.get_dummies(self.train_data.events)

        self.outcomes = self._make_rectangular(
            pd.concat(
                [
                    self.train_data.balls,
                    self.train_data.strikes,
                    events_onehot,
                ],
                axis=1,
            )
            .groupby(["balls", "strikes"])
            .mean()
        )

        self.outcomes_tensor = self.outcomes.to_numpy().reshape(4, 3, 7)

    def predict_df(self, test_data: pd.DataFrame) -> pd.DataFrame:
        prepped_test_data = self._preprocess_test(test_data)
        predictions = self._pick_from_tensor(
            self.outcomes_tensor,
            np.array(prepped_test_data.balls),
            np.array(prepped_test_data.strikes),
        )
        return pd.DataFrame(predictions, columns=bucketed_outcomes)

    def _pick_from_tensor(self, outcomes_tensor, balls, strikes):
        return np.take_along_axis(
            np.take_along_axis(
                outcomes_tensor,
                balls[:, None, None],
                axis=0,
            ),
            strikes[:, None, None],
            axis=1,
        )[:, 0, :]

    def _preprocess(self, data):
        data = self._clean_data(data.copy())
        data.loc[:, "events"] = data["events"].ffill().apply(bucket_outcomes)
        # drop data that we don't have a label for:
        unknown_events = data["events"] == "unknown"
        data = data[~unknown_events]

        num_unknown_events = unknown_events.sum()
        if num_unknown_events > 0 and self.verbose:
            print(f"{num_unknown_events} events dropped due to being unknown")
        return data[["balls", "strikes", "events"]]

    def _clean_data(self, data):
        data["balls"] = data["balls"].astype(int)
        data["strikes"] = data["strikes"].astype(int)
        data.loc[data["strikes"] > 2, "strikes"] = 2
        data.loc[data["balls"] > 3, "balls"] = 3
        return data

    def _preprocess_test(self, data):
        data = self._clean_data(data.copy())
        return data[["pitcher", "batter", "balls", "strikes"]]

    def _make_rectangular(self, df: pd.DataFrame) -> pd.DataFrame:
        new_indices = pd.MultiIndex.from_product(df.index.levels, names=df.index.names)
        return df.reindex(new_indices, fill_value=0.0)
