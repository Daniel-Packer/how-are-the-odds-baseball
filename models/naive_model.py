import pandas as pd
import numpy as np
from utils.utils import bucketed_outcomes, bucket_outcomes
from models.naivest_model import NaivestModel


class NaiveModel(NaivestModel):
    """The Naivest possible model:

    Each pitcher has a given event (such as a strikeout) probability for a count,
    each batter has a given event probability for a given count as well.

    The average of these two probabilities is our prediction!
    """

    def fit(self, train_data):
        self.train_data = self._preprocess(train_data.copy())
        events_onehot = pd.get_dummies(self.train_data.events)
        self.pitcher_outcomes = self._make_rectangular(
            pd.concat(
                [
                    self.train_data.pitcher,
                    self.train_data.balls,
                    self.train_data.strikes,
                    events_onehot,
                ],
                axis=1,
            )
            .groupby(["pitcher", "balls", "strikes"])
            .mean()
        )

        self.batter_outcomes = self._make_rectangular(
            pd.concat(
                [
                    self.train_data.batter,
                    self.train_data.balls,
                    self.train_data.strikes,
                    events_onehot,
                ],
                axis=1,
            )
            .groupby(["batter", "balls", "strikes"])
            .mean()
        )

        self.batter_outcomes_tensor = self._concat_avg(
            self.batter_outcomes.to_numpy().reshape(-1, 4, 3, 7)
        )

        self.pitcher_outcomes_tensor = self._concat_avg(
            self.pitcher_outcomes.to_numpy().reshape(-1, 4, 3, 7)
        )

        self.pitcher_indexer = dict(
            (p, i)
            for (i, p) in enumerate(
                np.array(self.pitcher_outcomes.index.get_level_values(0).unique())
            )
        )
        self.batter_indexer = dict(
            (p, i)
            for (i, p) in enumerate(
                np.array(self.batter_outcomes.index.get_level_values(0).unique())
            )
        )

        self.avg_batter_outcomes = self.batter_outcomes.groupby(
            ["balls", "strikes"]
        ).mean()
        self.avg_pitcher_outcomes = self.pitcher_outcomes.groupby(
            ["balls", "strikes"]
        ).mean()

    def predict_entry(self, pitcher_id, batter_id, balls, strikes):
        try:
            pitcher_probs = self.pitcher_outcomes.loc[pitcher_id, balls, strikes]
        except:
            pitcher_probs = self.avg_pitcher_outcomes.loc[balls, strikes]
        try:
            batter_probs = self.batter_outcomes.loc[batter_id, balls, strikes]
        except:
            batter_probs = self.avg_batter_outcomes.loc[balls, strikes]
        return (pitcher_probs + batter_probs) / 2

    def predict_df(self, test_data: pd.DataFrame):
        prepped_test_data = self._preprocess_test(test_data.copy())
        pitcher_inds = np.vectorize(lambda p: self.pitcher_indexer.get(p, -1))(
            np.array(prepped_test_data["pitcher"])
        )
        batter_inds = np.vectorize(lambda b: self.batter_indexer.get(b, -1))(
            np.array(prepped_test_data["batter"])
        )

        pitcher_preds = self._pick_from_tensor(
            self.pitcher_outcomes_tensor,
            pitcher_inds,
            np.array(prepped_test_data.balls),
            np.array(prepped_test_data.strikes),
        )
        batter_preds = self._pick_from_tensor(
            self.batter_outcomes_tensor,
            batter_inds,
            np.array(prepped_test_data.balls),
            np.array(prepped_test_data.strikes),
        )

        return pd.DataFrame(
            (pitcher_preds + batter_preds) / 2.0, columns=bucketed_outcomes
        )

    @property
    def pitcher_ids(self):
        return list(self.data["pitcher"].value_counts().index)

    @property
    def batter_ids(self):
        return list(self.data["batter"].value_counts().index)

    def pitches_by_pitcher(self, pitcher_id):
        return self.data[self.data["pitcher"] == pitcher_id]

    def pitches_by_batter(self, batter_id):
        return self.data[self.data["batter"] == batter_id]

    def _preprocess(self, data):
        self._clean_data(data)
        data.loc[:, "events"] = data["events"].ffill().apply(bucket_outcomes)
        # drop data that we don't have a label for:
        unknown_events = data["events"] == "unknown"
        data = data[~unknown_events]

        num_unknown_events = unknown_events.sum()
        if num_unknown_events > 0 and self.verbose:
            print(f"{num_unknown_events} events dropped due to being unknown")
        return data[["pitcher", "batter", "balls", "strikes", "events"]]

    def _concat_avg(self, tensor: np.ndarray, axis=0) -> np.ndarray:
        return np.concatenate(
            [tensor, np.expand_dims(np.mean(tensor, axis=0), axis=0)], axis=0
        )

    def _pick_from_tensor(self, outcomes_tensor, player_inds, balls, strikes):
        return np.take_along_axis(
            np.take_along_axis(
                np.take_along_axis(
                    outcomes_tensor,
                    player_inds[:, None, None, None],
                    axis=0,
                ),
                balls[:, None, None, None],
                axis=1,
            ),
            strikes[:, None, None, None],
            axis=2,
        )[:, 0, 0]
