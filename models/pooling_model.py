import pandas as pd
import numpy as np
from utils.utils import bucketed_outcomes, bucket_outcomes, weighted_avg
from models.naivest_model import NaivestModel
import torch


class PoolingModel(NaivestModel):
    """A more careful Naive model:

    Each pitcher has a given event (such as a strikeout) probability for a count,
    Each batter has a given event probability for a given count as well.

    The average of these two probabilities is our prediction, tempered by a
    regularization parameter proportional to the number of observations.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

        self.logit_params = torch.tensor(-1.0, 1_000.0)

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

        self.avg_outcomes_tensor = torch.tensor(
            pd.concat(
                [self.train_data.balls, self.train_data.strikes, events_onehot],
                axis=1,
            )
            .groupby(["balls", "strikes"])
            .mean()
        ).reshape(4, 3, 7)

        self.batter_outcomes_tensor = self._concat_avg(
            torch.tensor(self.batter_outcomes).reshape(-1, 4, 3, 7)
        )

        self.pitcher_outcomes_tensor = self._concat_avg(
            torch.tensor(self.pitcher_outcomes).reshape(-1, 4, 3, 7)
        )

        self.pitchers_in_order = torch.tensor(
            self.pitcher_outcomes.index.get_level_values(0).unique()
        )
        self.batters_in_order = torch.tensor(
            self.batter_outcomes.index.get_level_values(0).unique()
        )

        self.pitcher_indexer = dict(
            (p, i) for (i, p) in enumerate(self.pitchers_in_order)
        )
        self.batter_indexer = dict(
            (p, i) for (i, p) in enumerate(self.batters_in_order)
        )

        self.pitcher_counts = torch.tensor(
            np.vectorize(dict(self.train_data.pitcher.value_counts()).get)(
                self.pitchers_in_order
            )
        )

        self.pitcher_counts = torch.concat(
            [self.pitcher_counts, torch.tensor([torch.sum(self.pitcher_counts)])]
        )

        self.batter_counts = torch.tensor(
            np.vectorize(dict(self.train_data.batter.value_counts()).get)(
                self.batters_in_order
            )
        )

        self.batter_counts = torch.concat(
            [self.batter_counts, torch.tensor([torch.sum(self.batter_counts)])]
        )

    def fit(self, n_epochs: int, lr: float):
        optim = torch.optim.Adam(params=self.logit_params, lr=lr)

    def logit(self, c):
        return 1 / (
            (1 + np.exp(-torch.log(self.logit_params[0]) * (c - self.logit_params[1])))
        )

    def predict_df(self, test_data: pd.DataFrame):
        prepped_test_data = self._preprocess_test(test_data.copy())
        pitcher_inds = np.vectorize(lambda p: self.pitcher_indexer.get(p, -1))(
            np.array(prepped_test_data["pitcher"])
        )
        batter_inds = np.vectorize(lambda b: self.batter_indexer.get(b, -1))(
            np.array(prepped_test_data["batter"])
        )

        logit = lambda x: 1 / ((1 + np.exp(-4e-3 * (x - 1000))))

        pitcher_outcomes_averaged_tensor = weighted_avg(
            self.pitcher_outcomes_tensor,
            self.avg_outcomes_tensor[None, :, :, :],
            logit(self.pitcher_counts)[:, None, None, None],
        )
        batter_outcomes_averaged_tensor = weighted_avg(
            self.batter_outcomes_tensor,
            self.avg_outcomes_tensor[None, :, :, :],
            logit(self.batter_counts)[:, None, None, None],
        )

        pitcher_preds = self._pick_from_tensor(
            pitcher_outcomes_averaged_tensor,
            pitcher_inds,
            np.array(prepped_test_data.balls),
            np.array(prepped_test_data.strikes),
        )
        batter_preds = self._pick_from_tensor(
            batter_outcomes_averaged_tensor,
            batter_inds,
            np.array(prepped_test_data.balls),
            np.array(prepped_test_data.strikes),
        )

        return pd.DataFrame(
            (pitcher_preds + batter_preds) / 2.0, columns=bucketed_outcomes
        )

    @property
    def pitcher_ids(self):
        return list(self.train_data["pitcher"].value_counts().index)

    @property
    def batter_ids(self):
        return list(self.train_data["batter"].value_counts().index)

    def pitches_by_pitcher(self, pitcher_id):
        return self.train_data[self.train_data["pitcher"] == pitcher_id]

    def pitches_by_batter(self, batter_id):
        return self.train_data[self.train_data["batter"] == batter_id]

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

    def _concat_avg(self, tensor: torch.Tensor, dim=0) -> torch.Tensor:
        return torch.concat(
            [tensor, torch.unsqueeze(torch.mean(tensor, dim=dim), dim=dim)], dim=dim
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
