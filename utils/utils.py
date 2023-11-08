from typing import Optional, Union
import pandas as pd
import numpy as np

bucket_outcomes_dict = {
    "field_out": "fielded_out",
    "strikeout": "strikeout",
    "single": "single",
    "walk": "walk",
    "double": "double_or_triple",
    "triple": "double_or_triple",
    "home_run": "home_run",
    "force_out": "fielded_out",
    "grounded_into_double_play": "fielded_out",
    "hit_by_pitch": "walk",
    "field_error": "other",
    "double_play": "fielded_out",
    "sac_fly": "other",
    "sac_bunt": "other",
    "fielders_choice": "fielded_out",
    "fielders_choice_out": "fielded_out",
    "strikeout_double_play": "fielded_out",
    "caught_stealing_2b": "other",
    "catcher_interf": "other",
    "other_out": "fielded_out",
    "sac_fly_double_play": "other",
    "caught_stealing_home": "other",
    "pickoff_1b": "other",
    "caught_stealing_3b": "other",
    "wild_pitch": "other",
    "pickoff_2b": "other",
    "sac_bunt_double_play": "other",
    "triple_play": "fielded_out",
    "pickoff_caught_stealing_home": "other",
    "runner_double_play": "fielded_out",
    "stolen_base_2b": "other",
    "pickoff_caught_stealing_3b": "other",
    "game_advisory": "other",
    "pickoff_3b": "other",
}

bucketed_outcomes = [
    "double_or_triple",
    "fielded_out",
    "home_run",
    "other",
    "single",
    "strikeout",
    "walk",
]

bucket_outcomes = lambda s: bucket_outcomes_dict.get(s, "unknown")


def weighted_avg(a: np.ndarray, b: np.ndarray, ell: float):
    return (a * ell) + ((1 - ell) * b)


def prep_labels(data: pd.DataFrame) -> pd.DataFrame:
    one_hot = pd.get_dummies(
        data["events"].copy().ffill().apply(bucket_outcomes)
    ).astype(int)
    if "unknown" in one_hot.columns:
        one_hot = one_hot.drop("unknown", axis=1)
    return one_hot


cross_entropy = lambda p, q: -np.sum(p * np.log(q + 1e-12))


def mean_cross_entropy(labels: np.ndarray, predictions: np.ndarray):
    return np.sum(np.mean(-labels * np.log(predictions + 1e-12), axis=0))


def mean_cross_entropy_df(
    labels: pd.DataFrame, predictions: pd.DataFrame, margin: Optional[int] = None
) -> Union[pd.Series, float]:
    loss_df = -labels * np.log(predictions + 1e-12)
    if margin == None:
        return (loss_df.sum() / len(loss_df)).sum()
    return loss_df.mean(axis=margin)

def get_frequent_players_only(data: pd.DataFrame):
  pitcher_counts =  data.pitcher.value_counts()
  batter_counts =  data.batter.value_counts()

  freq_pitchers = list(pitcher_counts[pitcher_counts > 2_000].index)
  freq_batters = list(batter_counts[batter_counts > 2_000].index)

  return data[
    data['pitcher'].apply(lambda p: p in freq_pitchers) & 
    data['batter'].apply(lambda p: p in freq_batters)
  ]