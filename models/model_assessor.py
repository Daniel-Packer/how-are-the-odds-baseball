from typing import Literal
from utils.utils import mean_cross_entropy, prep_labels
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from models.abstract_model import AbstractModel


class ModelAssessor:
    def __init__(self, data, verbose: bool = False):
        self.data = data
        self.verbose = verbose

    def assess(
        self,
        model: AbstractModel,
        method: Literal["forward_slide_cut", "forward_slide_forget"],
        n_splits=10,
        **kwargs,
    ):
        match method:
            case "forward_slide_cut":
                return self._forward_slide(
                    model, method="cut", n_splits=n_splits, **kwargs
                )
            case "forward_slide_forget":
                return self._forward_slide(
                    model, method="forget", n_splits=n_splits, **kwargs
                )

    def _forward_slide(
        self, Model, method: Literal["cut", "forget"], n_splits=10, **kwargs
    ):
        split_df = np.array_split(self.data, n_splits)
        cross_entropies = np.zeros(n_splits - 1)

        pbar = range(1, n_splits)
        pbar = tqdm(pbar) if self.verbose else pbar

        for i in pbar:
            if method == "cut":
                train_data = pd.concat(split_df[:i])
                test_data = pd.concat(split_df[i:])
            if method == "forget":
                train_data = pd.concat(split_df[i - 1 : i])
                test_data = pd.concat(split_df[i:])

            model = Model(verbose=self.verbose, **kwargs)
            model.fit(train_data)
            predictions = model.predict_df(test_data)
            true_labels = prep_labels(test_data)

            cross_entropies[i - 1] = mean_cross_entropy(
                true_labels.to_numpy(), predictions.to_numpy()
            )

        return cross_entropies
