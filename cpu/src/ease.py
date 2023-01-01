import implicit
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from typing import Tuple, List
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from multiprocessing import Pool, cpu_count
from eval.eval_func import *

from util.dataloader import Data


class EASE:
    def __init__(self, user_df, item_df):
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()

        self.users = self.user_enc.fit_transform(user_df.loc[:, "user_id"])
        self.items = self.item_enc.fit_transform(item_df.loc[:, "item_id"])

    def _get_users_and_items(self, df):
        users = self.user_enc.transform(df.loc[:, "user_id"])
        items = self.item_enc.transform(df.loc[:, "item_id"])

        return users, items

    def train(self, df, lambda_: float = 0.5, implicit=True):
        """
        df: pandas.DataFrame with columns user_id, item_id and (rating)
        lambda_: l2-regularization term
        implicit: if True, ratings are ignored and taken as 1, else normalized ratings are used
        """
        self.train_df = df
        users, items = self._get_users_and_items(df)
        values = (
            np.ones(df.shape[0])
            if implicit
            else df["rating"].to_numpy() / df["rating"].max()
        )

        X = csr_matrix((values, (users, items)))
        self.X = X

        G = X.T.dot(X).toarray()
        diagIndices = np.diag_indices(G.shape[0])
        G[diagIndices] += lambda_
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        B[diagIndices] = 0

        self.B = B
        self.pred = X.dot(B)

    def predict(self, test_df: pd.DataFrame):
        train = self.train_df
        users = test_df["user_id"]
        items = test_df["item_id"]
        k = np.unique(self.items).size

        users = self.user_enc.transform(users)
        items = self.item_enc.transform(items)
        dd = train.loc[train.user_id.isin(users)]
        dd["ci"] = self.item_enc.transform(dd.item_id)
        dd["cu"] = self.user_enc.transform(dd.user_id)
        g = dd.groupby("cu")
        with Pool(cpu_count()) as p:
            user_preds = p.starmap(
                self.predict_for_user,
                [(user, group, self.pred[user, :], items, k) for user, group in g],
            )
        df = pd.concat(user_preds)
        df["item_id"] = self.item_enc.inverse_transform(df["item_id"])
        df["user_id"] = self.user_enc.inverse_transform(df["user_id"])
        result = test_df.merge(df, on=["user_id", "item_id"], how="left")
        return result["score"]

    @staticmethod
    def predict_for_user(user, group, pred, items, k):
        watched = set(group["ci"])
        candidates = [item for item in items if item not in watched]
        pred = np.take(pred, candidates)
        res = np.argpartition(pred, -k)[-k:]
        r = pd.DataFrame(
            {
                "user_id": [user] * len(res),
                "item_id": np.take(candidates, res),
                "score": np.take(pred, res),
            }
        ).sort_values("score", ascending=False)
        return r


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return Data().load()  # user_df, item_df, train_df, test_df


if __name__ == "__main__":
    user_df, item_df, train_df, test_df = load_data()
    ease = EASE(user_df, item_df)
    ease.train(train_df, lambda_=2000)

    # predict
    pred_df = test_df.copy()
    pred_df["y_pred"] = ease.predict(pred_df[["user_id", "item_id"]])
    pred_df = pred_df.dropna(subset=["y_pred"])
    pred_df["predicted_rank"] = pred_df.groupby(["user_id"])["y_pred"].rank(
        ascending=False, method="first"
    )
    pred_df = pred_df[["user_id", "item_id", "y_pred", "predicted_rank"]]
    true_df = test_df.copy()
    true_df = true_df[["user_id", "item_id", "rating"]].rename(
        columns={"rating": "y_true"}
    )
    true_df["optimal_rank"] = true_df.groupby(["user_id"])["y_true"].rank(
        ascending=False, method="first"
    )
    true_df = true_df[["user_id", "item_id", "y_true", "optimal_rank"]]

    result = eval(predict_recom=pred_df, true_recom=true_df)
    print(result)
