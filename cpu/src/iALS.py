import implicit
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from typing import Tuple, List
from scipy import sparse
from eval.eval_func import *


from util.dataloader import Data

class iALS:
    def __init__(self, user_df, item_df) -> None:
        self.user_df = user_df
        self.item_df = item_df

        self.users = self.user_df["user_id"].unique()
        self.movies = self.item_df["item_id"].unique()
        self.shape = (len(self.users), len(self.movies))

        # Create indices for users and movies
        self.user_cat = CategoricalDtype(categories=sorted(self.users), ordered=True)
        self.movie_cat = CategoricalDtype(categories=sorted(self.movies), ordered=True)

        self.user_index = train_df["user_id"].astype(self.user_cat).cat.codes
        self.movie_index = train_df["item_id"].astype(self.movie_cat).cat.codes

    def train(self, train_df):
        train_df = train_df[["user_id", "item_id", "rating"]]


        # Conversion via COO matrix
        coo = sparse.coo_matrix(
            (train_df["rating"], (self.user_index, self.movie_index)), shape=self.shape
        )
        self.csr = coo.tocsr()

        data = sparse.csr_matrix(train_df.values)
        self.model = implicit.als.AlternatingLeastSquares(factors=64)

        # train
        self.model.fit(self.csr)

        self.result = pd.DataFrame()
        for user_id in self.users:
            df = self.predict_user(user_id)
            self.result = pd.concat([self.result, df])

        self.result = self.result.set_index(["user_id", "item_id"])

    def predict_user(self, user_id):
        try:
            items, scores = self.model.recommend(userid=user_id, user_items=self.csr[user_id], N=self.shape[1])
            return pd.DataFrame({"user_id": user_id, "item_id": items, "value": scores})
        except:
            return pd.DataFrame({"user_id": user_id, "item_id": self.movies , "value": 3})


    def predict(self, test_df):

        return [self.predict_score(user_id, item_id) for user_id, item_id in test_df.values]

    def predict_score(self, user_id:int, item_id:int) -> float:
        """ user_id, item_id毎の推論したratingを返す """
        try:
            return self.result.loc[(user_id, item_id)]["value"]
        except:
            return np.nan



def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return Data().load() # user_df, item_df, train_df, test_df

if __name__ == "__main__":
    user_df, item_df, train_df, test_df = load_data()

    ials = iALS(user_df, item_df)
    ials.train(train_df)

    # predict
    result = ials.predict(test_df[["user_id", "item_id"]])
    test_df["rating_predict"] = result
    test_df = test_df.dropna(subset=['rating_predict'])

    ####### Evaluate #######
    # RMSE
    # わかっているratingについて、推定したratingとのRMSEを取得する
    rmse_score = rmse(
        test_df["rating_predict"].values.tolist(), test_df["rating"].values.tolist()
    )
    print(f"rmse: {rmse_score}")

    # Recall
    # 推定ratingを上からk件とってきてそれのrecallを取る
    k = 10
    true_df = (
        test_df.sort_values(["user_id", "rating", "item_id"], ascending=False)
        .groupby("user_id")
        .head(k)
    )
    y_true = (
        true_df[["user_id", "item_id"]]
        .groupby("user_id")["item_id"]
        .apply(list)
        .to_dict()
    )

    candidate = (
        test_df.sort_values(["user_id", "rating_predict", "item_id"], ascending=False)
        .groupby("user_id")
        .head(k)
    )
    y_pred = (
        candidate[["user_id", "item_id"]]
        .groupby("user_id")["item_id"]
        .apply(list)
        .to_dict()
    )
    recall_socre = recall(y_pred, y_true)
    print(f"recall@{k}: {recall_socre}")

    # nDCG
    k = 10
    pred_df = test_df[["user_id", "item_id"]]

    pred_df = pred_df.merge(
        test_df[["user_id", "item_id", "rating_predict"]], on=["user_id", "item_id"], how="left"
    )
    pred_df = pred_df.sort_values(["user_id", "rating_predict", "item_id"], ascending=False)
    pred_df = (
        pred_df.reset_index(drop=True).reset_index().rename(columns={"index": "rank"})
    )
    ndcg_score = ndcg(y_pred=pred_df, y_true=test_df, k=k)
    print(f"ndcg@{k}: {ndcg_score}")

