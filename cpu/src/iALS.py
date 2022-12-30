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

        self.model = implicit.als.AlternatingLeastSquares(factors=64)

        # train
        self.model.fit(self.csr)

        user_index = self.user_df["user_id"].drop_duplicates().astype(self.user_cat).cat.codes
        ids, scores = self.model.recommend(
            userid=user_index,
            user_items=self.csr[user_index],
            N=len(self.movies),
        )

        self.result = pd.DataFrame()
        for user_id, id, score in zip(user_index, ids, scores):
            # もとのcompany_idに復元
            user = self.user_cat.categories[user_id]
            id = [self.movie_cat.categories[i] for i in id]
            df = pd.DataFrame(
                {"user_id": [user] * len(id), "item_id": id, "value": score}
            )
            self.result = pd.concat([self.result, df])

        self.result = self.result.set_index(["user_id", "item_id"])

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
    pred_df = test_df.copy()
    pred_df["y_pred"] = ials.predict(pred_df[["user_id", "item_id"]])
    pred_df = pred_df.dropna(subset=['y_pred'])
    pred_df["predicted_rank"] = pred_df.groupby(["user_id"])["y_pred"].rank(ascending=False, method='first')
    pred_df = pred_df[["user_id", "item_id", "y_pred", "predicted_rank"]]

    true_df = test_df.copy()
    true_df = true_df[["user_id", "item_id", "rating"]].rename(columns={"rating":"y_true"})
    true_df["optimal_rank"] = true_df.groupby(["user_id"])["y_true"].rank(ascending=False, method='first')
    true_df = true_df[["user_id", "item_id", "y_true", "optimal_rank"]]

    result = eval(predict_recom=pred_df, true_recom=true_df)
    print(result)
