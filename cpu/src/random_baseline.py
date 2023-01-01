from typing import Tuple, List
from util.dataloader import Data
from eval.eval_func import *


class RandomRecommender:
    def __init__(self):
        # predict時のもとになるdatafrme
        self.model = None

    def train(self, train_df: pd.DataFrame):

        df = train_df.pivot_table(index="user_id", columns="item_id", values="rating")

        # random sampling
        ratings = list(range(1, 5))
        rating = np.random.choice(ratings, df.size)

        pred_df = pd.DataFrame(
            rating.reshape(df.shape[0], df.shape[1]), columns=df.columns, index=df.index
        )

        # trainに値がある箇所以外はrandomな値でflllする
        self.model = df.where(~df.isna(), pred_df)
        self.model = pd.melt(
            self.model.reset_index(), id_vars="user_id", value_vars=self.model.columns
        )
        self.model = self.model.set_index(["user_id", "item_id"])

        return

    def predict(self, test_df: pd.DataFrame) -> List[float]:
        """ あたえられたdfのすべてのratingを返す """
        return [
            self.predict_score(user_id, item_id) for user_id, item_id in test_df.values
        ]

    def predict_score(self, user_id: int, item_id: int) -> pd.DataFrame:
        """ user_id, item_id毎の推論したratingを返す """
        res = None
        try:
            res = self.model.loc[(user_id, item_id)]["value"]
        except:
            pass

        return res


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return Data().load()  # user_df, item_df, train_df, test_df


if __name__ == "__main__":

    user_df, item_df, train_df, test_df = load_data()

    # train
    # 未評価のところにrandomで値を埋めるだけ
    random_recommender = RandomRecommender()
    random_recommender.train(train_df=train_df)

    # predict
    pred_df = test_df.copy()
    pred_df["y_pred"] = random_recommender.predict(test_df[["user_id", "item_id"]])
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
