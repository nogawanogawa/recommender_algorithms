from typing import Tuple, List
from util.dataloader import Data
from eval.eval_func import *

class RandomRecommender():
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
        self.model = pd.melt(self.model.reset_index(), id_vars="user_id", value_vars=self.model.columns)
        self.model = self.model.set_index(["user_id", "item_id"])

        return

    def predict(self, test_df:pd.DataFrame) -> List[float]:
        """ あたえられたdfのすべてのratingを返す """
        return [self.predict_score(user_id, item_id) for user_id, item_id in test_df.values]

    def predict_score(self, user_id:int, item_id:int) -> pd.DataFrame:
        """ user_id, item_id毎の推論したratingを返す """
        res = None
        try:
            res = self.model.loc[(user_id, item_id)]["value"]
        except:
            pass

        return res

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return Data().load() # user_df, item_df, train_df, test_df

if __name__ == "__main__":

    user_df, item_df, train_df, test_df = load_data()

    # train
    # 未評価のところにrandomで値を埋めるだけ
    random_recommender = RandomRecommender()
    random_recommender.train(train_df=train_df)

    # predict
    result = random_recommender.predict(test_df[["user_id", "item_id"]])
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
