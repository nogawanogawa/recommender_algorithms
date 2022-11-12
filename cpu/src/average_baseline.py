from util.dataloader import DataLoader
from eval.eval_func import *
from typing import Tuple, List
from util.dataloader import Data
from eval.eval_func import *

class AverageRecommender():
    def __init__(self):
        # predict時のもとになるdatafrme
        self.model = None
        self.train_df = None

    def train(self, train_df: pd.DataFrame) -> None:
        """ 平均値で埋める """
        self.model = train_df.groupby("item_id")["rating"].mean()
        self.train_df = train_df.set_index(["user_id", "item_id"])

        return

    def predict(self, test_df:pd.DataFrame) -> List[float]:
        """ あたえられたdfのすべてのratingを返す """
        return [self.predict_score(user_id, item_id) for user_id, item_id in test_df.values]

    def predict_score(self, user_id:int, item_id:int) -> pd.DataFrame:
        """
        user_id, item_id毎の推論したratingを返す
        値が入っていない場合はNoneを返す
        """

        if (user_id, item_id) in self.train_df.index:
            return self.train_df.loc[(user_id, item_id)]["rating"]
        else:
            try:
                return self.model[item_id]
            except:
                return

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return Data().load() # user_df, item_df, train_df, test_df

if __name__ == "__main__":

    user_df, item_df, train_df, test_df = load_data()

    # train
    # 未評価のところは評価済みのユーザーのratingの平均とする
    average_recommender = AverageRecommender()
    average_recommender.train(train_df=train_df)

    # predict
    result = average_recommender.predict(test_df[["user_id", "item_id"]])
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
