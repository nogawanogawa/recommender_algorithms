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
    pred_df = test_df.copy()
    pred_df["y_pred"] = average_recommender.predict(test_df[["user_id", "item_id"]])
    pred_df = pred_df.dropna(subset=['y_pred'])

    pred_df["predicted_rank"] = pred_df.groupby(["user_id"])["y_pred"].rank(ascending=False, method='first')
    pred_df = pred_df[["user_id", "item_id", "y_pred", "predicted_rank"]]

    true_df = test_df.copy()
    true_df = true_df[["user_id", "item_id", "rating"]].rename(columns={"rating":"y_true"})
    true_df["optimal_rank"] = true_df.groupby(["user_id"])["y_true"].rank(ascending=False, method='first')
    true_df = true_df[["user_id", "item_id", "y_true", "optimal_rank"]]

    result = eval(predict_recom=pred_df, true_recom=true_df)
    print(result)

