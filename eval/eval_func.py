from typing import List, Dict
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd


def rmse(y_pred: List[float], y_true: List[float]) -> float:
    assert len(y_pred) == len(y_true)
    return np.sqrt(mean_squared_error(y_true, y_pred))


def recall(y_pred: Dict[int, List[int]], y_true: Dict[int, List[int]], k:int=10) -> float:
    """ しきい値設けてそれ以上の値が取れているかを考えたほうが良い気がしてきた """
    """ y_true に含まれるユーザーのrecallの平均 """
    recall = 0.0
    for key, value in y_true.items():
        if len(value) < k or key not in y_pred.keys():
            continue
        s = set(y_pred[key]).intersection(set(value))  # y_trueとy_predに共通で含まれる要素を取得
        recall += len(s) / len(set(value))
    return recall / len(y_true)


def dcg(gain, k=None):
    """ calc dcg value """
    if k is None:
        k = gain.shape[0]

    ret = gain[0]
    for i in range(1, k):
        ret += gain[i] / np.log2(i + 1)
    return ret


def _ndcg(y: np.array, k=None) -> float:
    """calc nDCG value 

    Args:
        y (np.array): 推定ratingの降順で並べたときの真のratingの配列
        k (_type_, optional): @k. Defaults to None.

    Returns:
        float: ndcg@k値
    """

    dcg_score = dcg(y, k=k)
    #print("pred_sorted_scores : {}".format(dcg_score))

    ideal_sorted_scores = np.sort(y)[::-1]
    ideal_dcg_score = dcg(ideal_sorted_scores, k=k)
    #print("ideal_dcg_score : {}".format(ideal_dcg_score))

    return dcg_score / ideal_dcg_score

def ndcg(y_pred: pd.DataFrame, y_true: pd.DataFrame, k: int = 10) -> float:
    """ y_predとy_trueを受け取って、ユーザーごとにndcgの平均したものを返す """

    # predのdfにy_trueのscoreを付与(columns = ["user_id", "item_id", "rank", "rating"])
    y = y_pred.merge(y_true, on=["user_id", "item_id"], how="left")

    score = 0.0
    count = 0
    for user_id in y["user_id"].unique():
        ranking = y[y["user_id"] == user_id].sort_values(
            by="rank", ascending=True
        )  # 順位なので昇順にsort
        ranking = ranking["rating"].values
        ranking = ranking[~np.isnan(ranking)]

        # scoreが不明のケースによって長さがk未満になった場合に平均の対象から除外する
        if len(ranking) < k:
            continue
        score += _ndcg(ranking, k)

        count += 1

    return score / count

def eval(predict_recom:pd.DataFrame, true_recom:pd.DataFrame) -> None:
    """
        predict_recom
            - user_id
            - item_id
            - y_pred
            - predicted_rank

        true_recom
            - user_id
            - item_id
            - y_true
            - optimal_rank
    """

    result = {}
    k = 10

    # 適合度がわかっているもの以外は除外
    # pred_y, predicted_rankはnullがあり得る
    y = true_recom.merge(predict_recom, on=["user_id", "item_id"], how="left")

    #  rmse
    eval_df = y.dropna(subset=['y_pred'])
    result["rmse"] = rmse(eval_df["y_pred"], eval_df['y_true'])

    # recall
    y_true_dict = y[y["y_true"] > 3].groupby("user_id")["item_id"].apply(list).to_dict()
    y_pred_dict = y[y["predicted_rank"] <= k].groupby("user_id")["item_id"].apply(list).to_dict()
    result['recall@10'] = recall(y_pred_dict, y_true_dict)

    # nDCG
    key = pd.concat([true_recom[["user_id", "item_id"]], predict_recom[["user_id", "item_id"]]]).drop_duplicates()
    y = key.merge(predict_recom, on=["user_id", "item_id"], how="left")
    y = y.merge(true_recom, on=["user_id", "item_id"], how="left")

    score = 0.0
    count = 0
    for user_id in y["user_id"].unique():
        ranking = y[y["user_id"] == user_id].sort_values(
            by="y_pred", ascending=False
        )
        # scoreが不明のケースによって長さがk未満になった場合に平均の対象から除外する
        if len(ranking.dropna(subset=['y_pred'])) < k:
            continue

        ranking = ranking["y_true"].values
        ranking = ranking[~np.isnan(ranking)]

        ndcg_score = _ndcg(ranking, k)
        score += ndcg_score
        count += 1
    result['ndcg@10']  = score / count

    return result

if __name__ == "__main__":
    # load data
    y_pred = [1, 2]
    y_true = [1, 3]
    print(rmse(y_pred, y_true))

    y_pred = {1: [1, 2], 2: [3, 4]}
    y_true = {1: [1, 2], 2: [5, 6]}
    print(recall(y_pred, y_true))

    pred = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            "item_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "rank": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )

    true = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            "item_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "score": [3, 3, 3, 3, 3, 0, 0, 0, 0, 5, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
        }
    )

    print(ndcg(pred, true))
