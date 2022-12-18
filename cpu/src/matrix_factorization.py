# ref: https://albertauyeung.github.io/2017/04/23/python-matrix-factorization.html/
import numpy as np
import pandas as pd
from util.dataloader import Data
from typing import Tuple, List
from eval.eval_func import *

class MF:
    def __init__(self, R: pd.DataFrame, K: int, alpha: float, beta: float, iterations: int):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """
        self.df = R.pivot_table(index="user_id", columns="item_id", values="rating")
        self.R = self.df.fillna(0).values # np.ndarrayに直す
        self.num_users, self.num_items = self.R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def train(self) -> None:
        # user latent feature matrix
        self.P = np.random.normal(scale=1.0 / self.K, size=(self.num_users, self.K))
        # item latent feature matrix
        self.Q = np.random.normal(scale=1.0 / self.K, size=(self.num_items, self.K))

        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        # Create a list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]

        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            rmse = self.rmse()
            training_process.append((i, rmse))
            if (i + 1) % 10 == 0:
                print("Iteration: %d ; error = %.4f" % (i + 1, rmse))

        pred_df = pd.DataFrame(self.full_matrix())
        # trainに値がある箇所以外はrandomな値でflllする
        self.model = self.df.where(~self.df.isna(), pred_df)
        self.model = pd.melt(self.model.reset_index(), id_vars="user_id", value_vars=self.model.columns)
        self.model = self.model.set_index(["user_id", "item_id"])

        return

    def rmse(self):
        """
        total root mean square error
        """
        xs, ys = self.R.nonzero()  # 欠損値は計算対象から外れている
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error / len(xs))

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = r - prediction

            # Update biases
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i, :])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j, :])

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        prediction = (
            self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        )
        return prediction

    def full_matrix(self) -> np.ndarray:
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        return (
            self.b
            + self.b_u[:, np.newaxis]
            + self.b_i[
                np.newaxis :,
            ]
            + self.P.dot(self.Q.T)
        )

    def predict(self, test_df:pd.DataFrame) -> List[float]:
        """ あたえられたdfのすべてのratingを返す """
        return [self.predict_score(user_id, item_id) for user_id, item_id in test_df.values]


    def predict_score(self, user_id:int, item_id:int) -> float:
        """ user_id, item_id毎の推論したratingを返す """
        try:
            return self.model.loc[(user_id, item_id)]["value"]
        except:
            return np.nan

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return Data().load() # user_df, item_df, train_df, test_df


if __name__ == "__main__":
    user_df, item_df, train_df, test_df = load_data()

    mf = MF(R=train_df, K=128, alpha=0.1, beta=0.1, iterations=50)
    mf.train()

    # predict
    pred_df = test_df.copy()
    pred_df["y_pred"] = mf.predict(test_df[["user_id", "item_id"]])
    pred_df = pred_df.dropna(subset=['y_pred'])

    pred_df["predicted_rank"] = pred_df.groupby(["user_id"])["y_pred"].rank(ascending=False, method='first')
    pred_df = pred_df[["user_id", "item_id", "y_pred", "predicted_rank"]]

    true_df = test_df.copy()
    true_df = true_df[["user_id", "item_id", "rating"]].rename(columns={"rating":"y_true"})
    true_df["optimal_rank"] = true_df.groupby(["user_id"])["y_true"].rank(ascending=False, method='first')
    true_df = true_df[["user_id", "item_id", "y_true", "optimal_rank"]]

    result = eval(predict_recom=pred_df, true_recom=true_df)
    print(result)

