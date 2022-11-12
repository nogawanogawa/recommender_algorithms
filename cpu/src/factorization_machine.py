import dataloader
import preprpcessor
import pandas as pd
import numpy as np
from tqdm import trange
from sklearn.base import BaseEstimator, ClassifierMixin
import scipy

import random
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os


class FactorizationMachineRegressor(nn.Module):
    """ Factorization Machine """

    def __init__(self,
                 n: int = None,
                 k: int = None):
        """
        Args:
            n (int, optional): weight matrixのサイズ
            k (int, optional): 学習データの行数
        """
        super().__init__()
        # Initially we fill V with random values sampled from Gaussian distribution
        # NB: use nn.Parameter to compute gradients
        self.V = nn.Parameter(torch.randn(n, k), requires_grad=True)
        self.lin = nn.Linear(n, 1)

    def forward(self, x):
        out_1 = torch.matmul(x, self.V).pow(2).sum(1, keepdim=True)  # S_1^2
        out_2 = torch.matmul(x.pow(2), self.V.pow(2)).sum(
            1, keepdim=True)  # S_2

        out_inter = 0.5*(out_1 - out_2)
        out_lin = self.lin(x)
        out = out_inter + out_lin

        return out


# convert to 32-bit numbers to send to GPU
X_train = train_df.values.astype(np.float32)
X_test = test_df.values.astype(np.float32)


# for reproducibility
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything()

# main training function


def train_mlp(X, X_test, y, folds, model_class=None, model_params=None, batch_size=128, epochs=1,
              criterion=None, optimizer_class=None, opt_params=None,
              #               clr=cyclical_lr(10000),
              device=None):

    seed_everything()
    models = []
    scores = []
    train_preds = np.zeros(y.shape)
    test_preds = np.zeros((X_test.shape[0], 1))

    X_tensor, X_test, y_tensor = torch.from_numpy(X).to(device), torch.from_numpy(
        X_test).to(device), torch.from_numpy(y).to(device)
    for n_fold, (train_ind, valid_ind) in enumerate(folds.split(X, y)):

        print(f'fold {n_fold+1}')

        train_set = TensorDataset(X_tensor[train_ind], y_tensor[train_ind])
        valid_set = TensorDataset(X_tensor[valid_ind], y_tensor[valid_ind])

        loaders = {'train': DataLoader(train_set, batch_size=batch_size, shuffle=True),
                   'valid': DataLoader(valid_set, batch_size=batch_size, shuffle=False)}

        model = model_class(**model_params)
        model.to(device)
        best_model_wts = copy.deepcopy(model.state_dict())

        optimizer = optimizer_class(model.parameters(), **opt_params)
#         scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])

        # training cycle
        best_score = 0.
        for epoch in range(epochs):
            losses = {'train': 0., 'valid': 0}

            for phase in ['train', 'valid']:

                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                for batch_x, batch_y in loaders[phase]:
                    optimizer.zero_grad()
                    out = model(batch_x)
                    loss = criterion(out, batch_y)
                    losses[phase] += loss.item()*batch_x.size(0)

                    with torch.set_grad_enabled(phase == 'train'):
                        if phase == 'train':
                            loss.backward()
#                             scheduler.step()
                            optimizer.step()

                losses[phase] /= len(loaders[phase].dataset)

            # after each epoch check if we improved roc auc and if yes - save model
            with torch.no_grad():
                model.eval()
                valid_preds = sigmoid(model(X_tensor[valid_ind]).cpu().numpy())
                epoch_score = roc_auc_score(y[valid_ind], valid_preds)
                if epoch_score > best_score:
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_score = epoch_score

            if ((epoch+1) % 30) == 0:
                print(
                    f'epoch {epoch+1} train loss: {losses["train"]:.3f} valid loss {losses["valid"]:.3f} valid roc auc {epoch_score:.3f}')

        # prediction on valid set
        with torch.no_grad():
            model.load_state_dict(best_model_wts)
            model.eval()

            train_preds[valid_ind] = sigmoid(
                model(X_tensor[valid_ind]).cpu().numpy())
            fold_score = roc_auc_score(y[valid_ind], train_preds[valid_ind])
            scores.append(fold_score)
            print(f'Best ROC AUC score {fold_score}')
            models.append(model)

            test_preds += sigmoid(model(X_test).cpu().numpy())

    print('CV AUC ROC', np.mean(scores), np.std(scores))

    test_preds /= folds.n_splits

    return models, train_preds, test_preds


folds = KFold(n_splits=5, random_state=17)


MS, train_preds, test_preds = train_mlp(X_train, X_test, y, folds,
                                        model_class=TorchFM,
                                        model_params={
                                            'n': X_train.shape[1], 'k': 5},
                                        batch_size=1024,
                                        epochs=300,
                                        criterion=nn.BCEWithLogitsLoss(),
                                        optimizer_class=torch.optim.SGD,
                                        opt_params={
                                            'lr': 0.01, 'momentum': 0.9},
                                        device=DEVICE
                                        )

if __name__ == '__main__':

    df = dataloader.data()
    user = dataloader.user()
    item = dataloader.item()

    # load train data
    train_df = pd.read_csv(
        '../input/dota-heroes-binary/dota_train_binary_heroes.csv', index_col='match_id_hash')
    test_df = pd.read_csv(
        '../input/dota-heroes-binary/dota_train_binary_heroes.csv', index_col='match_id_hash')
    target = pd.read_csv(
        '../input/dota-heroes-binary/train_targets.csv', index_col='match_id_hash')
    y = target['radiant_win'].values.astype(np.float32)
    y = y.reshape(-1, 1)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 各dataframeからtraining用データを作成
    df = preprpcessor.preprocess(df, user, item)

    fm = FactorizationMachineClassifier(n_iter=10, learning_rate=0.001)
    fm.fit(scipy.sparse.csr_matrix(
        df.drop(['rating'], axis=1).values), df['rating'].values)

    y_pred_prob = fm.predict_proba(
        scipy.sparse.csr_matrix(df.drop(['rating'], axis=1).values))

    print(y_pred_prob)
