import lightgbm as lgb
import sklearn.preprocessing as sp
from dtreeviz.trees import *
import mlflow
from sklearn.model_selection import train_test_split
import dataloader, preprpcessor


class LGBMRegressor:
    def __init__(self):
        pass

    def train(self, X_train, X_valid, y_train, y_valid):
        mlflow.lightgbm.autolog()

        train_data = lgb.Dataset(X_train, label=y_train)
        validation_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

        params = {
            'objective': 'regression',      
            'metric': "rmse",               
            'learning_rate': 0.05,
            'num_leaves': 21,               
            'min_data_in_leaf': 3,    
            'early_stopping':100,    
            'num_iteration': 1000           
            }

        self.model = lgb.train(params,
               train_set=train_data,
               valid_sets=validation_data,
               verbose_eval=50)
        
        viz = dtreeviz(self.model,
                    x_data = X_valid,
                    y_data = y_valid,
                    target_name = 'rating',
                    feature_names = X_train.columns.tolist(),
                    tree_index = 0)

        filename = 'lgb_tree.svg'
        viz.save(filename)
        mlflow.log_artifact(filename)
        
    def predict(self):
        pass

if __name__ == '__main__':
    df = dataloader.data()
    user = dataloader.user()
    item = dataloader.item()

    df = preprpcessor.preprocess(df, user, item)

    y = df['rating']
    X = df.drop(['rating'], axis=1)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=1,
                                                    stratify=y)

    lgb_regressor = LGBMRegressor()
    lgb_regressor.train(X_train, X_valid, y_train, y_valid )
    