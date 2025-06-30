# Recommender Algorithm API Documentation

## Overview

This documentation provides comprehensive coverage of all public APIs, functions, and components in the Recommender Algorithm project. The project implements various recommendation algorithms including collaborative filtering, matrix factorization, factorization machines, and baseline methods.

## Table of Contents

1. [Setup and Installation](#setup-and-installation)
2. [Data Loading and Preprocessing](#data-loading-and-preprocessing)
3. [Recommendation Algorithms](#recommendation-algorithms)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Utilities](#utilities)
6. [Usage Examples](#usage-examples)

---

## Setup and Installation

### Prerequisites

The project uses Docker for containerization and Poetry for dependency management.

### Setup Commands

```bash
# Run setup script to download datasets
./setup.sh

# Run algorithms using Docker
docker-compose run sandbox python src/<algorithm>.py
```

---

## Data Loading and Preprocessing

### `util.dataloader.Data`

**Description**: Main data loading class for the MovieLens-100k dataset.

#### Constructor

```python
Data()
```

Creates a new Data instance with home directory set to `/home`.

#### Methods

##### `user() -> pd.DataFrame`

**Description**: Loads user demographic data.

**Returns**: DataFrame with columns: `['user_id', 'age', 'gender', 'occupation', 'zip_code']`

**Example**:
```python
from util.dataloader import Data
dl = Data()
user_df = dl.user()
print(user_df.head())
```

##### `item() -> pd.DataFrame`

**Description**: Loads movie item data with genres.

**Returns**: DataFrame with movie information and genre features.

**Example**:
```python
item_df = dl.item()
print(item_df.columns)
```

##### `genre() -> pd.Series`

**Description**: Loads available movie genres.

**Returns**: Series of genre names sorted by ID.

##### `data(is_test: bool = False) -> pd.DataFrame`

**Description**: Loads rating data.

**Parameters**:
- `is_test` (bool): If True, loads test set; otherwise loads training set

**Returns**: DataFrame with columns: `['user_id', 'item_id', 'rating', 'timestamp']`

**Example**:
```python
train_df = dl.data(is_test=False)
test_df = dl.data(is_test=True)
```

##### `load() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]`

**Description**: Loads all datasets at once.

**Returns**: Tuple of (user_df, item_df, train_df, test_df)

**Example**:
```python
user_df, item_df, train_df, test_df = dl.load()
```

### `util.preprpcessor.Preprocessor`

**Description**: Data preprocessing utilities for feature engineering.

#### Constructor

```python
Preprocessor()
```

#### Methods

##### `fit(df: pd.DataFrame, user: pd.DataFrame, item: pd.DataFrame) -> pd.DataFrame`

**Description**: Fits encoders and transforms the data.

**Parameters**:
- `df`: Rating dataframe
- `user`: User dataframe  
- `item`: Item dataframe

**Returns**: Transformed dataframe ready for machine learning

##### `trenasform(df: pd.DataFrame, user: pd.DataFrame, item: pd.DataFrame) -> pd.DataFrame`

**Description**: Transforms data using fitted encoders.

**Example**:
```python
from util.preprpcessor import Preprocessor
preprocessor = Preprocessor()
train_processed = preprocessor.fit(train_df, user_df, item_df)
test_processed = preprocessor.trenasform(test_df, user_df, item_df)
```

---

## Recommendation Algorithms

### Collaborative Filtering

#### `cpu.src.collaborative_filtering.CF`

**Description**: User-based collaborative filtering using Pearson correlation.

##### Constructor

```python
CF(rating: pd.DataFrame)
```

**Parameters**:
- `rating`: DataFrame with columns ['user_id', 'item_id', 'rating']

##### Methods

###### `train() -> None`

**Description**: Trains the collaborative filtering model.

**Example**:
```python
from cpu.src.collaborative_filtering import CF
cf = CF(train_df)
cf.train()
```

###### `predict(test_df: pd.DataFrame) -> List[float]`

**Description**: Predicts ratings for user-item pairs.

**Parameters**:
- `test_df`: DataFrame with ['user_id', 'item_id'] columns

**Returns**: List of predicted ratings

###### `predict_score(user_id: int, item_id: int) -> float`

**Description**: Predicts rating for a single user-item pair.

**Parameters**:
- `user_id`: User identifier
- `item_id`: Item identifier

**Returns**: Predicted rating or NaN if unavailable

### Matrix Factorization

#### `cpu.src.matrix_factorization.MF`

**Description**: Matrix factorization using stochastic gradient descent.

##### Constructor

```python
MF(R: pd.DataFrame, K: int, alpha: float, beta: float, iterations: int)
```

**Parameters**:
- `R`: Rating matrix
- `K`: Number of latent dimensions
- `alpha`: Learning rate
- `beta`: Regularization parameter
- `iterations`: Number of training iterations

##### Methods

###### `train() -> None`

**Description**: Trains the matrix factorization model using SGD.

**Example**:
```python
from cpu.src.matrix_factorization import MF
mf = MF(R=train_df, K=128, alpha=0.1, beta=0.1, iterations=50)
mf.train()
```

###### `predict(test_df: pd.DataFrame) -> List[float]`

**Description**: Predicts ratings for test set.

###### `get_rating(i: int, j: int) -> float`

**Description**: Gets predicted rating using learned factors.

**Parameters**:
- `i`: User index
- `j`: Item index

**Returns**: Predicted rating

### Factorization Machine

#### `cpu.src.fm.FactorizationMachine`

**Description**: Deep learning implementation of Factorization Machines using PyTorch Lightning.

##### Constructor

```python
FactorizationMachine()
```

##### Methods

###### `train(train_df: pd.DataFrame) -> None`

**Description**: Trains the factorization machine model.

**Parameters**:
- `train_df`: Preprocessed training dataframe

**Example**:
```python
from cpu.src.fm import FactorizationMachine
fm = FactorizationMachine()
fm.train(train_df)
```

###### `predict(test_df: pd.DataFrame) -> List[float]`

**Description**: Predicts ratings using trained model.

### EASE (Embarrassingly Shallow AutoEncoders)

#### `cpu.src.ease.EASE`

**Description**: Linear autoencoder for collaborative filtering.

##### Constructor

```python
EASE(user_df: pd.DataFrame, item_df: pd.DataFrame)
```

**Parameters**:
- `user_df`: User dataframe for label encoding
- `item_df`: Item dataframe for label encoding

##### Methods

###### `train(df: pd.DataFrame, lambda_: float = 0.5, implicit: bool = True) -> None`

**Description**: Trains the EASE model.

**Parameters**:
- `df`: Rating dataframe
- `lambda_`: L2 regularization parameter
- `implicit`: Whether to treat ratings as implicit feedback

**Example**:
```python
from cpu.src.ease import EASE
ease = EASE(user_df, item_df)
ease.train(train_df, lambda_=2000)
```

###### `predict(test_df: pd.DataFrame) -> pd.Series`

**Description**: Generates predictions for test set.

### iALS (Implicit Alternating Least Squares)

#### `cpu.src.iALS.iALS`

**Description**: Implicit feedback matrix factorization using ALS.

##### Constructor

```python
iALS(user_df: pd.DataFrame, item_df: pd.DataFrame)
```

##### Methods

###### `train(train_df: pd.DataFrame) -> None`

**Description**: Trains the iALS model using implicit library.

**Example**:
```python
from cpu.src.iALS import iALS
ials = iALS(user_df, item_df)
ials.train(train_df)
```

### Baseline Methods

#### `cpu.src.average_baseline.AverageRecommender`

**Description**: Simple baseline using item average ratings.

##### Methods

###### `train(train_df: pd.DataFrame) -> None`

**Description**: Computes item average ratings.

###### `predict_score(user_id: int, item_id: int) -> float`

**Description**: Returns item average rating or training rating if available.

**Example**:
```python
from cpu.src.average_baseline import AverageRecommender
avg_rec = AverageRecommender()
avg_rec.train(train_df)
```

#### `cpu.src.random_baseline.RandomRecommender`

**Description**: Random baseline for comparison.

##### Methods

###### `train(train_df: pd.DataFrame) -> None`

**Description**: Fills missing values with random ratings.

### Two-Tower Model

#### `cpu.src.two_tower.TwoTower`

**Description**: Neural two-tower architecture for recommendations.

##### Constructor

```python
TwoTower(user_df: pd.DataFrame, item_df: pd.DataFrame)
```

#### `cpu.src.model.two_tower_model.UserTower`

**Description**: User tower neural network component.

##### Constructor

```python
UserTower(ln: List[int])
```

**Parameters**:
- `ln`: List of layer dimensions

#### `cpu.src.model.two_tower_model.ItemTower`

**Description**: Item tower neural network component.

##### Constructor

```python
ItemTower(n_items: int, ln: List[int])
```

---

## Evaluation Metrics

### `eval.eval_func`

**Description**: Comprehensive evaluation metrics for recommendation systems.

#### Functions

##### `rmse(y_pred: List[float], y_true: List[float]) -> float`

**Description**: Calculates Root Mean Square Error.

**Parameters**:
- `y_pred`: Predicted ratings
- `y_true`: True ratings

**Returns**: RMSE value

**Example**:
```python
from eval.eval_func import rmse
error = rmse([1, 2, 3], [1.1, 2.2, 2.9])
```

##### `recall(y_pred: Dict[int, List[int]], y_true: Dict[int, List[int]], k: int = 10) -> float`

**Description**: Calculates Recall@K for recommendation lists.

**Parameters**:
- `y_pred`: Predicted items per user
- `y_true`: True relevant items per user  
- `k`: Cutoff for top-k evaluation

**Returns**: Average recall@k across users

##### `ndcg(y_pred: pd.DataFrame, y_true: pd.DataFrame, k: int = 10) -> float`

**Description**: Calculates Normalized Discounted Cumulative Gain.

**Parameters**:
- `y_pred`: Predictions with ranking
- `y_true`: True ratings
- `k`: Cutoff for evaluation

**Returns**: Average nDCG@k

##### `eval(predict_recom: pd.DataFrame, true_recom: pd.DataFrame) -> Dict`

**Description**: Comprehensive evaluation returning multiple metrics.

**Parameters**:
- `predict_recom`: DataFrame with columns ['user_id', 'item_id', 'y_pred', 'predicted_rank']
- `true_recom`: DataFrame with columns ['user_id', 'item_id', 'y_true', 'optimal_rank']

**Returns**: Dictionary with RMSE, Recall@10, and nDCG@10

**Example**:
```python
from eval.eval_func import eval

# Prepare prediction dataframe
pred_df = test_df.copy()
pred_df['y_pred'] = model.predict(test_df[['user_id', 'item_id']])
pred_df['predicted_rank'] = pred_df.groupby('user_id')['y_pred'].rank(ascending=False)

# Prepare true dataframe  
true_df = test_df[['user_id', 'item_id', 'rating']].rename(columns={'rating': 'y_true'})
true_df['optimal_rank'] = true_df.groupby('user_id')['y_true'].rank(ascending=False)

# Evaluate
results = eval(predict_recom=pred_df, true_recom=true_df)
print(results)  # {'rmse': 0.95, 'recall@10': 0.12, 'ndcg@10': 0.08}
```

---

## Usage Examples

### Complete Workflow Example

```python
from util.dataloader import Data
from cpu.src.matrix_factorization import MF
from eval.eval_func import eval

# 1. Load data
user_df, item_df, train_df, test_df = Data().load()

# 2. Train model
mf = MF(R=train_df, K=128, alpha=0.1, beta=0.1, iterations=50)
mf.train()

# 3. Generate predictions
pred_df = test_df.copy()
pred_df['y_pred'] = mf.predict(test_df[['user_id', 'item_id']])
pred_df = pred_df.dropna(subset=['y_pred'])

# 4. Prepare evaluation format
pred_df['predicted_rank'] = pred_df.groupby('user_id')['y_pred'].rank(
    ascending=False, method='first'
)
pred_df = pred_df[['user_id', 'item_id', 'y_pred', 'predicted_rank']]

true_df = test_df.copy()
true_df = true_df[['user_id', 'item_id', 'rating']].rename(columns={'rating': 'y_true'})
true_df['optimal_rank'] = true_df.groupby('user_id')['y_true'].rank(
    ascending=False, method='first'  
)
true_df = true_df[['user_id', 'item_id', 'y_true', 'optimal_rank']]

# 5. Evaluate
results = eval(predict_recom=pred_df, true_recom=true_df)
print(f"RMSE: {results['rmse']:.3f}")
print(f"Recall@10: {results['recall@10']:.3f}")  
print(f"nDCG@10: {results['ndcg@10']:.3f}")
```

### Algorithm Comparison Example

```python
from util.dataloader import Data
from cpu.src.collaborative_filtering import CF
from cpu.src.matrix_factorization import MF
from cpu.src.ease import EASE
from eval.eval_func import eval

# Load data
user_df, item_df, train_df, test_df = Data().load()

algorithms = {
    'CF': CF(train_df),
    'MF': MF(R=train_df, K=64, alpha=0.1, beta=0.1, iterations=30),
    'EASE': EASE(user_df, item_df)
}

results = {}
for name, model in algorithms.items():
    # Train
    if name == 'EASE':
        model.train(train_df, lambda_=1000)
    else:
        model.train()
    
    # Predict and evaluate
    pred_df = test_df.copy()
    pred_df['y_pred'] = model.predict(test_df[['user_id', 'item_id']])
    pred_df = pred_df.dropna(subset=['y_pred'])
    
    # Format for evaluation
    pred_df['predicted_rank'] = pred_df.groupby('user_id')['y_pred'].rank(
        ascending=False, method='first'
    )
    
    true_df = test_df[['user_id', 'item_id', 'rating']].rename(columns={'rating': 'y_true'})
    true_df['optimal_rank'] = true_df.groupby('user_id')['y_true'].rank(ascending=False)
    
    results[name] = eval(predict_recom=pred_df, true_recom=true_df)

# Compare results
for name, metrics in results.items():
    print(f"{name}: RMSE={metrics['rmse']:.3f}, Recall@10={metrics['recall@10']:.3f}")
```

### Custom Evaluation Example

```python
from eval.eval_func import rmse, recall, ndcg
import pandas as pd

# Custom evaluation for specific metrics
y_pred = [4.2, 3.8, 4.5, 2.1]
y_true = [4.0, 4.0, 5.0, 2.0]

# RMSE
error = rmse(y_pred, y_true)
print(f"RMSE: {error:.3f}")

# Recall@5 with user-item lists
pred_items = {1: [101, 102, 103], 2: [201, 202, 203]}
true_items = {1: [101, 104, 105], 2: [201, 204, 205]}
recall_score = recall(pred_items, true_items, k=5)
print(f"Recall@5: {recall_score:.3f}")

# nDCG@10 with ranking data
pred_ranking = pd.DataFrame({
    'user_id': [1, 1, 1, 2, 2, 2],
    'item_id': [101, 102, 103, 201, 202, 203], 
    'rank': [1, 2, 3, 1, 2, 3]
})
true_ratings = pd.DataFrame({
    'user_id': [1, 1, 1, 2, 2, 2],
    'item_id': [101, 102, 103, 201, 202, 203],
    'score': [5, 4, 3, 4, 5, 2]
})
ndcg_score = ndcg(pred_ranking, true_ratings, k=10)
print(f"nDCG@10: {ndcg_score:.3f}")
```

---

## Error Handling and Best Practices

### Common Issues and Solutions

1. **Missing Values**: Always check for NaN values in predictions and handle appropriately
2. **Data Format**: Ensure DataFrames have required columns before calling evaluation functions
3. **Memory Usage**: For large datasets, consider batch processing in prediction methods
4. **Reproducibility**: Set random seeds for consistent results across runs

### Performance Tips

1. **Use appropriate batch sizes** for neural network models (FM, Two-Tower)
2. **Tune hyperparameters** (learning rates, regularization) for each algorithm
3. **Consider implicit vs explicit feedback** when choosing between algorithms
4. **Use early stopping** to prevent overfitting in iterative methods

---

## API Reference Summary

| Component | Main Classes | Key Methods |
|-----------|-------------|-------------|
| Data Loading | `Data` | `load()`, `user()`, `item()` |
| Preprocessing | `Preprocessor` | `fit()`, `trenasform()` |
| Collaborative Filtering | `CF` | `train()`, `predict()` |
| Matrix Factorization | `MF` | `train()`, `predict()` |
| Factorization Machine | `FactorizationMachine` | `train()`, `predict()` |
| EASE | `EASE` | `train()`, `predict()` |
| iALS | `iALS` | `train()`, `predict()` |
| Baselines | `AverageRecommender`, `RandomRecommender` | `train()`, `predict()` |
| Evaluation | Module functions | `rmse()`, `recall()`, `ndcg()`, `eval()` |

This documentation provides comprehensive coverage of all public APIs with practical examples and usage patterns for building recommendation systems.