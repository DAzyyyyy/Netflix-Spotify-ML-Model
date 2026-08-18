# =============================================================================
# Capstone Project: Netflix Movie Rating Regression
# N-Number: 14732891
# =============================================================================

import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── 1. Set random seed
random.seed(14732891)
np.random.seed(14732891)

# ── 2. Parse data.txt
# Format:
#   MovieID:
#   UserID,Rating,Date
#   ...

print("Parsing data.txt (this may take a minute)...")

DATA_PATH   = '/Users/dazydai/Downloads/data.txt'
TITLES_PATH = '/Users/dazydai/Downloads/movieTitles.csv'

records = []
current_movie = None

with open(DATA_PATH, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.endswith(':'):
            current_movie = int(line[:-1])
        else:
            parts = line.split(',')
            if len(parts) == 3:
                user_id = int(parts[0])
                rating  = int(parts[1])
                date    = parts[2].strip()
                records.append((current_movie, user_id, rating, date))

df = pd.DataFrame(records, columns=['movie_id', 'user_id', 'rating', 'date'])
print(f"Total ratings loaded: {len(df):,}")
print(f"Movies: {df['movie_id'].nunique()},  Users: {df['user_id'].nunique()}")

# ── 3. Load movie titles
titles = pd.read_csv(TITLES_PATH, header=None, encoding='latin-1',
                     usecols=[0, 1, 2], names=['movie_id', 'release_year', 'title'])

# ── 4. Parse dates and merge release year
df['date']         = pd.to_datetime(df['date'], errors='coerce')
df['rating_year']  = df['date'].dt.year
df['rating_month'] = df['date'].dt.month

df = df.merge(titles[['movie_id', 'release_year']], on='movie_id', how='left')
df['years_since_release'] = df['rating_year'] - df['release_year']

# ── 5. Train / test split
# For each movie: randomly pick 1 rating as test, rest as train
print("Splitting train / test sets")

test_idx  = []
train_idx = []

for movie_id, group in df.groupby('movie_id'):
    sampled = group.sample(n=1, random_state=14732891)
    test_idx.extend(sampled.index.tolist())
    train_idx.extend(group.drop(sampled.index).index.tolist())

train_df = df.loc[train_idx].copy()
test_df  = df.loc[test_idx].copy()
print(f"Train: {len(train_df):,} ratings,  Test: {len(test_df):,} ratings")

# ── 6. Compute aggregate features on training set only (no leakage)
global_mean = train_df['rating'].mean()
print(f"Global mean rating: {global_mean:.4f}")

movie_stats = train_df.groupby('movie_id')['rating'].agg(
    movie_mean='mean',
    movie_std='std',
    movie_count='count'
).reset_index()
movie_stats['movie_std'] = movie_stats['movie_std'].fillna(0)

user_stats = train_df.groupby('user_id')['rating'].agg(
    user_mean='mean',
    user_std='std',
    user_count='count'
).reset_index()
user_stats['user_std'] = user_stats['user_std'].fillna(0)

def add_features(data, movie_stats, user_stats, global_mean):
    data = data.merge(movie_stats, on='movie_id', how='left')
    data = data.merge(user_stats,  on='user_id',  how='left')
    data['movie_mean']          = data['movie_mean'].fillna(global_mean)
    data['movie_std']           = data['movie_std'].fillna(0)
    data['movie_count']         = data['movie_count'].fillna(0)
    data['user_mean']           = data['user_mean'].fillna(global_mean)
    data['user_std']            = data['user_std'].fillna(0)
    data['user_count']          = data['user_count'].fillna(0)
    data['release_year']        = data['release_year'].fillna(data['release_year'].median())
    data['years_since_release'] = data['years_since_release'].fillna(
                                      data['years_since_release'].median())
    data['rating_year']         = data['rating_year'].fillna(data['rating_year'].median())
    data['rating_month']        = data['rating_month'].fillna(6)
    return data

train_df = add_features(train_df, movie_stats, user_stats, global_mean)
test_df  = add_features(test_df,  movie_stats, user_stats, global_mean)

FEATURES = ['movie_mean', 'movie_std', 'movie_count',
            'user_mean',  'user_std',  'user_count',
            'release_year', 'years_since_release',
            'rating_year', 'rating_month']

X_train = train_df[FEATURES].values
y_train = train_df['rating'].values.astype(float)
X_test  = test_df[FEATURES].values
y_test  = test_df['rating'].values.astype(float)
print(f"Feature matrix — Train: {X_train.shape},  Test: {X_test.shape}")

# ── 7. Model: HistGradientBoosting Regressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

print("Training HistGradientBoosting Regressor...")
model = HistGradientBoostingRegressor(
    max_iter=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=14732891
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\n{'='*50}")
print(f"  Final RMSE: {rmse:.4f}")
print(f"{'='*50}")

baseline_rmse = np.sqrt(mean_squared_error(y_test, np.full_like(y_test, global_mean)))
print(f"  Baseline RMSE (global mean): {baseline_rmse:.4f}")

# ── 8. Feature importance
import inspect
if hasattr(model, 'feature_importances_'):
    feat_imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
else:
    # HistGradientBoosting: use permutation importance instead
    from sklearn.inspection import permutation_importance
    result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=14732891)
    feat_imp = pd.Series(result.importances_mean, index=FEATURES).sort_values(ascending=False)

print("\nFeature importances:")
print(feat_imp.to_string())

# ── 9. Plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Predicted vs Actual
ax = axes[0]
jitter = np.random.uniform(-0.15, 0.15, size=len(y_test))
ax.scatter(y_test + jitter, y_pred, alpha=0.3, s=8, color='steelblue')
ax.plot([1, 5], [1, 5], 'r--', lw=1.5, label='Perfect prediction')
ax.set_xlabel('Actual Rating', fontsize=12)
ax.set_ylabel('Predicted Rating', fontsize=12)
ax.set_title(f'Predicted vs Actual  (RMSE = {rmse:.4f})', fontsize=13, fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)

# Plot 2: Feature importance
ax2 = axes[1]
feat_imp.plot(kind='barh', ax=ax2, color='steelblue', edgecolor='white')
ax2.invert_yaxis()
ax2.set_xlabel('Importance', fontsize=12)
ax2.set_title('Feature Importances (HistGradientBoosting)', fontsize=13, fontweight='bold')
ax2.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('/Users/dazydai/Downloads/netflix_regression_results.png')
print("\nPlot saved: netflix_regression_results.png")