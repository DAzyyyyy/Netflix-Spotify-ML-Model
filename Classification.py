# =============================================================================
# Capstone Project: Music Genre Classification
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

# ── 2. Load data
df = pd.read_csv('/Users/dazydai/Downloads/musicData.csv')
print(f"Raw data shape: {df.shape}")

# ── 3. Data cleaning

# Drop rows where music_genre is missing
df = df.dropna(subset=['music_genre'])
print(f"After dropping missing genre: {df.shape}")

# Drop non-feature columns
drop_cols = ['instance_id', 'artist_name', 'track_name', 'obtained_date']
df = df.drop(columns=drop_cols)

# tempo column contains '?' sentinel values -> convert to NaN
df['tempo'] = pd.to_numeric(df['tempo'], errors='coerce')

# key: string -> semitone integer (0-11)
key_map = {'C':0,'C#':1,'D':2,'D#':3,'E':4,'F':5,
           'F#':6,'G':7,'G#':8,'A':9,'A#':10,'B':11}
df['key'] = df['key'].map(key_map)

# mode: categorical -> dummy variables (will NOT be standardized)
df['mode_Major'] = (df['mode'] == 'Major').astype(int)
df['mode_Minor'] = (df['mode'] == 'Minor').astype(int)
df = df.drop(columns=['mode'])

# Encode target variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['genre_label'] = le.fit_transform(df['music_genre'])
genres = le.classes_
print(f"Genres: {genres}")

# Impute missing numeric values with median
dummy_cols = ['mode_Major', 'mode_Minor']
feature_cols = [c for c in df.columns if c not in ['music_genre', 'genre_label']]
num_cols = [c for c in feature_cols if c not in dummy_cols]

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
df[num_cols] = imputer.fit_transform(df[num_cols])
print(f"Remaining missing values: {df.isnull().sum().sum()}")

# ── 4. Train / test split (500 test per genre, 4500 train per genre)
train_idx, test_idx = [], []
for genre in genres:
    g = df[df['music_genre'] == genre]
    test_s  = g.sample(n=500, random_state=14732891)
    train_s = g.drop(test_s.index)
    test_idx.extend(test_s.index.tolist())
    train_idx.extend(train_s.index.tolist())

train_df = df.loc[train_idx]
test_df  = df.loc[test_idx]

X_train = train_df[feature_cols].values.astype(float)
X_test  = test_df[feature_cols].values.astype(float)
y_train = train_df['genre_label'].values
y_test  = test_df['genre_label'].values
print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# ── 5. Standardize (numeric columns only, not dummy variables)
from sklearn.preprocessing import StandardScaler
dummy_idx = [feature_cols.index(c) for c in dummy_cols]
scale_idx = [i for i in range(len(feature_cols)) if i not in dummy_idx]

scaler = StandardScaler()
X_train_s = X_train.copy()
X_test_s  = X_test.copy()
X_train_s[:, scale_idx] = scaler.fit_transform(X_train[:, scale_idx])
X_test_s[:, scale_idx]  = scaler.transform(X_test[:, scale_idx])

# ── 6. Dimensionality reduction: PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=10, random_state=14732891)
X_tr_pca = pca.fit_transform(X_train_s)
X_te_pca = pca.transform(X_test_s)
print(f"PCA (10 components) cumulative explained variance: {pca.explained_variance_ratio_.sum():.3f}")

# ── 7. Classification: Random Forest + Extra Trees soft-voting ensemble
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.preprocessing import label_binarize

print("Training Random Forest (200 trees)...")
rf = RandomForestClassifier(
    n_estimators=200, max_depth=None,
    min_samples_split=4, class_weight='balanced',
    random_state=14732891, n_jobs=-1
)
rf.fit(X_tr_pca, y_train)

print("Training Extra Trees (200 trees)...")
et = ExtraTreesClassifier(
    n_estimators=200, max_depth=None,
    class_weight='balanced',
    random_state=14732891, n_jobs=-1
)
et.fit(X_tr_pca, y_train)

# Average predicted probabilities from both models
rf_prob = rf.predict_proba(X_te_pca)
et_prob = et.predict_proba(X_te_pca)
y_prob  = (rf_prob + et_prob) / 2
y_pred  = np.argmax(y_prob, axis=1)

# ── 8. Evaluation
y_test_bin = label_binarize(y_test, classes=list(range(len(genres))))
auc_macro  = roc_auc_score(y_test_bin, y_prob, multi_class='ovr', average='macro')
acc        = accuracy_score(y_test, y_pred)

print(f"\n{'='*50}")
print(f"  AUC (macro OvR): {auc_macro:.4f}")
print(f"  Accuracy:        {acc:.4f}")
print(f"{'='*50}")
print(classification_report(y_test, y_pred, target_names=genres))

# ── 9. Plots
from sklearn.metrics import roc_curve

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

colors = ['#e6194b','#3cb44b','#4363d8','#f58231','#911eb4',
          '#42d4f4','#f032e6','#9a6324','#808000','#469990']

# Plot 1: ROC curves per genre
ax = axes[0]
for i, (gname, color) in enumerate(zip(genres, colors)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    auc_i = roc_auc_score(y_test_bin[:, i], y_prob[:, i])
    ax.plot(fpr, tpr, color=color, lw=1.5, label=f'{gname} (AUC={auc_i:.3f})')
ax.plot([0,1],[0,1], 'k--', lw=1)
ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title(f'ROC Curves — Macro AUC = {auc_macro:.4f}', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=8)
ax.grid(alpha=0.3)

# Plot 2: PCA 2D cluster visualization
ax2 = axes[1]
pca2 = PCA(n_components=2, random_state=14732891)
X_2d = pca2.fit_transform(X_train_s)
for i, (gname, color) in enumerate(zip(genres, colors)):
    mask = y_train == i
    ax2.scatter(X_2d[mask,0], X_2d[mask,1], s=2, alpha=0.25, color=color, label=gname)
ax2.set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
ax2.set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
ax2.set_title('Genre Clusters in PCA-reduced Space', fontsize=13, fontweight='bold')
ax2.legend(loc='upper right', fontsize=8, markerscale=4)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/dazydai/Downloads/music_classification_results.png', dpi=150, bbox_inches='tight')
print("Plot saved: music_classification_results.png")

# ── 10. Feature importance (Extra Trees on raw features)
et_raw = ExtraTreesClassifier(n_estimators=100, random_state=14732891, n_jobs=-1)
et_raw.fit(X_train_s, y_train)
feat_imp = pd.Series(et_raw.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature importances (Extra Trees on raw features):")
print(feat_imp.to_string())