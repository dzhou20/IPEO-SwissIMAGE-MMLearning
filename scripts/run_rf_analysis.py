import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from config import CSV_PATH, IMAGE_DIR
from sweco_group_of_variables import sweco_variables_dict

def run_rf_analysis():
    print("1. Loading data...")
    df = pd.read_csv(CSV_PATH)
    df["img_path"] = df["id"].apply(lambda x: str(Path(IMAGE_DIR) / f"{x}.tif"))
    df["img_exists"] = df["img_path"].apply(lambda p: Path(p).exists())
    df = df[df["img_exists"]].reset_index(drop=True)
    
    train_df = df[df["split"] == "train"]
    y = train_df["EUNIS_cls"].values
    
    print(f"Training samples: {len(train_df)}")

    all_unique_features = set()
    feature_to_groups = {}

    for group_name, cols in sweco_variables_dict.items():
        for col in cols:
            if col in train_df.columns:
                actual_col = col
            else:
                matches = [c for c in train_df.columns if c.startswith(f"{col}.")]
                if matches:
                    actual_col = matches[0]
                else:
                    print(f"[Warn] Column {col} not found in CSV, skipping.")
                    continue
            if actual_col:
                all_unique_features.add(actual_col)
                if actual_col not in feature_to_groups:
                    feature_to_groups[actual_col] = []
                feature_to_groups[actual_col].append(group_name)

    all_features = sorted(list(all_unique_features))

    X = train_df[all_features].fillna(0).values
    
    print(f"Features count: {len(all_features)}")
    print("Variable groups found:", list(sweco_variables_dict.keys()))

    print("3. Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    acc = rf.score(X, y)
    print(f"RF Training Accuracy: {acc:.4f}")

    importances = rf.feature_importances_
    
    group_importance = {g: 0.0 for g in sweco_variables_dict.keys()}
    
    feature_imp_df = pd.DataFrame({
        'feature': all_features,
        'importance': importances,
        'group': [", ".join(feature_to_groups[f]) for f in all_features]
    }).sort_values('importance', ascending=False)
    
    print("\n=== Top 10 Single Variables ===")
    print(feature_imp_df.head(10))

    for idx, feat_name in enumerate(all_features):
        groups_containing_feat = feature_to_groups[feat_name]
        for g in groups_containing_feat:
            group_importance[g] += importances[idx]

    group_series = pd.Series(group_importance).sort_values(ascending=False)
    
    print("\n=== Variable Group Importance (Aggregated) ===")
    print(group_series)

    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(group_series)))
    group_series.plot(kind='bar', color=colors)
    plt.title("Importance of Variable Groups (based on Random Forest)")
    plt.ylabel("Total Importance Score")
    plt.xlabel("Variable Group")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("group_importance.png")
    plt.show()
    print("Plot saved to group_importance.png")

if __name__ == "__main__":
    run_rf_analysis()