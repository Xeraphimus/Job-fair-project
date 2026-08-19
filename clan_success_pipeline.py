import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42

PAYMENT_RANK = {
    "Negative": -1,
    "0) NonPayer": 0,
    "1) ExPayer": 1,
    "2) Minnow": 2,
    "3) Dolphin": 3,
    "4) Whale": 4,
}

AGG_STAT_COLS = [
    "avg_stars_top_11_players",
    "avg_stars_top_3_players",
    "avg_training_bonus",
    "clan_multiplier",
    "days_active_last_28_days",
    "days_active_last_7_days",
    "days_since_last_active",
    "training_count_last_28_days",
    "payment_rank",
]

RANK_STAT_COLS = [
    "avg_stars_top_11_players",
    "avg_stars_top_3_players",
    "avg_training_bonus",
    "clan_multiplier",
    "days_active_last_28_days",
    "days_active_last_7_days",
    "days_since_last_active",
    "training_count_last_28_days",
    "payment_rank",
]

def load_member_stats(path):
    df = pd.read_csv(path)
    df["payment_rank"] = df["dynamic_payment_segment"].map(PAYMENT_RANK)
    return df


def aggregate_clan_features(df):
    """Per-clan aggregate stats (mean/sum/min/max/std) across its 6 managers."""
    g = df.groupby("clan_id")
    feats = pd.DataFrame(index=g.size().index)
    feats["n_members"] = g.size()

    feats["sum_stars_top11"] = g["avg_stars_top_11_players"].sum()
    feats["mean_stars_top11"] = g["avg_stars_top_11_players"].mean()
    feats["min_stars_top11"] = g["avg_stars_top_11_players"].min()
    feats["max_stars_top11"] = g["avg_stars_top_11_players"].max()
    feats["std_stars_top11"] = g["avg_stars_top_11_players"].std().fillna(0)

    feats["sum_stars_top3"] = g["avg_stars_top_3_players"].sum()
    feats["mean_stars_top3"] = g["avg_stars_top_3_players"].mean()

    feats["sum_training_bonus"] = g["avg_training_bonus"].sum()
    feats["mean_training_bonus"] = g["avg_training_bonus"].mean()
    feats["min_training_bonus"] = g["avg_training_bonus"].min()

    feats["sum_multiplier"] = g["clan_multiplier"].sum()
    feats["mean_multiplier"] = g["clan_multiplier"].mean()
    feats["max_multiplier"] = g["clan_multiplier"].max()
    feats["min_multiplier"] = g["clan_multiplier"].min()

    tmp = df.copy()
    tmp["w_quality11"] = tmp["clan_multiplier"] * tmp["avg_stars_top_11_players"]
    tmp["w_quality3"] = tmp["clan_multiplier"] * tmp["avg_stars_top_3_players"]
    tmp["w_training"] = tmp["clan_multiplier"] * tmp["avg_training_bonus"]
    wq = tmp.groupby("clan_id")[["w_quality11", "w_quality3", "w_training"]].sum()
    feats["weighted_quality11"] = wq["w_quality11"]
    feats["weighted_quality3"] = wq["w_quality3"]
    feats["weighted_training"] = wq["w_training"]

    feats["mean_days_active_28"] = g["days_active_last_28_days"].mean()
    feats["min_days_active_28"] = g["days_active_last_28_days"].min()
    feats["mean_days_active_7"] = g["days_active_last_7_days"].mean()
    feats["min_days_active_7"] = g["days_active_last_7_days"].min()
    feats["mean_days_since_last_active"] = g["days_since_last_active"].mean()
    feats["max_days_since_last_active"] = g["days_since_last_active"].max()

    feats["sum_training_count_28"] = g["training_count_last_28_days"].sum()
    feats["mean_training_count_28"] = g["training_count_last_28_days"].mean()
    feats["min_training_count_28"] = g["training_count_last_28_days"].min()

    feats["mean_cohort_day"] = g["cohort_day"].mean()
    feats["min_cohort_day"] = g["cohort_day"].min()

    feats["frac_payer"] = g["is_payer_lifetime"].mean()
    feats["mean_payment_rank"] = g["payment_rank"].mean()
    feats["max_payment_rank"] = g["payment_rank"].max()

    return feats.reset_index()


def rank_based_clan_features(df):
    """Pivot manager stats by in-clan quality rank (r1=strongest ... r6=weakest),
    approximating the tournament's default best-vs-best pairing."""
    s = df.sort_values(["clan_id", "avg_stars_top_11_players"], ascending=[True, False]).copy()
    s["rank_in_clan"] = s.groupby("clan_id").cumcount() + 1
    pivot = s.pivot(index="clan_id", columns="rank_in_clan", values=RANK_STAT_COLS)
    pivot.columns = [f"{col}_r{rank}" for col, rank in pivot.columns]
    return pivot.reset_index()


def build_clan_features(member_stats_df):
    agg = aggregate_clan_features(member_stats_df)
    rnk = rank_based_clan_features(member_stats_df)
    feats = agg.merge(rnk, on="clan_id", how="left")
    return feats


def build_match_diff_features(matches_df, clan_feats, has_target=True):
    feature_cols = [c for c in clan_feats.columns if c != "clan_id"]
    df = matches_df.merge(clan_feats.add_prefix("c1_"), left_on="clan_1_id", right_on="c1_clan_id", how="left")
    df = df.merge(clan_feats.add_prefix("c2_"), left_on="clan_2_id", right_on="c2_clan_id", how="left")

    diff = {f"diff_{c}": df[f"c1_{c}"] - df[f"c2_{c}"] for c in feature_cols}
    diff_df = pd.DataFrame(diff)
    diff_df["clan_1_id"] = df["clan_1_id"]
    diff_df["clan_2_id"] = df["clan_2_id"]
    if has_target:
        diff_df["target"] = (df["clan_winner"] == 1).astype(int)
    return diff_df, feature_cols


def augment_mirror(X, y):
    """Double the dataset by mirroring clan_1/clan_2 (negate diffs, flip label)."""
    return np.vstack([X, -X]), np.concatenate([y, 1 - y])

class EnsembleModel:
    """Simple average-probability ensemble of Logistic Regression + HistGBM."""

    def __init__(self):
        self.lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.3))
        self.hgb = HistGradientBoostingClassifier(
            max_iter=300, max_depth=3, learning_rate=0.03,
            l2_regularization=1.0, random_state=RANDOM_STATE,
        )

    def fit(self, X, y):
        self.lr.fit(X, y)
        self.hgb.fit(X, y)
        return self

    def predict_proba(self, X):
        p_lr = self.lr.predict_proba(X)[:, 1]
        p_hgb = self.hgb.predict_proba(X)[:, 1]
        return (p_lr + p_hgb) / 2

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)


def cross_validate(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    accs = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_tr_aug, y_tr_aug = augment_mirror(X_tr, y_tr)
        model = EnsembleModel().fit(X_tr_aug, y_tr_aug)
        accs.append(accuracy_score(y_val, model.predict(X_val)))
    return np.array(accs)

def main(data_dir, out_path):
    member_train = load_member_stats(f"{data_dir}/member_stats_training.csv")
    member_test = load_member_stats(f"{data_dir}/member_stats_test.csv")
    matches_train = pd.read_csv(f"{data_dir}/clan_matches_training.csv")
    matches_test = pd.read_csv(f"{data_dir}/clan_matches_test.csv")

    clan_feats_train = build_clan_features(member_train)
    clan_feats_test = build_clan_features(member_test)

    train_diff, feature_cols = build_match_diff_features(matches_train, clan_feats_train, has_target=True)
    test_diff, _ = build_match_diff_features(matches_test, clan_feats_test, has_target=False)

    diff_feature_cols = [f"diff_{c}" for c in feature_cols]
    X = train_diff[diff_feature_cols].values
    y = train_diff["target"].values

    # --- validation ---
    cv_accs = cross_validate(X, y, n_splits=5)
    print(f"5-fold CV accuracy: {cv_accs.mean():.4f} +/- {cv_accs.std():.4f}")

    # --- final fit on all training data ---
    X_aug, y_aug = augment_mirror(X, y)
    final_model = EnsembleModel().fit(X_aug, y_aug)

    # --- predict on test set ---
    X_test = test_diff[diff_feature_cols].values
    proba_clan1_wins = final_model.predict_proba(X_test)
    pred_winner = np.where(proba_clan1_wins > 0.5, 1, 2)

    submission = pd.DataFrame({
        "clan_1_id": test_diff["clan_1_id"],
        "clan_2_id": test_diff["clan_2_id"],
        "predicted_clan_winner": pred_winner,
    })
    submission.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path} ({len(submission)} rows)")
    print(submission["predicted_clan_winner"].value_counts())

    return final_model, clan_feats_train, diff_feature_cols, cv_accs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=".")
    parser.add_argument("--out", default="clan_winner_predictions.csv")
    args = parser.parse_args()
    main(args.data_dir, args.out)
