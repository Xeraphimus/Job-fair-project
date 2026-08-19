import pandas as pd
import numpy as np
from clan_success_pipeline import (
    load_member_stats, build_clan_features, build_match_diff_features,
    augment_mirror, EnsembleModel, PAYMENT_RANK,
)


def win_probability(clan_members_df, opp_members_df, model, feature_cols):
    """Compute model win probability for clan (clan_members_df) vs opp_members_df."""
    c1_feats = build_clan_features(clan_members_df.assign(clan_id="THIS_CLAN"))
    c2_feats = build_clan_features(opp_members_df.assign(clan_id="OPPONENT"))
    c1 = c1_feats.iloc[0]
    c2 = c2_feats.iloc[0]
    diffs = np.array([[c1[c] - c2[c] for c in feature_cols]])
    return float(model.predict_proba(diffs)[0])


def simulate_interventions(clan_members_df, opp_members_df, model, feature_cols):
    """Try a menu of concrete roster changes and measure the win-probability delta."""
    baseline = win_probability(clan_members_df, opp_members_df, model, feature_cols)
    weakest_training_idx = clan_members_df["avg_training_bonus"].idxmin()
    weakest_quality_idx = clan_members_df["avg_stars_top_11_players"].idxmin()
    least_active_idx = clan_members_df["days_active_last_7_days"].idxmin()

    clan_avg_training = clan_members_df["avg_training_bonus"].mean()

    scenarios = {}

    # 1. Bring weakest trainer up to the clan's own average (no external investment, just consistency)
    m = clan_members_df.copy()
    m.loc[weakest_training_idx, "avg_training_bonus"] = max(
        m.loc[weakest_training_idx, "avg_training_bonus"], clan_avg_training
    )
    scenarios["Bring weakest manager's training bonus up to clan average (1 manager)"] = m

    # 2. Raise everyone's training bonus by +2 levels
    m = clan_members_df.copy()
    m["avg_training_bonus"] = (m["avg_training_bonus"] + 2).clip(upper=30)
    scenarios["Raise training bonus +2 for ALL 6 managers"] = m

    # 3. Raise weakest manager's roster quality by +1 star
    m = clan_members_df.copy()
    m.loc[weakest_quality_idx, "avg_stars_top_11_players"] += 1
    scenarios["Improve weakest manager's team quality by +1 star (1 manager)"] = m

    # 4. Raise everyone's roster quality by +0.5 star
    m = clan_members_df.copy()
    m["avg_stars_top_11_players"] = m["avg_stars_top_11_players"] + 0.5
    scenarios["Improve team quality +0.5 star for ALL 6 managers"] = m

    # 5. Get the least active manager back to full weekly activity
    m = clan_members_df.copy()
    m.loc[least_active_idx, "days_active_last_7_days"] = 7
    scenarios["Get least-active manager back to daily logins (1 manager)"] = m

    results = []
    for name, modified_df in scenarios.items():
        p = win_probability(modified_df, opp_members_df, model, feature_cols)
        results.append({"intervention": name, "new_win_prob": p, "delta": p - baseline})

    results_df = pd.DataFrame(results).sort_values("delta", ascending=False).reset_index(drop=True)
    return baseline, results_df


def advise(clan_id, opp_id, member_stats_df, model, feature_cols):
    clan_members = member_stats_df[member_stats_df["clan_id"] == clan_id].reset_index(drop=True)
    opp_members = member_stats_df[member_stats_df["clan_id"] == opp_id].reset_index(drop=True)
    baseline, results = simulate_interventions(clan_members, opp_members, model, feature_cols)

    print(f"\n=== Advisory report: {clan_id} vs {opp_id} ===")
    print(f"Current predicted win probability for {clan_id}: {baseline:.1%}")
    print("\nRanked interventions (by win-probability gain):")
    for _, row in results.iterrows():
        sign = "+" if row["delta"] >= 0 else ""
        print(f"  {sign}{row['delta']*100:5.2f} pp -> {row['intervention']} (new: {row['new_win_prob']:.1%})")
    return baseline, results


if __name__ == "__main__":
    member_train = load_member_stats("/home/claude/work/data/member_stats_training.csv")
    matches_train = pd.read_csv("/home/claude/work/data/clan_matches_training.csv")
    member_test = load_member_stats("/home/claude/work/data/member_stats_test.csv")
    matches_test = pd.read_csv("/home/claude/work/data/clan_matches_test.csv")

    clan_feats_train = build_clan_features(member_train)
    train_diff, feature_cols = build_match_diff_features(matches_train, clan_feats_train, has_target=True)
    diff_feature_cols = [f"diff_{c}" for c in feature_cols]
    X = train_diff[diff_feature_cols].values
    y = train_diff["target"].values
    X_aug, y_aug = augment_mirror(X, y)
    model = EnsembleModel().fit(X_aug, y_aug)

    # Demonstrate on 3 real upcoming test-set matches
    sample_matches = matches_test.sample(3, random_state=7)
    for _, row in sample_matches.iterrows():
        advise(row["clan_1_id"], row["clan_2_id"], member_test, model, feature_cols)
