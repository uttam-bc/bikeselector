"""
🏍️  Bike Selector using Machine Learning
=========================================
Uses TOPSIS multi-criteria decision making + ML scoring to rank bikes
based on user-defined priorities: mileage, cost, CC, curb weight, likeliness.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1.  Load & Inspect Dataset
# ─────────────────────────────────────────────

def load_data(path: str = "bikes.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"✅  Loaded {len(df)} bikes with {df.shape[1]} features.\n")
    return df


# ─────────────────────────────────────────────
# 2.  TOPSIS – Multi-Criteria Decision Making
# ─────────────────────────────────────────────

def topsis(df: pd.DataFrame, criteria: dict) -> pd.Series:
    """
    TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution).

    criteria: {column_name: (weight, 'max'|'min')}
      - 'max'  → higher is better  (mileage, power, likeliness)
      - 'min'  → lower  is better  (price, weight)
    """
    cols      = list(criteria.keys())
    weights   = np.array([v[0] for v in criteria.values()])
    is_max    = np.array([v[1] == "max" for v in criteria.values()])

    matrix = df[cols].values.astype(float)

    # Step 1 – Normalise
    norms  = np.sqrt((matrix ** 2).sum(axis=0))
    norm_m = matrix / norms

    # Step 2 – Weight
    weighted = norm_m * weights

    # Step 3 – Ideal best / worst
    ideal_best  = np.where(is_max, weighted.max(axis=0), weighted.min(axis=0))
    ideal_worst = np.where(is_max, weighted.min(axis=0), weighted.max(axis=0))

    # Step 4 – Euclidean distances
    d_best  = np.sqrt(((weighted - ideal_best)  ** 2).sum(axis=1))
    d_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))

    # Step 5 – Closeness score  (0→worst, 1→best)
    score = d_worst / (d_best + d_worst + 1e-9)
    return pd.Series(score, index=df.index, name="topsis_score")


# ─────────────────────────────────────────────
# 3.  ML – Random Forest Feature Importance
#     (learns which attributes matter most)
# ─────────────────────────────────────────────

def ml_score(df: pd.DataFrame, feature_weights: dict) -> pd.Series:
    """
    Trains a Random Forest to predict a synthetic 'ideal_score'
    derived from weighted features, then outputs its predicted score.
    """
    features = ["cc", "mileage_kmpl", "price_inr",
                "curb_weight_kg", "power_hp", "likeliness_score"]

    X = df[features].copy()

    # Build a weighted target (higher = better bike overall)
    scaler   = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)

    # Cost & weight → invert so higher still means better
    X_scaled["price_inr"]     = 1 - X_scaled["price_inr"]
    X_scaled["curb_weight_kg"] = 1 - X_scaled["curb_weight_kg"]

    # Apply feature weights
    target = sum(
        X_scaled[col] * feature_weights.get(col, 1.0)
        for col in features
    )

    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_scaled, target)
    preds = rf.predict(X_scaled)

    print("\n📊  Random Forest Feature Importances:")
    importances = dict(zip(features, rf.feature_importances_))
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"   {feat:<20} {bar}  ({imp:.3f})")

    return pd.Series(
        MinMaxScaler().fit_transform(preds.reshape(-1, 1)).flatten(),
        index=df.index,
        name="ml_score"
    )


# ─────────────────────────────────────────────
# 4.  KNN – Find Similar Bikes to Top Pick
# ─────────────────────────────────────────────

def find_similar_bikes(df: pd.DataFrame, top_idx: int, n: int = 3) -> pd.DataFrame:
    features = ["cc", "mileage_kmpl", "price_inr", "curb_weight_kg"]
    X = MinMaxScaler().fit_transform(df[features])
    knn = NearestNeighbors(n_neighbors=n + 1, metric="euclidean")
    knn.fit(X)
    distances, indices = knn.kneighbors([X[top_idx]])
    similar_idx = [i for i in indices[0] if i != top_idx][:n]
    return df.iloc[similar_idx][["name", "brand", "cc", "mileage_kmpl",
                                  "price_inr", "curb_weight_kg", "type"]]


# ─────────────────────────────────────────────
# 5.  Main – Run Everything
# ─────────────────────────────────────────────

def select_bike(
    budget_inr:    int   = 300_000,
    min_mileage:   float = 30.0,
    preferred_cc:  int   = 0,       # 0 = no preference
    max_weight_kg: float = 999.0,
    top_n:         int   = 5,
    # Priority weights  (1 = normal, 2 = high priority, 3 = very high)
    w_mileage:     float = 2.0,
    w_cost:        float = 2.0,
    w_cc:          float = 1.0,
    w_weight:      float = 1.0,
    w_likeliness:  float = 1.5,
):
    print("=" * 60)
    print("       🏍️   AI-POWERED BIKE SELECTOR")
    print("=" * 60)

    df = load_data()

    # ── Pre-filter on hard constraints ──────────────────────────
    df_filtered = df[df["price_inr"] <= budget_inr].copy()
    df_filtered = df_filtered[df_filtered["mileage_kmpl"] >= min_mileage]
    df_filtered = df_filtered[df_filtered["curb_weight_kg"] <= max_weight_kg]

    if preferred_cc > 0:
        # Keep bikes within ±100cc of preference
        df_filtered = df_filtered[
            (df_filtered["cc"] >= preferred_cc - 100) &
            (df_filtered["cc"] <= preferred_cc + 100)
        ]

    if df_filtered.empty:
        print("⚠️   No bikes match your filters. Try relaxing constraints.")
        return

    df_filtered = df_filtered.reset_index(drop=True)
    print(f"\n🔍  {len(df_filtered)} bikes passed your filters.\n")

    # ── TOPSIS scoring ──────────────────────────────────────────
    criteria = {
        "mileage_kmpl":    (w_mileage,   "max"),
        "price_inr":       (w_cost,      "min"),
        "cc":              (w_cc,        "max"),
        "curb_weight_kg":  (w_weight,    "min"),
        "likeliness_score":(w_likeliness,"max"),
    }
    df_filtered["topsis_score"] = topsis(df_filtered, criteria)

    # ── ML scoring ──────────────────────────────────────────────
    feature_weights = {
        "mileage_kmpl":    w_mileage,
        "price_inr":       w_cost,
        "cc":              w_cc,
        "curb_weight_kg":  w_weight,
        "likeliness_score":w_likeliness,
        "power_hp":        1.0,
    }
    df_filtered["ml_score"] = ml_score(df_filtered, feature_weights)

    # ── Combined score (60% TOPSIS + 40% ML) ────────────────────
    df_filtered["final_score"] = (
        0.60 * df_filtered["topsis_score"] +
        0.40 * df_filtered["ml_score"]
    )

    df_ranked = df_filtered.sort_values("final_score", ascending=False).reset_index(drop=True)

    # ── Display Top N ───────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  🏆  TOP {top_n} RECOMMENDED BIKES")
    print(f"{'=' * 60}")

    medals = ["🥇", "🥈", "🥉", "4️⃣ ", "5️⃣ "]
    for i, (_, row) in enumerate(df_ranked.head(top_n).iterrows()):
        medal = medals[i] if i < len(medals) else f"{i+1}."
        print(f"\n  {medal}  {row['name']}  ({row['brand']})")
        print(f"      Type         : {row['type']}")
        print(f"      Engine       : {int(row['cc'])} cc")
        print(f"      Mileage      : {row['mileage_kmpl']} kmpl")
        print(f"      Price        : ₹{int(row['price_inr']):,}")
        print(f"      Curb Weight  : {row['curb_weight_kg']} kg")
        print(f"      Power        : {row['power_hp']} hp")
        print(f"      Likeliness   : {row['likeliness_score']}/10")
        print(f"      ✅ Final Score: {row['final_score']:.4f}")

    # ── Winner detail ────────────────────────────────────────────
    winner     = df_ranked.iloc[0]
    winner_idx = df_ranked.index[0]

    print(f"\n{'=' * 60}")
    print(f"  🎯  BEST MATCH:  {winner['name']}")
    print(f"{'=' * 60}")
    print(f"\n  Similar bikes you might also consider:")
    similar = find_similar_bikes(df_filtered, winner_idx)
    for _, s in similar.iterrows():
        print(f"   • {s['name']}  ({s['type']}, {int(s['cc'])}cc, "
              f"₹{int(s['price_inr']):,}, {s['mileage_kmpl']} kmpl)")

    print(f"\n{'=' * 60}\n")
    return df_ranked


# ─────────────────────────────────────────────
# 6.  Entry Point – Edit priorities here
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ┌─────────────────────────────────────────────┐
    # │        🔧  CONFIGURE YOUR PREFERENCES        │
    # └─────────────────────────────────────────────┘

    result = select_bike(
        # Hard Filters
        budget_inr    = 250_000,   # Max budget in rupees
        min_mileage   = 30,        # Minimum mileage (kmpl)
        preferred_cc  = 0,         # Preferred engine CC (0 = any)
        max_weight_kg = 200,       # Max curb weight (kg)
        top_n         = 5,         # How many bikes to show

        # Priority Weights  (1=normal  2=high  3=very high)
        w_mileage     = 3.0,       # How much mileage matters
        w_cost        = 3.0,       # How much cost matters
        w_cc          = 1.0,       # How much engine size matters
        w_weight      = 1.5,       # How much low weight matters
        w_likeliness  = 2.0,       # How much overall likeliness matters
    )
