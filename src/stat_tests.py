# src/stat_tests.py
"""
Utility functions for Task-3: hypothesis testing on insurance data.

Functions:
 - prepare_kpis(df): compute LossRatio, HasClaim, ClaimSeverity, Margin
 - agg_by_group(df, group_col, min_count=30): aggregated KPIs per group with counts
 - chi2_test_frequency(df, group_col): chi-square test on claim frequency across groups
 - proportion_ztest_pair(df, group_col, group_a, group_b): z-test for proportions between two groups
 - kruskal_test_numeric(df, group_col, numeric_col, min_count=30): non-parametric test across groups
 - ttest_or_mannwhitney(df, group_col, group_a, group_b, numeric_col): two-sample test for numeric
 - summarize_test_result(...) returns dict with p-value, statistic, interpretation
"""

from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest

def prepare_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure core KPI columns exist:
    - HasClaim: 1 if TotalClaims > 0 else 0
    - ClaimSeverity: TotalClaims / HasClaim (NaN when HasClaim==0)
    - LossRatio: TotalClaims / TotalPremium (0 or NaN when premium 0)
    - Margin: TotalPremium - TotalClaims
    Returns df with new columns (does not copy by default).
    """
    df = df.copy()
    # numeric coercion
    for c in ["TotalClaims", "TotalPremium"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["HasClaim"] = (df["TotalClaims"] > 0).astype(int)
    # ClaimSeverity: average claim amount given a claim occurred.
    # If multiple claims per policy are recorded in TotalClaims (sum), it's the same.
    df["ClaimSeverity"] = df["TotalClaims"] / df["HasClaim"]
    df.loc[df["HasClaim"] == 0, "ClaimSeverity"] = np.nan

    # Loss ratio: avoid division by zero
    df["LossRatio"] = np.where(df["TotalPremium"] > 0, df["TotalClaims"] / df["TotalPremium"], np.nan)

    df["Margin"] = df["TotalPremium"] - df["TotalClaims"]
    return df

def agg_by_group(df: pd.DataFrame, group_col: str, min_count: int = 30) -> pd.DataFrame:
    """
    Aggregate core KPIs by group_col; return groups with counts >= min_count.
    KPIs: n_policies, claim_freq, mean_claim_severity, mean_lossratio, mean_margin
    """
    g = df.groupby(group_col).agg(
        n_policies=("PolicyID", "nunique") if "PolicyID" in df.columns else ("HasClaim", "count"),
        n_claims=("HasClaim", "sum"),
        mean_claim_severity=("ClaimSeverity", "mean"),
        mean_lossratio=("LossRatio", "mean"),
        mean_margin=("Margin", "mean")
    ).reset_index()
    g["claim_freq"] = g["n_claims"] / g["n_policies"]
    g = g.sort_values("n_policies", ascending=False)
    return g[g["n_policies"] >= min_count].copy()

def chi2_test_frequency(df: pd.DataFrame, group_col: str, min_count:int=30) -> Dict[str, Any]:
    """
    Chi-square test of independence between group_col and HasClaim.
    Only uses groups with at least min_count policies.
    Returns {statistic, pvalue, table, groups_used}.
    """
    agg = df.groupby(group_col).agg(n_policies=("HasClaim", "count"), n_claims=("HasClaim", "sum"))
    agg = agg[agg["n_policies"] >= min_count]
    if agg.shape[0] < 2:
        return {"error": "not enough groups with min_count", "groups": agg.index.tolist()}

    # construct contingency table: rows=groups, cols=[no_claim, has_claim]
    contingency = np.vstack([agg["n_policies"] - agg["n_claims"], agg["n_claims"]]).T
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    return {
        "statistic": float(chi2),
        "pvalue": float(p),
        "dof": int(dof),
        "expected": expected,
        "groups_used": agg.index.tolist(),
        "contingency_table": contingency
    }

def proportion_ztest_pair(df: pd.DataFrame, group_col: str, group_a, group_b) -> Dict[str, Any]:
    """
    Two-sample z-test for proportions (claim frequency) between group_a and group_b.
    group_a/group_b: exact values from group_col (e.g., 'Gauteng')
    Returns statistic, pvalue, counts and nobs.
    """
    sub = df[df[group_col].isin([group_a, group_b])]
    counts = sub.groupby(group_col)["HasClaim"].sum()
    nobs = sub.groupby(group_col)["HasClaim"].count()
    if len(counts) != 2:
        return {"error": "groups not found or insufficient data", "counts": counts.to_dict(), "nobs": nobs.to_dict()}

    count = np.array([int(counts.iloc[0]), int(counts.iloc[1])])
    nobs = np.array([int(nobs.iloc[0]), int(nobs.iloc[1])])
    stat, pval = proportions_ztest(count, nobs)
    return {
        "statistic": float(stat),
        "pvalue": float(pval),
        "count": count.tolist(),
        "nobs": nobs.tolist()
    }

def kruskal_test_numeric(df: pd.DataFrame, group_col: str, numeric_col: str, min_count:int=30) -> Dict[str, Any]:
    """
    Kruskal-Wallis H-test for independent samples: non-parametric alternative to ANOVA
    Groups used must have at least min_count observations (non-NaN numeric_col).
    Returns statistic, pvalue, groups used.
    """
    groups = []
    labels = []
    for name, grp in df.groupby(group_col):
        arr = grp[numeric_col].dropna().values
        if len(arr) >= min_count:
            groups.append(arr)
            labels.append(name)
    if len(groups) < 2:
        return {"error": "not enough groups with required observations", "groups_found": labels}

    stat, p = stats.kruskal(*groups)
    return {"statistic": float(stat), "pvalue": float(p), "groups_used": labels}

def ttest_or_mannwhitney(df: pd.DataFrame, group_col: str, group_a, group_b, numeric_col: str) -> Dict[str, Any]:
    """
    For numeric_col compare group_a vs group_b.
    Uses t-test if both groups pass normality (Shapiro) and have > 30 samples, otherwise Mann-Whitney U.
    Returns test name, statistic, pvalue, and group sizes.
    """
    sub = df[df[group_col].isin([group_a, group_b])]
    a = sub[sub[group_col] == group_a][numeric_col].dropna()
    b = sub[sub[group_col] == group_b][numeric_col].dropna()
    na, nb = len(a), len(b)
    if na < 5 or nb < 5:
        return {"error": "too few observations", "n_a": na, "n_b": nb}

    # normality tests (Shapiro) for moderate sizes; if large (>5000) assume non-normal by default? use sample
    try:
        sh_a = stats.shapiro(a.sample(5000) if len(a) > 5000 else a)
        sh_b = stats.shapiro(b.sample(5000) if len(b) > 5000 else b)
        normal = (sh_a.pvalue > 0.05) and (sh_b.pvalue > 0.05)
    except Exception:
        normal = False

    if normal and na >= 30 and nb >= 30:
        stat, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        test = "t-test (Welch)"
    else:
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        test = "Mann-Whitney U"

    return {
        "test": test,
        "statistic": float(stat),
        "pvalue": float(p),
        "n_a": na,
        "n_b": nb
    }

def pretty_interpret(pvalue: float, alpha: float = 0.05) -> str:
    if pvalue < alpha:
        return f"p = {pvalue:.4g} < {alpha} -> Reject H0 (statistically significant)"
    else:
        return f"p = {pvalue:.4g} >= {alpha} -> Fail to reject H0 (not statistically significant)"
