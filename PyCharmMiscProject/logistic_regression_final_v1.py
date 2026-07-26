"""
Final Logistic regression for the Steam Action RPG dissertation.

Input:
    steam_reviews_analysis_v5_excl_1300_20260724(2).csv

Primary analysis:
    1. Keep recommended reviews only.
    2. Keep reviews with at least 10 words.
    3. Keep playtime_at_review_hours > 0.
    4. Define high_playtime within each game:
           1 if playtime_at_review_hours >= that game's Q75
           0 otherwise
    5. Fit an unpenalised Logistic regression with HC3 robust standard errors:
           high_playtime ~ six NLP factors + log review length
                           + game fixed effects
                           + steam_purchase + received_for_free

The script does not retrain the frozen V5 NLP model. It uses only the factor
predictions already contained in the input CSV.

Before the first run, install the packages in the PyCharm terminal:
    python -m pip install pandas numpy scipy statsmodels matplotlib

Only the INPUT_CSV setting below should normally need to be changed.
Every run creates a new timestamped output folder and never edits the input CSV.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import statsmodels
    import statsmodels.formula.api as smf
    from scipy.stats import chi2
    from statsmodels.stats.multitest import multipletests
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except ImportError as exc:
    print("\nA required package is missing:", exc)
    print("Run this command in the PyCharm terminal, then run the script again:")
    print(
        f'"{sys.executable}" -m pip install '
        "pandas numpy scipy statsmodels matplotlib"
    )
    raise SystemExit(1) from exc


# ============================================================================
# 1. USER SETTING
# ============================================================================

INPUT_CSV = Path(
    r"C:\Users\14499\PyCharmMiscProject"
    r"\steam_reviews_analysis_v5_excl_1300_20260724.csv"
)

# Keep True for the dissertation analysis. It prevents accidental use of an
# altered file. Renaming or copying the same file does not change its hash.
STRICT_INPUT_CHECKS = True


# ============================================================================
# 2. FROZEN ANALYSIS SETTINGS — DO NOT CHANGE FOR THE PRIMARY MODEL
# ============================================================================

EXPECTED_SHA256 = "f7cff7190248910b26fa991c994cff51dd478152ac4dcbb20f70a05ba3853aea"
EXPECTED_ROWS = 98_173
EXPECTED_COLUMNS = 31

EXPECTED_GAMES = [
    "Cyberpunk 2077",
    "Elden Ring",
    "Hogwarts Legacy",
    "Monster Hunter Wilds",
    "The Witcher 3: Wild Hunt",
]

FACTORS = [
    "combat",
    "challenge",
    "progression",
    "exploration",
    "narrative",
    "immersion",
]

MIN_REVIEW_WORDS = 10
PRIMARY_QUANTILE = 0.75
SENSITIVITY_QUANTILES = [0.70, 0.75, 0.80]
REFERENCE_GAME = "Cyberpunk 2077"
ALPHA = 0.05
MAX_ITERATIONS = 200
TOP_COOKS_FRACTION = 0.001

REQUIRED_COLUMNS = [
    "game",
    "recommendation_id",
    "voted_up",
    "playtime_at_review_hours",
    "review_word_count",
    "steam_purchase",
    "received_for_free",
    *FACTORS,
]

FACTOR_TERMS = " + ".join(FACTORS)
GAME_TERM = f"C(game, Treatment(reference='{REFERENCE_GAME}'))"
CONTROL_TERMS = (
    f"log_review_words + {GAME_TERM} + "
    "steam_purchase + received_for_free"
)
MAIN_FORMULA = f"high_playtime ~ {FACTOR_TERMS} + {CONTROL_TERMS}"
BASELINE_FORMULA = f"high_playtime ~ {CONTROL_TERMS}"
NO_WORD_LENGTH_FORMULA = (
    f"high_playtime ~ {FACTOR_TERMS} + {GAME_TERM} + "
    "steam_purchase + received_for_free"
)
SPLINE_WORD_LENGTH_FORMULA = (
    f"high_playtime ~ {FACTOR_TERMS} + "
    "bs(log_review_words, df=4, degree=3, include_intercept=False) + "
    f"{GAME_TERM} + steam_purchase + received_for_free"
)
INTERACTION_FORMULA = (
    f"high_playtime ~ ({FACTOR_TERMS}) * {GAME_TERM} + "
    "log_review_words + steam_purchase + received_for_free"
)
LOO_FORMULA = (
    f"high_playtime ~ {FACTOR_TERMS} + log_review_words + "
    "C(game) + steam_purchase + received_for_free"
)
GAME_SPECIFIC_FORMULA = (
    f"high_playtime ~ {FACTOR_TERMS} + log_review_words + "
    "steam_purchase + received_for_free"
)


FIT_WARNING_ROWS: list[dict[str, str]] = []
SCENARIO_ERROR_ROWS: list[dict[str, str]] = []


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest without loading the whole file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def strict_or_warn(condition: bool, message: str) -> None:
    """Raise under frozen checks; otherwise display a conspicuous warning."""
    if condition:
        return
    if STRICT_INPUT_CHECKS:
        raise ValueError(message)
    print("WARNING:", message)


def to_binary(series: pd.Series, name: str) -> pd.Series:
    """Convert common Boolean/0-1 representations to an integer 0/1 series."""
    if pd.api.types.is_bool_dtype(series):
        converted = series.astype("int8")
    elif pd.api.types.is_numeric_dtype(series):
        converted = pd.to_numeric(series, errors="coerce")
    else:
        normalised = series.astype("string").str.strip().str.lower()
        mapping = {
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0,
            "yes": 1,
            "no": 0,
        }
        converted = normalised.map(mapping)

    if converted.isna().any():
        bad = series.loc[converted.isna()].astype(str).value_counts().head(10)
        raise ValueError(
            f"{name} contains missing or unrecognised values:\n{bad.to_string()}"
        )

    unique_values = set(pd.Series(converted).astype(float).unique().tolist())
    if not unique_values.issubset({0.0, 1.0}):
        raise ValueError(
            f"{name} must contain only 0/1 values; found {sorted(unique_values)}"
        )
    return pd.Series(converted, index=series.index, name=name).astype("int8")


def to_required_numeric(series: pd.Series, name: str) -> pd.Series:
    """Convert a required field to numeric and reject missing/invalid values."""
    converted = pd.to_numeric(series, errors="coerce")
    if converted.isna().any():
        raise ValueError(
            f"{name} contains {int(converted.isna().sum())} missing/invalid values."
        )
    return converted


def json_default(value: Any) -> Any:
    """Convert NumPy, pandas and Path objects for the run manifest."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Cannot serialise {type(value).__name__}")


def save_csv(
    frame: pd.DataFrame,
    output_dir: Path,
    filename: str,
    *,
    index: bool = False,
) -> None:
    """Save Excel-friendly UTF-8 CSV."""
    frame.to_csv(
        output_dir / filename,
        index=index,
        encoding="utf-8-sig",
        float_format="%.10g",
    )


def load_and_validate_input(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read the exact frozen analysis file and verify its modelling fields."""
    if not path.exists():
        raise FileNotFoundError(
            "\nInput CSV was not found.\n"
            f"Current setting: {path}\n"
            "Edit INPUT_CSV near the top of this script so it matches the "
            "file's location on your computer."
        )

    file_hash = sha256_file(path)
    strict_or_warn(
        file_hash == EXPECTED_SHA256,
        "Input SHA-256 does not match the validated frozen V5 main table.\n"
        f"Expected: {EXPECTED_SHA256}\nActual:   {file_hash}",
    )

    data = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    strict_or_warn(
        data.shape == (EXPECTED_ROWS, EXPECTED_COLUMNS),
        "Unexpected input dimensions. "
        f"Expected {(EXPECTED_ROWS, EXPECTED_COLUMNS)}, found {data.shape}.",
    )

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data]
    if missing_columns:
        raise ValueError(f"Required columns are missing: {missing_columns}")

    if data["recommendation_id"].isna().any():
        raise ValueError("recommendation_id contains missing values.")
    duplicate_ids = int(data["recommendation_id"].duplicated().sum())
    strict_or_warn(
        duplicate_ids == 0,
        f"recommendation_id contains {duplicate_ids} duplicates.",
    )

    data["game"] = data["game"].astype("string").str.strip()
    actual_games = sorted(data["game"].dropna().unique().tolist())
    strict_or_warn(
        actual_games == sorted(EXPECTED_GAMES),
        f"Unexpected games. Expected {EXPECTED_GAMES}; found {actual_games}.",
    )

    for column in ["voted_up", "steam_purchase", "received_for_free", *FACTORS]:
        data[column] = to_binary(data[column], column)

    data["review_word_count"] = to_required_numeric(
        data["review_word_count"], "review_word_count"
    )
    data["playtime_at_review_hours"] = to_required_numeric(
        data["playtime_at_review_hours"], "playtime_at_review_hours"
    )

    key_missing = data[REQUIRED_COLUMNS].isna().sum()
    if int(key_missing.sum()) != 0:
        raise ValueError(
            "Required modelling fields contain missing values:\n"
            + key_missing[key_missing > 0].to_string()
        )

    validation = {
        "input_path": str(path.resolve()),
        "sha256": file_hash,
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "unique_recommendation_ids": int(data["recommendation_id"].nunique()),
        "duplicate_recommendation_ids": duplicate_ids,
        "games": actual_games,
        "key_field_missing_total": int(key_missing.sum()),
    }
    return data, validation


def make_analysis_sample(
    data: pd.DataFrame,
    quantile: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the frozen sample rules and construct a game-specific outcome."""
    mask = (
        data["voted_up"].eq(1)
        & data["review_word_count"].ge(MIN_REVIEW_WORDS)
        & data["playtime_at_review_hours"].gt(0)
    )
    sample = data.loc[mask].copy()

    threshold_name = f"game_q{int(round(quantile * 100))}"
    thresholds = (
        sample.groupby("game", observed=True)["playtime_at_review_hours"]
        .quantile(quantile)
        .rename(threshold_name)
    )
    sample = sample.join(thresholds, on="game")
    sample["high_playtime"] = (
        sample["playtime_at_review_hours"] >= sample[threshold_name]
    ).astype("int8")
    sample["log_review_words"] = np.log1p(sample["review_word_count"])

    threshold_table = (
        sample.groupby("game", observed=True)
        .agg(
            n=("recommendation_id", "size"),
            threshold_hours=(threshold_name, "first"),
            high_playtime_n=("high_playtime", "sum"),
            playtime_min=("playtime_at_review_hours", "min"),
            playtime_median=("playtime_at_review_hours", "median"),
            playtime_max=("playtime_at_review_hours", "max"),
        )
        .reset_index()
    )
    threshold_table.insert(0, "quantile", quantile)
    threshold_table["low_playtime_n"] = (
        threshold_table["n"] - threshold_table["high_playtime_n"]
    )
    threshold_table["high_playtime_rate"] = (
        threshold_table["high_playtime_n"] / threshold_table["n"]
    )
    return sample, threshold_table


def fit_logit(
    formula: str,
    data: pd.DataFrame,
    label: str,
) -> tuple[Any, dict[str, Any]]:
    """Fit an unpenalised Logit model with HC3 robust covariance."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = smf.logit(formula=formula, data=data).fit(
            method="newton",
            maxiter=MAX_ITERATIONS,
            disp=False,
            cov_type="HC3",
        )

    for item in caught:
        FIT_WARNING_ROWS.append(
            {
                "model": label,
                "warning_category": type(item.message).__name__,
                "warning_message": str(item.message),
            }
        )

    if int(result.nobs) != len(data):
        raise RuntimeError(
            f"{label}: model used {int(result.nobs)} of {len(data)} rows. "
            "Silent row deletion is not allowed."
        )

    converged = bool(result.mle_retvals.get("converged", False))
    if not converged:
        raise RuntimeError(f"{label}: Logistic regression did not converge.")

    exog = np.asarray(result.model.exog, dtype=float)
    matrix_rank = int(np.linalg.matrix_rank(exog))
    matrix_columns = int(exog.shape[1])
    if matrix_rank != matrix_columns:
        raise RuntimeError(
            f"{label}: model matrix is not full rank "
            f"({matrix_rank}/{matrix_columns})."
        )

    if not np.isfinite(result.params.to_numpy()).all():
        raise RuntimeError(f"{label}: non-finite coefficients were produced.")
    if not np.isfinite(result.bse.to_numpy()).all():
        raise RuntimeError(f"{label}: non-finite standard errors were produced.")

    diagnostics = {
        "model": label,
        "formula": formula,
        "n": int(result.nobs),
        "events_high_playtime": int(np.asarray(result.model.endog).sum()),
        "parameters": int(len(result.params)),
        "matrix_columns": matrix_columns,
        "matrix_rank": matrix_rank,
        "condition_number": float(np.linalg.cond(exog)),
        "converged": converged,
        "iterations": int(result.mle_retvals.get("iterations", -1)),
        "covariance_type": str(result.cov_type),
    }
    return result, diagnostics


def coefficient_table(result: Any) -> pd.DataFrame:
    """Create a publication-ready coefficient/OR table for every model term."""
    confidence = result.conf_int(alpha=ALPHA)
    table = pd.DataFrame(
        {
            "term": result.params.index,
            "beta": result.params.to_numpy(),
            "hc3_robust_se": result.bse.to_numpy(),
            "z_value": result.tvalues.to_numpy(),
            "p_value": result.pvalues.to_numpy(),
            "ci95_beta_low": confidence.iloc[:, 0].to_numpy(),
            "ci95_beta_high": confidence.iloc[:, 1].to_numpy(),
        }
    )
    table["odds_ratio"] = np.exp(table["beta"])
    table["or_ci95_low"] = np.exp(table["ci95_beta_low"])
    table["or_ci95_high"] = np.exp(table["ci95_beta_high"])
    table["odds_percent_change"] = (table["odds_ratio"] - 1.0) * 100.0
    return table


def factor_table(result: Any) -> pd.DataFrame:
    """Return the six pre-specified factors with FDR correction and AMEs."""
    all_terms = coefficient_table(result).set_index("term")
    factors = all_terms.loc[FACTORS].reset_index()

    reject, adjusted, _, _ = multipletests(
        factors["p_value"].to_numpy(),
        alpha=ALPHA,
        method="fdr_bh",
    )
    factors["p_fdr_bh"] = adjusted
    factors["significant_raw_0_05"] = factors["p_value"] < ALPHA
    factors["significant_fdr_0_05"] = reject

    margins = result.get_margeff(
        at="overall",
        method="dydx",
        dummy=True,
    ).summary_frame(alpha=ALPHA)
    margin_table = pd.DataFrame(
        {
            "term": margins.index,
            "average_marginal_effect": margins.iloc[:, 0].to_numpy(),
            "ame_se": margins.iloc[:, 1].to_numpy(),
            "ame_p_value": margins.iloc[:, 3].to_numpy(),
            "ame_ci95_low": margins.iloc[:, 4].to_numpy(),
            "ame_ci95_high": margins.iloc[:, 5].to_numpy(),
        }
    )
    factors = factors.merge(margin_table, on="term", how="left")
    factors["factor_order"] = factors["term"].map(
        {factor: index for index, factor in enumerate(FACTORS)}
    )
    return factors.sort_values("factor_order").drop(columns="factor_order")


def model_fit_statistics(
    result: Any,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Return likelihood, information criteria and supplemental fit measures."""
    outcome = np.asarray(result.model.endog, dtype=float)
    fitted = np.asarray(result.predict(), dtype=float)
    brier = float(np.mean((outcome - fitted) ** 2))
    tjur = float(fitted[outcome == 1].mean() - fitted[outcome == 0].mean())

    return {
        **diagnostics,
        "log_likelihood": float(result.llf),
        "null_log_likelihood": float(result.llnull),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "mcfadden_pseudo_r2": float(result.prsquared),
        "model_lr_chi2": float(result.llr),
        "model_lr_df": int(result.df_model),
        "model_lr_p_value": float(result.llr_pvalue),
        "brier_score_in_sample_supplemental": brier,
        "tjur_r2_in_sample_supplemental": tjur,
    }


def robust_wald_test(
    result: Any,
    terms: list[str],
    label: str,
) -> dict[str, Any]:
    """Joint HC3 Wald test that all selected coefficients equal zero."""
    parameter_names = result.params.index.tolist()
    restriction = np.zeros((len(terms), len(parameter_names)))
    for row, term in enumerate(terms):
        if term not in parameter_names:
            raise KeyError(f"{label}: term not found in model: {term}")
        restriction[row, parameter_names.index(term)] = 1.0
    test = result.wald_test(restriction, scalar=True)
    return {
        "test": label,
        "statistic": float(test.statistic),
        "df": int(test.df_denom),
        "p_value": float(test.pvalue),
        "covariance_type": str(result.cov_type),
    }


def lr_test(
    larger: Any,
    smaller: Any,
    label: str,
) -> dict[str, Any]:
    """Likelihood-ratio comparison for two nested unpenalised models."""
    statistic = 2.0 * (float(larger.llf) - float(smaller.llf))
    degrees = int(round(float(larger.df_model - smaller.df_model)))
    return {
        "test": label,
        "statistic": statistic,
        "df": degrees,
        "p_value": float(chi2.sf(statistic, degrees)),
        "covariance_type": "Likelihood ratio (MLE log-likelihoods)",
    }


def descriptive_tables(
    sample: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build game summaries, factor rates and simple separation cell counts."""
    game_summary = (
        sample.groupby("game", observed=True)
        .agg(
            n=("recommendation_id", "size"),
            high_playtime_n=("high_playtime", "sum"),
            playtime_mean=("playtime_at_review_hours", "mean"),
            playtime_median=("playtime_at_review_hours", "median"),
            review_words_mean=("review_word_count", "mean"),
            steam_purchase_rate=("steam_purchase", "mean"),
            received_for_free_rate=("received_for_free", "mean"),
        )
        .reset_index()
    )
    game_summary["low_playtime_n"] = (
        game_summary["n"] - game_summary["high_playtime_n"]
    )
    game_summary["high_playtime_rate"] = (
        game_summary["high_playtime_n"] / game_summary["n"]
    )

    factor_rows: list[dict[str, Any]] = []
    for factor in FACTORS:
        high_group = sample.loc[sample["high_playtime"].eq(1), factor]
        low_group = sample.loc[sample["high_playtime"].eq(0), factor]
        factor_rows.append(
            {
                "factor": factor,
                "overall_mentions": int(sample[factor].sum()),
                "overall_mention_rate": float(sample[factor].mean()),
                "high_group_mentions": int(high_group.sum()),
                "high_group_mention_rate": float(high_group.mean()),
                "low_group_mentions": int(low_group.sum()),
                "low_group_mention_rate": float(low_group.mean()),
            }
        )
    factor_summary = pd.DataFrame(factor_rows)

    cell_rows: list[dict[str, Any]] = []
    for variable in [*FACTORS, "steam_purchase", "received_for_free"]:
        counts = pd.crosstab(sample[variable], sample["high_playtime"])
        for level in [0, 1]:
            cell_rows.append(
                {
                    "variable": variable,
                    "level": level,
                    "low_playtime_n": int(counts.loc[level, 0])
                    if level in counts.index and 0 in counts.columns
                    else 0,
                    "high_playtime_n": int(counts.loc[level, 1])
                    if level in counts.index and 1 in counts.columns
                    else 0,
                }
            )
    for game in EXPECTED_GAMES:
        counts = sample.loc[sample["game"].eq(game), "high_playtime"].value_counts()
        cell_rows.append(
            {
                "variable": "game",
                "level": game,
                "low_playtime_n": int(counts.get(0, 0)),
                "high_playtime_n": int(counts.get(1, 0)),
            }
        )
    separation_cells = pd.DataFrame(cell_rows)
    return game_summary, factor_summary, separation_cells


def calculate_vif(result: Any) -> pd.DataFrame:
    """Calculate VIF from the exact main-model design matrix."""
    matrix = np.asarray(result.model.exog, dtype=float)
    names = result.model.exog_names
    rows = []
    for index, term in enumerate(names):
        if term == "Intercept":
            continue
        rows.append(
            {
                "term": term,
                "vif": float(variance_inflation_factor(matrix, index)),
            }
        )
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def make_forest_plot(factors: pd.DataFrame, output_path: Path) -> None:
    """Save a forest plot for the six primary adjusted odds ratios."""
    plot_data = factors.copy()
    labels = [label.title() for label in plot_data["term"]]
    y_positions = np.arange(len(plot_data))
    point = plot_data["odds_ratio"].to_numpy()
    lower = plot_data["or_ci95_low"].to_numpy()
    upper = plot_data["or_ci95_high"].to_numpy()
    errors = np.vstack([point - lower, upper - point])

    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    axis.errorbar(
        point,
        y_positions,
        xerr=errors,
        fmt="o",
        color="#1f4e79",
        ecolor="#4f81bd",
        elinewidth=1.8,
        capsize=4,
        markersize=6,
    )
    axis.axvline(1.0, color="#8c8c8c", linestyle="--", linewidth=1.2)
    axis.set_yticks(y_positions, labels)
    axis.invert_yaxis()
    axis.set_xscale("log")
    axis.set_xlabel("Adjusted odds ratio (log scale), with 95% HC3 CI")
    axis.set_title("Game-design factor mentions and high playtime")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def add_scenario_metadata(
    table: pd.DataFrame,
    *,
    scenario_type: str,
    scenario: str,
    result: Any,
) -> pd.DataFrame:
    """Attach reproducibility fields to a six-factor model result."""
    output = table.copy()
    output.insert(0, "scenario", scenario)
    output.insert(0, "scenario_type", scenario_type)
    output["n"] = int(result.nobs)
    output["high_playtime_n"] = int(np.asarray(result.model.endog).sum())
    output["converged"] = bool(result.mle_retvals.get("converged", False))
    return output


def safe_factor_scenario(
    *,
    data: pd.DataFrame,
    formula: str,
    scenario_type: str,
    scenario: str,
) -> tuple[pd.DataFrame, Any | None, dict[str, Any] | None]:
    """Fit a secondary analysis without allowing one failure to erase all output."""
    label = f"{scenario_type}: {scenario}"
    try:
        result, diagnostics = fit_logit(formula, data, label)
        table = add_scenario_metadata(
            factor_table(result),
            scenario_type=scenario_type,
            scenario=scenario,
            result=result,
        )
        return table, result, diagnostics
    except Exception as exc:  # secondary analyses are recorded, not hidden
        SCENARIO_ERROR_ROWS.append(
            {
                "scenario_type": scenario_type,
                "scenario": scenario,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        return pd.DataFrame(), None, None


def package_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def main() -> None:
    print("=" * 88)
    print("FINAL LOGISTIC REGRESSION — FROZEN V5 EXCLUDED-1,300 ANALYSIS")
    print("=" * 88)
    print(f"Input: {INPUT_CSV}")

    data, validation = load_and_validate_input(INPUT_CSV)
    print(
        f"PASS: validated input ({validation['rows']:,} rows × "
        f"{validation['columns']} columns; SHA-256 matched)."
    )

    run_started = datetime.now()
    output_dir = INPUT_CSV.parent / (
        "logistic_results_" + run_started.strftime("%Y%m%d_%H%M%S_%f")
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    # ----------------------------------------------------------------------
    # Sample flow and primary Q75 outcome
    # ----------------------------------------------------------------------
    recommended_mask = data["voted_up"].eq(1)
    word_mask = data["review_word_count"].ge(MIN_REVIEW_WORDS)
    positive_playtime_mask = data["playtime_at_review_hours"].gt(0)
    sample_flow = pd.DataFrame(
        [
            {
                "step": "Validated frozen V5 excluded-1,300 input",
                "n": len(data),
            },
            {
                "step": "Recommended reviews (voted_up = 1)",
                "n": int(recommended_mask.sum()),
            },
            {
                "step": f"Recommended and review_word_count >= {MIN_REVIEW_WORDS}",
                "n": int((recommended_mask & word_mask).sum()),
            },
            {
                "step": "Plus playtime_at_review_hours > 0",
                "n": int(
                    (recommended_mask & word_mask & positive_playtime_mask).sum()
                ),
            },
        ]
    )
    sample_flow["removed_from_previous_step"] = (
        sample_flow["n"].shift(1) - sample_flow["n"]
    ).fillna(0).astype(int)
    save_csv(sample_flow, output_dir, "01_sample_flow.csv")

    primary, primary_thresholds = make_analysis_sample(data, PRIMARY_QUANTILE)
    strict_or_warn(
        len(primary) == 37_457,
        f"Expected 37,457 primary rows; found {len(primary):,}.",
    )
    strict_or_warn(
        int(primary["high_playtime"].sum()) == 9_366,
        "Expected 9,366 high-playtime cases; found "
        f"{int(primary['high_playtime'].sum()):,}.",
    )

    save_csv(primary_thresholds, output_dir, "02_primary_q75_thresholds.csv")
    game_summary, factor_summary, separation_cells = descriptive_tables(primary)
    save_csv(game_summary, output_dir, "03_descriptive_by_game.csv")
    save_csv(factor_summary, output_dir, "04_factor_descriptives.csv")
    save_csv(separation_cells, output_dir, "05_simple_separation_cells.csv")
    save_csv(
        primary[FACTORS].corr(),
        output_dir,
        "06_factor_correlations.csv",
        index=True,
    )

    minimal_columns = [
        "recommendation_id",
        "game",
        "playtime_at_review_hours",
        "game_q75",
        "high_playtime",
        "review_word_count",
        "log_review_words",
        *FACTORS,
        "steam_purchase",
        "received_for_free",
    ]
    save_csv(
        primary[minimal_columns],
        output_dir,
        "07_primary_model_sample_q75.csv",
    )

    print(
        f"PASS: primary sample n={len(primary):,}; "
        f"high_playtime=1: {int(primary['high_playtime'].sum()):,}; "
        f"high_playtime=0: "
        f"{int(primary['high_playtime'].eq(0).sum()):,}."
    )

    # ----------------------------------------------------------------------
    # Main and baseline models
    # ----------------------------------------------------------------------
    model_fit_rows: list[dict[str, Any]] = []
    main_model, main_diagnostics = fit_logit(
        MAIN_FORMULA,
        primary,
        "Primary Q75 full model",
    )
    baseline_model, baseline_diagnostics = fit_logit(
        BASELINE_FORMULA,
        primary,
        "Primary Q75 controls-only model",
    )
    model_fit_rows.extend(
        [
            model_fit_statistics(main_model, main_diagnostics),
            model_fit_statistics(baseline_model, baseline_diagnostics),
        ]
    )

    all_terms = coefficient_table(main_model)
    primary_factors = factor_table(main_model)
    save_csv(all_terms, output_dir, "08_main_model_all_terms.csv")
    save_csv(primary_factors, output_dir, "09_main_model_six_factors.csv")
    (output_dir / "10_main_model_summary.txt").write_text(
        main_model.summary().as_text(),
        encoding="utf-8",
    )

    main_vif = calculate_vif(main_model)
    save_csv(main_vif, output_dir, "11_main_model_vif.csv")
    make_forest_plot(primary_factors, output_dir / "12_main_model_or_forest.png")

    statistical_tests = [
        lr_test(
            main_model,
            baseline_model,
            "LR: add six factors to controls-only model",
        ),
        robust_wald_test(
            main_model,
            FACTORS,
            "HC3 Wald: six factor coefficients jointly equal zero",
        ),
    ]

    # ----------------------------------------------------------------------
    # Influence diagnostics; no observation is deleted from the main model
    # ----------------------------------------------------------------------
    influence = main_model.get_influence().summary_frame()
    influence_core = influence[
        ["cooks_d", "standard_resid", "hat_diag", "dffits_internal"]
    ].copy()
    influence_core = primary[
        [
            "recommendation_id",
            "game",
            "playtime_at_review_hours",
            "high_playtime",
            *FACTORS,
        ]
    ].join(influence_core)
    top_influence = influence_core.nlargest(50, "cooks_d")
    save_csv(top_influence, output_dir, "13_influence_top50_by_cooks_d.csv")

    n_primary = len(primary)
    parameter_count = len(main_model.params)
    cooks_cutoff = 4.0 / n_primary
    leverage_cutoff = 2.0 * parameter_count / n_primary
    influence_summary = pd.DataFrame(
        [
            {
                "n": n_primary,
                "parameters": parameter_count,
                "cooks_d_flag_rule": "Cook's D > 4/n",
                "cooks_d_cutoff": cooks_cutoff,
                "cooks_d_flagged_n": int(
                    (influence_core["cooks_d"] > cooks_cutoff).sum()
                ),
                "leverage_flag_rule": "hat diagonal > 2p/n",
                "leverage_cutoff": leverage_cutoff,
                "leverage_flagged_n": int(
                    (influence_core["hat_diag"] > leverage_cutoff).sum()
                ),
                "abs_standard_resid_gt_3_n": int(
                    (influence_core["standard_resid"].abs() > 3).sum()
                ),
                "maximum_cooks_d": float(influence_core["cooks_d"].max()),
                "maximum_hat_diag": float(influence_core["hat_diag"].max()),
                "maximum_abs_standard_resid": float(
                    influence_core["standard_resid"].abs().max()
                ),
            }
        ]
    )
    save_csv(influence_summary, output_dir, "14_influence_summary.csv")

    # ----------------------------------------------------------------------
    # Robustness 1: Q70 / Q75 / Q80
    # ----------------------------------------------------------------------
    threshold_tables: list[pd.DataFrame] = []
    quantile_factor_tables: list[pd.DataFrame] = []
    for quantile in SENSITIVITY_QUANTILES:
        quantile_sample, threshold_table = make_analysis_sample(data, quantile)
        threshold_tables.append(threshold_table)
        scenario = f"Q{int(round(quantile * 100))}"
        if quantile == PRIMARY_QUANTILE:
            quantile_table = add_scenario_metadata(
                factor_table(main_model),
                scenario_type="playtime_threshold",
                scenario=scenario,
                result=main_model,
            )
        else:
            quantile_table, quantile_model, quantile_diagnostics = (
                safe_factor_scenario(
                    data=quantile_sample,
                    formula=MAIN_FORMULA,
                    scenario_type="playtime_threshold",
                    scenario=scenario,
                )
            )
            if quantile_model is not None and quantile_diagnostics is not None:
                model_fit_rows.append(
                    model_fit_statistics(quantile_model, quantile_diagnostics)
                )
        if not quantile_table.empty:
            quantile_factor_tables.append(quantile_table)

    threshold_sensitivity = pd.concat(threshold_tables, ignore_index=True)
    quantile_sensitivity = pd.concat(quantile_factor_tables, ignore_index=True)
    save_csv(
        threshold_sensitivity,
        output_dir,
        "15_sensitivity_q70_q75_q80_thresholds.csv",
    )
    save_csv(
        quantile_sensitivity,
        output_dir,
        "16_sensitivity_q70_q75_q80_factors.csv",
    )

    # ----------------------------------------------------------------------
    # Robustness 2: leave one game out
    # ----------------------------------------------------------------------
    loo_tables: list[pd.DataFrame] = []
    for omitted_game in EXPECTED_GAMES:
        subset = primary.loc[~primary["game"].eq(omitted_game)].copy()
        table, result, diagnostics = safe_factor_scenario(
            data=subset,
            formula=LOO_FORMULA,
            scenario_type="leave_one_game_out",
            scenario=f"omit {omitted_game}",
        )
        if not table.empty:
            loo_tables.append(table)
        if result is not None and diagnostics is not None:
            model_fit_rows.append(model_fit_statistics(result, diagnostics))

    leave_one_out = pd.concat(loo_tables, ignore_index=True)
    if not leave_one_out.empty:
        _, adjusted, _, _ = multipletests(
            leave_one_out["p_value"],
            alpha=ALPHA,
            method="fdr_bh",
        )
        leave_one_out["p_fdr_bh_across_all_loo_tests"] = adjusted
    save_csv(
        leave_one_out,
        output_dir,
        "17_sensitivity_leave_one_game_out.csv",
    )

    # ----------------------------------------------------------------------
    # Robustness 3: five exploratory game-specific models
    # ----------------------------------------------------------------------
    game_tables: list[pd.DataFrame] = []
    for game in EXPECTED_GAMES:
        subset = primary.loc[primary["game"].eq(game)].copy()
        table, result, diagnostics = safe_factor_scenario(
            data=subset,
            formula=GAME_SPECIFIC_FORMULA,
            scenario_type="game_specific_exploratory",
            scenario=game,
        )
        if not table.empty:
            game_tables.append(table)
        if result is not None and diagnostics is not None:
            model_fit_rows.append(model_fit_statistics(result, diagnostics))

    game_specific = pd.concat(game_tables, ignore_index=True)
    if not game_specific.empty:
        _, adjusted, _, _ = multipletests(
            game_specific["p_value"],
            alpha=ALPHA,
            method="fdr_bh",
        )
        game_specific["p_fdr_bh_across_30_game_factor_tests"] = adjusted
    save_csv(
        game_specific,
        output_dir,
        "18_exploratory_game_specific_models.csv",
    )

    # ----------------------------------------------------------------------
    # Robustness 4: word-length specification and top 0.1% Cook's D
    # ----------------------------------------------------------------------
    specification_tables: list[pd.DataFrame] = []

    no_words_table, no_words_model, no_words_diagnostics = safe_factor_scenario(
        data=primary,
        formula=NO_WORD_LENGTH_FORMULA,
        scenario_type="model_specification",
        scenario="without log_review_words",
    )
    if not no_words_table.empty:
        specification_tables.append(no_words_table)
    if no_words_model is not None and no_words_diagnostics is not None:
        model_fit_rows.append(
            model_fit_statistics(no_words_model, no_words_diagnostics)
        )

    spline_table, spline_model, spline_diagnostics = safe_factor_scenario(
        data=primary,
        formula=SPLINE_WORD_LENGTH_FORMULA,
        scenario_type="model_specification",
        scenario="review length as cubic spline (df=4)",
    )
    if not spline_table.empty:
        specification_tables.append(spline_table)
    if spline_model is not None and spline_diagnostics is not None:
        model_fit_rows.append(
            model_fit_statistics(spline_model, spline_diagnostics)
        )
        statistical_tests.append(
            lr_test(
                spline_model,
                main_model,
                "LR sensitivity: cubic-spline versus linear log review length",
            )
        )

    n_remove = max(1, int(np.ceil(n_primary * TOP_COOKS_FRACTION)))
    cook_indices = influence_core.nlargest(n_remove, "cooks_d").index
    cook_subset = primary.drop(index=cook_indices).copy()
    cooks_table, cooks_model, cooks_diagnostics = safe_factor_scenario(
        data=cook_subset,
        formula=MAIN_FORMULA,
        scenario_type="influence_sensitivity",
        scenario=f"exclude top {TOP_COOKS_FRACTION:.1%} Cook's D ({n_remove} rows)",
    )
    if not cooks_table.empty:
        specification_tables.append(cooks_table)
    if cooks_model is not None and cooks_diagnostics is not None:
        model_fit_rows.append(
            model_fit_statistics(cooks_model, cooks_diagnostics)
        )

    specifications = pd.concat(specification_tables, ignore_index=True)
    save_csv(
        specifications,
        output_dir,
        "19_sensitivity_specification_and_influence.csv",
    )

    # ----------------------------------------------------------------------
    # Robustness 5: pooled factor × game interaction test
    # ----------------------------------------------------------------------
    interaction_model, interaction_diagnostics = fit_logit(
        INTERACTION_FORMULA,
        primary,
        "Primary Q75 factor-by-game interaction model",
    )
    model_fit_rows.append(
        model_fit_statistics(interaction_model, interaction_diagnostics)
    )
    # Do not use a simple ":" test here: the game value
    # "The Witcher 3: Wild Hunt" itself contains a colon. Actual Patsy
    # interaction terms start with one of the six factor names followed by ":".
    interaction_terms = [
        term
        for term in interaction_model.params.index
        if any(term.startswith(f"{factor}:") for factor in FACTORS)
    ]
    interaction_table = coefficient_table(interaction_model)
    interaction_table = interaction_table.loc[
        interaction_table["term"].isin(interaction_terms)
    ].reset_index(drop=True)
    save_csv(
        interaction_table,
        output_dir,
        "20_exploratory_game_interaction_terms.csv",
    )
    statistical_tests.extend(
        [
            lr_test(
                interaction_model,
                main_model,
                "LR: add all factor-by-game interactions",
            ),
            robust_wald_test(
                interaction_model,
                interaction_terms,
                "HC3 Wald: all factor-by-game interactions jointly equal zero",
            ),
        ]
    )

    save_csv(
        pd.DataFrame(statistical_tests),
        output_dir,
        "21_joint_and_model_comparison_tests.csv",
    )
    save_csv(
        pd.DataFrame(model_fit_rows),
        output_dir,
        "22_model_fit_statistics.csv",
    )

    warning_table = pd.DataFrame(
        FIT_WARNING_ROWS,
        columns=["model", "warning_category", "warning_message"],
    )
    error_table = pd.DataFrame(
        SCENARIO_ERROR_ROWS,
        columns=["scenario_type", "scenario", "error"],
    )
    save_csv(warning_table, output_dir, "23_fit_warnings.csv")
    save_csv(error_table, output_dir, "24_secondary_analysis_errors.csv")

    results_guide = f"""RESULTS GUIDE
=============

Primary analysis:
  02_primary_q75_thresholds.csv
      Exact within-game Q75 thresholds used to create high_playtime.
  09_main_model_six_factors.csv
      Main dissertation table: beta, HC3 SE, OR, 95% CI, raw p-value,
      BH-FDR p-value, and average marginal effect for the six factors.
  12_main_model_or_forest.png
      Forest plot of the six adjusted odds ratios.

Diagnostics:
  05_simple_separation_cells.csv
  06_factor_correlations.csv
  11_main_model_vif.csv
  13_influence_top50_by_cooks_d.csv
  14_influence_summary.csv
  21_joint_and_model_comparison_tests.csv
  22_model_fit_statistics.csv
  23_fit_warnings.csv

Robustness / exploratory:
  16_sensitivity_q70_q75_q80_factors.csv
  17_sensitivity_leave_one_game_out.csv
  18_exploratory_game_specific_models.csv
  19_sensitivity_specification_and_influence.csv
  20_exploratory_game_interaction_terms.csv

Interpretation:
  OR > 1 means higher odds of being in the within-game high-playtime group.
  OR < 1 means lower odds.
  This is an association, not a causal effect. A factor value of 1 means the
  review mentions the factor; it does not mean the mention is positive.

Primary formula:
  {MAIN_FORMULA}
"""
    (output_dir / "README_RESULTS.txt").write_text(
        results_guide,
        encoding="utf-8",
    )

    run_finished = datetime.now()
    manifest = {
        "analysis_name": "Final Logistic regression — frozen V5 excluded-1,300",
        "run_started": run_started.isoformat(),
        "run_finished": run_finished.isoformat(),
        "input_validation": validation,
        "settings": {
            "recommended_reviews_only": True,
            "minimum_review_words": MIN_REVIEW_WORDS,
            "playtime_at_review_hours_must_be_positive": True,
            "primary_quantile": PRIMARY_QUANTILE,
            "high_playtime_rule": "playtime_at_review_hours >= within-game Q75",
            "factors": FACTORS,
            "reference_game": REFERENCE_GAME,
            "covariance_type": "HC3",
            "alpha": ALPHA,
            "strict_input_checks": STRICT_INPUT_CHECKS,
        },
        "sample": {
            "primary_n": len(primary),
            "primary_high_playtime_n": int(primary["high_playtime"].sum()),
            "primary_low_playtime_n": int(primary["high_playtime"].eq(0).sum()),
        },
        "formulas": {
            "main": MAIN_FORMULA,
            "baseline": BASELINE_FORMULA,
            "no_word_length": NO_WORD_LENGTH_FORMULA,
            "spline_word_length": SPLINE_WORD_LENGTH_FORMULA,
            "interaction": INTERACTION_FORMULA,
            "leave_one_game_out": LOO_FORMULA,
            "game_specific": GAME_SPECIFIC_FORMULA,
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": package_version("scipy"),
            "statsmodels": statsmodels.__version__,
            "patsy": package_version("patsy"),
            "matplotlib": matplotlib.__version__,
        },
        "main_model": {
            "converged": bool(main_model.mle_retvals.get("converged", False)),
            "n": int(main_model.nobs),
            "log_likelihood": float(main_model.llf),
            "aic": float(main_model.aic),
            "mcfadden_pseudo_r2": float(main_model.prsquared),
        },
        "fit_warning_count": len(FIT_WARNING_ROWS),
        "secondary_analysis_error_count": len(SCENARIO_ERROR_ROWS),
        "output_directory": str(output_dir.resolve()),
        "output_files": sorted(path.name for path in output_dir.iterdir()),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            default=json_default,
        ),
        encoding="utf-8",
    )

    console_columns = [
        "term",
        "beta",
        "odds_ratio",
        "or_ci95_low",
        "or_ci95_high",
        "p_value",
        "p_fdr_bh",
        "average_marginal_effect",
    ]
    print("\nPRIMARY SIX-FACTOR RESULTS")
    print(primary_factors[console_columns].to_string(index=False))
    print("\nPASS: main model converged; design matrix is full rank.")
    print(f"Maximum VIF: {main_vif['vif'].max():.3f}")
    print(f"Results folder: {output_dir}")
    print("=" * 88)
    print(
        "Important: interpret the coefficients as associations among "
        "recommended reviews with at least 10 words, not as causal effects."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\n" + "=" * 88)
        print("ANALYSIS STOPPED")
        print(f"{type(error).__name__}: {error}")
        print("=" * 88)
        raise
