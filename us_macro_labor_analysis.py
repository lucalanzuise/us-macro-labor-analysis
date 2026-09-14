"""US macroeconomic, labour-market and inflation analysis.

This single-file project downloads public FRED series, validates and documents
the data, constructs worker-flow measures, plots the Beveridge curve, estimates
Phillips-curve relationships, tracks rolling coefficients and evaluates a simple
out-of-sample inflation forecast.

Author: Luca Lanzuise

Run:
    python us_macro_labor_analysis.py

Optional arguments:
    python us_macro_labor_analysis.py --refresh
    python us_macro_labor_analysis.py --end 2025-12-31 --output-dir results

Dependencies:
    pandas, numpy, matplotlib
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# 1. Configuration and metadata
# =============================================================================

DEFAULT_START = "1960-01-01"
DEFAULT_END = "2025-12-31"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


SERIES_METADATA = {
    "CPIAUCSL": {
        "label": "Consumer Price Index for All Urban Consumers",
        "short_name": "cpi",
        "units": "Index 1982-1984=100",
        "frequency": "Monthly",
        "seasonal_adjustment": "Seasonally adjusted",
        "source": "U.S. Bureau of Labor Statistics via FRED",
    },
    "PCEPI": {
        "label": "Personal Consumption Expenditures Price Index",
        "short_name": "pce_price_index",
        "units": "Index 2017=100",
        "frequency": "Monthly",
        "seasonal_adjustment": "Seasonally adjusted",
        "source": "U.S. Bureau of Economic Analysis via FRED",
    },
    "UNRATE": {
        "label": "Civilian Unemployment Rate",
        "short_name": "unemployment_rate",
        "units": "Percent",
        "frequency": "Monthly",
        "seasonal_adjustment": "Seasonally adjusted",
        "source": "U.S. Bureau of Labor Statistics via FRED",
    },
    "JTSJOL": {
        "label": "Job Openings: Total Nonfarm",
        "short_name": "vacancies",
        "units": "Thousands",
        "frequency": "Monthly",
        "seasonal_adjustment": "Seasonally adjusted",
        "source": "U.S. Bureau of Labor Statistics via FRED",
    },
    "UNEMPLOY": {
        "label": "Unemployment Level",
        "short_name": "unemployment_level",
        "units": "Thousands of persons",
        "frequency": "Monthly",
        "seasonal_adjustment": "Seasonally adjusted",
        "source": "U.S. Bureau of Labor Statistics via FRED",
    },
    "CLF16OV": {
        "label": "Civilian Labor Force Level",
        "short_name": "labour_force",
        "units": "Thousands of persons",
        "frequency": "Monthly",
        "seasonal_adjustment": "Seasonally adjusted",
        "source": "U.S. Bureau of Labor Statistics via FRED",
    },
    "UEMPLT5": {
        "label": "Number Unemployed for Less Than 5 Weeks",
        "short_name": "short_term_unemployment",
        "units": "Thousands of persons",
        "frequency": "Monthly",
        "seasonal_adjustment": "Seasonally adjusted",
        "source": "U.S. Bureau of Labor Statistics via FRED",
    },
    "CES0500000003": {
        "label": "Average Hourly Earnings of All Employees: Total Private",
        "short_name": "nominal_hourly_earnings",
        "units": "Dollars per hour",
        "frequency": "Monthly",
        "seasonal_adjustment": "Seasonally adjusted",
        "source": "U.S. Bureau of Labor Statistics via FRED",
    },
    "CIVPART": {
        "label": "Labor Force Participation Rate",
        "short_name": "participation_rate",
        "units": "Percent",
        "frequency": "Monthly",
        "seasonal_adjustment": "Seasonally adjusted",
        "source": "U.S. Bureau of Labor Statistics via FRED",
    },
    "FEDFUNDS": {
        "label": "Federal Funds Effective Rate",
        "short_name": "federal_funds_rate",
        "units": "Percent",
        "frequency": "Monthly",
        "seasonal_adjustment": "Not seasonally adjusted",
        "source": "Board of Governors of the Federal Reserve System via FRED",
    },
    "GDPC1": {
        "label": "Real Gross Domestic Product",
        "short_name": "real_gdp",
        "units": "Billions of chained 2017 dollars, annual rate",
        "frequency": "Quarterly",
        "seasonal_adjustment": "Seasonally adjusted",
        "source": "U.S. Bureau of Economic Analysis via FRED",
    },
    "NFIRSAXDCUSQ": {
        "label": "Real Gross Fixed Capital Formation for the United States",
        "short_name": "real_investment",
        "units": "Millions of domestic currency",
        "frequency": "Quarterly",
        "seasonal_adjustment": "Seasonally adjusted",
        "source": "International Monetary Fund via FRED",
    },
    "PCECC96": {
        "label": "Real Personal Consumption Expenditures",
        "short_name": "real_consumption",
        "units": "Billions of chained 2017 dollars, annual rate",
        "frequency": "Quarterly",
        "seasonal_adjustment": "Seasonally adjusted",
        "source": "U.S. Bureau of Economic Analysis via FRED",
    },
    "NROU": {
        "label": "Estimate of the Natural Rate of Unemployment",
        "short_name": "natural_unemployment_rate",
        "units": "Percent",
        "frequency": "Quarterly",
        "seasonal_adjustment": "Not seasonally adjusted",
        "source": "U.S. Congressional Budget Office via FRED",
    },
}


DERIVED_VARIABLE_METADATA = [
    {
        "variable": "cpi_inflation_yoy",
        "frequency": "Quarterly",
        "construction": "100 * (CPI_t / CPI_(t-12) - 1), averaged within quarter",
        "purpose": "Headline consumer-price inflation",
    },
    {
        "variable": "pce_inflation_yoy",
        "frequency": "Quarterly",
        "construction": "100 * (PCEPI_t / PCEPI_(t-12) - 1), averaged within quarter",
        "purpose": "PCE price inflation",
    },
    {
        "variable": "real_*_growth_yoy",
        "frequency": "Quarterly",
        "construction": "100 * (X_t / X_(t-4) - 1)",
        "purpose": "Year-on-year growth in real GDP, consumption and investment",
    },
    {
        "variable": "job_finding_probability",
        "frequency": "Monthly",
        "construction": "1 - (U_(t+1) - Us_(t+1)) / U_t",
        "purpose": "Shimer-style unemployment outflow probability",
    },
    {
        "variable": "separation_probability",
        "frequency": "Monthly",
        "construction": "Us_(t+1) / (E_t * (1 - 0.5 * F_t))",
        "purpose": "Shimer-style inflow probability with time-aggregation correction",
    },
    {
        "variable": "unemployment_rate_constructed",
        "frequency": "Monthly",
        "construction": "100 * U_t / L_t",
        "purpose": "Horizontal axis of the Beveridge curve",
    },
    {
        "variable": "vacancy_rate",
        "frequency": "Monthly",
        "construction": "100 * V_t / (E_t + V_t)",
        "purpose": "Vertical axis of the Beveridge curve",
    },
    {
        "variable": "unemployment_gap",
        "frequency": "Quarterly",
        "construction": "Unemployment rate minus CBO natural rate",
        "purpose": "Labour-market slack in Phillips-curve models",
    },
    {
        "variable": "adaptive_expected_inflation",
        "frequency": "Quarterly",
        "construction": "Mean CPI inflation over the previous four quarters",
        "purpose": "Simple backward-looking expectations measure",
    },
    {
        "variable": "inflation_surprise",
        "frequency": "Quarterly",
        "construction": "CPI inflation minus adaptive expected inflation",
        "purpose": "Outcome in expectations-augmented Phillips regressions",
    },
]


MONTHLY_SERIES = [
    "CPIAUCSL",
    "PCEPI",
    "UNRATE",
    "JTSJOL",
    "UNEMPLOY",
    "CLF16OV",
    "UEMPLT5",
    "CES0500000003",
    "CIVPART",
    "FEDFUNDS",
]

QUARTERLY_SERIES = ["GDPC1", "NFIRSAXDCUSQ", "PCECC96", "NROU"]


BEVERIDGE_PERIODS = {
    "Pre-Great Recession (Dec 2000-Dec 2007)": (
        "2000-12-01",
        "2007-12-31",
        "#2E8B57",
    ),
    "Great Recession (Jan 2008-Jun 2009)": (
        "2008-01-01",
        "2009-06-30",
        "#E69F00",
    ),
    "Recovery (Jul 2009-Feb 2020)": (
        "2009-07-01",
        "2020-02-29",
        "#C44E52",
    ),
    "Pandemic and reopening (Mar 2020-Apr 2022)": (
        "2020-03-01",
        "2022-04-30",
        "#4C72B0",
    ),
    "Monetary tightening (May 2022-Dec 2025)": (
        "2022-05-01",
        "2025-12-31",
        "#222222",
    ),
}


PHILLIPS_PERIODS = {
    "1961-1986": ("1961-01-01", "1986-12-31", "#4C72B0"),
    "1987-2007": ("1987-01-01", "2007-12-31", "#55A868"),
    "2008-2019": ("2008-01-01", "2019-12-31", "#C44E52"),
    "2020-2025": ("2020-01-01", "2025-12-31", "#222222"),
}


@dataclass
class OLSResult:
    coefficients: pd.Series
    standard_errors: pd.Series
    fitted_values: pd.Series
    residuals: pd.Series
    r_squared: float
    n_obs: int


# =============================================================================
# 2. Data collection, caching, validation and documentation
# =============================================================================


def fred_download_url(series_id: str) -> str:
    """Return FRED's public CSV-download URL for one series."""
    return f"{FRED_CSV_URL}?{urlencode({'id': series_id})}"


def load_fred_series(
    series_id: str,
    cache_dir: Path,
    start: str,
    end: str,
    refresh: bool = False,
) -> pd.Series:
    """Download a FRED series or load its cached raw CSV file."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{series_id}.csv"

    if refresh or not cache_file.exists():
        try:
            raw = pd.read_csv(fred_download_url(series_id), na_values=".")
        except Exception as exc:
            if cache_file.exists():
                print(f"Warning: download failed for {series_id}; using cached data.")
                raw = pd.read_csv(cache_file, na_values=".")
            else:
                raise RuntimeError(
                    f"Could not download {series_id} from FRED and no cache exists."
                ) from exc
        else:
            raw.to_csv(cache_file, index=False)
    else:
        raw = pd.read_csv(cache_file, na_values=".")

    if raw.shape[1] != 2:
        raise ValueError(
            f"Expected two columns for {series_id}; received {raw.shape[1]}."
        )

    raw.columns = ["date", series_id]
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw[series_id] = pd.to_numeric(raw[series_id], errors="coerce")
    raw = raw.dropna(subset=["date"]).sort_values("date")

    if raw["date"].duplicated().any():
        duplicates = int(raw["date"].duplicated().sum())
        raise ValueError(f"{series_id} contains {duplicates} duplicate dates.")

    series = raw.set_index("date")[series_id].loc[start:end]
    series.name = SERIES_METADATA[series_id]["short_name"]
    return series


def load_all_series(
    series_ids: Iterable[str],
    cache_dir: Path,
    start: str,
    end: str,
    refresh: bool,
) -> dict[str, pd.Series]:
    """Load all requested series and retain their FRED IDs as dictionary keys."""
    return {
        series_id: load_fred_series(
            series_id, cache_dir=cache_dir, start=start, end=end, refresh=refresh
        )
        for series_id in series_ids
    }


def build_quality_report(series_data: dict[str, pd.Series]) -> pd.DataFrame:
    """Summarise coverage and basic quality checks for every raw series."""
    records = []
    for series_id, series in series_data.items():
        observed = series.dropna()
        records.append(
            {
                "series_id": series_id,
                "short_name": series.name,
                "first_observation": observed.index.min(),
                "last_observation": observed.index.max(),
                "rows_in_requested_window": len(series),
                "non_missing_observations": int(series.notna().sum()),
                "missing_observations": int(series.isna().sum()),
                "duplicate_dates": int(series.index.duplicated().sum()),
                "monotonic_dates": bool(series.index.is_monotonic_increasing),
            }
        )
    return pd.DataFrame(records).sort_values("series_id")


def build_metadata_table() -> pd.DataFrame:
    """Create human-readable documentation for all source series."""
    records = []
    for series_id, metadata in SERIES_METADATA.items():
        records.append({"series_id": series_id, **metadata})
    return pd.DataFrame(records).sort_values("series_id")


def build_derived_variable_table() -> pd.DataFrame:
    """Document the transformations used to construct analytical variables."""
    return pd.DataFrame(DERIVED_VARIABLE_METADATA)


def align_monthly(series_data: dict[str, pd.Series], ids: Iterable[str]) -> pd.DataFrame:
    """Combine monthly series while preserving missing observations."""
    frame = pd.concat([series_data[series_id] for series_id in ids], axis=1)
    return frame.sort_index().asfreq("MS")


def quarter_end(series: pd.Series) -> pd.Series:
    """Place an already-quarterly series on a common quarter-end index."""
    result = series.copy()
    result.index = result.index.to_period("Q").to_timestamp("Q")
    return result[~result.index.duplicated(keep="last")].sort_index()


# =============================================================================
# 3. Descriptive macroeconomic dataset
# =============================================================================


def build_macro_dataset(series_data: dict[str, pd.Series]) -> pd.DataFrame:
    """Construct a quarterly dataset with consistent economic transformations."""
    monthly = align_monthly(series_data, MONTHLY_SERIES)

    monthly["cpi_inflation_yoy"] = (
        monthly["cpi"] / monthly["cpi"].shift(12) - 1.0
    ) * 100.0
    monthly["pce_inflation_yoy"] = (
        monthly["pce_price_index"] / monthly["pce_price_index"].shift(12) - 1.0
    ) * 100.0
    monthly["nominal_wage_growth_yoy"] = (
        monthly["nominal_hourly_earnings"]
        / monthly["nominal_hourly_earnings"].shift(12)
        - 1.0
    ) * 100.0

    monthly_quarterly_average = monthly[
        [
            "cpi_inflation_yoy",
            "pce_inflation_yoy",
            "unemployment_rate",
            "participation_rate",
            "nominal_wage_growth_yoy",
            "federal_funds_rate",
        ]
    ].resample("QE").mean()

    real_gdp = quarter_end(series_data["GDPC1"])
    real_investment = quarter_end(series_data["NFIRSAXDCUSQ"])
    real_consumption = quarter_end(series_data["PCECC96"])

    quarterly = pd.concat(
        [real_gdp, real_investment, real_consumption, monthly_quarterly_average],
        axis=1,
    ).sort_index()

    quarterly["real_gdp_growth_yoy"] = (
        quarterly["real_gdp"] / quarterly["real_gdp"].shift(4) - 1.0
    ) * 100.0
    quarterly["real_investment_growth_yoy"] = (
        quarterly["real_investment"]
        / quarterly["real_investment"].shift(4)
        - 1.0
    ) * 100.0
    quarterly["real_consumption_growth_yoy"] = (
        quarterly["real_consumption"]
        / quarterly["real_consumption"].shift(4)
        - 1.0
    ) * 100.0
    return quarterly


def plot_macro_dashboard(macro: pd.DataFrame, figure_dir: Path) -> None:
    """Plot a compact overview of real activity, inflation and labour markets."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)

    macro[
        [
            "real_gdp_growth_yoy",
            "real_consumption_growth_yoy",
            "real_investment_growth_yoy",
        ]
    ].plot(ax=axes[0, 0], linewidth=1.3)
    axes[0, 0].set_title("Real activity: year-on-year growth")
    axes[0, 0].set_ylabel("Percent")

    macro[["cpi_inflation_yoy", "pce_inflation_yoy"]].plot(
        ax=axes[0, 1], linewidth=1.3
    )
    axes[0, 1].set_title("Inflation")
    axes[0, 1].set_ylabel("Percent")

    macro[["unemployment_rate", "participation_rate"]].plot(
        ax=axes[1, 0], linewidth=1.3
    )
    axes[1, 0].set_title("Labour-market rates")
    axes[1, 0].set_ylabel("Percent")

    macro[["federal_funds_rate", "nominal_wage_growth_yoy"]].plot(
        ax=axes[1, 1], linewidth=1.3
    )
    axes[1, 1].set_title("Policy rate and nominal wage growth")
    axes[1, 1].set_ylabel("Percent")

    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        ax.set_xlabel("")

    fig.suptitle("United States macroeconomic indicators", fontsize=15)
    fig.tight_layout()
    fig.savefig(figure_dir / "01_macro_dashboard.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 4. Worker flows using unemployment-duration data
# =============================================================================


def build_worker_flows(series_data: dict[str, pd.Series]) -> pd.DataFrame:
    """Estimate monthly job-finding and separation probabilities.

    The job-finding probability follows Shimer (2005, equation 1):
        F_t = 1 - (U_{t+1} - U^s_{t+1}) / U_t.

    The separation probability applies Shimer's approximate time-aggregation
    correction (equation 2):
        S_t = U^s_{t+1} / [E_t * (1 - 0.5 * F_t)].
    """
    monthly = align_monthly(
        series_data, ["JTSJOL", "UNEMPLOY", "CLF16OV", "UEMPLT5"]
    ).loc["2000-12-01":"2025-12-31"]
    monthly = monthly.dropna().copy()

    monthly["employment"] = monthly["labour_force"] - monthly["unemployment_level"]
    monthly["unemployment_next"] = monthly["unemployment_level"].shift(-1)
    monthly["short_term_unemployment_next"] = monthly[
        "short_term_unemployment"
    ].shift(-1)

    monthly["job_finding_probability"] = 1.0 - (
        monthly["unemployment_next"]
        - monthly["short_term_unemployment_next"]
    ) / monthly["unemployment_level"]

    denominator = monthly["employment"] * (
        1.0 - 0.5 * monthly["job_finding_probability"]
    )
    monthly["separation_probability"] = (
        monthly["short_term_unemployment_next"] / denominator
    )

    for column in ["job_finding_probability", "separation_probability"]:
        invalid = ~monthly[column].between(0.0, 1.0) & monthly[column].notna()
        if invalid.any():
            print(
                f"Warning: {int(invalid.sum())} observations of {column} "
                "fall outside [0, 1] and were set to missing."
            )
            monthly.loc[invalid, column] = np.nan

    monthly["job_finding_hazard"] = -np.log1p(
        -monthly["job_finding_probability"]
    )
    monthly["separation_hazard"] = -np.log1p(
        -monthly["separation_probability"]
    )
    return monthly.dropna(
        subset=["job_finding_probability", "separation_probability"]
    )


def plot_worker_flows(flows: pd.DataFrame, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(
        flows.index,
        flows["job_finding_probability"],
        label="Job-finding probability",
        linewidth=1.4,
    )
    ax.plot(
        flows.index,
        flows["separation_probability"],
        label="Separation probability",
        linewidth=1.4,
    )
    ax.set_title("Monthly worker-flow probabilities")
    ax.set_ylabel("Probability")
    ax.set_xlabel("")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "02_worker_flows.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def worker_flow_summary(flows: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "job_finding_probability",
        "separation_probability",
        "job_finding_hazard",
        "separation_hazard",
    ]
    summary = flows[columns].agg(["count", "mean", "std", "min", "max"]).T
    summary["correlation_with_vacancy_unemployment_ratio"] = [
        flows[column].corr(flows["vacancies"] / flows["unemployment_level"])
        for column in columns
    ]
    return summary


# =============================================================================
# 5. Beveridge curve and business-cycle periods
# =============================================================================


def build_beveridge_data(flows: pd.DataFrame) -> pd.DataFrame:
    result = flows.copy()
    result["unemployment_rate_constructed"] = (
        result["unemployment_level"] / result["labour_force"] * 100.0
    )
    result["vacancy_rate"] = (
        result["vacancies"] / (result["employment"] + result["vacancies"]) * 100.0
    )
    result["period"] = pd.NA
    for label, (start, end, _color) in BEVERIDGE_PERIODS.items():
        mask = result.index.to_series().between(start, end)
        result.loc[mask, "period"] = label
    return result.dropna(subset=["period"])


def plot_beveridge_curve(beveridge: pd.DataFrame, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    for label, (_start, _end, color) in BEVERIDGE_PERIODS.items():
        subset = beveridge.loc[beveridge["period"] == label]
        ax.scatter(
            subset["unemployment_rate_constructed"],
            subset["vacancy_rate"],
            s=26,
            alpha=0.75,
            color=color,
            label=label,
        )
    ax.set_title("United States Beveridge curve, 2000-2025")
    ax.set_xlabel("Unemployment rate (percent)")
    ax.set_ylabel("Job-openings rate (percent)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "03_beveridge_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def beveridge_summary(beveridge: pd.DataFrame) -> pd.DataFrame:
    records = []
    for label in BEVERIDGE_PERIODS:
        subset = beveridge.loc[beveridge["period"] == label]
        records.append(
            {
                "period": label,
                "observations": len(subset),
                "mean_unemployment_rate": subset[
                    "unemployment_rate_constructed"
                ].mean(),
                "mean_vacancy_rate": subset["vacancy_rate"].mean(),
                "correlation": subset["unemployment_rate_constructed"].corr(
                    subset["vacancy_rate"]
                ),
            }
        )
    return pd.DataFrame(records)


# =============================================================================
# 6. Small OLS implementation used by the remaining sections
# =============================================================================


def fit_ols(data: pd.DataFrame, outcome: str, regressors: list[str]) -> OLSResult:
    """Estimate OLS with an intercept using NumPy and return core diagnostics."""
    clean = data[[outcome, *regressors]].dropna()
    if len(clean) <= len(regressors) + 1:
        raise ValueError("Not enough complete observations for OLS estimation.")

    names = ["intercept", *regressors]
    x = np.column_stack(
        [np.ones(len(clean)), *[clean[column].to_numpy() for column in regressors]]
    )
    y = clean[outcome].to_numpy()

    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    residuals = y - fitted
    degrees_of_freedom = len(y) - x.shape[1]
    residual_variance = float(residuals @ residuals / degrees_of_freedom)
    covariance = residual_variance * np.linalg.pinv(x.T @ x)
    standard_errors = np.sqrt(np.diag(covariance))

    total_sum_squares = float(((y - y.mean()) ** 2).sum())
    residual_sum_squares = float((residuals**2).sum())
    r_squared = 1.0 - residual_sum_squares / total_sum_squares

    return OLSResult(
        coefficients=pd.Series(beta, index=names),
        standard_errors=pd.Series(standard_errors, index=names),
        fitted_values=pd.Series(fitted, index=clean.index),
        residuals=pd.Series(residuals, index=clean.index),
        r_squared=r_squared,
        n_obs=len(clean),
    )


def ols_result_row(label: str, result: OLSResult) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "sample": label,
        "observations": result.n_obs,
        "r_squared": result.r_squared,
    }
    for name, value in result.coefficients.items():
        row[f"coefficient_{name}"] = value
        row[f"standard_error_{name}"] = result.standard_errors[name]
    return row


# =============================================================================
# 7. Phillips curves and structural change
# =============================================================================


def build_phillips_data(series_data: dict[str, pd.Series]) -> pd.DataFrame:
    monthly = align_monthly(series_data, ["CPIAUCSL", "PCEPI", "UNRATE"])
    monthly["cpi_inflation_yoy"] = (
        monthly["cpi"] / monthly["cpi"].shift(12) - 1.0
    ) * 100.0
    monthly["pce_inflation_yoy"] = (
        monthly["pce_price_index"] / monthly["pce_price_index"].shift(12) - 1.0
    ) * 100.0

    quarterly = monthly[
        ["cpi_inflation_yoy", "pce_inflation_yoy", "unemployment_rate"]
    ].resample("QE").mean()
    natural_rate = quarter_end(series_data["NROU"])
    quarterly = quarterly.join(natural_rate, how="left")
    quarterly["unemployment_gap"] = (
        quarterly["unemployment_rate"]
        - quarterly["natural_unemployment_rate"]
    )

    # A simple adaptive expectation: the mean inflation rate over the four
    # quarters known before the current quarter.
    quarterly["adaptive_expected_inflation"] = (
        quarterly["cpi_inflation_yoy"].shift(1).rolling(4, min_periods=4).mean()
    )
    quarterly["inflation_surprise"] = (
        quarterly["cpi_inflation_yoy"]
        - quarterly["adaptive_expected_inflation"]
    )
    return quarterly


def estimate_phillips_subsamples(phillips: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, (start, end, _color) in PHILLIPS_PERIODS.items():
        sample = phillips.loc[start:end]
        result = fit_ols(sample, "inflation_surprise", ["unemployment_gap"])
        rows.append(ols_result_row(label, result))
    return pd.DataFrame(rows)


def plot_phillips_subsamples(phillips: pd.DataFrame, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    for label, (start, end, color) in PHILLIPS_PERIODS.items():
        subset = phillips.loc[start:end].dropna(
            subset=["inflation_surprise", "unemployment_gap"]
        )
        result = fit_ols(subset, "inflation_surprise", ["unemployment_gap"])
        ax.scatter(
            subset["unemployment_gap"],
            subset["inflation_surprise"],
            alpha=0.35,
            s=22,
            color=color,
        )
        x_grid = np.linspace(
            subset["unemployment_gap"].min(),
            subset["unemployment_gap"].max(),
            100,
        )
        y_grid = (
            result.coefficients["intercept"]
            + result.coefficients["unemployment_gap"] * x_grid
        )
        ax.plot(
            x_grid,
            y_grid,
            color=color,
            linewidth=2,
            label=(
                f"{label}: slope="
                f"{result.coefficients['unemployment_gap']:.2f}"
            ),
        )
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.axvline(0, color="#888888", linewidth=0.8)
    ax.set_title("Expectations-augmented Phillips curve by period")
    ax.set_xlabel("Unemployment gap (percentage points)")
    ax.set_ylabel("CPI inflation minus adaptive expectation (percentage points)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(
        figure_dir / "04_phillips_curve_subsamples.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


# =============================================================================
# 8. Rolling Phillips-curve estimates
# =============================================================================


def rolling_phillips_estimates(
    phillips: pd.DataFrame, window: int = 40
) -> pd.DataFrame:
    """Estimate the Phillips-curve slope in rolling 10-year windows."""
    clean = phillips[["inflation_surprise", "unemployment_gap"]].dropna()
    records = []
    for end_position in range(window, len(clean) + 1):
        sample = clean.iloc[end_position - window : end_position]
        result = fit_ols(sample, "inflation_surprise", ["unemployment_gap"])
        slope = result.coefficients["unemployment_gap"]
        standard_error = result.standard_errors["unemployment_gap"]
        records.append(
            {
                "date": sample.index[-1],
                "slope": slope,
                "standard_error": standard_error,
                "lower_95": slope - 1.96 * standard_error,
                "upper_95": slope + 1.96 * standard_error,
                "r_squared": result.r_squared,
                "observations": result.n_obs,
            }
        )
    return pd.DataFrame(records).set_index("date")


def plot_rolling_phillips(rolling: pd.DataFrame, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(rolling.index, rolling["slope"], color="#4C72B0", linewidth=1.6)
    ax.fill_between(
        rolling.index,
        rolling["lower_95"],
        rolling["upper_95"],
        color="#4C72B0",
        alpha=0.2,
        label="Approximate 95% confidence interval",
    )
    ax.axhline(0, color="#222222", linewidth=0.9)
    ax.set_title("Rolling 10-year Phillips-curve slope")
    ax.set_ylabel("Coefficient on unemployment gap")
    ax.set_xlabel("")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(
        figure_dir / "05_rolling_phillips_slope.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


# =============================================================================
# 9. Expanding-window inflation forecast
# =============================================================================


def expanding_inflation_forecast(
    phillips: pd.DataFrame, initial_window: int = 80
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Produce genuine one-quarter-ahead, expanding-window forecasts.

    The forecasting regression uses only information dated t-1:
        inflation_t = a + b * inflation_(t-1) + c * unemployment_gap_(t-1).
    It is compared with a random-walk benchmark equal to inflation_(t-1).
    """
    data = phillips[["cpi_inflation_yoy", "unemployment_gap"]].copy()
    data["inflation_lag_1"] = data["cpi_inflation_yoy"].shift(1)
    data["unemployment_gap_lag_1"] = data["unemployment_gap"].shift(1)
    data = data.dropna()

    if len(data) <= initial_window:
        raise ValueError("The forecast sample is shorter than the initial window.")

    records = []
    regressors = ["inflation_lag_1", "unemployment_gap_lag_1"]
    for position in range(initial_window, len(data)):
        training = data.iloc[:position]
        target = data.iloc[position]
        result = fit_ols(training, "cpi_inflation_yoy", regressors)
        forecast = result.coefficients["intercept"] + sum(
            result.coefficients[name] * target[name] for name in regressors
        )
        records.append(
            {
                "date": data.index[position],
                "actual_inflation": target["cpi_inflation_yoy"],
                "model_forecast": forecast,
                "naive_forecast": target["inflation_lag_1"],
            }
        )

    forecasts = pd.DataFrame(records).set_index("date")
    forecasts["model_error"] = (
        forecasts["actual_inflation"] - forecasts["model_forecast"]
    )
    forecasts["naive_error"] = (
        forecasts["actual_inflation"] - forecasts["naive_forecast"]
    )

    metrics = pd.DataFrame(
        {
            "model": ["Phillips forecasting model", "Naive lagged-inflation benchmark"],
            "rmse": [
                np.sqrt(np.mean(forecasts["model_error"] ** 2)),
                np.sqrt(np.mean(forecasts["naive_error"] ** 2)),
            ],
            "mae": [
                np.mean(np.abs(forecasts["model_error"])),
                np.mean(np.abs(forecasts["naive_error"])),
            ],
            "forecast_observations": [len(forecasts), len(forecasts)],
        }
    )
    return forecasts, metrics


def plot_inflation_forecasts(forecasts: pd.DataFrame, figure_dir: Path) -> None:
    display = forecasts.loc["2008-01-01":]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(
        display.index,
        display["actual_inflation"],
        label="Actual CPI inflation",
        color="#222222",
        linewidth=1.8,
    )
    ax.plot(
        display.index,
        display["model_forecast"],
        label="Expanding-window forecast",
        color="#4C72B0",
        linewidth=1.4,
    )
    ax.plot(
        display.index,
        display["naive_forecast"],
        label="Naive benchmark",
        color="#C44E52",
        linewidth=1.1,
        alpha=0.8,
    )
    ax.set_title("One-quarter-ahead CPI inflation forecasts")
    ax.set_ylabel("Year-on-year percent change")
    ax.set_xlabel("")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(
        figure_dir / "06_inflation_forecasts.png", dpi=200, bbox_inches="tight"
    )
    plt.close(fig)


# =============================================================================
# 10. Orchestration and exported outputs
# =============================================================================


def ensure_output_directories(output_dir: Path) -> tuple[Path, Path, Path]:
    data_dir = output_dir / "data"
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    for directory in [data_dir, table_dir, figure_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    return data_dir, table_dir, figure_dir


def run_analysis(
    start: str,
    end: str,
    output_dir: Path,
    refresh: bool = False,
) -> None:
    data_dir, table_dir, figure_dir = ensure_output_directories(output_dir)
    cache_dir = data_dir / "raw"

    print("Loading and validating FRED data...")
    series_data = load_all_series(
        [*MONTHLY_SERIES, *QUARTERLY_SERIES],
        cache_dir=cache_dir,
        start=start,
        end=end,
        refresh=refresh,
    )
    build_metadata_table().to_csv(data_dir / "series_metadata.csv", index=False)
    build_derived_variable_table().to_csv(
        data_dir / "derived_variable_metadata.csv", index=False
    )
    build_quality_report(series_data).to_csv(
        table_dir / "data_quality_report.csv", index=False
    )

    print("Building the quarterly macroeconomic dataset...")
    macro = build_macro_dataset(series_data)
    macro.to_csv(data_dir / "macroeconomic_quarterly.csv")
    plot_macro_dashboard(macro, figure_dir)

    print("Estimating worker flows...")
    flows = build_worker_flows(series_data)
    flows.to_csv(data_dir / "worker_flows_monthly.csv")
    worker_flow_summary(flows).to_csv(table_dir / "worker_flow_summary.csv")
    plot_worker_flows(flows, figure_dir)

    print("Constructing the Beveridge curve...")
    beveridge = build_beveridge_data(flows)
    beveridge.to_csv(data_dir / "beveridge_curve_monthly.csv")
    beveridge_summary(beveridge).to_csv(
        table_dir / "beveridge_period_summary.csv", index=False
    )
    plot_beveridge_curve(beveridge, figure_dir)

    print("Estimating Phillips curves and structural changes...")
    phillips = build_phillips_data(series_data)
    phillips.to_csv(data_dir / "phillips_curve_quarterly.csv")
    estimate_phillips_subsamples(phillips).to_csv(
        table_dir / "phillips_subsample_estimates.csv", index=False
    )
    plot_phillips_subsamples(phillips, figure_dir)

    print("Estimating rolling Phillips-curve coefficients...")
    rolling = rolling_phillips_estimates(phillips, window=40)
    rolling.to_csv(table_dir / "rolling_phillips_estimates.csv")
    plot_rolling_phillips(rolling, figure_dir)

    print("Producing and evaluating inflation forecasts...")
    forecasts, forecast_metrics = expanding_inflation_forecast(
        phillips, initial_window=80
    )
    forecasts.to_csv(table_dir / "inflation_forecasts.csv")
    forecast_metrics.to_csv(table_dir / "forecast_accuracy.csv", index=False)
    plot_inflation_forecasts(forecasts, figure_dir)

    print(f"Analysis complete. Outputs saved in: {output_dir.resolve()}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproducible analysis of U.S. macro and labour-market data."
    )
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("analysis_outputs")
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Redownload all FRED series instead of using cached raw files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_analysis(
        start=args.start,
        end=args.end,
        output_dir=args.output_dir,
        refresh=args.refresh,
    )
