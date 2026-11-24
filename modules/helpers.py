import re
import pandas as pd


def clean_number(series):
    """Convert text-based numbers into numeric values."""

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )

    return pd.to_numeric(cleaned, errors="coerce")


def normalized_name(name):
    """Normalize column names and text for matching."""

    return re.sub(
        r"[^a-z0-9]+",
        " ",
        str(name).lower()
    ).strip()


def guess_column(columns, include=(), exclude=()):
    """Automatically identify columns using keywords."""

    for col in columns:
        name = normalized_name(col)

        if any(term in name for term in exclude):
            continue

        if any(term == name for term in include):
            return col

    for col in columns:
        name = normalized_name(col)

        if (
            any(term in name for term in include)
            and not any(term in name for term in exclude)
        ):
            return col

    return None


def money(value):
    """Format a number with commas and two decimal places."""

    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value)


def safe_sum(series):
    """Calculate the sum safely."""

    return float(
        pd.to_numeric(series, errors="coerce").sum()
    )