import pandas as pd
from modules.helpers import clean_number, guess_column


def prepare_dataset(df):
    """Clean the dataset and identify numeric and date columns."""

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    df = df.loc[:, ~pd.Index(df.columns).duplicated()].copy()

    for col in df.columns:
        if any(word in str(col).lower() for word in ["date", "time"]):
            parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

            if parsed.notna().sum() >= max(2, int(len(df) * 0.4)):
                df[col] = parsed

    for col in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            converted = clean_number(df[col])
            nonempty = df[col].notna().sum()

            if nonempty and converted.notna().sum() >= max(2, int(nonempty * 0.8)):
                df[col] = converted

    return df


def detect_column_types(df):
    """Identify numeric, date, and categorical columns."""

    numeric_cols = list(df.select_dtypes(include="number").columns)

    date_cols = [
        c for c in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[c])
    ]

    categorical_cols = [
        c for c in df.columns
        if c not in numeric_cols
        and c not in date_cols
        and df[c].nunique(dropna=True) <= max(100, int(len(df) * 0.5))
    ]

    return numeric_cols, date_cols, categorical_cols


def suggest_column_mappings(df, numeric_cols, date_cols, categorical_cols):
    """Suggest business-related columns automatically."""

    return {
        "sales": guess_column(numeric_cols, ["sales", "revenue", "total sales", "total revenue"]),
        "profit": guess_column(numeric_cols, ["profit", "net profit", "margin", "earnings"]),
        "quantity": guess_column(numeric_cols, ["quantity", "qty", "units"]),
        "discount": guess_column(numeric_cols, ["discount", "discount rate", "markdown"]),
        "loss": guess_column(numeric_cols, ["loss", "loss amount"]),
        "date": guess_column(date_cols, ["order date", "transaction date", "date", "purchase date"]),
        "category": guess_column(categorical_cols, ["category", "product category", "sub category", "department"]),
        "product": guess_column(categorical_cols, ["product name", "product", "item name", "item"]),
        "region": guess_column(categorical_cols, ["region", "territory", "state", "city", "location", "country"]),
        "id": guess_column(df.columns, ["order id", "transaction id", "record id", "invoice"]),
    }