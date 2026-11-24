import pandas as pd
from modules.helpers import clean_number, safe_sum


def calculate_business_metrics(df, mapping):
    """Calculate important business KPIs."""

    sales_col = mapping.get("sales")
    profit_col = mapping.get("profit")
    quantity_col = mapping.get("quantity")
    id_col = mapping.get("id")

    total_sales = safe_sum(df[sales_col]) if sales_col else 0
    total_profit = safe_sum(df[profit_col]) if profit_col else 0

    total_orders = (
        df[id_col].nunique()
        if id_col else len(df)
    )

    loss_records = (
        (clean_number(df[profit_col]) < 0).sum()
        if profit_col else 0
    )

    units_sold = (
        safe_sum(df[quantity_col])
        if quantity_col else 0
    )

    return {
        "total_sales": total_sales,
        "total_profit": total_profit,
        "total_orders": total_orders,
        "loss_records": int(loss_records),
        "units_sold": units_sold
    }


def dataframe_summary(df):
    """Generate a summary of the dataset for AI analysis."""

    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "data_types": {
            col: str(dtype)
            for col, dtype in df.dtypes.items()
        }
    }

    return summary