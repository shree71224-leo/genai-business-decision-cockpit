import pandas as pd


def analyze_patterns(df, mapping):
    """Identify important business patterns."""

    insights = []

    sales_col = mapping.get("sales")
    profit_col = mapping.get("profit")
    category_col = mapping.get("category")
    region_col = mapping.get("region")

    # Sales analysis
    if sales_col:
        sales = pd.to_numeric(df[sales_col], errors="coerce")

        if sales.notna().any():
            insights.append(
                f"Total Sales: {sales.sum():,.2f}"
            )

            if category_col:
                category_sales = df.assign(
                    _sales=sales
                ).groupby(category_col)["_sales"].sum()

                if not category_sales.empty:
                    best = category_sales.idxmax()
                    insights.append(
                        f"Top Sales Category: {best}"
                    )

            if region_col:
                region_sales = df.assign(
                    _sales=sales
                ).groupby(region_col)["_sales"].sum()

                if not region_sales.empty:
                    best_region = region_sales.idxmax()
                    insights.append(
                        f"Best Performing Region: {best_region}"
                    )

    # Profit analysis
    if profit_col:
        profit = pd.to_numeric(
            df[profit_col], errors="coerce"
        )

        if profit.notna().any():
            insights.append(
                f"Total Profit: {profit.sum():,.2f}"
            )

            loss_count = (profit < 0).sum()

            insights.append(
                f"Loss-Making Records: {loss_count}"
            )

    if not insights:
        insights.append(
            "Insufficient numerical data for pattern analysis."
        )

    return insights