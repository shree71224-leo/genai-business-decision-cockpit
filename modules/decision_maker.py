def generate_recommendations(df, mapping):
    """Generate actionable business recommendations."""

    recommendations = []

    sales_col = mapping.get("sales")
    profit_col = mapping.get("profit")
    category_col = mapping.get("category")
    region_col = mapping.get("region")

    # Profit analysis
    if profit_col:
        profit = df[profit_col].sum()

        if profit < 0:
            recommendations.append(
                "Overall profit is negative. Review operating costs, "
                "pricing, and discount strategies."
            )
        elif profit > 0:
            recommendations.append(
                "The business is profitable. Focus on maintaining "
                "healthy profit margins."
            )

    # Sales analysis
    if sales_col and category_col:
        category_sales = df.groupby(category_col)[sales_col].sum()

        if not category_sales.empty:
            best_category = category_sales.idxmax()

            recommendations.append(
                f"Focus on the {best_category} category, "
                "which generates the highest sales."
            )

    # Regional analysis
    if sales_col and region_col:
        region_sales = df.groupby(region_col)[sales_col].sum()

        if not region_sales.empty:
            best_region = region_sales.idxmax()

            recommendations.append(
                f"Study successful strategies in {best_region} "
                "and consider applying them to other regions."
            )

    if not recommendations:
        recommendations.append(
            "Upload a dataset containing sales, profit, "
            "or category information for recommendations."
        )

    return recommendations