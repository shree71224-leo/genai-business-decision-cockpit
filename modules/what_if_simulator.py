def simulate_discount(df, mapping, discount_percent):
    """Estimate the impact of an additional discount."""

    sales_col = mapping.get("sales")
    profit_col = mapping.get("profit")

    if not sales_col or not profit_col:
        return {
            "error": "Sales and Profit columns are required."
        }

    if not 0 <= discount_percent <= 100:
        return {
            "error": "Discount must be between 0 and 100."
        }

    sales = df[sales_col].sum()
    profit = df[profit_col].sum()

    estimated_cost = sales - profit

    new_sales = sales * (1 - discount_percent / 100)

    new_profit = new_sales - estimated_cost

    return {
        "original_sales": float(sales),
        "original_profit": float(profit),
        "discount_percent": discount_percent,
        "simulated_sales": float(new_sales),
        "simulated_profit": float(new_profit),
        "profit_change": float(new_profit - profit)
    }