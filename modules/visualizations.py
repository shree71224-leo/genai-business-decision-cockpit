import plotly.express as px


def sales_by_category(df, mapping):
    category = mapping.get("category")
    sales = mapping.get("sales")

    if not category or not sales:
        return None

    data = df.groupby(category, as_index=False)[sales].sum()

    return px.bar(
        data,
        x=category,
        y=sales,
        title="Sales by Category",
        color=category
    )


def sales_by_region(df, mapping):
    region = mapping.get("region")
    sales = mapping.get("sales")

    if not region or not sales:
        return None

    data = df.groupby(region, as_index=False)[sales].sum()

    return px.bar(
        data,
        x=region,
        y=sales,
        title="Sales by Region",
        color=region
    )


def profit_by_category(df, mapping):
    category = mapping.get("category")
    profit = mapping.get("profit")

    if not category or not profit:
        return None

    data = df.groupby(category, as_index=False)[profit].sum()

    return px.bar(
        data,
        x=category,
        y=profit,
        title="Profit by Category",
        color=category
    )


def monthly_sales_trend(df, mapping):
    date = mapping.get("date")
    sales = mapping.get("sales")

    if not date or not sales:
        return None

    data = df.copy()
    data[date] = __import__("pandas").to_datetime(
        data[date], errors="coerce"
    )

    data = data.dropna(subset=[date])
    data["Month"] = data[date].dt.to_period("M").astype(str)

    monthly = data.groupby("Month", as_index=False)[sales].sum()

    return px.line(
        monthly,
        x="Month",
        y=sales,
        title="Monthly Sales Trend",
        markers=True
    )


def sales_by_category_pie(df, mapping):
    category = mapping.get("category")
    sales = mapping.get("sales")

    if not category or not sales:
        return None

    data = df.groupby(category, as_index=False)[sales].sum()

    return px.pie(
        data,
        names=category,
        values=sales,
        title="Sales Distribution by Category",
        hole=0.35
    )
