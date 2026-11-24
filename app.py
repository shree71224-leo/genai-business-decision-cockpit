import os
import re
import io
import pandas as pd
import streamlit as st
from pathlib import Path

try:
    from google import genai
except ImportError:
    genai = None

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="GenAI Business Decision Cockpit",
    page_icon="📊",
    layout="wide",
)

st.title("📊 GenAI-Powered Business Decision Cockpit")
st.caption("Upload business data, explore interactive charts, ask questions, and generate AI-assisted insights.")

# Load environment variables from .env
env_file = Path(__file__).parent / ".env"

if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
# ============================================================
# GEMINI CONFIGURATION
# ============================================================
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
client = genai.Client(api_key=API_KEY) if genai and API_KEY else None

# ============================================================
# HELPERS
# ============================================================
def read_dataset(upload):
    upload.seek(0)
    name = upload.name.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(upload)
        except UnicodeDecodeError:
            upload.seek(0)
            return pd.read_csv(upload, encoding="latin-1")
    if name.endswith(".xls"):
        return pd.read_excel(upload, engine="xlrd")
    return pd.read_excel(upload)

def clean_number(series):
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
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()

def guess_column(columns, include=(), exclude=()):
    for col in columns:
        name = normalized_name(col)
        if any(term in name for term in exclude):
            continue
        if any(term == name for term in include):
            return col
    for col in columns:
        name = normalized_name(col)
        if any(term in name for term in include) and not any(term in name for term in exclude):
            return col
    return None

def money(value):
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value)

def safe_sum(series):
    return float(pd.to_numeric(series, errors="coerce").sum())

def dataframe_summary(data, sales, profit, quantity, discount, date, category, region, product, loss):
    parts = [
        f"Dataset has {len(data):,} rows and {len(data.columns)} columns.",
        f"Total missing cells: {int(data.isna().sum().sum()):,}.",
        f"Column names: {', '.join(map(str, data.columns))}.",
    ]
    for label, col in [
        ("Sales/revenue", sales), ("Profit", profit), ("Quantity", quantity),
        ("Discount", discount), ("Loss amount", loss)
    ]:
        if col:
            vals = clean_number(data[col])
            parts.append(
                f"{label} column '{col}': sum={vals.sum():,.2f}, "
                f"average={vals.mean():,.2f}, minimum={vals.min():,.2f}, maximum={vals.max():,.2f}."
            )
    if profit:
        profits = clean_number(data[profit])
        parts.append(
            f"Rows with negative profit: {int((profits < 0).sum()):,}; "
            f"rows with zero or positive profit: {int((profits >= 0).sum()):,}."
        )
    for label, group_col in [("category", category), ("region", region), ("product", product)]:
        if group_col:
            for metric_label, metric_col in [("sales", sales), ("profit", profit), ("quantity", quantity)]:
                if metric_col and metric_col != group_col:
                    temp = data[[group_col]].copy()
                    temp["__metric"] = clean_number(data[metric_col])
                    grouped = temp.groupby(group_col, dropna=False)["__metric"].sum().sort_values(ascending=False)
                    parts.append(f"Total {metric_label} by {label} ({group_col}), all groups:\n{grouped.to_string()}")
    if date:
        parsed = pd.to_datetime(data[date], errors="coerce")
        if parsed.notna().any():
            parts.append(f"Date range in '{date}': {parsed.min().date()} through {parsed.max().date()}.")
    return "\n\n".join(parts)

# ============================================================
# DATASET UPLOAD
# ============================================================
uploaded = st.file_uploader("Upload a CSV or Excel dataset", type=["csv", "xlsx", "xls"])
if uploaded is None:
    st.info("Upload your dataset to begin. The app accepts CSV, XLSX, and XLS files.")
    st.stop()

try:
    df = read_dataset(uploaded)
except Exception as exc:
    st.error(f"Could not read the uploaded file: {exc}")
    st.stop()

df.columns = [str(c).strip() for c in df.columns]
df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
df = df.loc[:, ~pd.Index(df.columns).duplicated()].copy()
if df.empty:
    st.error("The uploaded file contains no usable data.")
    st.stop()

# Detect and parse likely date columns.
for col in df.columns:
    name = normalized_name(col)
    if any(token in name for token in ["date", "time"]):
        parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
        if parsed.notna().sum() >= max(2, int(len(df) * 0.4)):
            df[col] = parsed

# Find numeric columns, including numeric values stored as text.
for col in df.columns:
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        converted = clean_number(df[col])
        nonempty = df[col].notna().sum()
        if nonempty and converted.notna().sum() >= max(2, int(nonempty * 0.8)):
            df[col] = converted

numeric_cols = list(df.select_dtypes(include="number").columns)
date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
categorical_cols = [
    c for c in df.columns
    if c not in numeric_cols and c not in date_cols
    and df[c].nunique(dropna=True) <= max(100, int(len(df) * 0.5))
]

st.success(f"Loaded **{uploaded.name}** — {len(df):,} rows × {len(df.columns)} columns.")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Rows", f"{len(df):,}")
k2.metric("Columns", f"{len(df.columns):,}")
k3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
k4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")

with st.expander("Preview uploaded data", expanded=False):
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)
with st.expander("Data quality details", expanded=False):
    quality = pd.DataFrame({
        "Column": df.columns,
        "Data type": [str(df[c].dtype) for c in df.columns],
        "Missing values": [int(df[c].isna().sum()) for c in df.columns],
        "Unique values": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    st.dataframe(quality, use_container_width=True, hide_index=True)

missing = pd.DataFrame({
    "Column": df.columns,
    "Missing Values": [int(df[c].isna().sum()) for c in df.columns],
    "Data Type": [str(df[c].dtype) for c in df.columns],
})
missing = missing[missing["Missing Values"] > 0]
st.subheader("Missing Values Diagnostic")
if missing.empty:
    st.success("No missing cells detected.")
else:
    st.dataframe(missing, use_container_width=True, hide_index=True)

# ============================================================
# COLUMN MAPPING
# ============================================================
st.header("⚙️ Configure Analysis")
st.caption("Check the suggested mappings. Change them if the app selected the wrong columns.")

def mapping(label, options, suggestion, key):
    choices = ["— Not selected —"] + list(options)
    idx = choices.index(suggestion) if suggestion in choices else 0
    value = st.selectbox(label, choices, index=idx, key=key)
    return None if value == "— Not selected —" else value

sales_guess = guess_column(numeric_cols, ["sales", "revenue", "total sales", "total revenue"])
profit_guess = guess_column(numeric_cols, ["profit", "net profit", "margin", "earnings"])
quantity_guess = guess_column(numeric_cols, ["quantity", "qty", "units"])
discount_guess = guess_column(numeric_cols, ["discount", "discount rate", "markdown"])
loss_guess = guess_column(numeric_cols, ["loss", "loss amount"])
date_guess = guess_column(date_cols, ["order date", "transaction date", "date", "purchase date"])
category_guess = guess_column(categorical_cols, ["category", "product category", "sub category", "department"])
product_guess = guess_column(categorical_cols, ["product name", "product", "item name", "item"])
region_guess = guess_column(categorical_cols, ["region", "territory", "state", "city", "location", "country"])
id_guess = guess_column(df.columns, ["order id", "transaction id", "record id", "invoice"])

with st.expander("Column mappings", expanded=True):
    a, b = st.columns(2)
    with a:
        sales_col = mapping("Sales / Revenue", numeric_cols, sales_guess, "map_sales")
        profit_col = mapping("Profit", numeric_cols, profit_guess, "map_profit")
        quantity_col = mapping("Quantity", numeric_cols, quantity_guess, "map_quantity")
        discount_col = mapping("Discount", numeric_cols, discount_guess, "map_discount")
        loss_col = mapping("Loss amount (optional)", numeric_cols, loss_guess, "map_loss")
    with b:
        date_col = mapping("Date", list(dict.fromkeys(date_cols + list(df.columns))), date_guess, "map_date")
        category_col = mapping("Category / Group", categorical_cols, category_guess, "map_category")
        product_col = mapping("Product / Item", categorical_cols, product_guess, "map_product")
        region_col = mapping("Region / Location", categorical_cols, region_guess, "map_region")
        id_col = mapping("Order / Record ID", list(df.columns), id_guess, "map_id")

# Normalize mapped numeric columns so all calculations work consistently.
for col in [sales_col, profit_col, quantity_col, discount_col, loss_col]:
    if col:
        df[col] = clean_number(df[col])

# ============================================================
# BUSINESS OVERVIEW
# ============================================================
st.header("📈 Business Overview")
metrics = []
if sales_col:
    metrics.append(("Total Sales", safe_sum(df[sales_col])))
if profit_col:
    metrics.append(("Total Profit", safe_sum(df[profit_col])))
if id_col:
    metrics.append(("Unique Orders", int(df[id_col].nunique(dropna=True))))
if profit_col:
    metrics.append(("Loss-Making Records", int((df[profit_col] < 0).sum())))
elif loss_col:
    metrics.append(("Loss-Making Records", int((df[loss_col] > 0).sum())))
if quantity_col:
    metrics.append(("Units Sold", safe_sum(df[quantity_col])))

if metrics:
    for start_idx in range(0, len(metrics), 4):
        cols = st.columns(min(4, len(metrics) - start_idx))
        for col_ui, (label, value) in zip(cols, metrics[start_idx:start_idx + 4]):
            col_ui.metric(label, f"{value:,.2f}" if isinstance(value, float) else f"{value:,}")
else:
    st.info("Map numeric sales, profit, or quantity columns to show overview metrics.")

# ============================================================
# INTERACTIVE VISUALIZATIONS
# ============================================================
st.header("📊 Visualizations & Charts")
if not numeric_cols:
    st.warning("No numeric columns were detected, so numeric charts cannot be generated.")
else:
    chart_a, chart_b, chart_c = st.columns(3)
    with chart_a:
        chart_metric = st.selectbox("Metric to visualize", numeric_cols, index=numeric_cols.index(sales_col) if sales_col in numeric_cols else 0)
    with chart_b:
        group_options = list(dict.fromkeys(categorical_cols + date_cols))
        chart_group = st.selectbox("Group / X-axis", group_options, index=group_options.index(category_col) if category_col in group_options else 0) if group_options else None
    with chart_c:
        chart_type = st.selectbox(
    "Chart type",
    ["Bar chart", "Line chart", "Area chart", "Scatter chart", "Histogram", "Pie chart"],
    index=0
)

    if chart_type == "Histogram":
        hist_data = pd.DataFrame({chart_metric: clean_number(df[chart_metric]).dropna()})
        if not hist_data.empty:
            st.bar_chart(hist_data[chart_metric].value_counts().sort_index().head(50), use_container_width=True)
        else:
            st.info("No valid numeric values for this histogram.")
    elif chart_type == "Scatter chart":
        scatter_x = st.selectbox("X-axis numeric field", numeric_cols, index=0, key="scatter_x")
        scatter_y = st.selectbox("Y-axis numeric field", numeric_cols, index=min(1, len(numeric_cols)-1), key="scatter_y")
        scatter = pd.DataFrame({"x": clean_number(df[scatter_x]), "y": clean_number(df[scatter_y])}).dropna()
        if not scatter.empty:
            st.scatter_chart(scatter, x="x", y="y", use_container_width=True)
        else:
            st.info("No valid values for the selected scatter chart.")
    elif chart_group:
        chart_df = df[[chart_group]].copy()
        chart_df[chart_metric] = clean_number(df[chart_metric])
        if chart_group in date_cols:
            chart_df[chart_group] = pd.to_datetime(chart_df[chart_group], errors="coerce").dt.to_period("M").astype(str)
        grouped = chart_df.dropna(subset=[chart_metric]).groupby(chart_group, dropna=False)[chart_metric].sum().sort_values(ascending=False)
        if chart_type in ["Line chart", "Area chart"] and chart_group in date_cols:
            grouped = chart_df.dropna(subset=[chart_metric]).groupby(chart_group)[chart_metric].sum().sort_index()
        grouped = grouped.head(30) if chart_type == "Bar chart" else grouped
        plot_df = grouped.rename(chart_metric).to_frame()
        if not plot_df.empty:
            if chart_type == "Line chart":
                st.line_chart(plot_df, use_container_width=True)
            elif chart_type == "Area chart":
                st.area_chart(plot_df, use_container_width=True)
            else:
                st.bar_chart(plot_df, use_container_width=True)
            st.dataframe(grouped.rename("Total").reset_index(), use_container_width=True, hide_index=True)
        else:
            st.info("No valid grouped values for this chart.")
    else:
        st.info("Add at least one categorical or date column to group the chart.")

# Quick business charts where mappings are available.
if category_col and sales_col:
    st.subheader("Sales by Category / Product")
    tmp = pd.DataFrame({"group": df[category_col].astype(str), "sales": clean_number(df[sales_col])})
    top_sales = tmp.groupby("group")["sales"].sum().sort_values(ascending=False).head(15)
    if not top_sales.empty:
        st.bar_chart(top_sales, use_container_width=True)
if region_col and profit_col:
    st.subheader("Profit by Region / Location")
    tmp = pd.DataFrame({"group": df[region_col].astype(str), "profit": clean_number(df[profit_col])})
    region_profit = tmp.groupby("group")["profit"].sum().sort_values(ascending=False)
    if not region_profit.empty:
        st.bar_chart(region_profit, use_container_width=True)

if date_col and sales_col:
    parsed_dates = pd.to_datetime(df[date_col], errors="coerce")
    if parsed_dates.notna().any():
        trend = pd.DataFrame({"month": parsed_dates.dt.to_period("M").astype(str), "sales": clean_number(df[sales_col])})
        trend = trend.dropna(subset=["sales"]).groupby("month")["sales"].sum().sort_index()
        if not trend.empty:
            st.subheader("Sales Trend Over Time")
            st.line_chart(trend, use_container_width=True)

# ============================================================
# DETERMINISTIC QUESTION ANSWERING + GEMINI EXPLANATION
# ============================================================
def answer_from_data(question_text):
    """Answer common business questions directly from the uploaded dataframe."""
    q = normalized_name(question_text)
    answers = []

    def has_any(terms):
        return any(normalized_name(term) in q for term in terms)

    def rank_for(metric_col, metric_name, group_col, label, highest):
        if not metric_col or not group_col:
            return None
        values = pd.DataFrame({
            "group": df[group_col].fillna("(Missing)").astype(str).str.strip(),
            "value": clean_number(df[metric_col]),
        }).dropna(subset=["value"])
        if values.empty:
            return None
        totals = values.groupby("group", dropna=False)["value"].sum()
        if totals.empty:
            return None
        name = totals.idxmax() if highest else totals.idxmin()
        amount = totals.max() if highest else totals.min()
        qualifier = "highest" if highest else "lowest"
        return f"The {label} with the {qualifier} total {metric_name} is '{name}', with {amount:,.2f}."

    # Simple dataset-wide questions
    if has_any(["missing value", "missing cell", "null value", "null cell"]):
        return f"The dataset contains {int(df.isna().sum().sum()):,} missing cells across {int((df.isna().sum() > 0).sum())} columns."
    if has_any(["how many rows", "number of rows", "total records", "total rows", "how many records"]):
        return f"The dataset contains {len(df):,} rows (records)."
    if has_any(["loss making", "loss-making", "negative profit", "number of losses"]):
        if profit_col:
            count = int((clean_number(df[profit_col]) < 0).sum())
            pct = count / len(df) * 100 if len(df) else 0
            return f"There are {count:,} loss-making records ({pct:.2f}% of {len(df):,} rows), defined as rows where '{profit_col}' is negative."
        if loss_col:
            count = int((clean_number(df[loss_col]) > 0).sum())
            return f"There are {count:,} loss-making records based on positive values in '{loss_col}'."
    if sales_col and has_any(["total sales", "sales total", "total revenue", "revenue total"]):
        return f"Total sales/revenue ({sales_col}) is {safe_sum(df[sales_col]):,.2f}."
    if profit_col and has_any(["total profit", "profit total", "overall profit", "total earnings"]):
        return f"Total profit ({profit_col}) is {safe_sum(df[profit_col]):,.2f}."
    if quantity_col and has_any(["total quantity", "total units", "units sold", "quantity sold"]):
        return f"Total quantity/units ({quantity_col}) is {safe_sum(df[quantity_col]):,.2f}."

    # Ranking questions must be handled BEFORE matching a product/region name.
    ranking_high = has_any(["highest", "maximum", "most", "best", "top"])
    ranking_low = has_any(["lowest", "minimum", "least", "worst"])
    metric_col = None
    metric_name = None
    if profit_col and has_any(["profit", "earnings", "most profitable", "least profitable"]):
        metric_col, metric_name = profit_col, "profit"
    elif sales_col and has_any(["sales", "revenue"]):
        metric_col, metric_name = sales_col, "sales"
    elif quantity_col and has_any(["quantity", "units sold"]):
        metric_col, metric_name = quantity_col, "quantity"

    if metric_col and (ranking_high or ranking_low):
        if has_any(["product", "item"]):
            candidates = [(product_col, "product")]
        elif has_any(["category", "categories"]):
            candidates = [(category_col, "category")]
        elif has_any(["region", "location", "city", "country"]):
            candidates = [(region_col, "region/location")]
        else:
            # Default to products for questions like "which made the most profit?"
            candidates = [(product_col, "product"), (category_col, "category"), (region_col, "region/location")]

        for group_col, label in candidates:
            if group_col:
                if ranking_high:
                    answer = rank_for(metric_col, metric_name, group_col, label, True)
                    if answer:
                        answers.append(answer)
                if ranking_low:
                    answer = rank_for(metric_col, metric_name, group_col, label, False)
                    if answer:
                        answers.append(answer)
                if answers:
                    return "\n\n".join(answers)

    # Exact name matching uses normalized whole phrases, preventing "West"
    # from matching the letters in "lowest".
    for group_col, label in [(product_col, "product"), (category_col, "category"), (region_col, "region/location")]:
        if not group_col:
            continue
        candidates = sorted(
            [str(v).strip() for v in df[group_col].dropna().unique() if str(v).strip()],
            key=len, reverse=True,
        )
        matched = None
        for candidate in candidates:
            candidate_norm = normalized_name(candidate)
            if candidate_norm and re.search(r"(?<![a-z0-9])" + re.escape(candidate_norm) + r"(?![a-z0-9])", q):
                matched = candidate
                break
        if matched:
            subset = df[df[group_col].astype(str).str.strip().str.lower() == matched.lower()]
            if profit_col and has_any(["profit", "earning", "profitable", "margin"]):
                return f"Total profit for {label} '{matched}' is {safe_sum(subset[profit_col]):,.2f}, calculated from {len(subset):,} matching rows."
            if sales_col and has_any(["sales", "revenue", "sold"]):
                return f"Total sales for {label} '{matched}' is {safe_sum(subset[sales_col]):,.2f}, calculated from {len(subset):,} matching rows."
            if quantity_col and has_any(["quantity", "units"]):
                return f"Total quantity for {label} '{matched}' is {safe_sum(subset[quantity_col]):,.2f}, calculated from {len(subset):,} matching rows."

    return None

st.header("🤖 AI Business Analyst")
st.write("Ask a question in everyday language. Exact totals and group comparisons are calculated from the uploaded data first.")
question = st.text_input(
    "Ask a question about this dataset",
    placeholder="Example: What is the total profit made by Mouse?",
    key="analyst_question"
)
if st.button("Analyze Question", type="primary", key="analyst_submit"):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        deterministic_answer = answer_from_data(question)
        if deterministic_answer:
            # Prefer exact local calculations. This also keeps common questions
            # working when Gemini is rate-limited or temporarily unavailable.
            st.session_state["analyst_answer"] = deterministic_answer
        elif client is None:
            st.session_state["analyst_answer"] = (
                "I could not calculate that question with the built-in rules. "
                "Gemini is not configured, so try asking about total sales, total profit, "
                "highest/lowest product profit, sales by product/category/region, "
                "loss-making records, quantity, or missing values."
            )
        else:
            evidence = dataframe_summary(df, sales_col, profit_col, quantity_col, discount_col, date_col, category_col, region_col, product_col, loss_col)
            prompt = f"""You are a careful business data analyst. Answer the user's question clearly using only the supplied evidence.
Never invent numbers or columns. Distinguish observed associations from proven causes.
If evidence is insufficient, say so and explain what field is needed.

DATASET EVIDENCE:
{evidence}

USER QUESTION:
{question}
"""
            try:
                with st.spinner("Analyzing your question..."):
                    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
                ai_text = getattr(response, "text", None)
                st.session_state["analyst_answer"] = ai_text or "The AI returned an empty response."
            except Exception:
                st.session_state["analyst_answer"] = (
                    "This question needs AI interpretation, but Gemini is temporarily unavailable "
                    "or its API quota has been reached. Try a question about totals, rankings, "
                    "products, categories, regions, quantities, losses, or missing values; "
                    "those are calculated locally without Gemini."
                )

if st.session_state.get("analyst_answer"):
    st.markdown("### AI Analyst Response")
    st.write(st.session_state["analyst_answer"])

# ============================================================
# PATTERN ANALYSIS
# ============================================================
st.header("🧠 Root-Cause / Pattern Analysis")
st.caption(
    "This section reports observed patterns; "
    "descriptive data alone cannot prove causation."
)

if st.button("Analyze Dataset Patterns", key="pattern_button"):
    evidence = dataframe_summary(
        df,
        sales_col,
        profit_col,
        quantity_col,
        discount_col,
        date_col,
        category_col,
        region_col,
        product_col,
        loss_col,
    )

    if client is None:
        st.session_state["pattern_result"] = (
            "AI analysis is unavailable because Gemini is not configured. "
            "Configure your API key to enable AI explanations."
        )
    else:
        prompt = f"""
Analyze the following dataset evidence.

Return:
1. Three observed findings with numbers.
2. Possible explanations clearly labelled as hypotheses.
3. Practical next steps.
4. Limitations.

Do not invent facts or claim causation.

DATASET EVIDENCE:
{evidence}
"""
        try:
            with st.spinner("Analyzing dataset patterns..."):
                result = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                )

            st.session_state["pattern_result"] = (
                getattr(result, "text", None)
                or "No response was returned. Please try again."
            )

        except Exception:
            # Keep this feature useful even when the external AI service is unavailable.
            local_findings = []
            if sales_col:
                local_findings.append(f"Total sales: {safe_sum(df[sales_col]):,.2f}.")
            if profit_col:
                p = clean_number(df[profit_col])
                local_findings.append(f"Total profit: {p.sum():,.2f}; negative-profit rows: {int((p < 0).sum()):,}.")
                if category_col:
                    g = pd.DataFrame({"group": df[category_col].astype(str), "profit": p}).groupby("group")["profit"].sum()
                    if not g.empty:
                        local_findings.append(f"Highest-profit category: {g.idxmax()} ({g.max():,.2f}); lowest-profit category: {g.idxmin()} ({g.min():,.2f}).")
                if product_col:
                    g = pd.DataFrame({"group": df[product_col].astype(str), "profit": p}).groupby("group")["profit"].sum()
                    if not g.empty:
                        local_findings.append(f"Highest-profit product: {g.idxmax()} ({g.max():,.2f}); lowest-profit product: {g.idxmin()} ({g.min():,.2f}).")
            st.session_state["pattern_result"] = (
                "Gemini is temporarily unavailable or its API quota is exhausted. "
                "Here are findings calculated directly from your dataset:\n\n- "
                + "\n- ".join(local_findings)
                + "\n\nThese are descriptive findings and do not establish causation."
            )

if st.session_state.get("pattern_result"):
    st.write(st.session_state["pattern_result"])


# ============================================================
# DECISION MAKER - AI-POWERED BUSINESS RECOMMENDATIONS
# ============================================================
st.header("🎯 Decision Maker — Business Recommendations")

st.caption(
    "Turn your sales and profit analysis into practical business actions. "
    "Recommendations are based on the uploaded dataset."
)

decision_goal = st.selectbox(
    "What is your main business objective?",
    [
        "Reduce losses",
        "Maximize profit",
        "Increase sales",
        "Optimize discount strategy",
        "Improve regional performance",
    ],
    key="decision_goal",
)

if st.button("Generate Business Decisions", key="decision_maker_button"):

    required_cols = [sales_col, profit_col]

    if not all(required_cols):
        st.warning("Please map the Sales and Profit columns first.")
    else:
        sales_values = clean_number(df[sales_col])
        profit_values = clean_number(df[profit_col])

        analysis_df = pd.DataFrame({
            "sales": sales_values,
            "profit": profit_values,
        }).dropna()

        if analysis_df.empty:
            st.warning("There is not enough valid data to generate recommendations.")
        else:
            total_sales = analysis_df["sales"].sum()
            total_profit = analysis_df["profit"].sum()
            loss_count = int((analysis_df["profit"] < 0).sum())
            loss_total = analysis_df.loc[
                analysis_df["profit"] < 0, "profit"
            ].sum()

            st.subheader("📊 Business Situation")
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Sales", f"₹{total_sales:,.2f}")
            m2.metric("Total Profit", f"₹{total_profit:,.2f}")
            m3.metric("Loss-Making Orders", loss_count)

            st.subheader("💡 Recommended Actions")

            if decision_goal == "Reduce losses":
                st.markdown(
                    "**Priority: HIGH**\n\n"
                    f"- Audit the {loss_count} orders with negative profit.\n"
                    f"- Investigate the transactions responsible for "
                    f"the combined negative profit of ₹{abs(loss_total):,.2f}.\n"
                    "- Check product costs, discounts, and unusual transactions "
                    "before changing prices."
                )

            elif decision_goal == "Maximize profit":
                st.markdown(
                    "**Priority: HIGH**\n\n"
                    "- Identify products with high sales and low or negative profit.\n"
                    "- Compare profit margins before allocating more inventory "
                    "or marketing budget.\n"
                    "- Prioritize products with consistently positive profit."
                )

            elif decision_goal == "Increase sales":
                st.markdown(
                    "**Priority: MEDIUM**\n\n"
                    "- Review products and categories generating the most sales.\n"
                    "- Compare sales across regions and customer types.\n"
                    "- Test promotional campaigns on a limited scale and "
                    "measure their effect on both sales and profit."
                )

            elif decision_goal == "Optimize discount strategy":
                st.markdown(
                    "**Priority: HIGH**\n\n"
                    "- Compare discounts against profit at product and order level.\n"
                    "- Identify transactions where discounts coincide with losses.\n"
                    "- Test lower discount levels using the What-If Simulator "
                    "before changing the actual pricing policy."
                )

            elif decision_goal == "Improve regional performance":
                st.markdown(
                    "**Priority: MEDIUM**\n\n"
                    "- Compare total sales, profit, and profit margin by region.\n"
                    "- Investigate regions with high sales but low or negative profit.\n"
                    "- Check whether product mix, discounting, or costs explain "
                    "the differences."
                )

            st.info(
                "These are initial recommendations based on descriptive data. "
                "Validate the relevant transactions and costs before taking action."
            )

            # Optional AI explanation using the Gemini client already configured
            if "client" in globals() and client is not None:
                try:
                    prompt = f"""
You are a careful business analyst.

Business objective: {decision_goal}
Total sales: {total_sales:.2f}
Total profit: {total_profit:.2f}
Loss-making orders: {loss_count}
Combined profit of loss-making orders: {loss_total:.2f}

Explain the key business implications and provide three concise,
actionable recommendations. Distinguish observed facts from hypotheses.
Do not invent facts or claim that correlation proves causation.
"""
                    response = client.models.generate_content(
                        model=MODEL_NAME,
                        contents=prompt,
                    )
                    st.subheader("🤖 AI Decision Explanation")
                    st.markdown(response.text)
               
                except Exception:
                    st.caption(
                        "AI explanation is temporarily unavailable or the API quota is exhausted. "
                        "The data-based recommendations above remain available."
                    )

            else:
                st.caption(
                    "Gemini is not configured. The data-based recommendations "
                    "above are still available."
                )



# ============================================================
# WHAT-IF DISCOUNT SIMULATOR
# ============================================================
st.header("🎛️ What-If Discount Simulator")

if sales_col and profit_col and discount_col:
    st.caption(
        "Estimate only: costs remain fixed, and demand changes are not modeled."
    )

    discount_values = clean_number(df[discount_col]).dropna()

    max_discount = (
        float(discount_values.max())
        if not discount_values.empty
        else 0.30
    )
    max_discount = min(max(max_discount, 0.30), 0.90)

    new_discount_pct = st.slider(
        "Scenario discount rate",
        min_value=0,
        max_value=int(round(max_discount * 100)),
        value=min(10, int(round(max_discount * 100))),
        step=1,
        format="%d%%",
        key="scenario_discount_slider"
    )

    new_discount = new_discount_pct / 100

    if st.button("Run What-If Analysis", key="what_if_button"):

        # Validate the required columns
        if "gross_amount" not in df.columns or "cost" not in df.columns:
            st.error(
                "This simulation requires gross_amount and cost columns. "
                "Please check your dataset or map these columns first."
            )
        else:
            base = pd.DataFrame({
                "gross": clean_number(df["gross_amount"]),
                "cost": clean_number(df["cost"]),
                "profit": clean_number(df[profit_col]),
            }).dropna()

            if base.empty:
                st.warning(
                    "No valid rows are available for this simulation."
                )
            else:
                # Estimate sales after the selected discount
                estimated_sales = base["gross"] * (1 - new_discount)

                # Estimate profit while holding costs fixed
                estimated_profit = estimated_sales - base["cost"]

                current_profit = base["profit"].sum()
                scenario_profit = estimated_profit.sum()
                profit_change = scenario_profit - current_profit

                w1, w2, w3 = st.columns(3)

                w1.metric(
                    "Current Profit",
                    f"₹{current_profit:,.2f}"
                )
                w2.metric(
                    "Estimated Profit",
                    f"₹{scenario_profit:,.2f}"
                )
                w3.metric(
                    "Estimated Change",
                    f"₹{profit_change:+,.2f}"
                )

                if profit_change > 0:
                    st.success(
                        "Under these assumptions, the selected discount "
                        "increases estimated profit."
                    )
                elif profit_change < 0:
                    st.warning(
                        "Under these assumptions, the selected discount "
                        "reduces estimated profit."
                    )
                else:
                    st.info(
                        "The selected scenario produces no change in profit."
                    )

else:
    st.info(
        "Map Sales, Profit, and Discount columns to enable the simulator."
    )

st.caption(
    "GenAI-Powered Business Decision Cockpit • "
    "Python calculations + Gemini explanations"
)