import os
from flask import Flask, render_template, request
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio

app = Flask(__name__)
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "superstore.csv")


def load_data():
    """Load the superstore CSV data and return a pandas DataFrame."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please add superstore.csv to the data folder.")

    df = pd.read_csv(DATA_PATH)
    return clean_data(df)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset and prepare it for KPI and chart generation."""
    df = df.copy()

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)

    # Ensure required columns exist
    required_columns = ["Order ID", "Order Date", "Sales", "Profit", "Quantity", "Category", "Region", "Product Name"]
    df = df[[col for col in required_columns if col in df.columns]].copy()

    # Convert Order Date to datetime
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")

    # Convert numeric columns and handle missing values
    df["Sales"] = pd.to_numeric(df.get("Sales", 0), errors="coerce")
    df["Profit"] = pd.to_numeric(df.get("Profit", 0), errors="coerce")
    df["Quantity"] = pd.to_numeric(df.get("Quantity", 0), errors="coerce")

    # Drop rows with invalid required fields
    df.dropna(subset=["Order ID", "Order Date", "Sales", "Profit", "Quantity", "Category", "Region", "Product Name"], inplace=True)

    # Replace any remaining numeric NaN values with zeros
    df["Sales"].fillna(0, inplace=True)
    df["Profit"].fillna(0, inplace=True)
    df["Quantity"].fillna(0, inplace=True)

    # Ensure Quantity is integer
    df["Quantity"] = df["Quantity"].astype(int)

    # Calculate Profit Margin
    df["Profit Margin"] = df.apply(lambda row: (row["Profit"] / row["Sales"]) if row["Sales"] else 0, axis=1)

    return df


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def format_percentage(value: float) -> str:
    return f"{value:.2f}%"


def compute_kpis(df: pd.DataFrame) -> dict:
    """Calculate dashboard KPI values."""
    total_sales = df["Sales"].sum()
    total_profit = df["Profit"].sum()
    total_orders = df["Order ID"].nunique()
    profit_margin_pct = (total_profit / total_sales * 100) if total_sales else 0

    return {
        "total_sales": format_currency(total_sales),
        "total_profit": format_currency(total_profit),
        "total_orders": f"{total_orders:,}",
        "profit_margin": format_percentage(profit_margin_pct),
    }


def aggregate_sales_by_category(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("Category")["Sales"].sum().sort_values(ascending=False).reset_index()


def aggregate_monthly_sales(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Year-Month"] = df["Order Date"].dt.to_period("M").astype(str)
    return df.groupby("Year-Month")["Sales"].sum().reset_index().sort_values("Year-Month")


def aggregate_sales_by_region(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("Region")["Sales"].sum().sort_values(ascending=False).reset_index()


def aggregate_top_products(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    return df.groupby("Product Name")["Sales"].sum().sort_values(ascending=False).head(top_n).reset_index()


def create_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, x_title: str, y_title: str) -> str:
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", showarrow=False, font=dict(size=16))
    else:
        fig = go.Figure(
            data=[go.Bar(x=df[x_col], y=df[y_col], marker_color="#4c78a8")],
            layout=go.Layout(
                title=title,
                xaxis=dict(title=x_title, tickangle=-45),
                yaxis=dict(title=y_title),
                margin=dict(l=40, r=20, t=60, b=120),
            ),
        )

    fig.update_layout(template="plotly_white", autosize=True)
    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def create_line_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, x_title: str, y_title: str) -> str:
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", showarrow=False, font=dict(size=16))
    else:
        fig = go.Figure(
            data=[go.Scatter(x=df[x_col], y=df[y_col], mode="lines+markers", line=dict(color="#e45756"))],
            layout=go.Layout(
                title=title,
                xaxis=dict(title=x_title),
                yaxis=dict(title=y_title),
                margin=dict(l=40, r=20, t=60, b=60),
            ),
        )

    fig.update_layout(template="plotly_white", autosize=True)
    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def create_pie_chart(df: pd.DataFrame, names_col: str, values_col: str, title: str) -> str:
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", showarrow=False, font=dict(size=16))
    else:
        fig = go.Figure(
            data=[go.Pie(labels=df[names_col], values=df[values_col], hole=0.45, marker=dict(colors=["#4c78a8", "#f58518", "#54a24b", "#72b7b2"]))],
            layout=go.Layout(title=title, margin=dict(l=20, r=20, t=60, b=20)),
        )

    fig.update_layout(template="plotly_white", autosize=True)
    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def filter_data(df: pd.DataFrame, region: str, category: str) -> pd.DataFrame:
    filtered = df.copy()
    if region and region != "All":
        filtered = filtered[filtered["Region"] == region]
    if category and category != "All":
        filtered = filtered[filtered["Category"] == category]
    return filtered


@app.route("/")
def index():
    selected_region = request.args.get("region", "All")
    selected_category = request.args.get("category", "All")

    df = load_data()
    region_options = ["All"] + sorted(df["Region"].dropna().unique().tolist())
    category_options = ["All"] + sorted(df["Category"].dropna().unique().tolist())

    filtered_df = filter_data(df, selected_region, selected_category)
    kpis = compute_kpis(filtered_df)

    sales_by_category = aggregate_sales_by_category(filtered_df)
    monthly_sales = aggregate_monthly_sales(filtered_df)
    sales_by_region = aggregate_sales_by_region(filtered_df)
    top_products = aggregate_top_products(filtered_df)

    category_chart = create_bar_chart(
        sales_by_category, "Category", "Sales", "Sales by Category", "Category", "Sales"
    )
    trend_chart = create_line_chart(
        monthly_sales, "Year-Month", "Sales", "Monthly Sales Trend", "Month", "Sales"
    )
    region_chart = create_pie_chart(
        sales_by_region, "Region", "Sales", "Sales by Region"
    )
    product_chart = create_bar_chart(
        top_products, "Product Name", "Sales", "Top 10 Products by Sales", "Product", "Sales"
    )

    return render_template(
        "index.html",
        kpis=kpis,
        category_chart=category_chart,
        trend_chart=trend_chart,
        region_chart=region_chart,
        product_chart=product_chart,
        region_options=region_options,
        category_options=category_options,
        selected_region=selected_region,
        selected_category=selected_category,
        no_data=filtered_df.empty,
    )


if __name__ == "__main__":
    app.run(debug=True)
