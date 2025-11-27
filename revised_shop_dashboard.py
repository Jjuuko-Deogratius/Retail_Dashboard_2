# retail_dashboard_fixed.py
import streamlit as st
import pandas as pd
import plotly.express as px
import os
from datetime import datetime

st.set_page_config(page_title="Retail Shop Dashboard", layout="wide")

# -------------------------
# Helper functions
# -------------------------
@st.cache_data
def load_and_prepare(csv_path="revised_shop.csv",
                     avg_retail_price=2000,
                     avg_wholesale_price=1200):
    """
    Load data, ensure expected columns exist, compute missing inventory / financial columns if needed,
    and validate cash flow & profit.
    """
    df = pd.read_csv(csv_path)

    # Accept either 'DATE' or 'Day' columns. Prefer DATE if present.
    if 'DATE' in df.columns:
        try:
            df['DATE'] = pd.to_datetime(df['DATE'])
        except Exception:
            # If DATE not parseable, create a DATE from an assumed year
            df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    else:
        # if user provided Day only, generate a DATE series starting today (fallback)
        df.insert(0, 'DATE', pd.date_range(start=datetime.today().date(), periods=len(df), freq='D'))

    # Normalise column names to upper (common expectation in previous code)
    df.columns = [c.strip().upper() for c in df.columns]

    # Required financial columns
    required = ['SALES', 'PURCHASES', 'UTILITIES', 'TRANSPORT']
    for col in required:
        if col not in df.columns:
            df[col] = 0

    # If ITEMS_SOLD or ITEMS_PURCHASED missing, auto-generate using avg prices (fallback)
    if 'ITEMS_SOLD' not in df.columns:
        df['ITEMS_SOLD'] = (df['SALES'] / avg_retail_price).round().astype(int)
    if 'ITEMS_PURCHASED' not in df.columns:
        df['ITEMS_PURCHASED'] = (df['PURCHASES'] / avg_wholesale_price).round().astype(int)

    # TURNOVER: by project definition we treat turnover = ITEMS_SOLD
    df['TURNOVER'] = df['ITEMS_SOLD']

    # Compute PROFIT if missing or to re-calc
    df['PROFIT'] = df['SALES'] - (df['PURCHASES'] + df['UTILITIES'] + df['TRANSPORT'])

    # Cash at start / end handling:
    # If CASH_AT_START present, we will recalc CASH_AT_END for consistency.
    # If CASH_AT_START not present but CASH_AT_END exists, we attempt to derive starts.
    if 'CASH_AT_START' not in df.columns:
        # If CASH_AT_END exists, compute a CASH_AT_START by reversing the formula if reasonable
        if 'CASH_AT_END' in df.columns:
            df['CASH_AT_START'] = df['CASH_AT_END'] - (df['SALES'] - (df['PURCHASES'] + df['UTILITIES'] + df['TRANSPORT']))
        else:
            # Otherwise assume cash_at_start = 0 for first row, then propagate
            df['CASH_AT_START'] = 0.0

    # Recalculate CASH_AT_END to ensure internal consistency
    df = df.sort_values('DATE').reset_index(drop=True)
    # If first CASH_AT_START is zero and user had meaningful cash, keep as-is; we still use formula
    recalculated_ends = []
    for i, row in df.iterrows():
        start = float(row.get('CASH_AT_START', 0.0))
        sales = float(row.get('SALES', 0.0))
        purchases = float(row.get('PURCHASES', 0.0))
        utilities = float(row.get('UTILITIES', 0.0))
        transport = float(row.get('TRANSPORT', 0.0))
        end = start + sales - (purchases + utilities + transport)
        recalculated_ends.append(end)
        # Ensure next row's CASH_AT_START matches current end unless user specified otherwise
        if i + 1 < len(df):
            # if next row has CASH_AT_START missing or zero, set it to end
            if pd.isna(df.loc[i + 1, 'CASH_AT_START']) or df.loc[i + 1, 'CASH_AT_START'] == 0:
                df.loc[i + 1, 'CASH_AT_START'] = end
    df['CASH_AT_END_CALC'] = recalculated_ends
    # Overwrite any provided CASH_AT_END for internal use with the recalculated values for consistency
    df['CASH_AT_END'] = df['CASH_AT_END_CALC']

    # Final tidy: set types
    for col in ['ITEMS_SOLD', 'ITEMS_PURCHASED', 'TURNOVER']:
        if col in df.columns:
            df[col] = df[col].astype(int)

    return df

# -------------------------
# Load data
# -------------------------
BASE_DIR = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
csv_path = os.path.join(BASE_DIR, "revised_shop.csv")
df = load_and_prepare(csv_path)

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.title("🖥️ DSMA Retail Dashboard")
st.sidebar.markdown("🛠️ Use the controls to filter dates and change chart type.")

min_date, max_date = df['DATE'].min(), df['DATE'].max()
start_date = st.sidebar.date_input("Start date", min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End date", max_date, min_value=min_date, max_value=max_date)

chart_type = st.sidebar.selectbox("Chart type", ["Line", "Bar", "Area", "Scatter", "Pie"])

# Filter data
mask = (df['DATE'] >= pd.Timestamp(start_date)) & (df['DATE'] <= pd.Timestamp(end_date))
dff = df.loc[mask].copy().sort_values('DATE')

# -------------------------
# Summary metrics row
# -------------------------
st.title("🛍️ Retail Shop Performance Dashboard")
st.markdown("*Cash at Hand On 1st January, 2025: 100,000 UGX*")
st.markdown("**Summary for selected period**")

total_revenue = dff['SALES'].sum()
total_purchases = dff['PURCHASES'].sum()
total_profit = dff['PROFIT'].sum()
avg_turnover = dff['TURNOVER'].mean()
total_items_sold = dff['ITEMS_SOLD'].sum()
total_items_purchased = dff['ITEMS_PURCHASED'].sum()
starting_cash = dff.iloc[0]['CASH_AT_START'] if not dff.empty else 0
ending_cash = dff.iloc[-1]['CASH_AT_END'] if not dff.empty else 0

c1, c2, c3, c4, c5 = st.columns([1.2,1.2,1.2,1.2,1.2])
c1.metric("💰Total Sales (UGX)", f"{total_revenue:,.0f}")
c2.metric("🛒Total Purchases (UGX)", f"{total_purchases:,.0f}")
c3.metric("📈Total Profit (UGX)", f"{total_profit:,.0f}")
c4.metric("🏷️Items Sold (units)", f"{total_items_sold:,}")
c5.metric("📦Avg Daily Turnover", f"{avg_turnover:.1f} items")


st.divider()

# -------------------------
# Main visualizations
# -------------------------
st.subheader("Sales, Purchases & Profit over time")
if not dff.empty:
    timeseries = dff[['DATE', 'SALES', 'PURCHASES', 'PROFIT']].set_index('DATE')
    if chart_type == "Line":
        st.line_chart(timeseries)
    elif chart_type == "Bar":
        st.bar_chart(timeseries)
    elif chart_type == "Area":
        st.area_chart(timeseries)
    elif chart_type == "Scatter":
        # scatter with Plotly for better control
        fig = px.scatter(dff, x='DATE', y='SALES', size='PROFIT', hover_data=['PURCHASES','PROFIT','TURNOVER'])
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "Pie":
        # show revenue distribution pie by day - only for small ranges
        pie_data = dff[['DATE','SALES']].copy()
        pie_data['DATE_STR'] = pie_data['DATE'].dt.strftime('%Y-%m-%d')
        fig = px.pie(pie_data, values='SALES', names='DATE_STR', title='Revenue by Day')
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No data in the selected date range.")

# Row: profit bar and turnover chart
colA, colB = st.columns(2)

with colA:
    st.subheader("Daily Profit")
    if not dff.empty:
        fig_profit = px.bar(dff, x='DATE', y='PROFIT', hover_data=['SALES','PURCHASES'], title='Daily Profit (UGX)')
        st.plotly_chart(fig_profit, use_container_width=True)
    else:
        st.info("No data to plot.")

with colB:
    st.subheader("Inventory Turnover (Items Sold / day)")
    if not dff.empty:
        fig_turn = px.bar(dff, x='DATE', y='TURNOVER', text='TURNOVER', title='Daily Turnover (units)')
        fig_turn.update_traces(textposition='outside')
        st.plotly_chart(fig_turn, use_container_width=True)
    else:
        st.info("No turnover data.")

# Expense breakdown
st.subheader("Expense Breakdown (Total for selected period)")
if not dff.empty:
    expense_sum = dff[['PURCHASES','UTILITIES','TRANSPORT']].sum().reset_index()
    expense_sum.columns = ['Category','Amount']
    fig_exp = px.pie(expense_sum, values='Amount', names='Category', title='Expense Distribution')
    st.plotly_chart(fig_exp, use_container_width=True)
else:
    st.info("No data for expenses.")

# Sales vs Profit scatter
st.subheader("Sales vs Profit (relationship)")
if not dff.empty:
    fig_sp = px.scatter(dff, x='SALES', y='PROFIT', size='TURNOVER',
                        hover_data=['DATE','ITEMS_SOLD','ITEMS_PURCHASED'],
                        title='Sales vs Profit (bubble size = turnover)')
    st.plotly_chart(fig_sp, use_container_width=True)

# -------------------------
# Daily summary table
# -------------------------
st.subheader("📋 Daily Summary Table")
if not dff.empty:
    summary_cols = ['DATE','SALES','PURCHASES','UTILITIES','TRANSPORT',
                    'ITEMS_PURCHASED','ITEMS_SOLD','TURNOVER',
                    'CASH_AT_START','CASH_AT_END','PROFIT']
    # ensure columns exist
    summary_cols = [c for c in summary_cols if c in dff.columns]
    st.dataframe(dff[summary_cols].assign(DATE=lambda x: x['DATE'].dt.strftime('%Y-%m-%d')), use_container_width=True)

    # Download corrected CSV
    csv_bytes = dff.to_csv(index=False).encode('utf-8')
    st.download_button("Download filtered CSV", data=csv_bytes, file_name="retail_shop_data_filtered_corrected.csv", mime="text/csv")

else:
    st.info("No records found for the selected period.")

# -------------------------
# Data validation and notes
# -------------------------
st.markdown("---")
st.subheader("Designed By Data Science Weekend Class.")

# Quick validation checks
errors = []
# Negative profit check
neg_profits = dff[dff['PROFIT'] < 0].shape[0] if 'PROFIT' in dff.columns else 0
if neg_profits:
    errors.append(f"⚠️ {neg_profits} day(s) with negative profit detected.")


    st.write("✅ Quick validation passed. Cash flow and profit values were computed/recomputed for consistency.")

# -------------------------
# Footer / credits
# -------------------------
st.markdown("---")
st.markdown("**Created for** `Data Science Tool Kit` Course Unit Project ")

