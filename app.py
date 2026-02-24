import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Careem Quik — AI Category Optimizer", layout="wide")

st.title("Careem Quik — AI-Driven Category Optimization Dashboard")
st.caption("Upload your SKU CSV (with AI Score + Recommended Action) to get an interactive category decision dashboard.")

REQUIRED_COLS = [
    "SKU ID","Product Name","Category","Monthly Orders","Revenue","Margin %","Fulfillment %",
    "Basket %","Returns %","Trend (%)","Days to Expiry","AI Score","Recommended Action"
]

@st.cache_data
def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]
    return df

def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in ["Monthly Orders","Revenue","Margin %","Fulfillment %","Basket %","Returns %","Trend (%)","Days to Expiry","AI Score"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d

def validate(df: pd.DataFrame):
    return [c for c in REQUIRED_COLS if c not in df.columns]

def kpi_block(df: pd.DataFrame):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("SKUs", f"{len(df):,}")
    c2.metric("Revenue", f"{df['Revenue'].sum():,.0f}")
    c3.metric("Avg Margin %", f"{df['Margin %'].mean():.1f}")
    c4.metric("Avg Fulfillment %", f"{df['Fulfillment %'].mean():.1f}")
    c5.metric("Avg AI Score", f"{df['AI Score'].mean():.1f}")

def top_lists(df: pd.DataFrame):
    a, b, c = st.columns(3, vertical_alignment="top")

    with a:
        st.subheader("Top 10 — INVEST")
        invest = df[df["Recommended Action"]=="INVEST"].sort_values("AI Score", ascending=False).head(10)
        st.dataframe(invest[["SKU ID","Product Name","Category","AI Score","Revenue","Margin %","Fulfillment %","Returns %","Trend (%)"]],
                     use_container_width=True, hide_index=True)

    with b:
        st.subheader("Top 10 — PROMOTE")
        promote = df[df["Recommended Action"]=="PROMOTE"].sort_values("AI Score", ascending=False).head(10)
        st.dataframe(promote[["SKU ID","Product Name","Category","AI Score","Revenue","Margin %","Fulfillment %","Returns %","Trend (%)"]],
                     use_container_width=True, hide_index=True)

    with c:
        st.subheader("Top 10 — RATIONALIZE")
        rat = df[df["Recommended Action"]=="RATIONALIZE"].sort_values("Returns %", ascending=False).head(10)
        st.dataframe(rat[["SKU ID","Product Name","Category","AI Score","Revenue","Margin %","Fulfillment %","Returns %","Trend (%)","Days to Expiry"]],
                     use_container_width=True, hide_index=True)

def executive_summary(df: pd.DataFrame):
    st.subheader("Executive Summary (auto)")
    d = df.copy()
    d["Profit Proxy"] = d["Revenue"] * (d["Margin %"]/100.0) * (d["Fulfillment %"]/100.0) * (1 - d["Returns %"]/100.0)
    cat = (d.groupby("Category", as_index=False)
             .agg(Revenue=("Revenue","sum"),
                  ProfitProxy=("Profit Proxy","sum"),
                  AvgScore=("AI Score","mean"),
                  AvgMargin=("Margin %","mean"),
                  AvgReturns=("Returns %","mean")))
    top_profit = cat.sort_values("ProfitProxy", ascending=False).head(3)
    worst_service = cat.sort_values("AvgReturns", ascending=False).head(3)

    action_mix = (d["Recommended Action"].value_counts(normalize=True) * 100).round(1)
    friction = d[d["Returns %"] > 7].shape[0]

    st.markdown(
        f"""
- **Top profit-driving categories:** {", ".join(top_profit["Category"].tolist())}
- **Highest service friction (returns):** {", ".join(worst_service["Category"].tolist())}
- **Portfolio action mix:** {", ".join([f"{k} {v}%" for k,v in action_mix.items()])}
- **High-returns SKUs (>7%):** {friction}
        """.strip()
    )

def make_download(df: pd.DataFrame):
    st.download_button(
        "Download filtered view (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="careem_ai_filtered_view.csv",
        mime="text/csv"
    )

# Sidebar: Upload
st.sidebar.header("Data Input")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded is None:
    demo_path = "riyadh_dark_store_260_skus_AI_scored.csv"
    try:
        df = load_csv(demo_path)
        st.sidebar.success("Loaded demo dataset bundled with the app.")
    except Exception:
        st.warning("Upload a CSV to begin (demo file not found).")
        st.stop()
else:
    df = load_csv(uploaded)

df = coerce_numeric(df)
missing = validate(df)
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.stop()

# Sidebar navigation and filters
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "SKU Explorer", "Category Dashboard", "AI Insights", "Executive Summary"])

st.sidebar.header("Global Filters")
categories = sorted(df["Category"].dropna().unique().tolist())
actions = ["INVEST","PROMOTE","MAINTAIN","RATIONALIZE","REPLACE"]

sel_cats = st.sidebar.multiselect("Category", categories, default=categories)
sel_actions = st.sidebar.multiselect("Recommended Action", actions, default=actions)

min_score, max_score = float(df["AI Score"].min()), float(df["AI Score"].max())
score_range = st.sidebar.slider("AI Score range", min_value=float(np.floor(min_score)), max_value=float(np.ceil(max_score)),
                                value=(float(np.floor(min_score)), float(np.ceil(max_score))))

search = st.sidebar.text_input("Search Product Name")

filtered = df[
    df["Category"].isin(sel_cats) &
    df["Recommended Action"].isin(sel_actions) &
    df["AI Score"].between(score_range[0], score_range[1])
].copy()

if search.strip():
    filtered = filtered[filtered["Product Name"].str.contains(search.strip(), case=False, na=False)].copy()

# Pages
if page == "Overview":
    kpi_block(filtered)
    st.divider()

    left, right = st.columns(2)
    with left:
        st.subheader("Action Mix")
        mix = filtered["Recommended Action"].value_counts().reindex(actions).fillna(0).astype(int).reset_index()
        mix.columns = ["Recommended Action","SKU Count"]
        st.plotly_chart(px.bar(mix, x="Recommended Action", y="SKU Count"), use_container_width=True)

    with right:
        st.subheader("AI Score Distribution")
        st.plotly_chart(px.histogram(filtered, x="AI Score", nbins=20), use_container_width=True)

    st.subheader("Top 20 SKUs by AI Score")
    st.dataframe(filtered.sort_values("AI Score", ascending=False).head(20)[REQUIRED_COLS],
                 use_container_width=True, hide_index=True)
    make_download(filtered)

elif page == "SKU Explorer":
    st.subheader("Interactive SKU Explorer")
    st.caption("Use sidebar filters to explore by Category, Action, AI Score, and Product search.")
    st.dataframe(filtered[REQUIRED_COLS].sort_values(["Recommended Action","AI Score"], ascending=[True, False]),
                 use_container_width=True, hide_index=True)
    make_download(filtered)

elif page == "Category Dashboard":
    kpi_block(filtered)
    st.divider()

    cat = (filtered.groupby("Category", as_index=False)
           .agg(Revenue=("Revenue","sum"),
                AvgMargin=("Margin %","mean"),
                AvgFulfillment=("Fulfillment %","mean"),
                AvgReturns=("Returns %","mean"),
                AvgScore=("AI Score","mean"),
                SKUCount=("SKU ID","count")))

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Revenue by Category")
        st.plotly_chart(px.bar(cat.sort_values("Revenue", ascending=False), x="Category", y="Revenue"), use_container_width=True)

        st.subheader("Avg AI Score by Category")
        st.plotly_chart(px.bar(cat.sort_values("AvgScore", ascending=False), x="Category", y="AvgScore"), use_container_width=True)

    with c2:
        st.subheader("Avg Margin % by Category")
        st.plotly_chart(px.bar(cat.sort_values("AvgMargin", ascending=False), x="Category", y="AvgMargin"), use_container_width=True)

        st.subheader("Returns % by Category")
        st.plotly_chart(px.bar(cat.sort_values("AvgReturns", ascending=False), x="Category", y="AvgReturns"), use_container_width=True)

    st.subheader("Action Mix by Category")
    pivot = filtered.pivot_table(index="Category", columns="Recommended Action", values="SKU ID", aggfunc="count", fill_value=0)
    st.dataframe(pivot, use_container_width=True)
    make_download(filtered)

elif page == "AI Insights":
    kpi_block(filtered)
    st.divider()
    top_lists(filtered)

    st.subheader("Friction Watchlist (Returns > 7%)")
    friction = filtered[filtered["Returns %"] > 7].sort_values("Returns %", ascending=False)
    st.dataframe(friction[["SKU ID","Product Name","Category","Recommended Action","Returns %","Fulfillment %","AI Score","Days to Expiry","Revenue"]],
                 use_container_width=True, hide_index=True)

    st.subheader("Expiry Watchlist (Days to Expiry ≤ 7)")
    exp = filtered[filtered["Days to Expiry"] <= 7].sort_values(["Days to Expiry","AI Score"], ascending=[True, False])
    st.dataframe(exp[["SKU ID","Product Name","Category","Days to Expiry","Recommended Action","AI Score","Revenue","Basket %"]],
                 use_container_width=True, hide_index=True)
    make_download(filtered)

elif page == "Executive Summary":
    kpi_block(filtered)
    st.divider()
    executive_summary(filtered)

    st.subheader("Category Table (with Profit Proxy)")
    tmp = filtered.copy()
    tmp["Profit Proxy"] = tmp["Revenue"] * (tmp["Margin %"]/100.0) * (tmp["Fulfillment %"]/100.0) * (1 - tmp["Returns %"]/100.0)
    cat = (tmp.groupby("Category", as_index=False)
           .agg(Revenue=("Revenue","sum"),
                ProfitProxy=("Profit Proxy","sum"),
                AvgMargin=("Margin %","mean"),
                AvgFulfillment=("Fulfillment %","mean"),
                AvgReturns=("Returns %","mean"),
                AvgScore=("AI Score","mean"),
                SKUCount=("SKU ID","count"))
           .sort_values("ProfitProxy", ascending=False))
    st.dataframe(cat, use_container_width=True, hide_index=True)
    make_download(filtered)
