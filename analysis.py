"""
E-Commerce Sales Analytics — Brazilian Olist Dataset
=====================================================
Tech: Python, Pandas, SQL (SQLite), Matplotlib, Seaborn, ETL
Dataset: 100,000 Orders | olist_* CSV files
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import sqlite3
import warnings
import os

warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR   = r"C:\Users\nsai6\OneDrive\Desktop\eco\ecommerce_analytics"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

PALETTE = {
    "green":     "#1D9E75",
    "blue":      "#378ADD",
    "amber":     "#EF9F27",
    "red":       "#E24B4A",
    "gray":      "#888780",
    "light":     "#F1EFE8",
    "dark":      "#2C2C2A",
    "purple":    "#7F77DD",
    "bg":        "#FAFAF8",
    "border":    "#D3D1C7",
}

plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["bg"],
    "axes.edgecolor":    PALETTE["border"],
    "axes.labelcolor":   PALETTE["dark"],
    "xtick.color":       PALETTE["gray"],
    "ytick.color":       PALETTE["gray"],
    "text.color":        PALETTE["dark"],
    "grid.color":        PALETTE["border"],
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ─── ETL PIPELINE ─────────────────────────────────────────────────────────────
def load_data(data_dir):
    print("📦 Loading CSVs...")
    files = {
        "orders":      "olist_orders_dataset.csv",
        "order_items": "olist_order_items_dataset.csv",
        "customers":   "olist_customers_dataset.csv",
        "products":    "olist_products_dataset.csv",
        "reviews":     "olist_order_reviews_dataset.csv",
        "payments":    "olist_order_payments_dataset.csv",
        "category":    "product_category_name_translation.csv",
        "sellers":     "olist_sellers_dataset.csv",
        "geolocation": "olist_geolocation_dataset.csv",
    }
    dfs = {}
    for key, fname in files.items():
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            dfs[key] = pd.read_csv(path)
            print(f"  ✓ {fname}  ({len(dfs[key]):,} rows)")
        else:
            print(f"  ✗ Missing: {fname}")
    return dfs


def transform(dfs):
    print("\n🔄 Running ETL transformations...")

    # Parse dates
    date_cols = ["order_purchase_timestamp", "order_delivered_customer_date",
                 "order_estimated_delivery_date", "order_approved_at",
                 "order_delivered_carrier_date"]
    for c in date_cols:
        if c in dfs["orders"].columns:
            dfs["orders"][c] = pd.to_datetime(dfs["orders"][c], errors="coerce")

    dfs["reviews"]["review_creation_date"] = pd.to_datetime(
        dfs["reviews"]["review_creation_date"], errors="coerce")

    # Delivery delay
    orders = dfs["orders"].copy()
    orders["delivery_delay_days"] = (
        orders["order_delivered_customer_date"] -
        orders["order_estimated_delivery_date"]
    ).dt.days

    # Merge category translation
    products = dfs["products"].merge(dfs["category"], on="product_category_name", how="left")
    products["product_category_name_english"] = products["product_category_name_english"].fillna(
        products["product_category_name"].str.replace("_", " ").str.title()
    )

    # Master fact table
    fact = (
        orders
        .merge(dfs["order_items"],  on="order_id",    how="left")
        .merge(dfs["customers"],    on="customer_id", how="left")
        .merge(products,            on="product_id",  how="left")
        .merge(dfs["payments"],     on="order_id",    how="left")
        .merge(dfs["reviews"][["order_id","review_score"]].drop_duplicates("order_id"),
               on="order_id", how="left")
    )

    # Time features
    fact["year"]    = fact["order_purchase_timestamp"].dt.year
    fact["month"]   = fact["order_purchase_timestamp"].dt.month
    fact["quarter"] = fact["order_purchase_timestamp"].dt.quarter
    fact["month_label"] = fact["order_purchase_timestamp"].dt.to_period("M").astype(str)

    # Filter 2017-2018 (complete years in dataset)
    fact = fact[fact["year"].isin([2017, 2018])].copy()

    print(f"  ✓ Fact table: {len(fact):,} rows × {fact.shape[1]} columns")
    return fact, products


def load_sqlite(fact):
    print("\n🗄  Loading into SQLite for SQL queries...")
    conn = sqlite3.connect(":memory:")
    fact.to_sql("orders_fact", conn, index=False, if_exists="replace")
    print("  ✓ Table 'orders_fact' ready")
    return conn


# ─── SQL QUERIES ──────────────────────────────────────────────────────────────
RFM_SQL = """
WITH rfm_base AS (
    SELECT
        customer_unique_id,
        MAX(order_purchase_timestamp)                              AS last_purchase,
        COUNT(DISTINCT order_id)                                   AS frequency,
        SUM(payment_value)                                         AS monetary
    FROM orders_fact
    WHERE order_status = 'delivered'
    GROUP BY customer_unique_id
),
rfm_scored AS (
    SELECT *,
        JULIANDAY('2018-10-01') - JULIANDAY(last_purchase)        AS recency_days,
        NTILE(5) OVER (ORDER BY JULIANDAY('2018-10-01') - JULIANDAY(last_purchase) DESC) AS r_score,
        NTILE(5) OVER (ORDER BY frequency)                        AS f_score,
        NTILE(5) OVER (ORDER BY monetary)                         AS m_score
    FROM rfm_base
),
rfm_labelled AS (
    SELECT *,
        (r_score + f_score + m_score) AS rfm_total,
        CASE
            WHEN r_score >= 4 AND f_score >= 4                     THEN 'Champions'
            WHEN r_score >= 3 AND f_score >= 3                     THEN 'Loyal Customers'
            WHEN r_score >= 3 AND f_score <= 2                     THEN 'Potential Loyalists'
            WHEN r_score <= 2 AND f_score >= 3                     THEN 'At-Risk'
            WHEN r_score = 1  AND f_score = 1                      THEN 'Lost'
            ELSE                                                        'Needs Attention'
        END AS segment
    FROM rfm_scored
)
SELECT segment,
       COUNT(*)              AS customers,
       ROUND(AVG(recency_days), 1)  AS avg_recency_days,
       ROUND(AVG(frequency), 2)     AS avg_frequency,
       ROUND(AVG(monetary), 2)      AS avg_monetary,
       ROUND(SUM(monetary), 2)      AS total_revenue
FROM rfm_labelled
GROUP BY segment
ORDER BY total_revenue DESC
"""

KPI_SQL = """
SELECT
    year,
    quarter,
    COUNT(DISTINCT order_id)           AS total_orders,
    ROUND(SUM(payment_value), 2)       AS total_revenue,
    ROUND(AVG(payment_value), 2)       AS avg_order_value,
    ROUND(AVG(review_score), 2)        AS avg_review_score
FROM orders_fact
WHERE order_status = 'delivered'
GROUP BY year, quarter
ORDER BY year, quarter
"""

CATEGORY_SQL = """
SELECT
    product_category_name_english              AS category,
    COUNT(DISTINCT order_id)                   AS orders,
    ROUND(SUM(payment_value), 2)               AS revenue,
    ROUND(AVG(review_score), 2)                AS avg_review
FROM orders_fact
WHERE order_status = 'delivered'
  AND product_category_name_english IS NOT NULL
GROUP BY product_category_name_english
ORDER BY revenue DESC
LIMIT 10
"""

MONTHLY_SQL = """
SELECT
    month_label,
    year,
    month,
    COUNT(DISTINCT order_id)       AS orders,
    ROUND(SUM(payment_value), 2)   AS revenue
FROM orders_fact
WHERE order_status = 'delivered'
GROUP BY month_label, year, month
ORDER BY year, month
"""


# ─── PLOT HELPERS ─────────────────────────────────────────────────────────────
def annotate_bars(ax, fmt="{:.0f}", offset=0.01, fontsize=9, color=None):
    for p in ax.patches:
        h = p.get_height()
        if pd.isna(h) or h == 0:
            continue
        ax.annotate(
            fmt.format(h),
            (p.get_x() + p.get_width() / 2, h),
            ha="center", va="bottom",
            fontsize=fontsize,
            color=color or PALETTE["gray"],
        )


def kpi_card(ax, title, value, subtitle="", color=None):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.04, 0.08), 0.92, 0.84,
        boxstyle="round,pad=0.02",
        facecolor="white", edgecolor=PALETTE["border"],
        linewidth=1, zorder=1
    ))
    ax.text(0.5, 0.72, title,  ha="center", va="center", fontsize=9,
            color=PALETTE["gray"], zorder=2)
    ax.text(0.5, 0.45, value,  ha="center", va="center", fontsize=18,
            fontweight="bold", color=color or PALETTE["dark"], zorder=2)
    ax.text(0.5, 0.22, subtitle, ha="center", va="center", fontsize=8,
            color=PALETTE["gray"], zorder=2)


# ─── DASHBOARD 1 : KPI + REVENUE TREND ───────────────────────────────────────
def plot_dashboard1(conn):
    print("\n📊 Dashboard 1 — KPI + Revenue Trend")
    kpi  = pd.read_sql(KPI_SQL, conn)
    mon  = pd.read_sql(MONTHLY_SQL, conn)

    fig = plt.figure(figsize=(18, 12), facecolor=PALETTE["bg"])
    fig.suptitle(
        "E-Commerce Sales Analytics — Brazilian Olist Dataset",
        fontsize=16, fontweight="bold", color=PALETTE["dark"], y=0.98
    )
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.4)

    # ── KPI Cards ──
    total_rev  = kpi["total_revenue"].sum()
    total_ord  = kpi["total_orders"].sum()
    avg_aov    = (kpi["total_revenue"] / kpi["total_orders"]).mean()
    avg_review = kpi["avg_review_score"].mean()

    card_data = [
        ("Total Revenue",   f"R$ {total_rev/1e6:.2f}M", "FY 2017–2018",      PALETTE["green"]),
        ("Total Orders",    f"{int(total_ord):,}",        "Delivered orders",  PALETTE["blue"]),
        ("Avg Order Value", f"R$ {avg_aov:.2f}",          "Per transaction",   PALETTE["amber"]),
        ("Avg Review",      f"{avg_review:.2f} / 5",      "Customer rating",   PALETTE["purple"]),
    ]
    for i, (title, val, sub, col) in enumerate(card_data):
        ax = fig.add_subplot(gs[0, i])
        kpi_card(ax, title, val, sub, col)

    # ── Monthly Revenue Trend ──
    ax2 = fig.add_subplot(gs[1, :])
    mon["month_dt"] = pd.to_datetime(mon["month_label"])
    mon_sorted = mon.sort_values("month_dt")

    colors_line = [PALETTE["red"] if (r.year == 2018 and r.month in [8,9])
                   else PALETTE["green"] for _, r in mon_sorted.iterrows()]

    ax2.fill_between(range(len(mon_sorted)), mon_sorted["revenue"]/1000,
                     alpha=0.12, color=PALETTE["green"])
    for i in range(len(mon_sorted)-1):
        ax2.plot([i, i+1],
                 [mon_sorted["revenue"].iloc[i]/1000, mon_sorted["revenue"].iloc[i+1]/1000],
                 color=colors_line[i], linewidth=2)
    ax2.scatter(range(len(mon_sorted)), mon_sorted["revenue"]/1000,
                color=colors_line, s=50, zorder=5)

    ax2.set_xticks(range(len(mon_sorted)))
    ax2.set_xticklabels(mon_sorted["month_label"], rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Revenue (R$ K)", fontsize=10)
    ax2.set_title("Monthly Revenue Trend  |  Red = anomaly months", fontsize=11,
                  color=PALETTE["dark"], pad=10)
    ax2.yaxis.grid(True); ax2.set_axisbelow(True)

    # ── Quarterly bar ──
    ax3 = fig.add_subplot(gs[2, :3])
    kpi["label"] = kpi["year"].astype(str) + " Q" + kpi["quarter"].astype(str)
    bar_colors = [PALETTE["red"] if (r.year == 2018 and r.quarter == 3)
                  else PALETTE["blue"] for _, r in kpi.iterrows()]
    bars = ax3.bar(kpi["label"], kpi["total_revenue"]/1000,
                   color=bar_colors, width=0.6, edgecolor="white", linewidth=0.5)
    annotate_bars(ax3, fmt="R${:.0f}K", fontsize=8)
    ax3.set_ylabel("Revenue (R$ K)"); ax3.set_title("Revenue by Quarter", fontsize=11)
    ax3.tick_params(axis="x", rotation=30)

    # ── Q3 callout ──
    ax4 = fig.add_subplot(gs[2, 3])
    ax4.axis("off")
    q3_18 = kpi[(kpi["year"]==2018) & (kpi["quarter"]==3)]["total_revenue"].sum()
    q3_17 = kpi[(kpi["year"]==2017) & (kpi["quarter"]==3)]["total_revenue"].sum()
    delta  = ((q3_18 - q3_17) / q3_17 * 100) if q3_17 > 0 else 0
    ax4.add_patch(mpatches.FancyBboxPatch(
        (0.05,0.1), 0.9, 0.8, boxstyle="round,pad=0.03",
        facecolor="#FCEBEB", edgecolor=PALETTE["red"], linewidth=1.5))
    ax4.text(0.5, 0.78, "⚠ Q3 Anomaly",   ha="center", fontsize=11, fontweight="bold",
             color=PALETTE["red"])
    ax4.text(0.5, 0.56, f"{delta:+.1f}% YoY",  ha="center", fontsize=18,
             fontweight="bold", color=PALETTE["red"])
    ax4.text(0.5, 0.36, f"2018 Q3: R${q3_18/1000:.0f}K", ha="center", fontsize=9,
             color=PALETTE["dark"])
    ax4.text(0.5, 0.22, f"2017 Q3: R${q3_17/1000:.0f}K", ha="center", fontsize=9,
             color=PALETTE["gray"])
    ax4.set_xlim(0,1); ax4.set_ylim(0,1)

    plt.savefig(os.path.join(OUTPUT_DIR, "dashboard1_kpi_revenue.png"),
                dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print("  ✓ Saved: dashboard1_kpi_revenue.png")


# ─── DASHBOARD 2 : RFM SEGMENTATION ──────────────────────────────────────────
def plot_dashboard2(conn):
    print("\n📊 Dashboard 2 — RFM Segmentation")
    rfm = pd.read_sql(RFM_SQL, conn)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=PALETTE["bg"])
    fig.suptitle("RFM Customer Segmentation  |  CTEs + Window Functions",
                 fontsize=14, fontweight="bold", color=PALETTE["dark"])

    seg_colors = {
        "Champions":          PALETTE["green"],
        "Loyal Customers":    PALETTE["blue"],
        "Potential Loyalists":PALETTE["purple"],
        "At-Risk":            PALETTE["amber"],
        "Needs Attention":    PALETTE["gray"],
        "Lost":               PALETTE["red"],
    }
    colors = [seg_colors.get(s, PALETTE["gray"]) for s in rfm["segment"]]

    # Customers per segment
    ax = axes[0]
    bars = ax.barh(rfm["segment"], rfm["customers"], color=colors, edgecolor="white")
    for bar, val in zip(bars, rfm["customers"]):
        ax.text(val + 50, bar.get_y() + bar.get_height()/2,
                f"{val:,}", va="center", fontsize=9, color=PALETTE["gray"])
    ax.set_title("Customers per Segment", fontsize=11)
    ax.set_xlabel("Customers"); ax.invert_yaxis()

    # Avg Monetary per segment
    ax = axes[1]
    bars = ax.barh(rfm["segment"], rfm["avg_monetary"], color=colors, edgecolor="white")
    for bar, val in zip(bars, rfm["avg_monetary"]):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                f"R${val:.0f}", va="center", fontsize=9, color=PALETTE["gray"])
    ax.set_title("Avg. Lifetime Value (R$)", fontsize=11)
    ax.set_xlabel("Avg Monetary Value"); ax.invert_yaxis()

    # Revenue contribution donut
    ax = axes[2]
    wedge_colors = [seg_colors.get(s, PALETTE["gray"]) for s in rfm["segment"]]
    wedges, texts, autotexts = ax.pie(
        rfm["total_revenue"], labels=rfm["segment"],
        colors=wedge_colors, autopct="%1.1f%%",
        startangle=140, pctdistance=0.82,
        wedgeprops=dict(width=0.5, edgecolor="white", linewidth=1.5)
    )
    for t in texts:     t.set_fontsize(8)
    for t in autotexts: t.set_fontsize(8); t.set_color("white"); t.set_fontweight("bold")
    ax.set_title("Revenue Contribution by Segment", fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, "dashboard2_rfm.png"),
                dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print("  ✓ Saved: dashboard2_rfm.png")


# ─── DASHBOARD 3 : CATEGORY + REVIEW EDA ─────────────────────────────────────
def plot_dashboard3(conn, fact):
    print("\n📊 Dashboard 3 — Category Performance + EDA")
    cat = pd.read_sql(CATEGORY_SQL, conn)

    fig = plt.figure(figsize=(18, 11), facecolor=PALETTE["bg"])
    fig.suptitle("Category Performance & Product/Review Analytics",
                 fontsize=14, fontweight="bold", color=PALETTE["dark"])
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

    # Top 10 categories by revenue
    ax1 = fig.add_subplot(gs[0, :2])
    bar_cols = [PALETTE["green"]] * 3 + [PALETTE["blue"]] * 7
    bars = ax1.barh(cat["category"][::-1], cat["revenue"][::-1]/1000,
                    color=bar_cols[::-1], edgecolor="white")
    for bar, val in zip(bars, cat["revenue"][::-1]/1000):
        ax1.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                 f"R${val:.0f}K", va="center", fontsize=8, color=PALETTE["gray"])
    ax1.set_title("Top 10 Categories by Revenue  |  Green = Top 3 drivers", fontsize=11)
    ax1.set_xlabel("Revenue (R$ K)")
    top3_patch = mpatches.Patch(color=PALETTE["green"], label="Top 3 revenue drivers")
    ax1.legend(handles=[top3_patch], fontsize=8, loc="lower right")

    # Review score distribution
    ax2 = fig.add_subplot(gs[0, 2])
    review_counts = fact.dropna(subset=["review_score"])["review_score"].value_counts().sort_index()
    rev_colors = [PALETTE["red"], PALETTE["amber"], PALETTE["amber"],
                  PALETTE["blue"], PALETTE["green"]]
    ax2.bar(review_counts.index.astype(int), review_counts.values,
            color=rev_colors, edgecolor="white", width=0.7)
    ax2.set_title("Review Score Distribution", fontsize=11)
    ax2.set_xlabel("Score (1–5)"); ax2.set_ylabel("Orders")
    ax2.set_xticks([1,2,3,4,5])

    # Category avg review heatmap style
    ax3 = fig.add_subplot(gs[1, :2])
    x = np.arange(len(cat))
    w = 0.35
    b1 = ax3.bar(x - w/2, cat["orders"],    width=w, label="Orders",  color=PALETTE["blue"],   alpha=0.85)
    b2 = ax3.bar(x + w/2, cat["revenue"]/1000, width=w, label="Revenue (K)", color=PALETTE["green"], alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels(cat["category"], rotation=35, ha="right", fontsize=8)
    ax3.set_title("Orders vs Revenue — Top 10 Categories", fontsize=11)
    ax3.legend(fontsize=9)
    ax3.yaxis.grid(True); ax3.set_axisbelow(True)

    # Payment type distribution
    ax4 = fig.add_subplot(gs[1, 2])
    ptype = (fact.dropna(subset=["payment_type"])["payment_type"]
             .value_counts().head(4))
    pcolors = [PALETTE["green"], PALETTE["blue"], PALETTE["amber"], PALETTE["purple"]]
    wedges, texts, autotexts = ax4.pie(
        ptype.values, labels=ptype.index,
        colors=pcolors, autopct="%1.1f%%", startangle=140,
        wedgeprops=dict(edgecolor="white", linewidth=1.5)
    )
    for t in texts:     t.set_fontsize(9)
    for t in autotexts: t.set_fontsize(8)
    ax4.set_title("Payment Type Breakdown", fontsize=11)

    plt.savefig(os.path.join(OUTPUT_DIR, "dashboard3_category_eda.png"),
                dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print("  ✓ Saved: dashboard3_category_eda.png")


# ─── DASHBOARD 4 : DELIVERY + CUSTOMER GEO ───────────────────────────────────
def plot_dashboard4(fact):
    print("\n📊 Dashboard 4 — Delivery & Geo Analytics")

    fig = plt.figure(figsize=(18, 10), facecolor=PALETTE["bg"])
    fig.suptitle("Delivery Performance & Customer Geography",
                 fontsize=14, fontweight="bold", color=PALETTE["dark"])
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

    # Delivery delay histogram
    ax1 = fig.add_subplot(gs[0, :2])
    delays = fact["delivery_delay_days"].dropna()
    delays = delays[(delays > -60) & (delays < 60)]
    ax1.hist(delays[delays <= 0], bins=40, color=PALETTE["green"],
             alpha=0.75, label="On-time / early")
    ax1.hist(delays[delays > 0],  bins=20, color=PALETTE["red"],
             alpha=0.75, label="Late delivery")
    ax1.axvline(0, color=PALETTE["dark"], linestyle="--", linewidth=1.2)
    ax1.set_title("Delivery Delay Distribution  (negative = early)", fontsize=11)
    ax1.set_xlabel("Days vs Estimated"); ax1.set_ylabel("Orders")
    ax1.legend(fontsize=9)

    # On-time rate by year
    ax2 = fig.add_subplot(gs[0, 2])
    for yr, col in [(2017, PALETTE["blue"]), (2018, PALETTE["green"])]:
        sub = fact[fact["year"] == yr]["delivery_delay_days"].dropna()
        rate = (sub <= 0).mean() * 100
        late = 100 - rate
        ax2.barh([yr], [rate], color=col, label=f"{yr} On-time {rate:.1f}%", alpha=0.85)
        ax2.barh([yr], [late], left=[rate], color=PALETTE["red"], alpha=0.4)
    ax2.set_xlim(0, 100); ax2.set_xlabel("Percentage")
    ax2.set_title("On-Time Delivery Rate", fontsize=11)
    ax2.legend(fontsize=8)

    # Top 10 states by orders
    ax3 = fig.add_subplot(gs[1, :2])
    state_orders = (fact.dropna(subset=["customer_state"])
                    .groupby("customer_state")["order_id"]
                    .nunique().sort_values(ascending=False).head(10))
    bar_colors = [PALETTE["green"]] + [PALETTE["blue"]] * 9
    ax3.bar(state_orders.index, state_orders.values,
            color=bar_colors, edgecolor="white")
    for i, v in enumerate(state_orders.values):
        ax3.text(i, v + 50, str(v), ha="center", fontsize=8, color=PALETTE["gray"])
    ax3.set_title("Top 10 States by Order Volume  |  SP dominates", fontsize=11)
    ax3.set_xlabel("State"); ax3.set_ylabel("Orders")
    ax3.yaxis.grid(True); ax3.set_axisbelow(True)

    # Monthly order heatmap
    ax4 = fig.add_subplot(gs[1, 2])
    pivot = (fact.groupby(["year","month"])["order_id"]
             .nunique().reset_index()
             .pivot(index="year", columns="month", values="order_id"))
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"][:len(pivot.columns)]
    sns.heatmap(pivot, ax=ax4, cmap="YlGn", annot=True, fmt=".0f",
                linewidths=0.5, cbar_kws={"shrink": 0.8}, annot_kws={"size": 8})
    ax4.set_title("Orders Heatmap (Year × Month)", fontsize=11)
    ax4.set_xlabel(""); ax4.set_ylabel("")

    plt.savefig(os.path.join(OUTPUT_DIR, "dashboard4_delivery_geo.png"),
                dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print("  ✓ Saved: dashboard4_delivery_geo.png")


# ─── PRINT SUMMARY REPORT ─────────────────────────────────────────────────────
def print_summary(conn):
    print("\n" + "="*60)
    print("  📋 KEY FINDINGS — STAKEHOLDER SUMMARY")
    print("="*60)

    kpi = pd.read_sql(KPI_SQL, conn)
    rfm = pd.read_sql(RFM_SQL, conn)
    cat = pd.read_sql(CATEGORY_SQL, conn)

    total_rev = kpi["total_revenue"].sum()
    total_ord = kpi["total_orders"].sum()
    q3_18 = kpi[(kpi["year"]==2018) & (kpi["quarter"]==3)]["total_revenue"].sum()
    q3_17 = kpi[(kpi["year"]==2017) & (kpi["quarter"]==3)]["total_revenue"].sum()
    delta  = (q3_18 - q3_17) / q3_17 * 100 if q3_17 else 0

    print(f"\n  KPIs")
    print(f"  • Total Revenue  : R$ {total_rev:,.0f}")
    print(f"  • Total Orders   : {total_ord:,}")
    print(f"  • Q3 2018 Decline: {delta:+.1f}% vs Q3 2017  ← FLAGGED")

    print(f"\n  TOP 3 REVENUE CATEGORIES")
    for i, row in cat.head(3).iterrows():
        print(f"  {i+1}. {row['category']:<30} R$ {row['revenue']:,.0f}")

    print(f"\n  RFM SEGMENTS (3 High-Value Cohorts)")
    for _, row in rfm.iterrows():
        print(f"  • {row['segment']:<22} {int(row['customers']):>5} customers  "
              f"Avg LTV: R${row['avg_monetary']:.0f}  "
              f"Total: R${row['total_revenue']:,.0f}")

    print("\n" + "="*60)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dfs          = load_data(DATA_DIR)
    fact, prods  = transform(dfs)
    conn         = load_sqlite(fact)

    plot_dashboard1(conn)
    plot_dashboard2(conn)
    plot_dashboard3(conn, fact)
    plot_dashboard4(fact)
    print_summary(conn)

    conn.close()
    print("\n✅ All dashboards saved. Open the PNG files to view.")
