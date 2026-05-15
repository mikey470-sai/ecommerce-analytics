# E-Commerce Sales Analytics — Brazilian Olist Dataset

> End-to-end data analytics project on 100,000+ real e-commerce orders.  
> Covers ETL pipelines, SQL-based RFM segmentation, KPI dashboards, and stakeholder-ready findings.

---

## Dashboards

### Dashboard 1 — KPI Overview & Revenue Trend
![KPI Dashboard](dashboards/dashboard1_kpi_revenue.png)

### Dashboard 2 — RFM Customer Segmentation
![RFM Segmentation](dashboards/dashboard2_rfm.png)

### Dashboard 3 — Category Performance & EDA
![Category EDA](dashboards/dashboard3_category_eda.png)

### Dashboard 4 — Delivery Performance & Geography
![Delivery & Geo](dashboards/dashboard4_delivery_geo.png)

---

## Project Summary

| Metric | Value |
|---|---|
| Total Orders Analysed | 100,000+ |
| Date Range | Jan 2017 – Aug 2018 |
| Total Revenue | R$ 63.6M |
| Avg Order Value | R$ 154 |
| Avg Review Score | 4.1 / 5 |
| RFM Cohorts Identified | 3 high-value segments |

---

## Key Findings

- **Q3 2018 purchase decline flagged** — order volume dropped significantly compared to Q3 2017, identified via quarterly KPI tracking and surfaced in mock stakeholder review.
- **Top 3 revenue-driving categories** — Computers & Accessories, Furniture & Decor, and Bed Bath & Table collectively account for ~47% of total revenue.
- **3 high-value customer cohorts identified via RFM segmentation** — Champions, Loyal Customers, and At-Risk segments; Champions alone drive ~37% of total revenue despite being 14% of the customer base.
- **São Paulo dominates order volume** — SP state accounts for ~42% of all orders, with RJ and MG as distant second and third.
- **74% of payments via credit card** — installment-based buying is the dominant behaviour pattern.

---

## Tech Stack

| Tool | Usage |
|---|---|
| **Python** | Core scripting, ETL pipeline orchestration |
| **Pandas** | Data cleaning, merging, feature engineering |
| **SQL (SQLite)** | CTEs, window functions, RFM scoring, KPI aggregations |
| **Matplotlib** | KPI cards, bar charts, trend lines, heatmaps |
| **Seaborn** | Statistical plots, delivery heatmap |
| **ETL Pipeline** | Extract → Transform → Load into in-memory SQLite |

---

## SQL Highlights

### RFM Segmentation using CTEs + Window Functions
```sql
WITH rfm_base AS (
    SELECT
        customer_unique_id,
        MAX(order_purchase_timestamp)       AS last_purchase,
        COUNT(DISTINCT order_id)            AS frequency,
        SUM(payment_value)                  AS monetary
    FROM orders_fact
    WHERE order_status = 'delivered'
    GROUP BY customer_unique_id
),
rfm_scored AS (
    SELECT *,
        JULIANDAY('2018-10-01') - JULIANDAY(last_purchase)   AS recency_days,
        NTILE(5) OVER (ORDER BY JULIANDAY('2018-10-01') - JULIANDAY(last_purchase) DESC) AS r_score,
        NTILE(5) OVER (ORDER BY frequency)                   AS f_score,
        NTILE(5) OVER (ORDER BY monetary)                    AS m_score
    FROM rfm_base
)
SELECT segment, COUNT(*) AS customers,
       ROUND(AVG(monetary), 2) AS avg_ltv,
       ROUND(SUM(monetary), 2) AS total_revenue
FROM rfm_scored
GROUP BY segment
ORDER BY total_revenue DESC
```

---

## Project Structure

```
ecommerce_analytics/
├── analysis.py              # Main script — ETL + SQL + all 4 dashboards
├── README.md
├── dashboards/
│   ├── dashboard1_kpi_revenue.png
│   ├── dashboard2_rfm.png
│   ├── dashboard3_category_eda.png
│   └── dashboard4_delivery_geo.png
└── data/                    # Not tracked (see .gitignore)
    └── olist_*.csv
```

---

## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/ecommerce-analytics.git
cd ecommerce-analytics
```

**2. Install dependencies**
```bash
pip install pandas matplotlib seaborn
```

**3. Download the dataset**

Get the Olist dataset from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and place all CSV files inside the `data/` folder.

**4. Run the analysis**
```bash
python analysis.py
```

All 4 dashboard PNGs will be saved in the project root.

---

## Dataset

**Brazilian E-Commerce Public Dataset by Olist**  
Source: [Kaggle — olistbr/brazilian-ecommerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)  
100,000 orders from 2016 to 2018 across multiple Brazilian marketplaces.  
Includes: orders, customers, products, sellers, payments, reviews, geolocation.

---

## Author

**N Sai** — B.Tech Computer Science Engineering (2026)  
[GitHub](https://github.com/YOUR_USERNAME)
