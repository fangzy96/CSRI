import pandas as pd
import numpy as np
from scipy.stats import fisher_exact, ranksums
from datetime import timedelta

# Rank-sum test for continuous variables
def ranksum_test(df, col):
    fast = df[df["cluster"] == 1][col].dropna()
    slow = df[df["cluster"] == 2][col].dropna()
    if len(fast) == 0 or len(slow) == 0:
        return "N/A"
    stat, p = ranksums(fast, slow)
    return f"{p:.4f}"

# Fisher's exact test for binary variables
def fisher_test(df, col, value=1):
    table = pd.crosstab(df[col] == value, df["cluster"])
    if table.shape == (2, 2):
        _, p = fisher_exact(table)
        return f"{p:.4f}"
    else:
        return "N/A"


# Load data
cluster_df = pd.read_csv("All_types_of_surgery_sf36_clustered_results_2_7.csv")
chart_df = pd.read_csv("Chart_review_20250709_processed.csv", encoding="ISO-8859-1")
merged_df = cluster_df.merge(chart_df, left_on="uuid", right_on="patient_uuid", how="left")

print(merged_df)
merged_df.to_csv('Combined.csv', index=False)

# Ensure date columns are datetime type
merged_df['surgery_date_x'] = pd.to_datetime(merged_df['surgery_date_x'], errors='coerce')

# Get all readmit date columns (e.g. readmit_1_date, readmit_2_date, ...)
readmit_cols = [col for col in merged_df.columns if col.startswith("readmit_") and col.endswith("_date")]

# Convert these columns to datetime
for col in readmit_cols:
    merged_df[col] = pd.to_datetime(merged_df[col], errors='coerce')

# Initialize statistics dict
readmit_stats = {}

# Iterate over two clusters: 1 = Fast, 2 = Slow
for cluster_label, cluster_name in [(1, "Fast"), (2, "Slow")]:
    group = merged_df[merged_df["cluster"] == cluster_label]
    count = 0

    for _, row in group.iterrows():
        surgery_date = row['surgery_date_x']
        if pd.isna(surgery_date):
            continue

        # Check if any readmit date is within 30 days (inclusive)
        for col in readmit_cols:
            readmit_date = row[col]
            if pd.notna(readmit_date) and (readmit_date - surgery_date).days <= 30 and (
                    readmit_date - surgery_date).days >= 0:
                count += 1
                break  # Count each patient only once
    total = len(group)
    proportion = count / total if total > 0 else 0
    readmit_stats[cluster_name] = {"count": count, "total": total, "proportion": proportion}

# Print results
for group_name in ["Fast", "Slow"]:
    c = readmit_stats[group_name]
    print(f"{group_name} group: {c['count']} / {c['total']} patients had 30-day readmission ({c['proportion']:.2%})")

# Create new column: 1/0 for 30-day readmission
def has_30d_readmit(row):
    surgery_date = row['surgery_date_x']
    if pd.isna(surgery_date):
        return 0
    for col in readmit_cols:
        readmit_date = row[col]
        if pd.notna(readmit_date):
            delta = (readmit_date - surgery_date).days
            if 0 <= delta <= 30:
                return 1
    return 0

merged_df['readmit_30d'] = merged_df.apply(has_30d_readmit, axis=1)

# Fisher exact test for significance
pval = fisher_test(merged_df, 'readmit_30d', value=1)
print(f"Fisher exact test p-value for 30-day readmission between clusters: {pval}")
