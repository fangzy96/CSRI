import pandas as pd
import numpy as np
from scipy.stats import fisher_exact, ranksums
from scipy.stats import chi2_contingency

# Chi-square test for multi-categorical variables
def chi2_test(df, col):
    table = pd.crosstab(df[col], df["cluster"])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return "N/A"  # Avoid computation when table has insufficient rows/cols
    try:
        chi2, p, _, _ = chi2_contingency(table)
        return f"{p:.4f}"
    except:
        return "N/A"

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
chart_df = pd.read_csv("Chart_review_20250616_processed.csv", encoding="ISO-8859-1")
merged_df = cluster_df.merge(chart_df, left_on="uuid", right_on="patient_uuid", how="left")

# Map sex
merged_df["Sex"] = merged_df["sex"].map({0: "Male", 1: "Female"})

# Age summary: median (IQR)
def age_summary(df):
    q1 = df["age"].quantile(0.25)
    q2 = df["age"].median()
    q3 = df["age"].quantile(0.75)
    return f"{int(q2)} ({int(q1)}, {int(q3)})"

# Sex summary
def sex_summary(df):
    total = len(df)
    counts = df["Sex"].value_counts()
    male = counts.get("Male", 0)
    female = counts.get("Female", 0)
    male_str = f"{male} ({male / total * 100:.1f}%)"
    female_str = f"{female} ({female / total * 100:.1f}%)"
    return male_str, female_str

# Print header
print("{:<12} {:<20} {:<20} {:<20}".format("Feature", "Total", "Fast (cluster=1)", "Slow (cluster=2)", "p-value"))


# Age
age_total = age_summary(merged_df)
age_fast = age_summary(merged_df[merged_df["cluster"] == 1])
age_slow = age_summary(merged_df[merged_df["cluster"] == 2])
# Rank-sum test p-value for age
age_pval = ranksum_test(merged_df, "age")

# Print results with p-value
print("-" * 100)
print("{:<20} {:<20} {:<20} {:<20} {:<10}".format("Age", "Total", "Fast (cluster=1)", "Slow (cluster=2)", "p-value"))
print("-" * 100)
print("{:<12} {:<20} {:<20} {:<20} {:<10}".format("Age", age_total, age_fast, age_slow, age_pval))


# Sex
male_total, female_total = sex_summary(merged_df)
male_fast, female_fast = sex_summary(merged_df[merged_df["cluster"] == 1])
male_slow, female_slow = sex_summary(merged_df[merged_df["cluster"] == 2])
# Fisher exact test p-value for sex (Male = True)
sex_pval = fisher_test(merged_df, "Sex", value="Male")
# sex_pval = chi2_test(merged_df, "Sex")

# Print header
print("-" * 100)
print("{:<20} {:<20} {:<20} {:<20} {:<10}".format("Sex", "Total", "Fast (cluster=1)", "Slow (cluster=2)", "p-value"))
print("-" * 100)

# Print each sex category with p-value
print("{:<12} {:<20} {:<20} {:<20} {:<10}".format("Male", male_total, male_fast, male_slow, sex_pval))
print("{:<12} {:<20} {:<20} {:<20} {:<10}".format("Female", female_total, female_fast, female_slow, sex_pval))



# Simplify patient_race to categories
def simplify_race(race):
    if pd.isna(race):
        return "Unknown"
    race = race.strip()
    # print(race)
    if race == "White":
        return "White"
    elif race == "Black":
        return "Other"
    elif race == "Asian":
        return "Other"
    else:
        return "Other"

merged_df["Race"] = merged_df["patient_race"].apply(simplify_race)

# Race summary
def race_summary(df):
    total = len(df)
    counts = df["Race"].value_counts()
    return {
        race: f"{counts.get(race, 0)} ({counts.get(race, 0) / total * 100:.1f}%)"
        for race in ["White", "Other", "Unknown"]
    }

# Get results by group
race_total = race_summary(merged_df)
race_fast = race_summary(merged_df[merged_df["cluster"] == 1])
race_slow = race_summary(merged_df[merged_df["cluster"] == 2])

# Fisher exact test p-value per race category
# race_pvals = {
#     race: fisher_test(merged_df.copy(), "Race", value=race)
#     for race in ["White", "Other", "Unknown"]
# }
race_pvals = chi2_test(merged_df.copy(), "Race")
# Print header
print("-" * 100)
print("{:<20} {:<20} {:<20} {:<20} {:<10}".format("Race", "Total", "Fast (cluster=1)", "Slow (cluster=2)", "p-value"))
print("-" * 100)

# # Print each race category with p-value
# for race in ["White", "Other", "Unknown"]:
#     print("{:<12} {:<20} {:<20} {:<20} {:<10}".format(
#         f"{race}", race_total[race], race_fast[race], race_slow[race], race_pvals[race]
#     ))
# Print each race category with p-value
for race in ["White", "Other", "Unknown"]:
    print("{:<12} {:<20} {:<20} {:<20} {:<10}".format(
        f"{race}", race_total[race], race_fast[race], race_slow[race], race_pvals
    ))

# Map ethnicity
def map_ethnicity(val):
    if val == 1:
        return "Hispanic"
    elif val == 0:
        return "Non-Hispanic"
    else:
        return "Unknown"

merged_df["Ethnicity"] = merged_df["patient_hispanic"].apply(map_ethnicity)

def ethnicity_summary(df):
    counts = df["Ethnicity"].value_counts()
    total = len(df)
    result = {}
    for status in ["Hispanic", "Non-Hispanic", "Unknown"]:
        count = counts.get(status, 0)
        result[status] = f"{count} ({count / total * 100:.1f}%)"
    return result

ethnicity_total = ethnicity_summary(merged_df)
ethnicity_fast = ethnicity_summary(merged_df[merged_df["cluster"] == 1])
ethnicity_slow = ethnicity_summary(merged_df[merged_df["cluster"] == 2])

# Fisher exact test p-value per ethnicity category
# ethnicity_pvals = {
#     status: fisher_test(merged_df.copy(), "Ethnicity", value=status)
#     for status in ["Hispanic", "Non-Hispanic", "Unknown"]
# }
ethnicity_pvals = chi2_test(merged_df.copy(), "Ethnicity")
# Print results with p-value
print("-" * 100)
print("{:<20} {:<20} {:<20} {:<20} {:<10}".format("Ethnicity", "Total", "Fast (cluster=1)", "Slow (cluster=2)", "p-value"))
print("-" * 100)
# for status in ["Hispanic", "Non-Hispanic", "Unknown"]:
#     print("{:<20} {:<20} {:<20} {:<20} {:<10}".format(
#         status,
#         ethnicity_total[status],
#         ethnicity_fast[status],
#         ethnicity_slow[status],
#         ethnicity_pvals[status]
#     ))
for status in ["Hispanic", "Non-Hispanic", "Unknown"]:
    print("{:<20} {:<20} {:<20} {:<20} {:<10}".format(
        status,
        ethnicity_total[status],
        ethnicity_fast[status],
        ethnicity_slow[status],
        ethnicity_pvals
    ))

# Comorbidity name mapping
comorbidity_names = {
    1: "Stroke/TIA",
    2: "Dementia",
    3: "CAD",
    4: "MI",
    5: "CHF",
    6: "AF",
    7: "HTN",
    8: "COPD",
    9: "PulmHTN",
    10: "OSA",
    11: "PUD",
    12: "IBD",
    13: "Liver",
    14: "CKD",
    15: "DVT/PE",
    16: "PVD/PAD",
    17: "Diabetes"
}

# Summary function
def comorbidity_summary(df):
    total = len(df)
    summary = {}
    for i in range(1, 18):
        col = f"patient_comorbidities___{i}"
        count = df[col].fillna(0).astype(int).sum()
        summary[comorbidity_names[i]] = f"{int(count)} ({count / total * 100:.1f}%)"
    return summary

# Three groups
comorb_total = comorbidity_summary(merged_df)
comorb_fast = comorbidity_summary(merged_df[merged_df["cluster"] == 1])
comorb_slow = comorbidity_summary(merged_df[merged_df["cluster"] == 2])


def collapse_comorbidities(df):
    comorb_cols = [f"patient_comorbidities___{i}" for i in range(1, 18)]
    # At most one value is 1 per row; find its position (category)
    comorb_type = df[comorb_cols].idxmax(axis=1)
    # Extract numeric part from column name
    comorb_type = comorb_type.str.extract(r'(\d+)', expand=False)
    return comorb_type.astype("Int64")
# merged_df["comorb_type"] = collapse_comorbidities(merged_df)

# Fisher exact test p-value per comorbidity
comorb_pvals = {
    comorbidity_names[i]: fisher_test(merged_df, f"patient_comorbidities___{i}")
    for i in range(1, 18)
}

# Print results with p-value
print("-" * 100)
print("{:<20} {:<20} {:<20} {:<20} {:<10}".format("Comorbidity", "Total", "Fast (cluster=1)", "Slow (cluster=2)", "p-value"))
print("-" * 100)
for name in comorbidity_names.values():
    print("{:<20} {:<20} {:<20} {:<20} {:<10}".format(
        name, comorb_total[name], comorb_fast[name], comorb_slow[name], comorb_pvals[name]
    ))


# BMI median (IQR)
def bmi_summary(df):
    q1 = df["bmi"].quantile(0.25)
    q2 = df["bmi"].median()
    q3 = df["bmi"].quantile(0.75)
    return f"{q2:.1f} ({q1:.1f}, {q3:.1f})"

# Three groups
bmi_total = bmi_summary(merged_df)
bmi_fast = bmi_summary(merged_df[merged_df["cluster"] == 1])
bmi_slow = bmi_summary(merged_df[merged_df["cluster"] == 2])

# Rank-sum test p-value for BMI
bmi_pval = ranksum_test(merged_df, "bmi")

# Print results with p-value
print("-" * 100)
print("{:<20} {:<20} {:<20} {:<20} {:<10}".format("BMI", "Total", "Fast (cluster=1)", "Slow (cluster=2)", "p-value"))
print("-" * 100)
print("{:<12} {:<20} {:<20} {:<20} {:<10}".format("BMI", bmi_total, bmi_fast, bmi_slow, bmi_pval))

# Employment Status mapping (with Unknown for missing)
employment_mapping = {
    1: "Full Time",
    2: "Part Time",
    3: "Not Employed",
    4: "Self Employed",
    5: "Retired",
    6: "Active Military",
    7: "Student Full Time",
    8: "Student Part Time",
    9: "Unknown",
    100: "Disabled",
    101: "Homemaker"
}

# Map and fill missing as Unknown
merged_df["Employment Status"] = merged_df["patient_job"].map(employment_mapping)
merged_df["Employment Status"] = merged_df["Employment Status"].fillna("Unknown")

def employment_summary(df):
    counts = df["Employment Status"].value_counts()
    total = len(df)
    result = {}
    for label in employment_mapping.values():
        count = counts.get(label, 0)
        result[label] = f"{count} ({count / total * 100:.1f}%)"
    # Ensure missing is counted in Unknown (including unmapped)
    if "Unknown" not in result:
        unknown_count = counts.get("Unknown", 0)
        result["Unknown"] = f"{unknown_count} ({unknown_count / total * 100:.1f}%)"
    return result

employment_total = employment_summary(merged_df)
employment_fast = employment_summary(merged_df[merged_df["cluster"] == 1])
employment_slow = employment_summary(merged_df[merged_df["cluster"] == 2])

# Fisher exact test p-value per Employment Status
# employment_pvals = {
#     status: fisher_test(merged_df.copy(), "Employment Status", value=status)
#     for status in set(employment_mapping.values()).union({"Unknown"})
# }
employment_pvals = chi2_test(merged_df.copy(), "Employment Status")

# Print results with p-value
print("-" * 100)
print("{:<25} {:<20} {:<20} {:<20} {:<10}".format("Employment Status", "Total", "Fast (cluster=1)", "Slow (cluster=2)", "p-value"))
print("-" * 100)
# for status in sorted(employment_pvals.keys()):
#     print("{:<25} {:<20} {:<20} {:<20} {:<10}".format(
#         status,
#         employment_total.get(status, "0 (0.0%)"),
#         employment_fast.get(status, "0 (0.0%)"),
#         employment_slow.get(status, "0 (0.0%)"),
#         employment_pvals[status]
#     ))
for status in employment_mapping.values():
    print("{:<25} {:<20} {:<20} {:<20} {:<10}".format(
        status,
        employment_total.get(status, "0 (0.0%)"),
        employment_fast.get(status, "0 (0.0%)"),
        employment_slow.get(status, "0 (0.0%)"),
        employment_pvals
    ))


# Map smoking status
smoking_map = {
    0: "Never",
    1: "Former (>30 days)",
    2: "Former (<30 days)",
    3: "Active"
}
merged_df["Smoking"] = merged_df["smoking_status"].map(smoking_map)

# Smoking status summary
def smoking_summary(df):
    total = len(df)
    counts = df["Smoking"].value_counts()
    result = {}
    for status in ["Never", "Former (>30 days)", "Former (<30 days)", "Active"]:
        count = counts.get(status, 0)
        result[status] = f"{count} ({count / total * 100:.1f}%)"
    return result

# Get results
smoking_total = smoking_summary(merged_df)
smoking_fast = smoking_summary(merged_df[merged_df["cluster"] == 1])
smoking_slow = smoking_summary(merged_df[merged_df["cluster"] == 2])

# Fisher exact test p-value per Smoking Status
# smoking_pvals = {
#     status: fisher_test(merged_df.copy(), "Smoking", value=status)
#     for status in ["Never", "Former (>30 days)", "Former (<30 days)", "Active"]
# }
smoking_pvals = chi2_test(merged_df.copy(), "Smoking")
# Print results with p-value
print("-" * 100)
print("{:<25} {:<20} {:<20} {:<20} {:<10}".format("Smoking Status", "Total", "Fast (cluster=1)", "Slow (cluster=2)", "p-value"))
print("-" * 100)
# for status in ["Never", "Former (>30 days)", "Former (<30 days)", "Active"]:
#     print("{:<25} {:<20} {:<20} {:<20} {:<10}".format(
#         status,
#         smoking_total[status],
#         smoking_fast[status],
#         smoking_slow[status],
#         smoking_pvals[status]
#     ))
for status in ["Never", "Former (>30 days)", "Former (<30 days)", "Active"]:
    print("{:<25} {:<20} {:<20} {:<20} {:<10}".format(
        status,
        smoking_total[status],
        smoking_fast[status],
        smoking_slow[status],
        smoking_pvals
    ))
# Lung function summary: median (IQR)
def lung_summary(df, col):
    clean = df[col].dropna()
    if len(clean) == 0:
        return "N/A"
    q1 = clean.quantile(0.25)
    q2 = clean.median()
    q3 = clean.quantile(0.75)
    return f"{q2:.1f} ({q1:.1f}, {q3:.1f})"


# Add significance test p-value output
for col, label in {
    "pft_fev1_x": "FEV1",
    "pft_fvc_x": "FVC",
    "pft_dlco_x": "DLCO"
}.items():
    total_val = lung_summary(merged_df, col)
    fast_val = lung_summary(merged_df[merged_df["cluster"] == 1], col)
    slow_val = lung_summary(merged_df[merged_df["cluster"] == 2], col)
    pval = ranksum_test(merged_df, col)

    print("-" * 100)
    print("{:<20} {:<20} {:<20} {:<20} {:<10}".format("Lung Metric", "Total", "Fast (cluster=1)", "Slow (cluster=2)",
                                                      "p-value"))
    print("-" * 100)
    print("{:<20} {:<20} {:<20} {:<20} {:<10}".format(label, total_val, fast_val, slow_val, pval))

# Lung resection type mapping
resection_mapping = {
    1: "Wedge Resection",
    2: "Segmentectomy",
    3: "Lobectomy",
    4: "Extended lobectomy",
    5: "Bilobectomy",
    6: "Sleeve Lobectomy",
    7: "Pneumonectomy"
}

# Get all related columns
resection_cols = [f"lung_resection___{i}" for i in range(1, 8)]

# Flatten data (one entry per surgery per row)
def get_long_lung_df(df):
    long_df = df[["uuid", "cluster"] + resection_cols].copy()
    long_df = long_df.melt(id_vars=["uuid", "cluster"], value_vars=resection_cols,
                           var_name="resection_code", value_name="has_resection")
    long_df["type_id"] = long_df["resection_code"].str.extract(r"(\d+)").astype(int)
    long_df = long_df[long_df["has_resection"] == 1]
    long_df["Resection Type"] = long_df["type_id"].map(resection_mapping)
    return long_df

long_lung_df = get_long_lung_df(merged_df)

# Count function
def count_lung_type(df):
    counts = df["Resection Type"].value_counts()
    total = len(df["uuid"].unique())
    return {k: f"{v} ({v / total * 100:.1f}%)" for k, v in counts.items()}

# Results by cluster
lung_total = count_lung_type(long_lung_df)
lung_fast = count_lung_type(long_lung_df[long_lung_df["cluster"] == 1])
lung_slow = count_lung_type(long_lung_df[long_lung_df["cluster"] == 2])

# Fisher exact test p-value per Lung Resection type
lung_pvals = {
    resection: fisher_test(long_lung_df.copy(), "Resection Type", value=resection)
    for resection in resection_mapping.values()
}

# Print results with p-value
print("-" * 100)
print("{:<25} {:<20} {:<20} {:<20} {:<10}".format("Lung Resection", "Total", "Fast (cluster=1)", "Slow (cluster=2)", "p-value"))
print("-" * 100)
for resection in resection_mapping.values():
    t = lung_total.get(resection, "0 (0.0%)")
    f = lung_fast.get(resection, "0 (0.0%)")
    s = lung_slow.get(resection, "0 (0.0%)")
    p = lung_pvals[resection]
    print("{:<25} {:<20} {:<20} {:<20} {:<10}".format(resection, t, f, s, p))


# Mapping
approach_mapping = {
    1: "Open",
    2: "VATS",
    3: "VATS converted to open",
    4: "Robotic",
    5: "Robotic converted to open"
}

# Extract related columns
approach_cols = [f"thoracic_approach___{i}" for i in range(1, 6)]

# Flatten data
def get_long_approach_df(df):
    long_df = df[["uuid", "cluster"] + approach_cols].copy()
    long_df = long_df.melt(id_vars=["uuid", "cluster"], value_vars=approach_cols,
                           var_name="approach_code", value_name="has_approach")
    long_df["approach_id"] = long_df["approach_code"].str.extract(r"(\d+)").astype(int)
    long_df = long_df[long_df["has_approach"] == 1]
    long_df["Surgical Approach"] = long_df["approach_id"].map(approach_mapping)
    return long_df

long_approach_df = get_long_approach_df(merged_df)

# Count function
def count_approach_type(df):
    counts = df["Surgical Approach"].value_counts()
    total = len(df["uuid"].unique())
    return {k: f"{v} ({v / total * 100:.1f}%)" for k, v in counts.items()}

# Summarize by group
approach_total = count_approach_type(long_approach_df)
approach_fast = count_approach_type(long_approach_df[long_approach_df["cluster"] == 1])
approach_slow = count_approach_type(long_approach_df[long_approach_df["cluster"] == 2])

# Fisher exact test p-value per Surgical Approach type
# approach_pvals = {
#     approach: fisher_test(long_approach_df.copy(), "Surgical Approach", value=approach)
#     for approach in approach_mapping.values()
# }
approach_pvals = chi2_test(long_approach_df.copy(), "Surgical Approach")
# Print results with p-value
print("-" * 100)
print("{:<25} {:<20} {:<20} {:<20} {:<10}".format("Surgical Approach", "Total", "Fast (cluster=1)", "Slow (cluster=2)", "p-value"))
print("-" * 100)
# for approach in approach_mapping.values():
#     t = approach_total.get(approach, "0 (0.0%)")
#     f = approach_fast.get(approach, "0 (0.0%)")
#     s = approach_slow.get(approach, "0 (0.0%)")
#     p = approach_pvals[approach]
#     print("{:<25} {:<20} {:<20} {:<20} {:<10}".format(approach, t, f, s, p))
for approach in approach_mapping.values():
    t = approach_total.get(approach, "0 (0.0%)")
    f = approach_fast.get(approach, "0 (0.0%)")
    s = approach_slow.get(approach, "0 (0.0%)")
    p = approach_pvals
    print("{:<25} {:<20} {:<20} {:<20} {:<10}".format(approach, t, f, s, p))

# Mapping
dispo_mapping = {
    0: "Same-day discharge",
    1: "Inpatient floor",
    2: "ICU"
}

# Create new column
merged_df["Inpatient Disposition"] = merged_df["surgery_inpatient_dispo"].map(dispo_mapping)

# Summary function
def dispo_summary(df):
    total = len(df)
    counts = df["Inpatient Disposition"].value_counts()
    return {k: f"{v} ({v / total * 100:.1f}%)" for k, v in counts.items()}

# Summarize by group
dispo_total = dispo_summary(merged_df)
dispo_fast = dispo_summary(merged_df[merged_df["cluster"] == 1])
dispo_slow = dispo_summary(merged_df[merged_df["cluster"] == 2])

# Fisher exact test p-value per Inpatient Disposition
# dispo_pvals = {
#     dispo: fisher_test(merged_df.copy(), "Inpatient Disposition", value=dispo)
#     for dispo in dispo_mapping.values()
# }
dispo_pvals = chi2_test(merged_df.copy(), "Inpatient Disposition")
# Print results with p-value
print("-" * 100)
print("{:<25} {:<20} {:<20} {:<20} {:<10}".format("Inpatient Disposition", "Total", "Fast (cluster=1)", "Slow (cluster=2)", "p-value"))
print("-" * 100)
# for dispo in dispo_mapping.values():
#     t = dispo_total.get(dispo, "0 (0.0%)")
#     f = dispo_fast.get(dispo, "0 (0.0%)")
#     s = dispo_slow.get(dispo, "0 (0.0%)")
#     p = dispo_pvals[dispo]
#     print("{:<25} {:<20} {:<20} {:<20} {:<10}".format(dispo, t, f, s, p))
for dispo in dispo_mapping.values():
    t = dispo_total.get(dispo, "0 (0.0%)")
    f = dispo_fast.get(dispo, "0 (0.0%)")
    s = dispo_slow.get(dispo, "0 (0.0%)")
    p = dispo_pvals
    print("{:<25} {:<20} {:<20} {:<20} {:<10}".format(dispo, t, f, s, p))

# Mapping
discharge_mapping = {
    0: "Home",
    1: "Acute rehabilitation facility",
    2: "Long-term care facility",
    3: "Nursing home",
    4: "Other"
}

# Create new column
merged_df["Discharge Location"] = merged_df["surgery_discharge_location"].map(discharge_mapping)

# Summary function
def discharge_summary(df):
    total = len(df)
    counts = df["Discharge Location"].value_counts()
    return {k: f"{v} ({v / total * 100:.1f}%)" for k, v in counts.items()}

# Summarize by group
discharge_total = discharge_summary(merged_df)
discharge_fast = discharge_summary(merged_df[merged_df["cluster"] == 1])
discharge_slow = discharge_summary(merged_df[merged_df["cluster"] == 2])

# Fisher exact test p-value per Discharge Location
# discharge_pvals = {
#     location: fisher_test(merged_df.copy(), "Discharge Location", value=location)
#     for location in discharge_mapping.values()
# }
discharge_pvals = chi2_test(merged_df.copy(), "Discharge Location")
# Print results with p-value
print("-" * 100)
print("{:<35} {:<20} {:<20} {:<20} {:<10}".format("Discharge Location", "Total", "Fast (cluster=1)", "Slow (cluster=2)", "p-value"))
print("-" * 100)
# for location in discharge_mapping.values():
#     t = discharge_total.get(location, "0 (0.0%)")
#     f = discharge_fast.get(location, "0 (0.0%)")
#     s = discharge_slow.get(location, "0 (0.0%)")
#     p = discharge_pvals[location]
#     print("{:<35} {:<20} {:<20} {:<20} {:<10}".format(location, t, f, s, p))
for location in discharge_mapping.values():
    t = discharge_total.get(location, "0 (0.0%)")
    f = discharge_fast.get(location, "0 (0.0%)")
    s = discharge_slow.get(location, "0 (0.0%)")
    p = discharge_pvals
    print("{:<35} {:<20} {:<20} {:<20} {:<10}".format(location, t, f, s, p))

# Map to Yes/No/Unknown
def map_lung_cancer(x):
    if pd.isna(x):
        return "Unknown"
    return "Yes" if x == 1 else "No"

merged_df["Primary Lung Cancer"] = merged_df["primary_lung_cancer"].apply(map_lung_cancer)

# Update summary function
def binary_summary(df, column):
    total = len(df)
    counts = df[column].value_counts()
    yes = counts.get("Yes", 0)
    no = counts.get("No", 0)
    unknown = counts.get("Unknown", 0)
    yes_str = f"{yes} ({yes / total * 100:.1f}%)"
    no_str = f"{no} ({no / total * 100:.1f}%)"
    unknown_str = f"{unknown} ({unknown / total * 100:.1f}%)"
    return yes_str, no_str, unknown_str

# Aggregate statistics
yes_total, no_total, unknown_total = binary_summary(merged_df, "Primary Lung Cancer")
yes_fast, no_fast, unknown_fast = binary_summary(merged_df[merged_df["cluster"] == 1], "Primary Lung Cancer")
yes_slow, no_slow, unknown_slow = binary_summary(merged_df[merged_df["cluster"] == 2], "Primary Lung Cancer")

# Fisher exact test p-value per category
primary_lung_pvals = {
    status: fisher_test(merged_df.copy(), "Primary Lung Cancer", value=status)
    for status in ["Yes", "No", "Unknown"]
}

# Print results with p-value
print("-" * 100)
print("{:<25} {:<20} {:<20} {:<20} {:<10}".format("Primary Lung Cancer", "Total", "Fast (cluster=1)", "Slow (cluster=2)", "p-value"))
print("-" * 100)
print("{:<25} {:<20} {:<20} {:<20} {:<10}".format("Yes", yes_total, yes_fast, yes_slow, primary_lung_pvals["Yes"]))
print("{:<25} {:<20} {:<20} {:<20} {:<10}".format("No", no_total, no_fast, no_slow, primary_lung_pvals["No"]))
print("{:<25} {:<20} {:<20} {:<20} {:<10}".format("Unknown", unknown_total, unknown_fast, unknown_slow, primary_lung_pvals["Unknown"]))



# Map to Yes/No
merged_df["Neoadjuvant Therapy"] = merged_df["neoadj_therapy_x"].map(map_lung_cancer)

# Uses binary_summary defined above
# Aggregate statistics
yes_total, no_total, unknown_total = binary_summary(merged_df, "Neoadjuvant Therapy")
yes_fast, no_fast, unknown_fast = binary_summary(merged_df[merged_df["cluster"] == 1], "Neoadjuvant Therapy")
yes_slow, no_slow, unknown_slow = binary_summary(merged_df[merged_df["cluster"] == 2], "Neoadjuvant Therapy")

# Fisher exact test p-value per Neoadjuvant Therapy category
# neoadj_pvals = {
#     status: fisher_test(merged_df.copy(), "Neoadjuvant Therapy", value=status)
#     for status in ["Yes", "No", "Unknown"]
# }
neoadj_pvals = chi2_test(merged_df.copy(), "Neoadjuvant Therapy")
# Print results with p-value
print("-" * 130)
print("{:<25} {:<20} {:<20} {:<20} {:<10}".format("Neoadjuvant Therapy", "Total", "Fast (cluster=1)", "Slow (cluster=2)", "p-value"))
print("-" * 130)
# print("{:<25} {:<20} {:<20} {:<20} {:<10}".format("Yes", yes_total, yes_fast, yes_slow, neoadj_pvals["Yes"]))
# print("{:<25} {:<20} {:<20} {:<20} {:<10}".format("No", no_total, no_fast, no_slow, neoadj_pvals["No"]))
# print("{:<25} {:<20} {:<20} {:<20} {:<10}".format("Unknown", unknown_total, unknown_fast, unknown_slow, neoadj_pvals["Unknown"]))
print("{:<25} {:<20} {:<20} {:<20} {:<10}".format("Yes", yes_total, yes_fast, yes_slow, neoadj_pvals))
print("{:<25} {:<20} {:<20} {:<20} {:<10}".format("No", no_total, no_fast, no_slow, neoadj_pvals))
print("{:<25} {:<20} {:<20} {:<20} {:<10}".format("Unknown", unknown_total, unknown_fast, unknown_slow, neoadj_pvals))


# Map to Yes/No
merged_df["Adjuvant Therapy"] = merged_df["adj_therapy_x"].map(map_lung_cancer)

# Summary
yes_total, no_total, unknown_total = binary_summary(merged_df, "Adjuvant Therapy")
yes_fast, no_fast, unknown_fast = binary_summary(merged_df[merged_df["cluster"] == 1], "Adjuvant Therapy")
yes_slow, no_slow, unknown_slow = binary_summary(merged_df[merged_df["cluster"] == 2], "Adjuvant Therapy")

# Fisher exact test p-value per Adjuvant Therapy category
# adjuvant_pvals = {
#     status: fisher_test(merged_df.copy(), "Adjuvant Therapy", value=status)
#     for status in ["Yes", "No", "Unknown"]
# }
adjuvant_pvals = chi2_test(merged_df.copy(), "Adjuvant Therapy")
# Print results with p-value
print("-" * 100)
print("{:<25} {:<20} {:<20} {:<20} {:<10}".format("Adjuvant Therapy", "Total", "Fast (cluster=1)", "Slow (cluster=2)", "p-value"))
print("-" * 100)
# print("{:<25} {:<20} {:<20} {:<20} {:<10}".format("Yes", yes_total, yes_fast, yes_slow, adjuvant_pvals["Yes"]))
# print("{:<25} {:<20} {:<20} {:<20} {:<10}".format("No", no_total, no_fast, no_slow, adjuvant_pvals["No"]))
# print("{:<25} {:<20} {:<20} {:<20} {:<10}".format("Unknown", unknown_total, unknown_fast, unknown_slow, adjuvant_pvals["Unknown"]))
print("{:<25} {:<20} {:<20} {:<20} {:<10}".format("Yes", yes_total, yes_fast, yes_slow, adjuvant_pvals))
print("{:<25} {:<20} {:<20} {:<20} {:<10}".format("No", no_total, no_fast, no_slow, adjuvant_pvals))
print("{:<25} {:<20} {:<20} {:<20} {:<10}".format("Unknown", unknown_total, unknown_fast, unknown_slow, adjuvant_pvals))



# Table 2 - primary_lung_cancer = Yes
# Keep only rows with primary_lung_cancer == 1
merged_df = merged_df[merged_df["primary_lung_cancer"] == 1]
print('Table 2: ', merged_df.shape)
# Tumor Location mapping
tumor_location_map = {
    1: "Right upper lobe",
    2: "Right middle lobe",
    3: "Right lower lobe",
    4: "Left upper lobe",
    5: "Left lower lobe"
}
merged_df["Tumor Location"] = merged_df["tumor_location_x"].map(tumor_location_map)

# Summary function
def tumor_location_summary(df):
    total = len(df)
    counts = df["Tumor Location"].value_counts()
    return {k: f"{v} ({v / total * 100:.1f}%)" for k, v in counts.items()}

# Summarize by group
tumor_total = tumor_location_summary(merged_df)
tumor_fast = tumor_location_summary(merged_df[merged_df["cluster"] == 1])
tumor_slow = tumor_location_summary(merged_df[merged_df["cluster"] == 2])

# Fisher exact test p-value
# tumor_pvals = {
#     loc: fisher_test(merged_df.copy(), "Tumor Location", value=loc)
#     for loc in tumor_location_map.values()
# }
tumor_pvals = chi2_test(merged_df.copy(), "Tumor Location")
# Print results with p-value
print("-" * 100)
print("{:<30} {:<20} {:<20} {:<20} {:<10}".format("Tumor Location", "Total", "Fast (cluster=1)", "Slow (cluster=2)", "p-value"))
print("-" * 100)
# for loc in tumor_location_map.values():
#     t = tumor_total.get(loc, "0 (0.0%)")
#     f = tumor_fast.get(loc, "0 (0.0%)")
#     s = tumor_slow.get(loc, "0 (0.0%)")
#     p = tumor_pvals[loc]
#     print("{:<30} {:<20} {:<20} {:<20} {:<10}".format(loc, t, f, s, p))
for loc in tumor_location_map.values():
    t = tumor_total.get(loc, "0 (0.0%)")
    f = tumor_fast.get(loc, "0 (0.0%)")
    s = tumor_slow.get(loc, "0 (0.0%)")
    p = tumor_pvals
    print("{:<30} {:<20} {:<20} {:<20} {:<10}".format(loc, t, f, s, p))

# Tumor Size summary: median (IQR)
def tumor_size_summary(df):
    q1 = df["tumor_size_x"].quantile(0.25)
    q2 = df["tumor_size_x"].median()
    q3 = df["tumor_size_x"].quantile(0.75)
    return f"{q2:.1f} ({q1:.1f}, {q3:.1f})"

# Print (commented out)
# print("\n{:<30} {:<20} {:<20} {:<20}".format("Tumor Size (cm)", "Total", "Fast (cluster=1)", "Slow (cluster=2)"))
# print("-" * 95)

tumor_total = tumor_size_summary(merged_df)
tumor_fast = tumor_size_summary(merged_df[merged_df["cluster"] == 1])
tumor_slow = tumor_size_summary(merged_df[merged_df["cluster"] == 2])

# Rank-sum test p-value for Tumor Size
tumor_size_pval = ranksum_test(merged_df, "tumor_size_x")

# Print results with p-value
print("-" * 100)
print("{:<30} {:<20} {:<20} {:<20} {:<10}".format("Tumor Size (cm)", "Total", "Fast (cluster=1)", "Slow (cluster=2)", "p-value"))
print("-" * 100)
print("{:<30} {:<20} {:<20} {:<20} {:<10}".format("Tumor Size", tumor_total, tumor_fast, tumor_slow, tumor_size_pval))


# Update mapping
t_stage_map = {
    0: "T0",
    1: "Tmi",
    2: "T1a",
    3: "T1b",
    4: "T1c",
    5: "T2a",
    6: "T2b",
    7: "T3",
    8: "T4"
}

# Apply mapping
merged_df["Clinical T Status"] = merged_df["clin_t_stage"].map(t_stage_map).fillna("Unknown")

# Summary function
def t_stage_summary(df):
    counts = df["Clinical T Status"].value_counts()
    total = len(df)
    result = {}
    for t in ["T0", "Tmi", "T1a", "T1b", "T1c", "T2a", "T2b", "T3", "T4", "Unknown"]:
        count = counts.get(t, 0)
        result[t] = f"{count} ({count / total * 100:.1f}%)"
    return result


# Compute statistics
t_total = t_stage_summary(merged_df)
t_fast = t_stage_summary(merged_df[merged_df["cluster"] == 1])
t_slow = t_stage_summary(merged_df[merged_df["cluster"] == 2])

# Fisher exact test p-value per Clinical T Status
# t_stage_pvals = {
#     t: fisher_test(merged_df.copy(), "Clinical T Status", value=t)
#     for t in ["T0", "Tmi", "T1a", "T1b", "T1c", "T2a", "T2b", "T3", "T4", "Unknown"]
# }
t_stage_pvals = chi2_test(merged_df.copy(), "Clinical T Status")
# Print results with p-value
print("-" * 100)
print("{:<20} {:<20} {:<20} {:<20} {:<10}".format("Clinical T Status", "Total", "Fast (cluster=1)", "Slow (cluster=2)", "p-value"))
print("-" * 100)
# for t in ["T0", "Tmi", "T1a", "T1b", "T1c", "T2a", "T2b", "T3", "T4", "Unknown"]:
#     print("{:<20} {:<20} {:<20} {:<20} {:<10}".format(
#         t,
#         t_total[t],
#         t_fast[t],
#         t_slow[t],
#         t_stage_pvals[t]
#     ))
for t in ["T0", "Tmi", "T1a", "T1b", "T1c", "T2a", "T2b", "T3", "T4", "Unknown"]:
    print("{:<20} {:<20} {:<20} {:<20} {:<10}".format(
        t,
        t_total[t],
        t_fast[t],
        t_slow[t],
        t_stage_pvals
    ))

# Mapping
n_stage_map = {
    0: "N0", 1: "N1", 2: "N2", 3: "N3"
}

# Apply mapping
merged_df["Clinical N Status"] = merged_df["clin_n_stage"].map(n_stage_map).fillna("Unknown")

# Summary function
def n_stage_summary(df):
    counts = df["Clinical N Status"].value_counts()
    total = len(df)
    result = {}
    for n in ["N0", "N1", "N2", "N3", "Unknown"]:
        count = counts.get(n, 0)
        result[n] = f"{count} ({count / total * 100:.1f}%)"
    return result

# Compute statistics
n_total = n_stage_summary(merged_df)
n_fast = n_stage_summary(merged_df[merged_df["cluster"] == 1])
n_slow = n_stage_summary(merged_df[merged_df["cluster"] == 2])

# Fisher exact test p-value per Clinical N Status
# n_stage_pvals = {
#     n: fisher_test(merged_df.copy(), "Clinical N Status", value=n)
#     for n in ["N0", "N1", "N2", "N3", "Unknown"]
# }
n_stage_pvals = chi2_test(merged_df.copy(), "Clinical N Status")
# Print results with p-value
print("-" * 100)
print("{:<20} {:<20} {:<20} {:<20} {:<10}".format("Clinical N Status", "Total", "Fast (cluster=1)", "Slow (cluster=2)", "p-value"))
print("-" * 100)
# for n in ["N0", "N1", "N2", "N3", "Unknown"]:
#     print("{:<20} {:<20} {:<20} {:<20} {:<10}".format(
#         n,
#         n_total[n],
#         n_fast[n],
#         n_slow[n],
#         n_stage_pvals[n]
#     ))
for n in ["N0", "N1", "N2", "N3", "Unknown"]:
    print("{:<20} {:<20} {:<20} {:<20} {:<10}".format(
        n,
        n_total[n],
        n_fast[n],
        n_slow[n],
        n_stage_pvals
    ))

# Mapping
m_stage_map = {
    0: "M0", 1: "M1a", 2: "M1b", 3: "M1c"
}

# Apply mapping
merged_df["Clinical M Status"] = merged_df["clin_m_stage"].map(m_stage_map).fillna("Unknown")

# Summary function
def m_stage_summary(df):
    counts = df["Clinical M Status"].value_counts()
    total = len(df)
    result = {}
    for m in ["M0", "M1a", "M1b", "M1c", "Unknown"]:
        count = counts.get(m, 0)
        result[m] = f"{count} ({count / total * 100:.1f}%)"
    return result

# Compute statistics
m_total = m_stage_summary(merged_df)
m_fast = m_stage_summary(merged_df[merged_df["cluster"] == 1])
m_slow = m_stage_summary(merged_df[merged_df["cluster"] == 2])

# Fisher exact test p-value per Clinical M Status
# m_stage_pvals = {
#     m: fisher_test(merged_df.copy(), "Clinical M Status", value=m)
#     for m in ["M0", "M1a", "M1b", "M1c", "Unknown"]
# }
m_stage_pvals = chi2_test(merged_df.copy(), "Clinical M Status")
# Print results with p-value
print("-" * 100)
print("{:<20} {:<20} {:<20} {:<20} {:<10}".format("Clinical M Status", "Total", "Fast (cluster=1)", "Slow (cluster=2)", "p-value"))
print("-" * 100)
# for m in ["M0", "M1a", "M1b", "M1c", "Unknown"]:
#     print("{:<20} {:<20} {:<20} {:<20} {:<10}".format(
#         m,
#         m_total[m],
#         m_fast[m],
#         m_slow[m],
#         m_stage_pvals[m]
#     ))
for m in ["M0", "M1a", "M1b", "M1c", "Unknown"]:
    print("{:<20} {:<20} {:<20} {:<20} {:<10}".format(
        m,
        m_total[m],
        m_fast[m],
        m_slow[m],
        m_stage_pvals
    ))


# Mapping
path_t_stage_map = {
    0: "T0",
    1: "Tmi",
    2: "T1a",
    3: "T1b",
    4: "T1c",
    5: "T2a",
    6: "T2b",
    7: "T3",
    8: "T4"
}

# Apply mapping
merged_df["Pathologic T Status"] = merged_df["path_t_stage"].map(path_t_stage_map).fillna("Unknown")

# Summary function
def path_t_stage_summary(df):
    counts = df["Pathologic T Status"].value_counts()
    total = len(df)
    result = {}
    for t in ["T0", "Tmi", "T1a", "T1b", "T1c", "T2a", "T2b", "T3", "T4", "Unknown"]:
        count = counts.get(t, 0)
        result[t] = f"{count} ({count / total * 100:.1f}%)"
    return result

# Get summary results
path_t_total = path_t_stage_summary(merged_df)
path_t_fast = path_t_stage_summary(merged_df[merged_df["cluster"] == 1])
path_t_slow = path_t_stage_summary(merged_df[merged_df["cluster"] == 2])

# Fisher exact test p-value per Pathologic T Status
# path_t_pvals = {
#     t: fisher_test(merged_df.copy(), "Pathologic T Status", value=t)
#     for t in ["T0", "Tmi", "T1a", "T1b", "T1c", "T2a", "T2b", "T3", "T4", "Unknown"]
# }
path_t_pvals = chi2_test(merged_df.copy(), "Pathologic T Status")
# Print results with p-value
print("-" * 100)
print("{:<25} {:<20} {:<20} {:<20} {:<10}".format("Pathologic T Status", "Total", "Fast (cluster=1)", "Slow (cluster=2)", "p-value"))
print("-" * 100)
# for t in ["T0", "Tmi", "T1a", "T1b", "T1c", "T2a", "T2b", "T3", "T4", "Unknown"]:
#     print("{:<25} {:<20} {:<20} {:<20} {:<10}".format(
#         t,
#         path_t_total[t],
#         path_t_fast[t],
#         path_t_slow[t],
#         path_t_pvals[t]
#     ))
for t in ["T0", "Tmi", "T1a", "T1b", "T1c", "T2a", "T2b", "T3", "T4", "Unknown"]:
    print("{:<25} {:<20} {:<20} {:<20} {:<10}".format(
        t,
        path_t_total[t],
        path_t_fast[t],
        path_t_slow[t],
        path_t_pvals
    ))

# Mapping
path_n_stage_map = {
    0: "N0",
    1: "N1",
    2: "N2",
    3: "N3"
}

# Apply mapping
merged_df["Pathologic N Status"] = merged_df["path_n_stage"].map(path_n_stage_map).fillna("Unknown")

# Summary function
def path_n_stage_summary(df):
    counts = df["Pathologic N Status"].value_counts()
    total = len(df)
    result = {}
    for n in ["N0", "N1", "N2", "N3", "Unknown"]:
        count = counts.get(n, 0)
        result[n] = f"{count} ({count / total * 100:.1f}%)"
    return result

# Aggregate statistics
path_n_total = path_n_stage_summary(merged_df)
path_n_fast = path_n_stage_summary(merged_df[merged_df["cluster"] == 1])
path_n_slow = path_n_stage_summary(merged_df[merged_df["cluster"] == 2])

# Fisher exact test p-value per Pathologic N Status
# path_n_pvals = {
#     n: fisher_test(merged_df.copy(), "Pathologic N Status", value=n)
#     for n in ["N0", "N1", "N2", "N3", "Unknown"]
# }
path_n_pvals = chi2_test(merged_df.copy(), "Pathologic N Status")
# Print results with p-value
print("-" * 100)
print("{:<25} {:<20} {:<20} {:<20} {:<10}".format("Pathologic N Status", "Total", "Fast (cluster=1)", "Slow (cluster=2)", "p-value"))
print("-" * 100)
# for n in ["N0", "N1", "N2", "N3", "Unknown"]:
#     print("{:<25} {:<20} {:<20} {:<20} {:<10}".format(
#         n,
#         path_n_total[n],
#         path_n_fast[n],
#         path_n_slow[n],
#         path_n_pvals[n]
#     ))
for n in ["N0", "N1", "N2", "N3", "Unknown"]:
    print("{:<25} {:<20} {:<20} {:<20} {:<10}".format(
        n,
        path_n_total[n],
        path_n_fast[n],
        path_n_slow[n],
        path_n_pvals
    ))

# Mapping
path_m_stage_map = {
    0: "M0",
    1: "M1a",
    2: "M1b",
    3: "M1c"
}

# Apply mapping
merged_df["Pathologic M Status"] = merged_df["path_m_stage"].map(path_m_stage_map).fillna("Unknown")

# Summary function
def path_m_stage_summary(df):
    counts = df["Pathologic M Status"].value_counts()
    total = len(df)
    result = {}
    for m in ["M0", "M1a", "M1b", "M1c", "Unknown"]:
        count = counts.get(m, 0)
        result[m] = f"{count} ({count / total * 100:.1f}%)"
    return result

# Aggregate statistics
path_m_total = path_m_stage_summary(merged_df)
path_m_fast = path_m_stage_summary(merged_df[merged_df["cluster"] == 1])
path_m_slow = path_m_stage_summary(merged_df[merged_df["cluster"] == 2])

# Fisher exact test p-value per Pathologic M Status
path_m_pvals = {
    m: fisher_test(merged_df.copy(), "Pathologic M Status", value=m)
    for m in ["M0", "M1a", "M1b", "M1c", "Unknown"]
}

# Print results with p-value
print("-" * 100)
print("{:<25} {:<20} {:<20} {:<20} {:<10}".format("Pathologic M Status", "Total", "Fast (cluster=1)", "Slow (cluster=2)", "p-value"))
print("-" * 100)
for m in ["M0", "M1a", "M1b", "M1c", "Unknown"]:
    print("{:<25} {:<20} {:<20} {:<20} {:<10}".format(
        m,
        path_m_total[m],
        path_m_fast[m],
        path_m_slow[m],
        path_m_pvals[m]
    ))

