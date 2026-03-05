import pandas as pd
import matplotlib.pyplot as plt

# fatigue top 50%
# df = pd.DataFrame({
#     "Feature": [
#         "age", "pft_dlco", "pft_fev1", "bmi", "cluster_Poor Sleep",
#         "sex_male", "surgery_type_Lobectomy", "surgery_type_Extended Lobectomy",
#         "patient_race_White", "approach_type_Open Surgery"
#     ],
#     "OR": [1.031, 1.004, 0.993, 1.020, 4.638, 0.531, 2.787, 6.383e9, 0.315, 0.188],
#     "CI_lower": [0.977, 0.990, 0.967, 0.921, 1.413, 0.172, 0.845, 0.000, 0.014, 0.012],
#     "CI_upper": [1.089, 1.019, 1.019, 1.131, 15.220, 1.632, 9.191, float("inf"), 7.001, 3.058],
#     "P": [0.257, 0.552, 0.580, 0.704, 0.011, 0.269, 0.092, 1.000, 0.465, 0.240]
# })

# general health top 25%
# df = pd.DataFrame({
#     "Feature": [
#         "age", "pft_dlco", "pft_fev1", "bmi", "cluster_Poor Sleep",
#         "sex_male", "surgery_type_Lobectomy", "surgery_type_Extended Lobectomy",
#         "patient_race_White", "approach_type_Open Surgery"
#     ],
#     "OR": [0.999873, 0.982908, 1.016703, 0.987305, 7.746638,0.768508, 2.640364, 1.169453e-14, 0.191880, 0.416875
#     ],
#     "CI_lower": [0.934348, 0.950084, 0.984680, 0.876132, 1.733671,0.208047, 0.708294, 0.000000, 0.008925, 0.025826
#     ],
#     "CI_upper": [1.069993, 1.017054, 1.049767, 1.112575, 34.615038,2.838809, 9.842317, float("inf"), 4.125325, 6.728973
#     ],
#     "P": [0.997064, 0.323461, 0.310353, 0.833902, 0.007353,0.692880, 0.148104, 0.999997, 0.291580, 0.537515
#     ]
# })

# general health top 25%  New 116 patients
#df = pd.DataFrame({
#    "Feature": [
#        "age", "pft_dlco", "pft_fev1", "bmi", "cluster_Poor Sleep",
#        "sex_male", "surgery_type_Lobectomy",
#        "patient_race_White", "approach_type_Open Surgery", "smoking_status Ever",
#    ],
#    "OR": [1.014166, 0.976252, 1.000615, 1.000826, 5.306612, 0.883116, 2.175880, 0.654746, 2.458035, 0.326580
#    ],
#    "CI_lower": [0.953574, 0.938332, 0.968418, 0.884412, 1.465716,0.253351, 0.661160, 0.083815, 0.284825, 0.080075
#    ],
#    "CI_upper": [1.078608, 1.015704, 1.033882, 1.132564, 19.212541,3.078314, 7.160826, 5.114754, 21.212764, 1.331936
#    ],
#    "P": [0.654484, 0.234408, 0.970613, 0.989554, 0.011008,0.845310, 0.200828, 0.686357, 0.413420, 0.118679
#    ]
#})

# general health <= 25%  New 113 patients
df = pd.DataFrame({
     "Feature": [
         "cluster_Poor Sleep", "surgery_type_Lobectomy", "approach_type_Open Surgery", "sex_male", "smoking_status Ever", "patient_race_White","age", "bmi", "pft_dlco", "pft_fev1", "SF-36 Baseline"
     ],
     "OR": [
         6.44, 1.78, 2.32, 1.08, 0.41, 0.83, 1.00, 1.01, 0.97, 1.00, 1.03
     ],
     "CI_lower": [
         1.67, 0.51, 0.22, 0.30, 0.09, 0.10, 0.94, 0.89, 0.93, 0.97, 0.99
     ],
     "CI_upper": [
         24.81, 6.15, 24.58, 3.95, 1.84, 6.82, 1.07, 1.15, 1.02, 1.04, 1.07
     ],
     "P": [
         0.007, 0.364, 0.483, 0.906, 0.243, 0.861, 0.929, 0.890, 0.223, 0.818, 0.120
     ]
})

# PCS 30 top 50%
# df = pd.DataFrame({
#     "Feature": [
#         "age", "pft_dlco", "pft_fev1", "bmi", "cluster_Poor Sleep",
#         "sex_male", "surgery_type_Lobectomy", "surgery_type_Extended Lobectomy",
#         "patient_race_White", "approach_type_Open Surgery"
#     ],
#     "OR": [
#         1.006751, 0.996887, 0.985952, 1.001448, 2.870023,1.032536, 2.528137, 1.426668, 1.109128, 0.151833
#     ],
#     "CI_lower": [
#         0.960763, 0.986730, 0.961053, 0.916880, 1.037308,0.356391, 0.892467, 0.052227, 0.100185, 0.012145
#     ],
#     "CI_upper": [
#         1.054960, 1.006945, 1.011495, 1.094065, 7.940780,2.991464, 7.161586, 38.971536, 12.278978, 1.898133
#     ],
#     "P": [
#         0.777964, 0.542690, 0.278297, 0.974415, 0.042301,0.952957, 0.080837, 0.833220, 0.932712, 0.143550
#     ]
# })

reference_groups = {
    "sex_male": "Female",
    "patient_race_White": "Other",
    "cluster_Poor Sleep": "Fast Sleep Recovery Group",
    "surgery_type_Lobectomy": "Sublobar Resection",
    "smoking_status Ever": "Never Smoke",
    # "approach_type_Minimally Invasive Surgery": "Open Surgery",
    "approach_type_Open Surgery": "Minimally Invasive Surgery"
}


feature_name_map = {
    "age": "Age",
    "pft_dlco": "DLCO",
    "pft_fev1": "FEV1",
    "bmi": "BMI",
    "cluster_Poor Sleep": "Slow Sleep Recovery Group",
    "sex_male": "Male",
    "surgery_type_Lobectomy": "Lobectomy",
    "surgery_type_Extended Lobectomy": "Extended Lobectomy",
    "patient_race_White": "White",
    "smoking_status Ever": "Ever Smoke",
    # "approach_type_Minimally Invasive Surgery": "Minimally Invasive Surgery",
    "approach_type_Open Surgery": "Open Surgery",
    "SF-36 Baseline": "SF-36 Baseline"
}

df["Feature"] = df["Feature"].apply(lambda x:
    f"{feature_name_map[x]} \n(Reference Group: {reference_groups[x]})" if x in reference_groups else feature_name_map[x]
)

include_inf_ci = False

if not include_inf_ci:
    df = df[df["CI_upper"] != float("inf")]


def mark_p(p):
    if p < 0.001:
        return '(***)'
    elif p < 0.01:
        return '(**)'
    elif p < 0.05:
        return '(*)'
    else:
        return '(NS)'
df["Significance"] = df["P"].apply(mark_p)
# df["color"] = df["OR"].apply(lambda x: "blue" if x > 1 else "red")
df["color"] = df["Feature"].apply(
    lambda x: "blue" if "Slow Sleep Recovery Group" in x else "black"
)

df = df.sort_values("OR")
df = df.reset_index(drop=True)

fig, ax = plt.subplots(figsize=(20, 7), dpi=300)

for i, row in df.iterrows():
    err_low = row['OR'] - row['CI_lower']
    err_high = row['CI_upper'] - row['OR'] if row['CI_upper'] != float("inf") else row['OR'] * 2
    linewidth = 3 if "Slow Sleep Recovery Group" in row["Feature"] else 1

    ax.errorbar(row["OR"], i, xerr=[[err_low], [err_high]],
                fmt='o', color='black', ecolor=row['color'],
                elinewidth=linewidth, capsize=4, alpha=0.7)

    is_significant = row["P"] < 0.05

    ax.text(row["OR"], i + 0.2, f"{row['OR']:.2f} {row['Significance']}",
            va='bottom',
            ha='left' if row["OR"] > 1 else 'right',
            fontsize=16,
            fontweight='bold' if is_significant else 'normal',
            color=row['color'],
            alpha=0.7)

yticks = []
for i, feature in enumerate(df["Feature"]):
    if "Slow Sleep Recovery Group" in feature:
        t = ax.text(0.005, i, feature, va='center', ha='right', fontsize=15, color='blue', fontweight='bold')
    else:
        t = ax.text(0.005, i, feature, va='center', ha='right', fontsize=15, color='black')
    yticks.append(t)

ax.set_yticks(range(len(df)))
ax.set_yticklabels([''] * len(df))


ax.axvline(x=1, color='black', linestyle='--', linewidth=2, alpha=0.7)
ax.set_xscale('log')
ax.set_xlabel("Odds Ratio (log scale)", fontsize=15)
ax.set_ylim(-1, len(df))

plt.tight_layout()
plt.savefig("plot_forester_multivariate.png")
# plt.show()
