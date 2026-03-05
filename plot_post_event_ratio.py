import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import fisher_exact

# Prepare data
data = pd.DataFrame({
    "Group": ["Group 1"] * 56 + ["Group 2"] * 57,
    "Event": [1]*4 + [0]*52 + [1]*6 + [0]*51
})

# Compute significance
p_value = fisher_exact([[4, 52], [6, 51]])[1]
print("p-value:", p_value)

# Set style
# sns.set(style="whitegrid")

# Plot
fig, ax = plt.subplots(figsize=(4.5, 5))
palette = {"Group 1": "#D55E00", "Group 2": "#0072B2"}
barplot = sns.barplot(data=data, x="Group", y="Event", ci=None,
                      estimator=lambda x: sum(x)/len(x), palette=palette, ax=ax)

# Adjust bar width and border
desired_width = 0.3
for patch in ax.patches:
    current_width = patch.get_width()
    diff = current_width - desired_width
    patch.set_width(desired_width)
    patch.set_x(patch.get_x() + diff / 2)
    patch.set_edgecolor("black")
    patch.set_linewidth(2)

# Y-axis settings
ax.set_ylim(0, 0.4)
ax.set_ylabel("Complication\n (POD 8-30)", fontsize=30)
ax.set_xlabel("Group", fontsize=30)
ax.tick_params(axis='y', labelsize=30, width=2, length=6)
ax.tick_params(axis='x', labelsize=0, width=2, length=6)  # Hide labels

# Significance annotation
group_means = data.groupby("Group")["Event"].mean().values
y_max = max(group_means)
ax.hlines(y_max + 0.065, 0, 1, color='black', linewidth=2)
ax.text(0.5, y_max + 0.07, 'NS', ha='center', va='bottom', fontsize=28)

# Add value labels
for i, patch in enumerate(ax.patches):
    height = patch.get_height()
    ax.text(patch.get_x() + patch.get_width()/2, height + 0.01,
            f"{height:.2f}", ha='center', va='bottom', fontsize=30)

# Spines/borders
for spine in ax.spines.values():
    spine.set_linewidth(2)

# Save
plt.tight_layout()
plt.savefig("post_event_sns_custom_barwidth.png", dpi=300)
# plt.show()
