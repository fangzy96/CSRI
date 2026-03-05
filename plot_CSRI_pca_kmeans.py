import os

# Workaround for threadpoolctl bug on macOS: get_config() returns None, causing AttributeError
import threadpoolctl
from contextlib import contextmanager

_orig_threadpool_info = threadpoolctl.threadpool_info
def _safe_threadpool_info(*args, **kwargs):
    try:
        return _orig_threadpool_info(*args, **kwargs)
    except AttributeError:
        return []
threadpoolctl.threadpool_info = _safe_threadpool_info

@contextmanager
def _noop_threadpool_limits(*args, **kwargs):
    yield
threadpoolctl.threadpool_limits = _noop_threadpool_limits

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from scipy.cluster.hierarchy import fcluster
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
# from tslearn.metrics import cdist_dtw  # Unused; causes threadpoolctl error on some macOS setups
from scipy.cluster.hierarchy import linkage, fcluster
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from scipy.stats import ranksums
#import umap
from mpl_toolkits.mplot3d import Axes3D

current_uuid_list = [
    "219cdf52-d51c-4aa0-adec-19904b95416e",
    "8843df7a-cd56-4b8d-94e0-b11c48104a23",
    "5baea122-2171-408c-931e-7dc41a8dd74e",
    "571fb276-2a39-49f5-8eeb-e1d2484e4a0b",
    "fc4500d4-1e52-41e1-b833-9ecfcb644a17",
    "4b5a8edd-544d-494a-a263-e7fbf6de1779",
    "282bc739-8e12-44f9-948d-bbc0e1dc4065",
    "46e75855-68d0-436c-9191-5f085da84374",
    "57df011e-d905-4c8f-876c-71ef1cf7643e",
    "f49f2a1b-524c-43b1-9cc0-4f3b72d55c95",
    "adb22ab4-e76c-439a-87b7-e55885fa91a9",
    "ecad863a-057c-4718-84b8-5d51a05383c5",
    "241b220c-b864-44fa-bb35-dbedd63ed545",
    "57b0d9cc-6fd8-44f2-819e-e7c62abb977d",
    "a480b97a-c778-4c26-8b89-b1f28a2ac5dc",
    "ead74f57-c022-4c79-aee7-cab87314c1f4",
    "d9c1ec3c-7e50-422c-b928-cd9f1a60423f",
    "cbfed99f-e726-45ef-97ef-929397cf69aa",
    "d8dd4ddc-e94d-4cde-a4e0-ad28ae7794fc",
    "359f0dbd-da44-4e27-b3e4-26da27152beb",
    "1146a2d1-91a1-441e-b30d-249595f37b46",
    "9c36749f-0a90-4601-a194-c573f081feb6",
    "fe45805c-c774-48fa-8b42-90ed990721d5",
    "83cde823-0634-4aae-946c-766878eb0b70",
    "63c974f9-78fb-486f-a176-c7c2faa9e397",
    "5233d6b7-1f99-4429-87b0-bd293174428c",
    "1caad2ae-712d-4399-8031-f119cbe19bd5",
    "e388fe16-2eeb-4c71-848a-4aa71530c21e",
    "8b16d1e6-3b40-4073-9287-81235c71b84e",
    # "3ecaf428-2a9c-4276-8a4c-48587ec796e2",
    "b8661332-ffc0-4838-865a-2f69f8e8061e",
    "29c04598-79df-428a-bab2-db4550af2e6a",
    "145f6b43-eb77-40f0-8816-ae4d0ce10e24",
    "403aeded-a19c-40da-bec8-1e9791b1bcc9",
    "a8e8553d-d3c4-4b81-824f-d06b8c3eca8d",
    "5034a984-4259-4e6e-815b-f57eb7bdd12b",
    "edb102cd-9724-4282-baf4-8992c0124a51",
    "7f243e5b-282a-4189-bab4-da945085a349",
    "d6563dbc-26bc-46e0-b07e-0fde59f25ff8",
    "10087e5b-6d95-416e-8307-caad0903510f",
    "2be2ad89-f47a-4ac6-9c6b-e5edee7c7eb8",
    "3022ab1e-3b8c-4357-98e8-438200b8a56b",
    "90f75c2c-54bb-450e-9891-293e0c45c21e",
    "f1994796-b849-4dc5-93f6-abefae7da512",
    "39713303-8d70-4e7d-bd9a-9bb8ccfb677d",
    "8bdaa521-95f4-40ff-b026-abd07dff3325",
    "846f52e2-f4a6-4f01-bb14-ed06860e350a",
    "ba279bb2-15f6-4747-9835-c0891a0e3f71",
    "91117ef7-528e-41d2-afdd-b4ae57a6a72c",
    "6766d63b-247f-417e-8361-8b69b3ee5e20",
    "5cf2e17c-e1d4-4e0c-b691-7856b0932d5e",
    "aafdd8e5-6db6-499a-88ac-5165faca2a57",
    "c69efd72-58d0-4f54-b235-ca6f57babc62",
    "90419681-ce22-400d-8d0b-38d2d114b99a",
    "3ba16b96-7c3b-45c8-ba00-d29e4e11de82",
    "d2e9b7fc-7dd9-4823-95b5-7a366d5fc012",
    "78f632e5-5380-4972-8be2-89a8ba7a64b1",
    "844f71b1-5a13-4b6d-a07d-5a7e2b9a66b7",
    "cfc3df55-336b-4cfd-a5db-3fc28189dc6c",
    "74dc6f71-8e76-4fde-8b14-647152c1eb64",
    "c381db98-1fed-415a-b71e-0e8f2631e39d",
    "6a39016f-b942-405d-b864-080a7f199f81",
    "6aaa9cdb-a79d-49bc-abea-650a2aa9720e",
    "7c795322-9137-40a4-b7bb-d659dd9a2ee7",
    # "092589fe-7441-4047-8f41-29652c8f2412",
    "9566b8c3-d62c-40a5-82a6-92ff56554a7c",
    "6bea1f57-9846-4f6c-a3bd-2b3fff2c324f",
    "b7e99f1a-4598-467f-87ac-0428423d7ddb",
    "724564ac-2b80-46cf-8bef-be932b598c28",
    "b338240d-0fc8-485d-886e-b3b55d7d738a",
    "b4cc2a92-229e-40ee-bf15-0fca3909d10b",
    "9a14da5e-7a0e-4e36-a25a-0658eba03df9",
    "a646360d-e122-45c2-848d-5ec5cd0589ca",
    "bc0a9a2d-6cf8-4e0f-9177-e57fa0af28e1",
    "6cb0708c-556b-446a-b92d-8c9214c3e35d",
    "96a1d3cc-c765-4034-9e98-d024d6f94d0b",
    "e7ff0f3b-14b0-491f-96a3-26bec962d031",
    "16a2227e-749f-4699-8279-9c29af77bc2b",
    "ff734900-1fd2-4d8f-ba78-4f11e818e39a",
    "9d7f35c3-94a9-46f8-bdc4-304e35a3dc5b",
    "3eb5c512-957d-47e9-a26c-58f916374bda",
    "80fa00c6-d39c-4b7e-a706-ba863c4feace",
    "42c35247-7fa6-4dbc-a675-b9d006f53923",
    "1b40c675-d667-48c5-978c-d80e425d3d11",
    "33fe3960-3bc3-4949-8216-8bfde1519ab3",
    "728dc1a3-741f-44d6-a3eb-c60e522e2b46",
    "6ec0aa72-2eee-4ec5-b26a-0b8b5dbd6a72",
    "5fa143f0-b299-4190-9e0b-858d2b1a43f4",
    "414567c3-aac9-4019-88a0-bca286fee882",
    "c59db9a2-acd0-4ac8-b1cf-3b5e3b8234f6",
    "cf2acc3e-5811-4ba7-b9de-50d161584656",
    "6e5054b9-0be8-4d66-ad2a-204da65463d8",
    # "bc57b4bc-421f-4cd9-8c27-3ce6a61d88a1",
    "273b88b6-c1c4-4eed-9445-c65378c549d0",
    "18e9c520-011a-4c72-83cb-fd14d8b0ba3b",
    "748c7b6f-06d8-403b-a0ac-178a2ad5c109",
    "0c570ad7-4f0f-4c1b-a76d-3b6b57103730",
    "9ee5db36-f6f8-48a2-bca1-6883e00e5f93",
    "aed992e8-d6b6-4386-8c67-cbef0da67bcf",
    "77691dff-0c95-40e2-afa3-fcb3b736c09c",
    "e5629436-d6d4-4a6c-bf13-e144688e3122",
    "e038012c-b6fc-4f6c-af9e-b4554eb3869d",
    "dd34c6e1-bb9f-4bd4-85a9-7e31f726bddc",
    "8f2ce515-6d1c-4aee-82e6-74445b338324",
    "7efc7912-b0ba-4ab4-9bb3-15ee1e5ed373",
    "64f4b992-9ecb-4e44-b056-ec136a22e731",
    "9ae36450-7d8a-499e-bb54-9776d50c6d02",
    "735f0dd3-5dc3-4231-a8ee-c4531210bbe9",
    "6e8138c6-56cb-41eb-b233-da84aaadb891",
    "ce694416-de79-4b8c-97bb-9de5c9da7edb",
    "b22964cb-a2a8-4bd9-8365-77f98de11c7e",
    "2b43ec6e-1ed2-4081-9b00-03b403b81071",
    "cc2aed84-d7a5-4751-9503-b5f0928a55ad",
    "9b1b0dff-e0b7-4b6e-b84f-afd330cbd80c",
    "e466ff25-fcd8-4ad3-ae56-cc593e3a117e",
    "1a99664a-02d6-48a3-b811-978e8bff9344"
]


font_size = 45
# Global font size and style
rcParams.update({
    'font.size': font_size,
    'axes.titlesize': font_size,
    'axes.labelsize': font_size,
    'xtick.labelsize': font_size,
    'ytick.labelsize': font_size,
    'legend.fontsize': 65,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial']
})

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import sem
from scipy.signal import savgol_filter

def plot_cluster_median_curve(reordered_full_data, output_folder, group_name):
    """
    Compute median + SEM per cluster and plot two curves.

    Parameters:
        reordered_full_data (pd.DataFrame): DataFrame with time series and cluster column.
        output_folder (str): Output path.
        group_name (str): Group name for file naming.
    """
    # print(reordered_full_data)
    input_file = "sleep_disturb_lessthan3hrs_correlation_heatmaps/All_types_of_surgery_sf36_clustered_results.csv"
    sf36_data = pd.read_csv(input_file)
    # sf36_data = sf36_data.dropna(subset=["limitation_phys_post", "limitation_phys_pre"])
    sf36_data = sf36_data.set_index("uuid")
    reordered_full_data = reordered_full_data.iloc[:, list(range(9, 15)) + [-1]]
    # Keep only UUIDs in sf36_data
    # print(sf36_data)
    # print(reordered_full_data)
    reordered_full_data = reordered_full_data[reordered_full_data.index.isin(sf36_data.index)]
    # print(reordered_full_data)
    print(reordered_full_data.shape)

    # Get time columns
    day_columns = [col for col in reordered_full_data.columns if col.isdigit()]
    sorted_days = sorted(day_columns, key=lambda x: int(x))

    # Compute median and SEM per cluster
    grouped_stats = reordered_full_data.groupby("cluster")[sorted_days].agg(['median', sem])

    # Get stats for Cluster 1 and Cluster 2
    median_group1 = grouped_stats.loc[1, (slice(None), 'median')].values.flatten()
    median_group2 = grouped_stats.loc[2, (slice(None), 'median')].values.flatten()

    sem_group1 = grouped_stats.loc[1, (slice(None), 'sem')].values.flatten()
    sem_group2 = grouped_stats.loc[2, (slice(None), 'sem')].values.flatten()

    # Smooth curves
    window_size = 3
    poly_order = 2
    #
    median_group1_smooth = savgol_filter(median_group1, window_size, poly_order)
    median_group2_smooth = savgol_filter(median_group2, window_size, poly_order)
    # median_group1_smooth = median_group1
    # median_group2_smooth = median_group2
    #
    sem_group1_smooth = savgol_filter(sem_group1, window_size, poly_order)
    sem_group2_smooth = savgol_filter(sem_group2, window_size, poly_order)
    # sem_group1_smooth = sem_group1
    # sem_group2_smooth = sem_group2

    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    x_values = np.arange(2, len(sorted_days) + 2)

    # Plot curves with shaded SEM
    ax.plot(x_values, median_group1_smooth, label="Group 1", color="#0072B2", linewidth=2)
    ax.fill_between(x_values,
                    median_group1_smooth - sem_group1_smooth.clip(min=0),
                    median_group1_smooth + sem_group1_smooth,
                    color="lightseagreen", alpha=0.2)

    ax.plot(x_values, median_group2_smooth, label="Group 2", color="#D55E00", linewidth=2)
    ax.fill_between(x_values,
                    median_group2_smooth - sem_group2_smooth.clip(min=0),
                    median_group2_smooth + sem_group2_smooth,
                    color="#D55E00", alpha=0.2)
    # Add markers
    ax.scatter(x_values, median_group1_smooth, color="#0072B2", edgecolors="black", s=100, zorder=3)
    ax.scatter(x_values, median_group2_smooth, color="#D55E00", edgecolors="black", s=100, zorder=3)

    # Spines
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # Axes appearance
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(axis='both', length=6, width=2)

    # Labels
    ax.set_xlabel("Days after surgery", fontsize=font_size)
    ax.set_ylabel("CSRI", fontsize=font_size)

    # Y-axis range
    # ax.set_ylim(0.4, 1.0)

    # X-axis ticks
    ax.set_xticks(range(2, 8))
    ax.set_xticklabels([str(i) for i in range(2, 8)], rotation=90)

    # Legend
    # ax.legend(fontsize=32, loc="lower right")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend().set_visible(False)
    # Cluster label order
    # ax.legend(handles, ["Poor Sleep Group", "Normal Sleep Group"], title="Group", fontsize=28, loc="lower right")

    # Layout
    plt.tight_layout()

    # Save
    output_path = f"{output_folder}/{group_name}_cluster_median_curve.png"
    plt.savefig(output_path, dpi=300, format="png")
    plt.close()

    print(f"Curve plot saved to {output_path}")





def plot_correlation_comparison_boxplot(reordered_full_data, output_folder, group_name):
    """
    Compare two groups across time windows (8~30 days, 31~89 days post-op).
    Compute Pearson correlation median per subject per window, visualize with boxplot,
    and run Wilcoxon rank-sum test for group comparison.

    Parameters:
        reordered_full_data (pd.DataFrame): DataFrame with time series and cluster column.
        output_folder (str): Output path.
        group_name (str): Group name for file naming.
    """
    input_file = "sleep_disturb_lessthan3hrs_correlation_heatmaps/All_types_of_surgery_sf36_clustered_results.csv"
    sf36_data = pd.read_csv(input_file)
    # sf36_data = sf36_data.dropna(subset=["limitation_phys_post", "limitation_phys_pre"])
    sf36_data = sf36_data.set_index("uuid")
    # reordered_full_data = reordered_full_data.iloc[:, list(range(8, 15)) + [-1]]
    # Keep only UUIDs in sf36_data
    # print(sf36_data)
    # print(reordered_full_data)
    reordered_full_data = reordered_full_data[reordered_full_data.index.isin(sf36_data.index)]
    # print(reordered_full_data)
    print(reordered_full_data.shape)

    # Time windows
    days_0_6 = [str(i) for i in range(0, 7) if str(i) in reordered_full_data.columns]
    days_7_8 = [str(i) for i in range(7, 9) if str(i) in reordered_full_data.columns]
    days_9_14 = [str(i) for i in range(9, 15) if str(i) in reordered_full_data.columns]
    days_15_37 = [str(i) for i in range(15, 38) if str(i) in reordered_full_data.columns]
    days_38_97 = [str(i) for i in range(38, 98) if str(i) in reordered_full_data.columns]

    # Ensure columns exist
    # if not days_8_30 or not days_31_89:
    #     print("Error: Required time period columns are missing in the dataset.")
    #     return
    # print(reordered_full_data)
    # jjj
    # Compute median per subject per window
    reordered_full_data["median_0_6"] = reordered_full_data[days_0_6].median(axis=1, skipna=True)
    reordered_full_data["median_7_8"] = reordered_full_data[days_7_8].median(axis=1, skipna=True)
    reordered_full_data["median_9_14"] = reordered_full_data[days_9_14].median(axis=1, skipna=True)
    reordered_full_data["median_15_37"] = reordered_full_data[days_15_37].median(axis=1, skipna=True)
    reordered_full_data["median_38_97"] = reordered_full_data[days_38_97].median(axis=1, skipna=True)

    # Keep cluster and median columns
    median_data = reordered_full_data[["cluster", "median_0_6", "median_7_8", "median_9_14", "median_15_37", "median_38_97"]].dropna()
    print(median_data)
    # Reshape for boxplot
    long_data = pd.melt(
        median_data,
        id_vars=["cluster"],
        value_vars=["median_0_6", "median_7_8", "median_9_14", "median_15_37", "median_38_97"],
        var_name="Time Period",
        value_name="Correlation Median"
    )
    long_data["Time Period"] = long_data["Time Period"].replace({
        "median_0_6": "-7~-1",
        "median_7_8": "0~1",
        "median_9_14": "2~7",
        "median_15_37": "8~30",
        "median_38_97": "31~90"
    })

    # Wilcoxon rank-sum test
    group1_0_6 = median_data.loc[median_data["cluster"] == 1, "median_0_6"].dropna()
    group2_0_6 = median_data.loc[median_data["cluster"] == 2, "median_0_6"].dropna()
    group1_7_8 = median_data.loc[median_data["cluster"] == 1, "median_7_8"].dropna()
    group2_7_8 = median_data.loc[median_data["cluster"] == 2, "median_7_8"].dropna()
    group1_9_14 = median_data.loc[median_data["cluster"] == 1, "median_9_14"].dropna()
    group2_9_14 = median_data.loc[median_data["cluster"] == 2, "median_9_14"].dropna()
    group1_15_37 = median_data.loc[median_data["cluster"] == 1, "median_15_37"].dropna()
    group2_15_37 = median_data.loc[median_data["cluster"] == 2, "median_15_37"].dropna()
    group1_38_97 = median_data.loc[median_data["cluster"] == 1, "median_38_97"].dropna()
    group2_38_97 = median_data.loc[median_data["cluster"] == 2, "median_38_97"].dropna()

    stat_0_6, p_0_6 = ranksums(group1_0_6, group2_0_6)
    stat_7_8, p_7_8 = ranksums(group1_7_8, group2_7_8)
    stat_9_14, p_9_14 = ranksums(group1_9_14, group2_9_14)
    stat_15_37, p_15_37 = ranksums(group1_15_37, group2_15_37)
    stat_38_97, p_38_97 = ranksums(group1_38_97, group2_38_97)
    print('Pre-op -7 to -1')
    print(np.median(group1_0_6), np.median(group2_0_6))
    print(f"Wilcoxon rank-sum test for 0_6 days: p={p_0_6:.5f}")
    print('POD 0 - 1')
    print(np.median(group1_7_8), np.median(group2_7_8))
    print(f"Wilcoxon rank-sum test for 7_8 days: p={p_7_8:.5f}")
    print('POD 2 -7')
    print(np.median(group1_9_14), np.median(group2_9_14))
    print(f"Wilcoxon rank-sum test for 9_14 days: p={p_9_14:.5f}")
    print('POD 8 - 30')
    print(np.median(group1_15_37), np.median(group2_15_37))
    print(f"Wilcoxon rank-sum test for 15_37 days: p={p_15_37:.5f}")
    print('POD 31 - 90')
    print(np.median(group1_38_97), np.median(group2_38_97))
    print(f"Wilcoxon rank-sum test for 38_97 days: p={p_38_97:.5f}")

    # Boxplot
    fig, ax = plt.subplots(figsize=(14, 10))

    sns.boxplot(
        x="Time Period", y="Correlation Median", hue="cluster",
        data=long_data, palette=["#0072B2", "#D55E00"], width=0.3, showfliers=False, ax=ax
    )
    # Boxplot styling
    for patch in ax.patches:
        patch.set_edgecolor('black')
        patch.set_linewidth(2)
    for line in ax.lines:
        line.set_color('black')
        line.set_linewidth(2)
    ax.legend().set_visible(False)
    ax.set_xlabel("Days after surgery", fontsize=font_size)
    ax.set_ylabel("Median CSRI", fontsize=font_size)
    # ax.set_title("Comparison of Sleep Correlation Between Two Groups", fontsize=16)
    # Legend handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Cluster label order
    # ax.legend(handles, ["Poor Sleep Group", "Normal Sleep Group"], title="Group", fontsize=24, loc="lower right")

    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(axis='both', length=6, width=2)
    # Significance markers
    x1, x2, x3, x4, x5 = 0, 1, 2, 3, 4
    max_y = long_data["Correlation Median"].max()
    if p_0_6 < 0.05:
        ax.text(0, max_y+0.03, "***" if p_0_6 < 0.001 else "**" if p_0_6 < 0.01 else "*" if p_0_6 < 0.05 else "NS",
                ha="center", fontsize=40, color="black")
        line_y = max_y + 0.02
        ax.hlines(line_y, x1-0.15/2, x1+0.15/2, color='black', linewidth=2)
    else:
        ax.text(0, max_y+0.05, "***" if p_0_6 < 0.001 else "**" if p_0_6 < 0.01 else "*" if p_0_6 < 0.05 else "NS",
                ha="center", fontsize=30, color="black")
        line_y = max_y + 0.02
        ax.hlines(line_y, x1-0.15/2, x1+0.15/2, color='black', linewidth=2)
    if p_7_8 < 0.05:
        ax.text(1, max_y+0.03, "*" if p_7_8 >= 0.01 else "**" if p_7_8 >= 0.001 else "***",
                ha="center", fontsize=40, color="black")
        line_y = max_y + 0.02
        ax.hlines(line_y, x2-0.15/2, x2+0.15/2, color='black', linewidth=2)
    if p_9_14 < 0.05:
        ax.text(2, max_y+0.03, "*" if p_9_14 >= 0.01 else "**" if p_9_14 >= 0.001 else "***",
                ha="center", fontsize=40, color="black")
        line_y = max_y + 0.02
        ax.hlines(line_y, x3-0.15/2, x3+0.15/2, color='black', linewidth=2)
    if p_15_37 < 0.05:
        ax.text(3, max_y+0.03, "*" if p_15_37 >= 0.01 else "**" if p_15_37 >= 0.001 else "***",
                ha="center", fontsize=40, color="black")
        line_y = max_y + 0.02
        ax.hlines(line_y, x4-0.15/2, x4+0.15/2, color='black', linewidth=2)
    if p_38_97 < 0.05:
        ax.text(4, max_y+0.03, "*" if p_38_97 >= 0.01 else "**" if p_38_97 >= 0.001 else "***",
                ha="center", fontsize=40, color="black")
        line_y = max_y + 0.02
        ax.hlines(line_y, x5-0.15/2, x5+0.15/2, color='black', linewidth=2)
    ax.set_ylim(-0.2, 1.2)
    plt.tight_layout()

    # Save
    output_path = f"{output_folder}/{group_name}_group_comparison_boxplot.png"
    plt.savefig(output_path, dpi=300, format="png")
    plt.close()

    print(f"Group comparison boxplot saved to {output_path}")



def knn_imputation(df, n_neighbors=5):
    # operation_types = df['operation_type'].unique()
    print(df)
    day_columns = [f'Day_{i}' for i in range(0, 98)]
    print(day_columns)
    type_data = df.copy()

    # Extract day columns
    type_day_data = type_data[day_columns]
    print(type_day_data.shape)
    print(type_day_data.head())

    # Similarity matrix
    similarity_matrix = np.zeros((type_day_data.shape[0], type_day_data.shape[0]))

    # Cosine similarity on overlapping non-missing values
    for i in range(type_day_data.shape[0]):
        for j in range(i + 1, type_day_data.shape[0]):
            # Mask for non-missing values
            non_nan_mask = ~np.isnan(type_day_data.iloc[i]) & ~np.isnan(type_day_data.iloc[j])

            # Cosine similarity on overlapping non-missing
            if np.any(non_nan_mask):
                similarity = cosine_similarity(
                    [type_day_data.iloc[i][non_nan_mask]],
                    [type_day_data.iloc[j][non_nan_mask]]
                )[0, 0]
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
            else:
                similarity_matrix[i, j] = 0
                similarity_matrix[j, i] = 0

    # print(len(similarity_matrix), len(similarity_matrix[0]))

    # Impute using k nearest neighbors
    for col in day_columns:
        missing_indices = type_day_data[type_day_data[col].isna()].index

        for idx in missing_indices:

            # Get top-k most similar rows
            similarities = similarity_matrix[idx]
            top_similar_indices = np.argsort(similarities)[-n_neighbors:]

            # Filter valid neighbors
            valid_indices = [i for i in top_similar_indices if not np.isnan(type_day_data.loc[i, col])]

            if valid_indices:
                # Median of neighbors
                imputed_value = type_day_data.loc[valid_indices, col].median()

                # Fallback to column median
                if pd.isna(imputed_value):
                    imputed_value = type_day_data[col].median(skipna=True)
                    print(f"No valid median from neighbors, using column median for column {col}: {imputed_value}")
            else:
                # No valid neighbors: use column median
                imputed_value = type_day_data[col].median(skipna=True)
                print(f"No valid neighbors for index {idx}, using column median for column {col}: {imputed_value}")

            # Impute
            type_day_data.loc[idx, col] = imputed_value

    # Update original dataframe
    df.loc[type_data.index, day_columns] = type_day_data

    return df

def plot_correlation_heatmap_cluster(data, output_folder, group_name, group_description, complication_dates_file, sf36_file):
    # Load sf36 data
    sf36_df = pd.read_csv(sf36_file)
    sf36_df['uuid'] = sf36_df['uuid'].str.lower()
    # data = data.drop(columns=["Day_0"])
    # Preprocess
    data = knn_imputation(data)
    data = data.set_index('uuid')
    data = data.drop(columns=['index'], errors='ignore')
    data['clean_uuid'] = data.index.map(lambda row: row.split('| uuid:')[-1].strip())
    data = data[~data['clean_uuid'].duplicated(keep='first')]
    data = data.drop(columns=['clean_uuid'])

    # Custom colormap
    custom_colors = ["#FFDD57", "#C4E48E", "#77DD77", "#A8D7FF", "#389FFF"]
    cmap = LinearSegmentedColormap.from_list("custom_heatmap", custom_colors, N=16)
    print(data)
    
    # Cluster on days 9-14
    columns_to_scale = [f'Day_{i}' for i in range(9, 15)]
    cluster_data = data[columns_to_scale].copy()
    # Feature engineering
    # cluster_data['num_bad_days1'] = (cluster_data[columns_to_scale] < 0.45).sum(axis=1)
    # cluster_data['num_bad_days2'] = (cluster_data[columns_to_scale] < 0.4).sum(axis=1)
    # cluster_data['num_bad_days3'] = (cluster_data[columns_to_scale] < 0.6).sum(axis=1)
    cluster_data['num_good_days1'] = (cluster_data[columns_to_scale] > 0.7).sum(axis=1)
    cluster_data['min_corr'] = cluster_data[columns_to_scale].min(axis=1)
    # cluster_data['max_corr'] = cluster_data[columns_to_scale].max(axis=1)
    cluster_data['median_corr'] = cluster_data[columns_to_scale].median(axis=1)
    cluster_data['mean_corr'] = cluster_data[columns_to_scale].mean(axis=1)
    cluster_data['range_corr'] = cluster_data[columns_to_scale].max(axis=1) - cluster_data[columns_to_scale].min(axis=1)
    # Std
    cluster_data['std_corr'] = cluster_data[columns_to_scale].std(axis=1)
    # Mean of worst 2 days
    cluster_data['mean_worst2'] = cluster_data[columns_to_scale].apply(
        lambda row: row.nsmallest(2).mean(), axis=1
    )
    # Gap area from ideal curve [1,1,1,1,1,1]
    x = np.array([9, 10, 11, 12, 13, 14])
    ideal_y = np.ones_like(x)
    actual_y = cluster_data[columns_to_scale].values  # shape: (N, 4)
    gap_area = np.trapz(np.abs(ideal_y - actual_y), x=x, axis=1)
    cluster_data['gap_area'] = gap_area
    # Num days with correlation >= 0.9
    # cluster_data['num_good_days'] = (cluster_data[columns_to_scale] >= 0.9).sum(axis=1)

    # For DTW: reshape data
    # cluster_data_3d = cluster_data.values[..., np.newaxis]  # (n_samples, time_steps, 1)

    # For DTW: pairwise distance matrix
    # dtw_distance_matrix = cdist_dtw(cluster_data_3d)

    # For DTW: linkage clustering
    # linkage_matrix = linkage(dtw_distance_matrix, method='average')

    # Apply PCA
    pca = PCA(n_components=3)
    cluster_data_pca = pca.fit_transform(cluster_data.values)
    print(pca.components_)  # shape: (n_components, n_features)
    print(pca.explained_variance_ratio_)
    
    # K-means
    n_clusters = 2
    # Fit K-means on PCA-reduced data
    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=10,
        max_iter=300,
        tol=1e-4,
        random_state=26,
        algorithm='lloyd'
    )
    cluster_labels = kmeans.fit_predict(cluster_data_pca)
    original_labels = cluster_labels.copy()
    cluster_labels = cluster_labels + 1
    temp = cluster_labels.copy()
    cluster_labels[temp == 1] = 2
    cluster_labels[temp == 2] = 1
    print(cluster_labels)
    print(len(cluster_labels))
    # Add cluster labels to data
    centers = kmeans.cluster_centers_
    # Visualize clusters
    plt.figure(figsize=(16, 12))
    # Define custom colors
    custom_colors = np.array(['#0072B2', '#D55E00'])  # or ['##0072B2', '##D55E00'] for hex
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    data_tsne = tsne.fit_transform(cluster_data_pca)
    # UMAP (optional)
    # reducer = umap.UMAP(n_components=2, random_state=42)
    # data_umap = reducer.fit_transform(cluster_data_pca)
    plt.scatter(data_tsne[:, 0],
                data_tsne[:, 1],
                c=custom_colors[original_labels],
                s=100,
                alpha=0.8)
    # Cluster centers
    # plt.scatter(centers[:, 0], centers[:, 1], facecolors='none',
    #         edgecolors='black', s=200, marker='o', label='Centers')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    # plt.title(f'K-Means Clustering (k={n_clusters})')
    plt.grid(True)
    output_path = os.path.join(output_folder, f"{group_name.replace(' ', '_')}_grouped_kmeans.png")
    plt.savefig(output_path, dpi=300)
    # cluster_data_pca: PCA-reduced data
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 3D scatter (swapped colors: cluster 0=orange, cluster 1=blue)
    pca_3d_colors = np.array(['#D55E00', '#0072B2'])
    ax.scatter(
        cluster_data_pca[:, 0],
        cluster_data_pca[:, 1],
        cluster_data_pca[:, 2],
        c=pca_3d_colors[original_labels],
        s=100,
        alpha=0.8
    )

    ax.set_xlabel("PCA 1", fontsize=14)
    ax.set_ylabel("PCA 2", fontsize=14)
    ax.set_zlabel("PCA 3", fontsize=14)
    ax.tick_params(axis='both', labelsize=14)
    ax.tick_params(axis='z', labelsize=14)
    # ax.set_title("3D PCA Visualization", fontsize=14)
    plt.tight_layout()

    # Save
    output_path = os.path.join(output_folder, f"{group_name.replace(' ', '_')}_pca_3d_kmeans.png")
    plt.savefig(output_path, dpi=300)
    
    # print(pca.components_)  # shape: (n_components, n_features)
    # print(pca.explained_variance_ratio_)
    
    # cluster_data shape: (n_samples, time_steps)
    # distance_matrix = pdist(cluster_data.values, metric='euclid')
    # linkage_matrix = linkage(distance_matrix, method='ward')
    # g = sns.clustermap(
    #     cluster_data,
    #     cmap=cmap,
    #     vmin=0.0,
    #     vmax=1.0,
    #     figsize=(24, 52),
    #     row_cluster=True,
    #     col_cluster=False,
    #     cbar_kws={'label': 'CSRI'},
    #     linewidths=2.5,
    #     row_linkage=linkage_matrix,
    # )

    # Extract cluster labels
    # num_clusters = 2
    # Example: use distance threshold
    # cluster_labels = fcluster(linkage_matrix, t=num_clusters, criterion='maxclust')
    # cluster_labels = fcluster(linkage_matrix, t=3.28, criterion='distance')
    # Sort by KMeans cluster labels
    data['kmeans_cluster'] = cluster_labels
    reordered_index = data.sort_values('kmeans_cluster').index
    reordered_full_data = data.loc[reordered_index]

    # Reorder rows
    # clustered_rows = g.dendrogram_row.reordered_ind
    # reordered_index = cluster_data.index[clustered_rows]
    columns_to_scale = [f'Day_{i}' for i in range(0, 98)] # 9-13
    showimg_data = data[columns_to_scale]
    # Reorder full data by cluster
    reordered_full_data = showimg_data.loc[reordered_index]

    # Cluster colors
    unique_clusters = np.unique(cluster_labels)
    print('unique_clusters', unique_clusters)
    palette = sns.color_palette("husl", len(unique_clusters))
    print('palette', palette)
    cluster_colors = {cluster: palette[i] for i, cluster in enumerate(unique_clusters)}
    # Custom colors
    cluster_colors = {
        unique_clusters[0]: '#0072B2',
        unique_clusters[1]: '#D55E00'
    }
    print('cluster_colors', cluster_colors)
    row_colors = pd.Series(cluster_labels, index=cluster_data.index).map(cluster_colors)
    print('row_colors', row_colors)
    day_columns = [col for col in reordered_full_data.columns if col.startswith('Day_')]
    print('day_columns', day_columns)
    new_day_columns = {col: col.split('_')[1] for col in day_columns}
    print('new_day_columns', new_day_columns)
    reordered_full_data.rename(columns=new_day_columns, inplace=True)
    print('reordered_full_data', reordered_full_data)
    print(reordered_full_data)
    
    # show complete figures
    custom_colors = ["#FFDD57", "#C4E48E", "#77DD77", "#A8D7FF", "#389FFF"]
    cmap = LinearSegmentedColormap.from_list("custom_heatmap", custom_colors, N=16)
    
    g = sns.clustermap(
        reordered_full_data,
        cmap=cmap,
        vmin=0.2,
        vmax=0.8,
        figsize=(48, 52),
        row_cluster=False,
        col_cluster=False,
        row_colors=row_colors,
        cbar_kws={'label': 'CSRI', 'shrink': 0.8},
        linewidths=2.5,
        yticklabels=False,
    )
    # Dendrogram line width
    for line in g.ax_row_dendrogram.collections:
        line.set_linewidth(3)
    g.ax_cbar.remove()
    # g = sns.clustermap(...) return value
    # Adjust row_colors position
    # g.ax_row_colors.set_position([
    #     g.ax_row_colors.get_position().x0 - 0.05,
    #     g.ax_row_colors.get_position().y0,
    #     0.5,
    #     g.ax_row_colors.get_position().height
    # ])


    # Custom axis labels
    # g.ax_heatmap.set_xlabel("Days after surgery", fontsize=70)
    # g.ax_heatmap.set_ylabel("Participants", fontsize=70)
    g.ax_cbar.set_ylabel("CSRI", fontsize=55)
    # Axes
    g.ax_heatmap.tick_params(axis='both', length=6, width=2)
    # plt.subplots_adjust(bottom=0.2)
    # g.ax_heatmap.set_position([0.2, 0.15, 0.6, 0.7])
    # g.ax_row_colors.set_position([0.1, 0.2, 0.05, 0.6])

    # Row labels
    new_row_labels = range(1, len(reordered_full_data) + 1)
    # Row labels
    # g.ax_heatmap.set_yticks(np.arange(len(new_row_labels)) + 0.5)
    # g.ax_heatmap.set_yticklabels(new_row_labels, rotation=0, fontsize=40)
    # g.ax_row_colors.set_position([0.1, 0.2, 0.05, 0.6])  # (x, y, width, height)
    new_column_labels = range(-7, 91)
    # Tick positions
    tick_positions = np.arange(len(new_column_labels))[::3] + 0.5
    tick_labels = [new_column_labels[i] for i in range(0, len(new_column_labels), 3)]

    # X-axis ticks
    g.ax_heatmap.set_xticks(tick_positions)
    g.ax_heatmap.set_xticklabels(tick_labels, rotation=90, fontsize=40)

    plt.tight_layout()
    
    # show complete figures
    correct_cluster_labels = pd.Series(cluster_labels, index=cluster_data.index).loc[reordered_full_data.index].values
    reordered_full_data["cluster"] = correct_cluster_labels
    reordered_full_data["cluster"] = reordered_full_data["cluster"].astype(int)

    # for i, row in enumerate(reordered_index):
    #     uuid = row
    
    #     # Look up complication_dates and surgery_date in sf36
    #     sf36_row = sf36_df[sf36_df['uuid'] == uuid]
    #     if sf36_row.empty:
    #         continue
    
    #     surgery_date = pd.to_datetime(sf36_row['surgery_date'].values[0])
    #     complication_dates = sf36_row['complication_dates'].values[0]
    
    #     if pd.notna(complication_dates):
    #         # Parse comma-separated dates
    #         complication_dates = [
    #             pd.to_datetime(date.strip())
    #             for date in complication_dates.split(',')
    #             if date.strip()
    #         ]
    
    #         # Mark complication dates
    #         for comp_date in complication_dates:
    #             day_offset = (comp_date - surgery_date).days
    #             if 0 <= day_offset <= 98:
    #                 g.ax_heatmap.scatter(day_offset + 0.5 + 7, i + 0.5, color='red', marker='*', s=200, zorder=5)

    # for i, row in enumerate(reordered_index):
    #     uuid = row
    #
    #     # Look up complication_dates and surgery_date in sf36
    #     sf36_row = sf36_df[sf36_df['uuid'] == uuid]
    #     if sf36_row.empty:
    #         continue
    #
    #     surgery_date = pd.to_datetime(sf36_row['surgery_date'].values[0])
    #     # print(sf36_row.columns)
    #     discharge_dates = sf36_row['surgery_discharge_date'].values[0]

        # if pd.notna(discharge_dates):
        #     # Parse comma-separated dates
        #     discharge_dates = [
        #         pd.to_datetime(date.strip())
        #         for date in discharge_dates.split(',')
        #         if date.strip()
        #     ]
        #
        #     for comp_date in discharge_dates:
        #         day_offset = (comp_date - surgery_date).days
        #         if 0 <= day_offset < 90:
        #             # Rectangle covering Day 0 to day_offset
        #             rect = Rectangle((0, i), day_offset + 1, 1, linewidth=2, edgecolor='red', facecolor='none',
        #                              zorder=5)
        #             g.ax_heatmap.add_patch(rect)

    # Sort by KMeans cluster
    sorted_index = data.sort_values('kmeans_cluster').index

    # Save cluster results
    clustered_data = pd.DataFrame({
        'uuid': sorted_index,
        'cluster': data.loc[sorted_index, 'kmeans_cluster'].values
    })

    # Data availability heatmap
    total_sleep_data = pd.read_csv('total_sleep_available.csv')
    total_sleep_data['uuid'] = total_sleep_data['uuid'].str.lower()
    # Deduplicate by uuid
    total_sleep_data = total_sleep_data.drop_duplicates(subset='uuid', keep='first')

    availability_data = total_sleep_data[total_sleep_data['uuid'].isin(reordered_index)]
    availability_data = availability_data.set_index('uuid').loc[reordered_index]

    day_columns = [col for col in availability_data.columns if col.startswith('Day_')]
    # columns_to_availability = [f'Day_{i}' for i in range(1, 30)]
    availability_matrix = availability_data[day_columns]

    # Data availability heatmap
    plt.figure(figsize=(26, 20))
    sns.heatmap(
        availability_matrix,
        cmap="YlGnBu",
        cbar_kws={'label': 'Data Availability'},
        linewidths=0.5,
        linecolor='grey'
    )
    # plt.title("Data Availability Heatmap (Day -7 - Day 90)", fontsize=16, color="darkblue")
    plt.xlabel("Days", fontsize=20)
    plt.ylabel("Patients", fontsize=20)
    plt.yticks(
        ticks=np.arange(availability_matrix.shape[0]),
        labels=availability_matrix.index,
        fontsize=10
    )

    # plt.yticks([])  # Hides y-axis tick labels
    plt.xticks([])

    # Save heatmap
    output_heatmap = os.path.join(output_folder, f"{group_name.replace(' ', '_')}_data_availability_heatmap.png")
    plt.tight_layout()
    plt.savefig(output_heatmap, dpi=300)
    plt.close()
    print(f"Data availability heatmap saved to: {output_heatmap}")
    print(availability_matrix)
    # Save merged results
    output_csv = os.path.join(output_folder, f"{group_name.replace(' ', '_')}_sf36_clustered_results.csv")
    merged_sf36 = clustered_data.merge(sf36_df, on='uuid', how='left')
    print(merged_sf36)
    # Add discharge days column
    merged_sf36['surgery_date'] = pd.to_datetime(merged_sf36['surgery_date'], errors='coerce')
    merged_sf36['surgery_discharge_date'] = pd.to_datetime(merged_sf36['surgery_discharge_date'], errors='coerce')
    # Compute discharge days
    merged_sf36['discharge_days'] = (merged_sf36['surgery_discharge_date'] - merged_sf36['surgery_date']).dt.days
    merged_sf36 = showimg_data.merge(merged_sf36, on='uuid', how='left')
    merged_sf36.to_csv(output_csv, index=False)
    print(f"Clustered group results with sf36 scores saved to: {output_csv}")
    
    # Save clustermap
    output_path = os.path.join(output_folder, f"{group_name.replace(' ', '_')}_grouped_clustermap.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Clustermap with group markers saved to {output_path}")

    print(reordered_full_data)

    plot_cluster_median_curve(reordered_full_data, output_folder, group_name)
    plot_correlation_comparison_boxplot(reordered_full_data, output_folder, group_name)

def process_csri_data(input_file, output_folder, sf36_file):
    """
    Load CSRI data and plot heatmaps.

    Parameters:
        input_file: str, path to CSRI_results.csv
        output_folder: str, output path for heatmaps.
    """
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    complication_dates_file = "Chart_review_20250616_processed.csv"
    group_name = "All types of surgery"
    group_description = "All types of surgery"

    # Load CSRI data directly
    data = pd.read_csv(input_file)
    data['uuid'] = data['uuid'].str.lower()
    print(f"Loaded {len(data)} rows from {input_file}")

    # Load complication data
    complication_df = pd.read_csv(complication_dates_file, encoding='ISO-8859-1')
    complication_df['uuid'] = complication_df['patient_uuid'].str.lower()

    # Age: based on surgery_date and dob
    def calculate_age(dob, surgery_date):
        if pd.isnull(dob) or pd.isnull(surgery_date):
            return None
        birth_date = datetime.strptime(dob, "%m/%d/%Y")
        surgery_date = datetime.strptime(surgery_date, "%m/%d/%Y")
        return surgery_date.year - birth_date.year - ((surgery_date.month, surgery_date.day) < (birth_date.month, birth_date.day))

    complication_df['age'] = complication_df.apply(
        lambda row: calculate_age(row['patient_dob'], row['surgery_date']), axis=1
    )

    # Merge with complication data
    matched_info = complication_df[['patient_uuid', 'age', 'patient_sex', 'patient_bmi']]
    matched_info = matched_info.rename(columns={'patient_uuid': 'uuid'})
    data = data.merge(matched_info, on='uuid', how='left')
    data = data.drop(columns=['patient_sex', 'patient_bmi'], errors='ignore')

    # Filter by current_uuid_list
    merged_data = data[data['uuid'].isin(current_uuid_list)]
    if merged_data.empty:
        print("No data after filtering by current_uuid_list")
        return

    # Filter by valid data count
    filter_merged_data = merged_data.copy().loc[:, 'Day_7':'Day_97']
    row_valid_counts = filter_merged_data.notna().sum(axis=1)
    k = 63  # Min valid data points (~70%)
    valid_rows = row_valid_counts[row_valid_counts >= k].index
    merged_data = merged_data.loc[valid_rows]
    merged_data = merged_data.reset_index(drop=True)

    plot_correlation_heatmap_cluster(merged_data, output_folder, group_name, group_description, complication_dates_file, sf36_file)



def main():
    input_file = "sleep_CSRI_csv_results/CSRI_results.csv"
    output_folder = "clustering_results"
    sf36_file = "sf36_calculated_scores.csv"

    process_csri_data(input_file, output_folder, sf36_file)



if __name__ == "__main__":
    main()
