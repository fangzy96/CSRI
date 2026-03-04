import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import wilcoxon, ttest_ind, mannwhitneyu
from scipy.signal import savgol_filter

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

# File mapping for sleep available data
file_mapping = {
    'sleep_day': 'sleep_day_available.csv',
    'sleep_night': 'sleep_night_available.csv',
    'nonsleep_day': 'nonsleep_day_available.csv',
    'nonsleep_night': 'nonsleep_night_available.csv',
    'combined_sleep': 'combined_sleep_available.csv',
    'combined_nonsleep': 'combined_nonsleep_available.csv'
}
from matplotlib import rcParams
font_size = 45
# Set global font size and style
rcParams.update({
    'font.size': font_size,
    'axes.titlesize': font_size,
    'axes.labelsize': font_size,
    'xtick.labelsize': 45,
    'ytick.labelsize': font_size,
    'legend.fontsize': font_size,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial']
})
def plot_boxplot(df, title, ax, alpha=0.05, ylim=None):
    """
    Plot median curve with significance markers (Mann-Whitney U test vs baseline).
    """
    # Select Day_1 to Day_98 columns for plotting
    print(df)
    data = df.loc[:, 'Day_1':'Day_98']

    if data.empty:
        print(f"No data available to plot for {title}")
        return  # Skip plotting

    # Reshape dataframe: each day as a category
    df_melted = data.melt(var_name="Day", value_name="RHR")

    # Extract first 7 days as baseline
    baseline_data = df.loc[:, 'Day_1':'Day_7'].values.flatten()
    baseline_data = baseline_data[~np.isnan(baseline_data)]  # Remove NaN

    # Compute median and standard error per day
    median_values = df_melted.groupby('Day')['RHR'].median()
    std_values = df_melted.groupby('Day')['RHR'].std()
    count_values = df_melted.groupby('Day')['RHR'].count()
    stderr_values = std_values / np.sqrt(count_values)

    # Ensure days are sorted in natural order
    sorted_median_values = median_values.reindex(sorted(median_values.index, key=lambda x: int(x.split('_')[1])))
    sorted_stderr_values = stderr_values.reindex(sorted(stderr_values.index, key=lambda x: int(x.split('_')[1])))

    # # Savitzky-Golay smoothing (commented out)
    # window_size = 5  # must be odd
    # poly_order = 4
    # smoothed_median_values = savgol_filter(sorted_median_values, window_size, poly_order, mode='interp')
    # smoothed_stderr_values = savgol_filter(sorted_stderr_values, window_size, poly_order, mode='interp')
    # sorted_median_values = pd.Series(smoothed_median_values, index=sorted_median_values.index)
    # sorted_stderr_values = pd.Series(smoothed_stderr_values, index=sorted_stderr_values.index)

    # Compute significance (Mann-Whitney U vs baseline)
    significance_markers = []
    for day in sorted_median_values.index:
        day_data = df[day].dropna().values  # Data for each day
        stat, p_value = mannwhitneyu(baseline_data, day_data, alternative='two-sided')
        print(day, p_value)
        # Set significance marker by p-value
        if p_value < 0.001:
            significance_markers.append('***')
        elif p_value < 0.01:
            significance_markers.append('**')
        elif p_value < 0.05:
            significance_markers.append('*')
        else:
            significance_markers.append('ns')

    # Surgery date marker (day 7 = day 0 post-op)
    ax.axvline(x=7, color='black', linestyle='--', linewidth=6)
    # sns.boxplot(x='Day', y='RHR', data=df_melted, showfliers=False, color='paleturquoise', ax=ax)

    # Plot median line
    x_values = range(len(sorted_median_values))
    ax.plot(x_values, sorted_median_values,
            color='blue', marker='o', markersize=15, linewidth=4, label='Median', alpha=0.6)

    # Plot standard error bounds (dashed lines)
    ax.plot(x_values, sorted_median_values - sorted_stderr_values,
            color='blue', linestyle='dashed', linewidth=3, label='Median - Std Error', alpha=0.6)
    ax.plot(x_values, sorted_median_values + sorted_stderr_values,
            color='blue', linestyle='dashed', linewidth=3, label='Median + Std Error', alpha=0.6)

    # Significance background color mapping (for optional use)
    significance_colors = {
        '***': 'coral',
        '**': 'bisque',
        '*': 'lightyellow'
    }
    # for i, marker in enumerate(significance_markers):
    #     if marker in significance_colors and i >= 8:
    #         ax.axvspan(i - 0.5, i + 0.5, color=significance_colors[marker], alpha=0.6)
    if ylim is not None:
        ax.set_ylim(*ylim)
    # ax.set_xticks(range(0, 98))
    # ax.set_xticklabels([f"{i+1}" for i in range(-8, 90)], rotation=90)
    # Show tick every 7 days
    xtick_positions = list(range(0, 98, 7))
    xtick_labels = [f"{i + 1}" for i in range(-8, 90, 7)]
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels)
    # ax.set_xlabel("Days", fontsize=font_size)
    # ax.set_ylabel("Median Sleep Ratio", fontsize=font_size)
    # Adjust x-axis to fit data range
    ax.set_xlim(min(x_values), max(x_values))

    # Reduce left/right margins for compact layout
    plt.subplots_adjust(left=0.05, right=0.95)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(axis='both', length=6, width=2)
    plt.tight_layout()
    # ax.legend(fontsize=font_size)
    # ax.set_title(title, fontsize=font_size)

def load_and_filter_data(filename, surgery_types):
    """
    Load CSV and filter by specified surgery types.
    """
    df = pd.read_csv(filename)
    return df[df['operation_type'].isin(surgery_types)].reset_index(drop=True)

def plot_boxplots(file_mapping, output_folder):
    """
    Merge group data and plot boxplots.
    """
    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Define surgery type groups
    group_1 = ["Wedge Resection", "Segmentectomy"]
    group_2 = ["Lobectomy", "Extended lobectomy", "Bilobectomy", "Sleeve Lobectomy"]
    group_3 = ["Wedge Resection", "Segmentectomy", "Lobectomy", "Extended lobectomy", "Bilobectomy", "Sleeve Lobectomy", 
               "Pneumonectomy", "Aortic aneurysm repair", "Aortic arch repair/graft placement", "Other"]

    for group_name, surgery_types in [("All types of surgery", group_3)]:
        # For each group, generate two figures with different metrics and y-limits
        metric_configs = [
            ("sleep_night", (0.55, 0.95), "nighttime_normalized_sleep_ratio"),
            ("combined_sleep", (0.0, 1.0), "normalized_total_sleep_ratio"),
        ]

        for metric_key, ylim, filename in metric_configs:
            fig, ax = plt.subplots(1, 1, figsize=(40, 12))

            print(f"Processing: {group_name} - {metric_key}")

            # Load and filter data for current group and metric
            df_metric = load_and_filter_data(file_mapping[metric_key], surgery_types)
            df_metric = df_metric[df_metric['uuid'].isin(current_uuid_list)]
            # Keep first row per uuid
            df_metric = df_metric.drop_duplicates(subset='uuid', keep='first')
            print(len(current_uuid_list))
            print(df_metric)

            # Plot
            plot_boxplot(df_metric, f"{group_name}", ax, ylim=ylim)

            # Save figure
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            output_path = os.path.join(output_folder, f"{filename}.png")
            plt.savefig(output_path, format='png', dpi=300)
            plt.close()
            print(f"Saved: {output_path}")

        # fig, axes = plt.subplots(1, 1, figsize=(30, 10))
        # axes = axes.flatten()
        # print(f"Processing: {group_name}")
        # Merge data for current group (alternative: sleep_night, sleep_day, combined_sleep)
        # df_sleep_night = load_and_filter_data(file_mapping['sleep_night'], surgery_types)
        # df_sleep_day = load_and_filter_data(file_mapping['sleep_day'], surgery_types)
        # df_combined_sleep = load_and_filter_data(file_mapping['combined_sleep'], surgery_types)

        # plot_boxplot(df_sleep_night, f"{group_name} - Sleep Night", axes[0])
        # plot_boxplot(df_sleep_day, f"{group_name} - Sleep Day", axes[1])
        # plot_boxplot(df_combined_sleep, f"{group_name} - Combined Sleep", axes[2])

        # fig.suptitle(...)
        # fig.suptitle(f"Sleep Analysis for {group_name}", fontsize=24)
        # plt.tight_layout(rect=[0, 0, 1, 0.95])
        # output_path = os.path.join(output_folder, f"{group_name.replace(' ', '_')}_sleep_available_boxplot.png")
        # plt.savefig(output_path, dpi=300)
        # plt.close()
        # print(f"Saved: {output_path}")

# Main
def main():
    output_folder = "sleep_available_boxplots"

    plot_boxplots(file_mapping, output_folder)

if __name__ == "__main__":
    main()
