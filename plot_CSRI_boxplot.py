import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import wilcoxon, ttest_ind, mannwhitneyu

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

def plot_combined_correlation_boxplot(input_folder, output_folder, group_name, surgery_types):
    """
    Plot Sleep Correlation Boxplot, merging data from specified surgery types.
    """
    combined_data = []

    # Iterate over specified surgery type files
    for surgery in surgery_types:
        if surgery == 'all_surgery_types':
            input_csv = os.path.join(input_folder, "CSRI_results.csv")
        else:
            input_csv = None
            for file in os.listdir(input_folder):
                if file.endswith("_correlation_results.csv") and surgery in file:
                    input_csv = os.path.join(input_folder, file)
                    break
            if input_csv is None:
                continue

        if os.path.exists(input_csv):
            # Read and reshape data
            data = pd.read_csv(input_csv)
            print(data)
            data = data.drop(columns=['surgery_subspecialty_type'], errors='ignore')
            data = data.melt(id_vars='uuid', var_name='Day', value_name='Correlation')
            data['Day'] = data['Day'].str.extract(r'(-?\d+)').astype(int)
            combined_data.append(data)

    # Merge all data
    if combined_data:

        combined_df = pd.concat(combined_data, ignore_index=True)
        print(combined_df)
        combined_df = combined_df[combined_df['Day'] <= 98]
        combined_df = combined_df[combined_df['uuid'].isin(current_uuid_list)]
        print(combined_df)
        
        # print(combined_df.columns)
        # 1. Print daily medians
        daily_medians = combined_df.groupby('Day')['Correlation'].median()
        for day, median_val in daily_medians.items():
            print(f"Day {day}: median = {median_val:.4f}")

        print("\n--- Range medians ---")

        # 2. Compute medians for specified day ranges
        ranges = {
            "Day 31 – Day 90":  (38, 97),
            "Day 4 – Day 7": (11, 14),
            "Day 8 – Day 30":  (15, 37),
        }

        for label, (start, end) in ranges.items():
            mask = (combined_df['Day'] >= start) & (combined_df['Day'] <= end)
            median_val = combined_df.loc[mask, 'Correlation'].median()
            print(f"{label}: median = {median_val:.4f}")
        
        # Custom color list
        custom_colors = ['mediumseagreen'] * 7 + ['silver'] * 2 + ['#FFDD57'] * 4 + ['#70A1D7'] * 85
        combined_df['Day'] = pd.to_numeric(combined_df['Day'], errors='coerce')

        # Compute median and standard error
        median_std_data = combined_df.groupby('Day')['Correlation'].agg(['median', 'sem']).reset_index()
        # print(median_std_data[:10])
        # print(median_std_data[-10:])

        baseline_days = list(range(0, 7))  # Baseline: first 7 days

        # Mann-Whitney U (Wilcoxon rank-sum) test
        significance_markers = []
        p_values = []

        for day in range(7, 98):  # Post-op days
            baseline_values = combined_df[combined_df['Day'].isin(baseline_days)]['Correlation'].dropna()
            daily_values = combined_df[combined_df['Day'] == day]['Correlation'].dropna()
            # print(baseline_values.shape, daily_values.shape)
            if len(daily_values) > 0 and len(baseline_values) > 0:
                _, p_value = mannwhitneyu(daily_values, baseline_values)  # Wilcoxon rank-sum test
                p_values.append(p_value)

                # Significance markers
                if p_value < 0.001:
                    significance_markers.append("***")
                elif p_value < 0.01:
                    significance_markers.append("**")
                elif p_value < 0.05:
                    significance_markers.append("*")
                else:
                    significance_markers.append("")
            else:
                significance_markers.append("")

        fig, ax = plt.subplots(figsize=(40, 12))

        # Boxplot
        # sns.boxplot(x='Day', y='Correlation', data=combined_df, showfliers=False, palette=custom_colors, ax=ax)

        # Median line
        ax.plot(median_std_data['Day'], median_std_data['median'], color='blue', marker='o', markersize=15, linewidth=4, alpha=0.6)

        # Standard error bounds (dashed lines)
        ax.plot(median_std_data['Day'], median_std_data['median'] - median_std_data['sem'],
                color='blue', linestyle='dashed', linewidth=3, label='Median - Std Error', alpha=0.6)

        ax.plot(median_std_data['Day'], median_std_data['median'] + median_std_data['sem'],
                color='blue', linestyle='dashed', linewidth=3, label='Median + Std Error', alpha=0.6)
        ax.axvline(x=7, color='black', linestyle='--', linewidth=6)  # Surgery date marker

        # Title and axis labels
        # ax.set_xlabel("Days after Surgery")
        # ax.set_ylabel("CSRI")

        # xticks = list(range(0, 98))  # Assuming Day 0 to Day 97 (total 98 days)
        # xticklabels = [str(i - 7) for i in xticks]  # Convert to -7 to 90
        # ax.set_xticks(xticks)
        # ax.set_xticklabels(xticklabels, rotation=90)

        xtick_positions = list(range(0, 98, 7))
        xtick_labels = [f"{i + 1}" for i in range(-8, 90, 7)]
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels)

        ax.set_ylim(0.0, 1.0)
        # Adjust x-axis to fit data range
        ax.set_xlim(min(median_std_data['Day']), max(median_std_data['Day']))

        # Reduce left/right margins for compact layout
        plt.subplots_adjust(left=0.05, right=0.95)
        # Set border line width
        for spine in ax.spines.values():
            spine.set_linewidth(2)

        # Set tick line width
        ax.tick_params(axis='both', length=6, width=2)

        # # Significance background colors (optional)
        # significance_colors = {'***': 'coral', '**': 'bisque', '*': 'lightyellow'}
        # for i, marker in enumerate(significance_markers):
        #     if marker in significance_colors:
        #         ax.axvspan(i + 7 - 0.5, i + 7 + 0.5, color=significance_colors[marker], alpha=0.6)

        plt.tight_layout()
        # plt.legend(loc="lower right", fontsize=font_size)

        # Save figure
        output_path = os.path.join(output_folder, "CSRI_boxplot.png")

        plt.savefig(output_path, format='png', dpi=300)
        plt.close()
        print(f"Boxplot with median and std error saved to {output_path}")
    else:
        print(f"No data found for group: {group_name}")


def main():
    input_folder = "sleep_CSRI_csv_results"
    output_folder = "sleep_CSRI_boxplots"

    # Create output folder if not exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Surgery type groups
    group_1 = ["Wedge_Resection", "Segmentectomy"]
    group_2 = ["Lobectomy", "Extended_lobectomy", "Bilobectomy", "Sleeve_Lobectomy"]
    # group_3 = ["Wedge Resection", "Segmentectomy", "Lobectomy", "Extended lobectomy", "Bilobectomy", "Sleeve Lobectomy", 
    #            "Pneumonectomy", "Aortic aneurysm repair", "Aortic arch repair/graft placement", "Other"]
    group_3 = ['all_surgery_types']

    # Plot boxplots
    # plot_combined_correlation_boxplot(input_folder, output_folder, "Group 1 - Wedge & Segmentectomy", group_1)
    # plot_combined_correlation_boxplot(input_folder, output_folder, "Group 2 - Lobectomy Types", group_2)
    plot_combined_correlation_boxplot(input_folder, output_folder, "All types of surgery", group_3)


if __name__ == "__main__":
    main()
