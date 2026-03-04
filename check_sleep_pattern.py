import os
import pandas as pd
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import timedelta
import matplotlib
matplotlib.use('Agg')

from matplotlib import rcParams
font_size = 45
# Set global font size and style
rcParams.update({
    'font.size': font_size,
    'axes.titlesize': font_size,
    'axes.labelsize': font_size,
    'xtick.labelsize': font_size,
    'ytick.labelsize': font_size,
    'legend.fontsize': font_size,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial']
})

# Operation type mapping
operation_mapping = {
    1: "Wedge Resection",
    2: "Segmentectomy",
    3: "Lobectomy",
    4: "Extended lobectomy",
    5: "Bilobectomy",
    6: "Sleeve Lobectomy",
    7: "Pneumonectomy"
    # 8: "Aortic aneurysm repair",
    # 9: "Aortic arch repair/graft placement",
    # 10: "Other"
}

def process_surgery_sleep_distribution(complication_dates_file, processed_data_folder, output_folder):
    complication_df = pd.read_csv(complication_dates_file, encoding='ISO-8859-1')
    complication_df['patient_uuid'] = complication_df['patient_uuid'].str.lower()
    # Get all processed patient folders
    processed_files = [os.path.join(processed_data_folder, f) for f in os.listdir(processed_data_folder) if
                       os.path.isdir(os.path.join(processed_data_folder, f))]
    sleep_hourly_distribution = []

    # Iterate over each processed patient folder
    for processed_file in processed_files:
        uuid = processed_file.split('/')[-1]
        print(f"Processing UUID: {uuid}")

        complication_row = complication_df[complication_df['patient_uuid'] == uuid]
        if complication_row.empty:
            print(f"No complication data found for UUID: {uuid}")
            continue

        date_of_surgery = pd.to_datetime(complication_row['surgery_date'].values[0])
        has_complication = not complication_row[['complication_1_date', 'complication_2_date',
                                                 'complication_3_date', 'complication_4_date',
                                                 'complication_5_date']].isnull().all(axis=None)

        operation_type = []
        for i in range(1, 8):  # lung_resection___1 to lung_resection___7
            if complication_row[f'lung_resection___{i}'].values[0] == 1:
                operation_type.append(operation_mapping[i])

        # Read _processed.csv files (surgery -7 days to +90 days)
        start_date = date_of_surgery - pd.Timedelta(days=7)
        end_date = date_of_surgery + pd.Timedelta(days=90)
        processed_df_path = processed_file  # processed_file is already full path
        # print('processed_df_path', processed_df_path)
        all_dataframes = []
        # Walk through all subfolders
        for root, dirs, files in os.walk(processed_df_path):
            # print('root, dirs, files', root, dirs, files)
            for file in files:
                if file.endswith(".csv") and "_processed_" in file:
                    # Extract date from filename
                    date_str = file.split("_processed_")[-1].replace(".csv", "")
                    try:
                        file_date = pd.Timestamp(date_str)
                    except ValueError:
                        print(f"Unable to parse file date: {file}")
                        continue

                    # Check if date is within range
                    file_date = pd.to_datetime(file_date)
                    # print(file_date)
                    if start_date <= file_date < end_date:
                        file_path = os.path.join(root, file)
                        # Load CSV and append to list
                        df = pd.read_csv(file_path)
                        # print(df.head())
                        all_dataframes.append(df)

        # Merge all data
        if all_dataframes:
            merged_df = pd.concat(all_dataframes, ignore_index=True)
        else:
            print(f"No data found for UUID: {uuid}")
            continue
        processed_df = merged_df
        # Generate full time range (minute frequency)
        full_time_range = pd.date_range(start=start_date, end=end_date, freq='T')  # 'T' = minutes
        # Create DataFrame with full time range
        full_time_df = pd.DataFrame({'datetime': full_time_range})
        full_time_df = full_time_df.drop(full_time_df.index[-1])
        # Ensure datetime column types are consistent
        full_time_df['datetime'] = pd.to_datetime(full_time_df['datetime'])
        merged_df['datetime'] = pd.to_datetime(merged_df['datetime'])
        # Merge
        processed_df = pd.merge(full_time_df, merged_df, on='datetime', how='left')
        # Set index for hour grouping
        filtered_df = processed_df  # processed_df[(processed_df.index >= baseline_start_date) & (processed_df.index < end_date)]
        filtered_df.set_index('datetime', inplace=True)
        print(filtered_df)
        print(filtered_df.shape)

        # filtered_df = processed_df[(processed_df.index >= start_date) & (processed_df.index < end_date)]

        print(start_date, end_date, len(filtered_df))
        # Compute hourly sleep ratio (Sleep_Level 0-6 minutes / total HR minutes)
        filtered_df['hour'] = filtered_df.index.hour
        # num_days_mins = (filtered_df.shape[0] / 1440) * 60
        print(filtered_df)

        heart_rate_with_hour = filtered_df[['Heart Rate', 'hour']].dropna()
        hourly_sleep_counts = (
                filtered_df[filtered_df['Sleep_Level'].isin([0, 1, 2, 3, 4, 5, 6])].groupby('hour').size()
                / heart_rate_with_hour.groupby('hour').size()
        )
        print(hourly_sleep_counts)

        # Fill missing hours with 0
        hourly_sleep_counts = hourly_sleep_counts.reindex(range(24), fill_value=0)
        # Clip max to 1.0
        hourly_sleep_counts = hourly_sleep_counts.clip(upper=1.0)
        # Append to results
        for hour, count in hourly_sleep_counts.items():
            for operation in operation_type:
                sleep_hourly_distribution.append({
                    'uuid': uuid,
                    'hour': hour,
                    'sleep_count': count,
                    'has_complication': 'Yes' if has_complication else 'No',
                    'operation_type': operation
                })

    # Convert to DataFrame
    if not sleep_hourly_distribution:
        print("No data collected. Check complication_dates_file and processed_data_folder paths.")
        return
    sleep_hourly_distribution_df = pd.DataFrame(sleep_hourly_distribution)
    print(sleep_hourly_distribution_df)
    print(sleep_hourly_distribution_df.head())
    print(sleep_hourly_distribution_df.columns)
    # Plot boxplot by hour
    # for operation in sleep_hourly_distribution_df['operation_type'].unique():
    #     operation_df = sleep_hourly_distribution_df[sleep_hourly_distribution_df['operation_type'] == operation]
    # Custom color mapping (night: blue, day: yellow)
    box_colors = ['#70A1D7' if (hour <= 8 or hour >= 21) else '#FFDD57' for hour in range(24)]
    # Plot boxplot
    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
    sns.boxplot(
        data=sleep_hourly_distribution_df,
        x='hour',
        y='sleep_count',
        showfliers=False,
        palette=box_colors
    )
    # plt.title(f"Hourly Sleep Distribution")
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(axis='both', length=6, width=2)
    for patch in ax.patches:
        patch.set_edgecolor('black')
        patch.set_linewidth(2)
    for line in ax.lines:
        line.set_color('black')
        line.set_linewidth(2)
    plt.xlabel("Hour of Day")
    plt.ylabel("Sleep Ratio")
    plt.xticks(rotation=90)
    # plt.legend(title="Complication")
    plt.tight_layout()
    plt.savefig("Hourly Sleep Distribution.png")
    # plt.show()
        # Optional: save per-operation plots
        # output_plot_path = os.path.join(output_folder, f"sleep_hourly_distribution_{operation.replace(' ', '_')}_boxplot.png")
        # plt.savefig(output_plot_path)
        # print(f"Boxplot for {operation} saved to {output_plot_path}")
        # plt.show()



# Main
def main():
    complication_dates_file = "Chart_review_20250616_processed.csv"
    processed_data_folder = "20250505_complete_incomplete/"
    output_files = "sleep_24hours_pattern_boxplots"

    process_surgery_sleep_distribution(complication_dates_file, processed_data_folder, output_files)

if __name__ == "__main__":
    main()
