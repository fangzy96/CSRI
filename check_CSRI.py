import os
import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
import time

# Operation type mapping
operation_mapping = {
    1: "Wedge Resection",
    2: "Segmentectomy",
    3: "Lobectomy",
    4: "Extended lobectomy",
    5: "Bilobectomy",
    6: "Sleeve Lobectomy",
    7: "Pneumonectomy",
    8: "Aortic aneurysm repair",
    9: "Aortic arch repair/graft placement",
    10: "Other"
}
# Surgery subspecialty type mapping
surgery_mapping = {
    1: "Thoracic",
    2: "Cardiac",
    3: "Colorectal",
    4: "Pancreatic",
    5: "Orthopedic",
    6: "Other"
}

def calculate_sleep_correlation(complication_dates_file, processed_data_folder, output_folder):
    # Read complication data
    complication_df = pd.read_csv(complication_dates_file, encoding='ISO-8859-1')
    complication_df['patient_uuid'] = complication_df['patient_uuid'].str.lower()
    # Drop rows with NaN uuid
    complication_df = complication_df[~complication_df['patient_uuid'].isna()]
    # Keep only uuids that appear exactly once
    uuid_counts = complication_df['patient_uuid'].value_counts()
    valid_uuids = uuid_counts[uuid_counts == 1].index
    # Keep only those uuids
    complication_df = complication_df[complication_df['patient_uuid'].isin(valid_uuids)]

    # Filter uuids with surgery_subspecialty_type == 1 and withdrew != 1
    # target_uuids = set(
    #     complication_df.loc[complication_df['surgery_subspecialty_type'].isin([1, 2, 3, 4, 5, 6]) & 
    #     (complication_df['withdrew'] != 1),
    #     'patient_uuid']
    # )
    target_uuids = set(
    complication_df.loc[
        (complication_df['surgery_subspecialty_type'] == 1) & 
        (complication_df['withdrew'] != 1),
        'patient_uuid'])
    # Print a sample to verify surgery_subspecialty_type mapping
    for uuid in list(target_uuids)[:10]:  # print first 10 to avoid too much output
        surgery_type = complication_df.loc[complication_df['patient_uuid'] == uuid, 'surgery_subspecialty_type'].values
        print(f"UUID: {uuid}, Surgery Type(s): {surgery_type}")
    
    # Get all processed patient folders
    all_folders = [os.path.join(processed_data_folder, f)
                for f in os.listdir(processed_data_folder)
                if os.path.isdir(os.path.join(processed_data_folder, f))]

    # Match folders to target UUIDs
    processed_files = [
        folder for folder in all_folders
        if os.path.basename(folder).lower() in target_uuids
    ]

    # # Alternative: get all processed folders without filtering by uuid
    # processed_files = [os.path.join(processed_data_folder, f) for f in os.listdir(processed_data_folder)
    #                    if os.path.isdir(os.path.join(processed_data_folder, f))]
    print(len(processed_files))
    
    # Containers for results by operation type and by surgery subspecialty
    operation_results = {operation: [] for operation in operation_mapping.values()}
    surgery_type_results = {surgery: [] for surgery in surgery_mapping.values()}
    for processed_file in processed_files:
        print(processed_file)
        uuid = os.path.basename(processed_file)
        complication_row = complication_df[complication_df['patient_uuid'] == uuid]
        if complication_row.empty:
            print(f"No complication data found for UUID: {uuid}")
            continue

        # --- Define surgery date
        date_of_surgery = pd.to_datetime(complication_row['surgery_date'].values[0])

        # --- Define time window for baseline and post-surgery data
        baseline_start = date_of_surgery - timedelta(days=21)  # start a bit earlier
        post_end = date_of_surgery + timedelta(days=91)

        # --- Step 1: collect all files within the time window
        file_date_pairs = []
        for root, dirs, files in os.walk(processed_file):
            for file in files:
                if file.endswith(".csv") and "_processed_" in file:
                    file_date = pd.to_datetime(file.split("_processed_")[-1].replace(".csv", ""), errors='coerce')
                    if pd.notna(file_date) and baseline_start <= file_date < post_end:
                        file_date_pairs.append((file_date, os.path.join(root, file)))

        # --- Step 2: split into baseline and post-surgery files
        baseline_files = []
        post_files = []
        for file_date, file_path in file_date_pairs:
            if file_date < date_of_surgery:
                baseline_files.append((file_date, file_path))
            else:
                post_files.append((file_date, file_path))

        # --- Step 3: process baseline files
        baseline_files.sort()  # ascending by date

        # Drop the earliest baseline day
        if baseline_files:
            baseline_files = baseline_files[1:]

        # Keep at most 7 baseline days
        if len(baseline_files) > 7:
            baseline_files = baseline_files[-7:]

        # --- Step 4: process post-surgery files
        post_files.sort()  # naturally 0–90 days after surgery

        # --- Step 5: read all files
        all_dataframes = []

        # baseline files
        for file_date, file_path in baseline_files:
            df = pd.read_csv(file_path)
            all_dataframes.append(df)

        # post-surgery files
        for file_date, file_path in post_files:
            df = pd.read_csv(file_path)
            all_dataframes.append(df)

        # --- Step 6: merge all dataframes
        if not all_dataframes:
            continue  # skip this patient if no data
        merged_df = pd.concat(all_dataframes, ignore_index=True)
        merged_df['datetime'] = pd.to_datetime(merged_df['datetime'])
        merged_df['day'] = (merged_df['datetime'] - pd.Timestamp(date_of_surgery)).dt.days
        merged_df['hour'] = merged_df['datetime'].dt.hour

        # Initialize matrix: 7 baseline days + 90 post-surgery days = 98 rows, 24 hours
        sleep_matrix = np.full((98, 24), np.nan)

        # Compute sleep-available ratio per hour
        for day in range(-7, 91):
            # print(day, type(day))
            for hour in range(24):
                hour_data = merged_df[(merged_df['day'] == day) & (merged_df['hour'] == hour)]
                if not hour_data.empty:
                    sleep_count = hour_data[(hour_data['Sleep_Level'].isin([0,1,2,3,4,5,6])) &
                                            hour_data['Heart Rate'].notna()].shape[0]
                    total_count = hour_data['Heart Rate'].notna().sum()
                    sleep_matrix[day + 7, hour] = sleep_count / total_count if total_count > 0 else np.nan
        # print('------------------------------------------')
        # print(day, sleep_matrix.shape, sleep_matrix[day + 7, :])
        # time.sleep(0.5)  

        # Compute baseline (first 7 days) median pattern across hours
        # print(sleep_matrix[:1, :])
        # print(sleep_matrix.shape)
        baseline_median = np.nanmedian(sleep_matrix[:7, :], axis=0)
        # print(baseline_median)
        # print(baseline_median.shape)

        # print(sleep_matrix)
        # print(sleep_matrix.shape)
        # Compute Pearson correlation between baseline pattern and each day
        correlation_values = []
        for day in range(0, 98):
            # print(day)
            daily_values = sleep_matrix[day, :]
            # print(day, daily_values)
            if not np.isnan(daily_values).all():
                # indices where both baseline and daily values are not NaN
                valid_indices = ~np.isnan(baseline_median) & ~np.isnan(daily_values)
                temp_baseline_median = baseline_median[valid_indices]
                temp_daily_values = daily_values[valid_indices]
                correlation = np.corrcoef(temp_baseline_median, temp_daily_values, rowvar=False)[0, 1]
                correlation_values.append(correlation)
            else:
                correlation_values.append(np.nan)
        # Save per-operation results
        for i in range(1, 8):
            if complication_row[f'lung_resection___{i}'].values[0] == 1:
                operation_results[operation_mapping[i]].append([uuid] + correlation_values)
        print(complication_row['surgery_subspecialty_type'])
        # complication_row may contain one or multiple rows
        surgery_types = complication_row['surgery_subspecialty_type']

        # If it's a single value, wrap into a list
        if not isinstance(surgery_types, (pd.Series, list)):
            surgery_types = [surgery_types]

        # Remove duplicates + remove NaN
        unique_types = pd.Series(surgery_types).dropna().astype(int).unique()

        # Record results for each surgery type label
        for st in unique_types:
            if st in surgery_mapping:
                surgery_label = surgery_mapping[st]
                surgery_type_results[surgery_label].append([uuid] + correlation_values)

    # Save per-operation results to CSV
    for operation, results in operation_results.items():
        if results:
            columns = ['uuid'] + [f"Day_{i}" for i in range(0, 98)]
            results_df = pd.DataFrame(results, columns=columns)
            output_csv_path = os.path.join(output_folder, f"{operation.replace(' ', '_')}_correlation_results.csv")
            results_df.to_csv(output_csv_path, index=False)
            print(f"Saved correlation results for {operation} to {output_csv_path}")
    
    # Aggregate all surgery types' results
    # Build mapping from uuid to surgery_subspecialty_type (if needed)
    uuid_to_type = dict(zip(
        complication_df['patient_uuid'],
        complication_df['surgery_subspecialty_type']
    ))

    # Collect all rows from surgery_type_results (not from operation_results)
    all_results = []

    for surgery_type_label, results in surgery_type_results.items():
        if results:
            for row in results:
                all_results.append([surgery_type_label] + row)

    # Column names: surgery_subspecialty_type_label + uuid + Day_0 ~ Day_97
    columns = ['surgery_subspecialty_type', 'uuid'] + [f"Day_{i}" for i in range(0, 98)]

    # Convert to DataFrame and save combined table
    results_df = pd.DataFrame(all_results, columns=columns)
    output_csv_path = os.path.join(output_folder, "CSRI_results.csv")
    results_df.to_csv(output_csv_path, index=False)
    print(f"Saved combined correlation results to {output_csv_path}")

# Main
def main():
    complication_dates_file = "Chart_review_20250616_processed.csv"
    processed_data_folder = "20250505_complete_incomplete/"
    output_folder = "sleep_CSRI_csv_results"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    calculate_sleep_correlation(complication_dates_file, processed_data_folder, output_folder)

if __name__ == "__main__":
    main()
