import os
import pandas as pd
from datetime import timedelta

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

def process_sleep_data(complication_dates_file, processed_data_folder, output_files):
    # Read complication dates file
    complication_df = pd.read_csv(complication_dates_file, encoding='ISO-8859-1')
    complication_df['patient_uuid'] = complication_df['patient_uuid'].str.lower()
    uuid_to_check = "e388fe16-2eeb-4c71-848a-4aa71530c21e"
    if uuid_to_check in complication_df['patient_uuid'].values:
        print(f"{uuid_to_check} exists in complication_df.")
    else:
        print(f"{uuid_to_check} does NOT exist in complication_df.")

    # Drop rows with NaN uuid
    complication_df = complication_df[~complication_df['patient_uuid'].isna()]
    # Keep only uuids that appear exactly once
    uuid_counts = complication_df['patient_uuid'].value_counts()
    valid_uuids = uuid_counts[uuid_counts == 1].index
    complication_df = complication_df[complication_df['patient_uuid'].isin(valid_uuids)]

    # Filter uuids with surgery_subspecialty_type == 1 and withdrew != 1
    target_uuids = set(
    complication_df.loc[
        (complication_df['surgery_subspecialty_type'] == 1) & 
        (complication_df['withdrew'] != 1),
        'patient_uuid'])
    if uuid_to_check in complication_df['patient_uuid'].values:
        print(f"{uuid_to_check} exists in complication_df.")
    else:
        print(f"{uuid_to_check} does NOT exist in complication_df.")
    
    # Get all patient folder paths
    all_folders = [os.path.join(processed_data_folder, f)
                for f in os.listdir(processed_data_folder)
                if os.path.isdir(os.path.join(processed_data_folder, f))]

    # Match folders to target UUIDs
    processed_files = [
        folder for folder in all_folders
        if os.path.basename(folder).lower() in target_uuids
    ]
    # processed_files = [os.path.join(processed_data_folder, f) for f in os.listdir(processed_data_folder) if
    #            os.path.isdir(os.path.join(processed_data_folder, f))]
    print(processed_files)
    print(len(processed_files))
    
    # Initialize result dataframes
    sleep_day_results = pd.DataFrame()
    sleep_night_results = pd.DataFrame()
    nonsleep_day_results = pd.DataFrame()
    nonsleep_night_results = pd.DataFrame()
    combined_sleep_results = pd.DataFrame()
    combined_nonsleep_results = pd.DataFrame()

    # Iterate over each processed patient folder
    for processed_file in processed_files:
        uuid = processed_file.split('/')[-1]
        if 'e388fe16-2eeb-4c71-848a-4aa71530c21e' == uuid:
            print('uuid', uuid)
        # Find date_of_surgery and operation type for this UUID
        complication_row = complication_df[complication_df['patient_uuid'] == uuid]
        if complication_row.empty:
            print(f"No complication data found for UUID: {uuid}")
            continue
        

        # Find operation type
        operations = []
        for i in range(1, 8):  # lung_resection___1 to lung_resection___7
            if complication_row[f'lung_resection___{i}'].values[0] == 1:
                operations.append(operation_mapping[i])

        # Collect complication dates and format as MM/DD/YYYY
        complication_columns = ['complication_1_date', 'complication_2_date', 'complication_3_date',
                                'complication_4_date', 'complication_5_date']
        complication_dates = complication_row[complication_columns].values.flatten()
        complication_dates = [pd.to_datetime(date).strftime('%m/%d/%Y') for date in complication_dates if
                              pd.notna(date)]
        complication_dates_str = ",".join(complication_dates)  # Join dates with comma

        date_of_surgery = pd.to_datetime(complication_row['surgery_date'].values[0])
        day1 = date_of_surgery - timedelta(days=7)  # day1 = 7 days before surgery

        # Define time range: 21 days before day1 for baseline, 91 days after surgery
        baseline_start_date = day1 - timedelta(days=21)
        end_date_filter = date_of_surgery + timedelta(days=91)
        # Read corresponding _processed.csv files
        processed_df_path = os.path.join(processed_data_folder, processed_file)
        # Step 1: Find all files within the time range
        file_date_pairs = []
        for root, dirs, files in os.walk(processed_df_path):
            for file in files:
                if file.endswith(".csv") and "_processed_" in file:
                    file_date = pd.to_datetime(file.split("_processed_")[-1].replace(".csv", ""), errors='coerce')
                    if pd.notna(file_date) and baseline_start_date <= file_date < end_date_filter:
                        file_date_pairs.append((file_date, os.path.join(root, file)))

        # Step 2: Sort by file_date (ascending)
        file_date_pairs.sort()

        if file_date_pairs:
            # Skip the first day
            file_date_pairs = file_date_pairs[1:]

        # Step 3: Read remaining files
        all_dataframes = []
        for file_date, file_path in file_date_pairs:
            df = pd.read_csv(file_path)
            all_dataframes.append(df)

        if not all_dataframes:
            continue
        merged_df = pd.concat(all_dataframes, ignore_index=True)

        merged_df['datetime'] = pd.to_datetime(merged_df['datetime'])
        merged_df['day'] = (merged_df['datetime'].dt.date - date_of_surgery.date()).dt.days
        merged_df['hour'] = merged_df['datetime'].dt.hour
        processed_df = merged_df
        # Generate full time range (minute frequency)
        full_time_range = pd.date_range(start=baseline_start_date, end=end_date_filter, freq='T')  # 'T' = minutes
        # Create DataFrame with full time range
        full_time_df = pd.DataFrame({'datetime': full_time_range})
        full_time_df = full_time_df.drop(full_time_df.index[-1])
        # print('full_time_df', full_time_df)
        # print(baseline_start_date, end_date)

        # Ensure 'datetime' column types are consistent
        full_time_df['datetime'] = pd.to_datetime(full_time_df['datetime'])
        merged_df['datetime'] = pd.to_datetime(merged_df['datetime'])
        # Perform merge
        processed_df = pd.merge(full_time_df, merged_df, on='datetime', how='left')
        # processed_df['Time'] = pd.to_datetime(processed_df['Time'])  # Ensure time column is datetime
        # processed_df.set_index('Time', inplace=True)

        # Set index for day-wise iteration
        filtered_df = processed_df # processed_df[(processed_df.index >= baseline_start_date) & (processed_df.index < end_date)]
        filtered_df.set_index('datetime', inplace=True)

        # Initialize sleep metric lists
        sleep_day_heart_rate = []
        sleep_night_heart_rate = []
        nonsleep_day_heart_rate = []
        nonsleep_night_heart_rate = []
        combined_sleep_heart_rate = []
        combined_nonsleep_heart_rate = []

        # Calculate metrics for each day
        for day in range(98):
            day_start = day1 + timedelta(days=day)
            day_end = day_start + timedelta(days=1)
            # print(filtered_df.index)
            # print(day_start, day_end)
            day_data = filtered_df[(filtered_df.index >= day_start) & (filtered_df.index < day_end)]

            # print(day_data)
            # 1. Sleep ratio during daytime: (light+deep+rem) / all sleep stages
            sleep_day_data_sleep = day_data[(day_data['Sleep_Level'].isin([3, 4, 5]))]['Heart Rate'].dropna()
            sleep_day_data = day_data[(day_data['Sleep_Level'].isin([0, 1, 2, 3, 4, 5, 6]))]['Heart Rate'].dropna()
            if not sleep_day_data.empty:
                sleep_day_heart_rate.append(sleep_day_data_sleep.shape[0] / sleep_day_data.shape[0])
            else:
                sleep_day_heart_rate.append(None)


            # 2. Sleep data availability during nighttime (hour < 9 or >= 21)
            sleep_night_data_sleep = day_data[(day_data['Sleep_Level'].isin([0, 1, 2, 3, 4, 5, 6])) & (day_data.index.hour < 9) | (day_data.index.hour >= 21)][
                'Heart Rate'].dropna()
            sleep_night_data = day_data[(day_data.index.hour < 9) | (day_data.index.hour >= 21)]['Heart Rate'].dropna()
            if not sleep_night_data.empty:
                sleep_night_heart_rate.append(sleep_night_data_sleep.shape[0] / sleep_night_data.shape[0])
            else:
                sleep_night_heart_rate.append(None)

            # 3. Combined sleep (day + night)
            combined_sleep_data_sleep = day_data[day_data['Sleep_Level'].isin([0, 1, 2, 3, 4, 5, 6])]['Heart Rate'].dropna()
            combined_sleep_data = day_data['Heart Rate'].dropna()
            if not combined_sleep_data_sleep.empty:
                combined_sleep_heart_rate.append(combined_sleep_data_sleep.shape[0] / combined_sleep_data.shape[0])
            else:
                combined_sleep_heart_rate.append(None)
            # 4. Nonsleep (no sleep stage) ratio during daytime (9-21h)
            nonsleep_day_data_sleep = day_data[(~day_data['Sleep_Level'].isin([0, 1, 2, 3, 4, 5, 6])) & (day_data.index.hour >= 9) & (day_data.index.hour < 21)][
                'Heart Rate'].dropna()
            nonsleep_day_data = day_data[(day_data.index.hour >= 9) & (day_data.index.hour < 21)][
                'Heart Rate'].dropna()
            # nonsleep_day_data_sleep = day_data[day_data['Sleep_Level'].isin([4])]['Heart Rate'].dropna()
            # nonsleep_day_data = day_data['Heart Rate'].dropna()
            if not nonsleep_day_data.empty:
                nonsleep_day_heart_rate.append(nonsleep_day_data_sleep.shape[0] / nonsleep_day_data.shape[0])
            else:
                nonsleep_day_heart_rate.append(None)

            # 5. REM (Sleep_Level 5) ratio for the day
            # nonsleep_night_data_sleep = day_data[(~day_data['Sleep_Level'].isin([0, 1, 2, 3, 4, 5, 6])) & (day_data.index.hour < 9) | (day_data.index.hour >= 21)][
            #     'Heart Rate'].dropna()
            # nonsleep_night_data = day_data[(day_data.index.hour < 9) | (day_data.index.hour >= 21)][
            #     'Heart Rate'].dropna()
            nonsleep_night_data_sleep = day_data[day_data['Sleep_Level'].isin([5])]['Heart Rate'].dropna()
            nonsleep_night_data = day_data['Heart Rate'].dropna()
            if not nonsleep_night_data.empty:
                nonsleep_night_heart_rate.append(nonsleep_night_data_sleep.shape[0] / nonsleep_night_data.shape[0])
            else:
                nonsleep_night_heart_rate.append(None)

            # 6. Wake (Sleep_Level 6) ratio for the day
            # combined_nonsleep_data_sleep = day_data[~day_data['Sleep_Level'].isin([0, 1, 2, 3, 4, 5, 6])]['Heart Rate'].dropna()
            # combined_nonsleep_data = day_data['Heart Rate'].dropna()
            combined_nonsleep_data_sleep = day_data[day_data['Sleep_Level'].isin([6])]['Heart Rate'].dropna()
            combined_nonsleep_data = day_data['Heart Rate'].dropna()
            if not combined_nonsleep_data.empty:
                combined_nonsleep_heart_rate.append(
                    combined_nonsleep_data_sleep.shape[0] / combined_nonsleep_data.shape[0])
            else:
                combined_nonsleep_heart_rate.append(None)

        # Add row for each operation and save to CSV
        for operation in operations:
            # Append to six different DataFrames
            sleep_day_results = pd.concat([sleep_day_results, pd.DataFrame([[uuid, operation, date_of_surgery, complication_dates_str, day1] + sleep_day_heart_rate],
                                                     columns=['uuid', 'operation_type', 'surgery_date', 'complication_dates', 'day1'] +
                                                             [f'Day_{i + 1}' for i in range(98)])],
                                ignore_index=True)

            sleep_night_results = pd.concat([sleep_night_results, pd.DataFrame([[uuid, operation, date_of_surgery, complication_dates_str, day1] + sleep_night_heart_rate],
                                                       columns=['uuid', 'operation_type', 'surgery_date', 'complication_dates', 'day1'] +
                                                               [f'Day_{i + 1}' for i in range(98)])],
                                ignore_index=True)

            nonsleep_day_results = pd.concat([nonsleep_day_results, pd.DataFrame([[uuid, operation, date_of_surgery, complication_dates_str, day1] + nonsleep_day_heart_rate],
                                                       columns=['uuid', 'operation_type', 'surgery_date', 'complication_dates', 'day1'] +
                                                               [f'Day_{i + 1}' for i in range(98)])],
                                ignore_index=True)

            nonsleep_night_results = pd.concat([nonsleep_night_results, pd.DataFrame([[uuid, operation, date_of_surgery, complication_dates_str, day1] + nonsleep_night_heart_rate],
                                                       columns=['uuid', 'operation_type', 'surgery_date', 'complication_dates', 'day1'] +
                                                               [f'Day_{i + 1}' for i in range(98)])],
                                ignore_index=True)

            combined_sleep_results = pd.concat([combined_sleep_results, pd.DataFrame([[uuid, operation, date_of_surgery, complication_dates_str, day1] + combined_sleep_heart_rate],
                                                       columns=['uuid', 'operation_type', 'surgery_date', 'complication_dates', 'day1'] +
                                                               [f'Day_{i + 1}' for i in range(98)])],
                                ignore_index=True)

            combined_nonsleep_results = pd.concat([combined_nonsleep_results, pd.DataFrame([[uuid, operation, date_of_surgery, complication_dates_str, day1] + combined_nonsleep_heart_rate],
                                                       columns=['uuid', 'operation_type', 'surgery_date', 'complication_dates', 'day1'] +
                                                               [f'Day_{i + 1}' for i in range(98)])],
                                ignore_index=True)

        # Save six results to six CSV files
        sleep_day_results.to_csv(output_files['sleep_day'], index=False)
        sleep_night_results.to_csv(output_files['sleep_night'], index=False)
        nonsleep_day_results.to_csv(output_files['nonsleep_day'], index=False)
        nonsleep_night_results.to_csv(output_files['nonsleep_night'], index=False)
        combined_sleep_results.to_csv(output_files['combined_sleep'], index=False)
        combined_nonsleep_results.to_csv(output_files['combined_nonsleep'], index=False)

    print(f"Results saved to {output_files['sleep_day']}, {output_files['sleep_night']}, {output_files['nonsleep_day']}, {output_files['nonsleep_night']}, {output_files['combined_sleep']}, {output_files['combined_nonsleep']}")

# Main
def main():
    complication_dates_file = "Chart_review_20250616_processed.csv"  # Complication dates file path
    processed_data_folder = "20250505_complete_incomplete/"  # Processed data folder path
    output_files = {
        'sleep_day': 'sleep_day_available.csv',
        'sleep_night': 'sleep_night_available.csv',
        'nonsleep_day': 'nonsleep_day_available.csv',
        'nonsleep_night': 'nonsleep_night_available.csv',
        'combined_sleep': 'combined_sleep_available.csv',
        'combined_nonsleep': 'combined_nonsleep_available.csv'
    }  # Output file paths

    process_sleep_data(complication_dates_file, processed_data_folder, output_files)

if __name__ == "__main__":
    main()
