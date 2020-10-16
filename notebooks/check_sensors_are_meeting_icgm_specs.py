import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tidepool_data_science_models.models.icgm_sensor_generator import iCGMSensorGenerator
from tidepool_data_science_models.models.icgm_sensor_generator_functions import create_dataset, generate_icgm_sensors
import plotly.express as px

# Set file locations and save paths and make directories that don't exist
data_path = os.path.join("..", ".data", "processed")

scenario_folder_name = "icgm-sensitivity-analysis-scenarios-2020-07-10-nogit"
scenario_files_path = os.path.join(data_path, scenario_folder_name)

save_percent_pass_path = os.path.join(data_path, "icgm_special_controls_checks")
save_accuracy_tables_path = os.path.join(save_percent_pass_path, "accuracy-tables-" + scenario_folder_name)

for path in [save_percent_pass_path, save_accuracy_tables_path]:
    if not os.path.exists(path):
        print("making directory " + path + "...")
        os.makedirs(path)

# Data frame to store percent pass results in
percent_pass_df = pd.DataFrame(columns=['training_scenario_filename', 'percent_pass'])

special_controls_not_passed_df = pd.DataFrame(columns=['training_scenario_filename', 'percent_pass_overall',
                                                       'special_control_not_passed', 'icgmSpecialControls',
                                                       'nPairs', 'icgmSensorResults'])

icgm_sensor_results_df = pd.DataFrame(columns=['training_scenario_filename', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'])

# Iterate through each scenario file, get batch sensors, and print/save desired output metrics
for i, filename in enumerate(os.listdir(scenario_files_path)[0:10]):
    print(i, filename)

    df = pd.read_csv(os.path.join(scenario_files_path, filename))

    true_bg_trace = df.iloc[50, 2:].astype(float).values
    #px.scatter(x=range(0, 2880), y=true_bg_trace).show()

    # Get batch sensors
    batch_training_size = 30
    sensor_generator = iCGMSensorGenerator(
        batch_training_size=batch_training_size,
        max_number_of_spurious_events_per_10_days=0,
        # use_g6_accuracy_in_loss=True,
        verbose=True
    )

    sensor_generator.fit(true_bg_trace)

    # Determine percent pass and add to dataframe of all files
    percent_pass = sensor_generator.percent_pass
    print("Percent Pass: " + str(percent_pass))
    percent_pass_df = percent_pass_df.append({'training_scenario_filename': filename, 'percent_pass': percent_pass}, ignore_index=True)

    # Get, print and save the icgm special controls data frame
    print("Accuracy Table")
    accuracy_table_df = pd.DataFrame(sensor_generator.icgm_special_controls_accuracy_table)
    print(accuracy_table_df)
    accuracy_table_df.to_csv(path_or_buf=os.path.join(save_accuracy_tables_path, "accuracy_table_"+filename+".csv"), index=True)

    # Add icgmSensorResults to overall table
    values_to_add = {'training_scenario_filename': filename}
    for special_control in accuracy_table_df.index:
        values_to_add[special_control] = accuracy_table_df['icgmSensorResults'][special_control]
    icgm_sensor_results_df = icgm_sensor_results_df.append({}, ignore_index=True)

    # If any of the special controls don't pass, add this to dataframe of special controls that didn't pass
    if percent_pass != 100:
        accuracy_table_df = accuracy_table_df.reset_index(name="special_control")
        accuracy_table_subset_df = accuracy_table_df[accuracy_table_df['icgmSensorResults'] < accuracy_table_df['icgmSpecialControls']]
        for ind in accuracy_table_subset_df.index:

            special_controls_not_passed_df = special_controls_not_passed_df.append({
                'training_scenario_filename': filename,
                'percent_pass_overall': percent_pass,
                'special_control_not_passed': accuracy_table_subset_df['special_control'][ind],
                'icgmSpecialControls':  accuracy_table_subset_df['icgmSpecialControls'][ind],
                'npairs': accuracy_table_subset_df['npairs'][ind],
                'icgmSensorResults': accuracy_table_subset_df['icgmSensorResults'][ind]}, ignore_index=True)


# Save the overall percent pass dataframe and the special controls not passed dataframe
percent_pass_df.to_csv(path_or_buf=os.path.join(save_percent_pass_path, "percent_pass_"+scenario_folder_name+".csv"), index=True)

special_controls_not_passed_df.to_csv(path_or_buf=os.path.join(save_percent_pass_path, "special_controls_not_passed_"+scenario_folder_name+".csv"), index=False)

#sensor_generator.g6_table
