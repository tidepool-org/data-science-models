import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tidepool_data_science_models.models.icgm_sensor_generator import iCGMSensorGenerator
from tidepool_data_science_models.models.icgm_sensor_generator_functions import create_dataset, generate_icgm_sensors
import plotly.express as px
import json

# To do a check with a "ideal sensor," use the following parameters and hardcode the noise
# coefficient and bias as zero (in icgm_sensor_generator_functions.py) and change the delay to 0 in (icgm_sensor_generator.py).
bias_drift_type = "none"

batch_training_size = 30

# Set file locations and save paths and make directories that don't exist
data_path = os.path.join("..", ".data", "processed")


scenario_folder_name = "icgm-sensitivity-analysis-scenarios-2020-07-10-nogit"
scenario_files_path = os.path.join(data_path, scenario_folder_name)


save_percent_pass_path = os.path.join(data_path, "icgm_special_controls_checks_batch_size_"+str(batch_training_size)+"_mean_shift_ideal_sensor_only_scenarios_with_ok_rate_not_passing_sc")
save_accuracy_tables_path = os.path.join(save_percent_pass_path, "accuracy-tables-" + scenario_folder_name)
save_rates_change_tables_path =  os.path.join(data_path, "rates-change-tables-" + scenario_folder_name)

for path in [save_percent_pass_path, save_accuracy_tables_path, save_rates_change_tables_path]:
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
if batch_training_size >= 30:
    scenarios_not_passing_df = pd.read_csv(os.path.join("..", ".data", "processed",
                                                "scenarios-not-passing-sc-summary_rates_change_icgm-sensitivity-analysis-scenarios-2020-07-10-nogit.csv"))
    files = scenarios_not_passing_df[scenarios_not_passing_df["invalid_rate_change_flag"] == 0]["training_scenario_filename"].unique().tolist()
    #batch_size_30_df = pd.read_csv(os.path.join("..", ".data", "processed", "icgm_special_controls_checks_batch_size_30", "percent_pass_per_scenario_icgm-sensitivity-analysis-scenarios-2020-07-10-nogit.csv"))
    #files = batch_size_30_df[batch_size_30_df["percent_pass"] < 100]["training_scenario_filename"].tolist()
    print(len(files))
    print(files)
else:
    files = os.listdir(scenario_files_path)

files.sort()

#For testing:
#files = ['train_e0d4c09a2c38269fa27362dc752d721a6fb820a6f0448d67dfc80cad6aee71bc.csv_condition8.csv']
#files = ['train_160102103a951851ef4f652ec2b2051b325148d5ca0736f6a17c5ad927b2952d.csv_condition5.csv'] #, 'train_e0d4c09a2c38269fa27362dc752d721a6fb820a6f0448d67dfc80cad6aee71bc.csv_condition8.csv']
#files = []

for i, filename in enumerate(files):
    if filename.endswith(".csv"):
        print(i, filename)

        df = pd.read_csv(os.path.join(scenario_files_path, filename))

        print(df.iloc[50, 2:].astype(float).values)

        true_bg_trace = df.iloc[50, 2:].astype(float).apply(lambda x: x-20).values

        print(true_bg_trace)

        #px.scatter(x=range(0, 2880), y=true_bg_trace).show()

        # Get batch sensors
        sensor_generator = iCGMSensorGenerator(
            batch_training_size=batch_training_size,
            max_number_of_spurious_events_per_10_days=0,
            bias_drift_type=bias_drift_type,
            #use_g6_accuracy_in_loss=True,
            verbose=True
        )

        sensor_generator.fit(true_bg_trace)

        print(sensor_generator.batch_sensor_brute_search_results)

        # Determine percent pass and add to dataframe of all files
        percent_pass = sensor_generator.percent_pass
        print("Percent Pass: " + str(percent_pass))
        percent_pass_df = percent_pass_df.append({'training_scenario_filename': filename, 'percent_pass': percent_pass}, ignore_index=True)

        # Get, print and save the icgm special controls data frame
        print("Accuracy Table")
        accuracy_table_df = pd.DataFrame(sensor_generator.icgm_special_controls_accuracy_table)
        print(accuracy_table_df)
        accuracy_table_df.to_csv(path_or_buf=os.path.join(save_accuracy_tables_path, "accuracy_table_"+filename), index=True)

        # Add icgmSensorResults to overall table
        values_to_add = {'training_scenario_filename': filename}
        for special_control in accuracy_table_df.index:
            values_to_add[special_control] = accuracy_table_df['icgmSensorResults'][special_control]
        icgm_sensor_results_df = icgm_sensor_results_df.append(values_to_add, ignore_index=True)

        # If any of the special controls don't pass, add this to dataframe of special controls that didn't pass
        if percent_pass != 100:
            accuracy_table_subset_df = accuracy_table_df[(accuracy_table_df['icgmSensorResults'] < accuracy_table_df['icgmSpecialControls']) | accuracy_table_df['icgmSensorResults'].isnull()]
            for special_control in accuracy_table_subset_df.index:
                special_controls_not_passed_df = special_controls_not_passed_df.append({
                    'training_scenario_filename': filename,
                    'percent_pass_overall': percent_pass,
                    'special_control_not_passed': special_control,
                    'icgmSpecialControls':  accuracy_table_subset_df['icgmSpecialControls'][special_control],
                    'nPairs': accuracy_table_subset_df['nPairs'][special_control],
                    'nPairs_divided_by_batch_size': accuracy_table_subset_df['nPairs'][special_control]/batch_training_size,
                    'icgmSensorResults': accuracy_table_subset_df['icgmSensorResults'][special_control]}, ignore_index=True)


# Save the overall percent pass dataframe and the special controls not passed dataframe
percent_pass_df.to_csv(path_or_buf=os.path.join(save_percent_pass_path, "percent_pass_per_scenario_"+scenario_folder_name+".csv"), index=True)

special_controls_not_passed_df.to_csv(path_or_buf=os.path.join(save_percent_pass_path, "special_controls_not_passed_"+scenario_folder_name+".csv"), index=False)

icgm_sensor_results_df.to_csv(path_or_buf=os.path.join(save_percent_pass_path, "icgm_sensor_results_all_"+scenario_folder_name+".csv"), index=False)

#sensor_generator.g6_table


scenario_files = ["train_c69a10e993c0452f04f80f5d3dd58ebac53b1448be3ba8cb2f2a1b9d341051bf.csv_condition1.csv"]#,
#"train_e0e0c1b4e8c754ee772f5226162800723cfedd98eff97ae50c8e58b037071610.csv_condition1.csv",
#"train_4023423e51054602b7a44930e9e8d8eb9332e62e8135747781b8be96915c4f67.csv_condition8.csv",
#"train_ed0d568a49546d64a23613a87ba5bc766477fe187781ef90b779fd5695c85c52.csv_condition5.csv",
#"train_376036c828808b881c9ba4e01ff31720e5ab269f73b8886a146f6028458c905f.csv_condition6.csv",
#"train_c4995e26671bb199994aeef0c0a8bf5239d6a75f43cdc72cba61d03e7dcbfc80.csv_condition6.csv"]

for filename in scenario_files:
    if filename.endswith(".csv"):
        print(filename)
        df = pd.read_csv(os.path.join(scenario_files_path, filename))
        time = df.iloc[40, 2:].astype(float).values
        true_bg_trace = df.iloc[50, 2:].astype(float).values
        d = {'true_bg': true_bg_trace[2:], 'cgm': true_bg_trace[:-2]}
        traces_df = pd.DataFrame(data=d)
        print(traces_df)

        traces_df.to_csv(path_or_buf=os.path.join(data_path, "test-"+filename), index=False)

        plt.plot(traces_df.index.values, traces_df["true_bg"], 'r--', traces_df.index.values, traces_df["cgm"], 'g--')
        #plt.legend()
        plt.show()

'''
# Determine if number of scenario files where there are physiologically impossible rates of change
scenario_files = os.listdir(scenario_files_path)

#If only want to check the scenarios that don't pass the special controls
#batch_size_30_df = pd.read_csv(os.path.join("..", ".data", "processed", "icgm_special_controls_checks_batch_size_30", "percent_pass_per_scenario_icgm-sensitivity-analysis-scenarios-2020-07-10-nogit.csv"))
#scenario_files = batch_size_30_df[batch_size_30_df["percent_pass"] < 100]["training_scenario_filename"].tolist()

# Data frame to store
rates_change_df = pd.DataFrame(columns=['training_scenario_filename', 'max_rate_change', 'min_rate_change', 'invalid_rate_change_flag'])

for filename in scenario_files[0:0]:
    if filename.endswith(".csv"):
        print(filename)
        df = pd.read_csv(os.path.join(scenario_files_path, filename))
        true_bg_trace = df.iloc[50, 2:].astype(float).values

        d = {'true_bg': true_bg_trace}
        df = pd.DataFrame(data=d)

        df['true_bg_diff'] = df['true_bg'].diff()

        df['rate_per_minute'] = df['true_bg_diff']/5

        #print(df)

        df.to_csv(path_or_buf=os.path.join(save_rates_change_tables_path, "rates_change_"+filename+".csv"), index=False)

        invalid_rate_change_flag = 0

        if any(np.where(abs(df['rate_per_minute']) > 5, True, False)):
            print("rates out of range")
            invalid_rate_change_flag = 1

        values_to_add = {'training_scenario_filename': filename, 'max_rate_change': np.nanmax(df['rate_per_minute']), 'min_rate_change': np.nanmin(df['rate_per_minute']), 'invalid_rate_change_flag': invalid_rate_change_flag}

        rates_change_df = rates_change_df.append(values_to_add, ignore_index=True)


print(rates_change_df)

rates_change_df.to_csv(path_or_buf=os.path.join(data_path, "all_scenarios_summary_rates_change_"+scenario_folder_name+".csv"), index=False)







# Specific scenerio/control testing
# For each of the main special controls testing, loading the file, pull out the bg trace and see how many of the idfferences
# are, for example, +/- 15

def find_corresponding_vp(filename):
    patient_characteristics_path = "/Users/anneevered/Desktop/Tidepool 2020/Tidepool Repositories/data-science--explore--risk-sim-figures/data/processed/icgm-sensitivity-analysis-results-2020-09-19-nogit"
    for vp in range(100):
        f = open(os.path.join(patient_characteristics_path, "vp" + str(vp) + ".json"), "r")
        json_data = json.loads(f.read())
        patient_characteristics_df = pd.DataFrame(json_data, index=['i', ])
        if patient_characteristics_df["patient_scenario_filename"].iloc[0].split("/")[-1].split("_")[1] == filename.split("_")[1]:
            return "vp" + str(vp)# Specific scenerio/control testing
# For each of the main special controls testing, loading the file, pull out the bg trace and see how many of the idfferences
# are, for example, +/- 15

def find_corresponding_vp(filename):
    patient_characteristics_path = "/Users/anneevered/Desktop/Tidepool 2020/Tidepool Repositories/data-science--explore--risk-sim-figures/data/processed/icgm-sensitivity-analysis-results-2020-09-19-nogit"
    for vp in range(100):
        f = open(os.path.join(patient_characteristics_path, "vp" + str(vp) + ".json"), "r")
        json_data = json.loads(f.read())
        patient_characteristics_df = pd.DataFrame(json_data, index=['i', ])
        if patient_characteristics_df["patient_scenario_filename"].iloc[0].split("/")[-1].split("_")[1] == filename.split("_")[1]:
            return "vp" + str(vp)
            
path = os.path.join(data_path, "icgm_special_controls_checks_batch_size_30_ideal_sensor_with_delay")

sc_not_passed_df = pd.read_csv(os.path.join(path, "special_controls_not_passed_"+scenario_folder_name+".csv"))

files_to_check = sc_not_passed_df[sc_not_passed_df["nPairs"] > 0][["training_scenario_filename", "special_control_not_passed"]]

for filename, sc_not_passed in zip(files_to_check["training_scenario_filename"], files_to_check["special_control_not_passed"]):
    df = pd.read_csv(os.path.join(scenario_files_path, filename))
    true_bg_trace = df.iloc[50, 2:].astype(float).values

    print(filename)
    print(find_corresponding_vp(filename))

    print("Special Control Not Passed: "+ sc_not_passed)

    d = {'true_bg': true_bg_trace[:-2], 'cgm': true_bg_trace[2:]}
    traces_df = pd.DataFrame(data=d)
    traces_df["difference"]=traces_df['true_bg']-traces_df['cgm']
    traces_df["percent_difference"] = 100*traces_df["difference"]/traces_df['cgm']

    if sc_not_passed == "A":
        n_in_range = len(traces_df[traces_df['true_bg'] < 70])
        n_meeting_criteria_and_in_range = len(traces_df[(traces_df['true_bg'] < 70) & (traces_df['difference'].abs() < 15)])

        print("N in Range: " + str(n_in_range))
        print("N Meeting Criteria and in Range: " + str(n_meeting_criteria_and_in_range))
        if n_in_range != 0:
            print("Percent: " + str(100*n_meeting_criteria_and_in_range/n_in_range))
        print("Special Controls Percent: 85%")

    elif sc_not_passed == "B":

        n_in_range = len(traces_df[(traces_df['true_bg'] > 70) & (traces_df['true_bg'] < 180)])
        n_meeting_criteria_and_in_range = len(traces_df[(traces_df['true_bg'] > 70) & (traces_df['true_bg'] < 180) & (traces_df['percent_difference'].abs() < 15)])

        print("N in Range: " + str(n_in_range))
        print("N Meeting Criteria and in Range: " + str(n_meeting_criteria_and_in_range))
        if n_in_range != 0:
            print("Percent: " + str(100*n_meeting_criteria_and_in_range/n_in_range))
        print("Special Controls Percent: 70%")

    elif sc_not_passed == "C":

        n_in_range = len(traces_df[(traces_df['true_bg'] > 180)])
        n_meeting_criteria_and_in_range = len(traces_df[(traces_df['true_bg'] > 180) & (traces_df['percent_difference'].abs() < 15)])

        print("N in Range: " + str(n_in_range))
        print("N Meeting Criteria and in Range: " + str(n_meeting_criteria_and_in_range))
        if n_in_range != 0:
            print("Percent: " + str(100*n_meeting_criteria_and_in_range/n_in_range))
        print("Special Controls Percent: 80%")

    elif sc_not_passed == "D":

        n_in_range = len(traces_df[traces_df['true_bg'] < 70])
        n_meeting_criteria_and_in_range = len(traces_df[(traces_df['true_bg'] < 70) & (traces_df['difference'].abs() < 40)])

        print("N in Range: " + str(n_in_range))
        print("N Meeting Criteria and in Range: " + str(n_meeting_criteria_and_in_range))
        if n_in_range != 0:
            print("Percent: " + str(100*n_meeting_criteria_and_in_range/n_in_range))

        print("Special Controls Percent: 98%")

    elif sc_not_passed == "E":
        n_in_range = len(traces_df[(traces_df['true_bg'] > 70) & (traces_df['true_bg'] < 180)])
        n_meeting_criteria_and_in_range = len(traces_df[(traces_df['true_bg'] > 70) & (traces_df['true_bg'] < 180) & (traces_df['percent_difference'].abs() < 40)])

        print("N in Range: " + str(n_in_range))
        print("N Meeting Criteria and in Range: " + str(n_meeting_criteria_and_in_range))
        if n_in_range != 0:
            print("Percent: " + str(100*n_meeting_criteria_and_in_range/n_in_range))
        print("Special Controls Percent: 99%")

    elif sc_not_passed == "F":
        n_in_range = len(traces_df[(traces_df['true_bg'] > 180)])
        n_meeting_criteria_and_in_range = len(traces_df[(traces_df['true_bg'] > 180) & (traces_df['percent_difference'].abs() < 40)])

        print("N in Range: " + str(n_in_range))
        print("N Meeting Criteria and in Range: " + str(n_meeting_criteria_and_in_range))
        if n_in_range != 0:
            print("Percent: " + str(100*n_meeting_criteria_and_in_range/n_in_range))
        print("Special Controls Percent: 99%")

    else:
        print("Special Control Not Found")


#May want to also write some code that will take what the filename is and get the corresponding
#virtual patient so can inspect some of those

'''