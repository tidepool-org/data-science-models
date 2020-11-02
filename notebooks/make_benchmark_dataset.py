import os
import pickle
from tidepool_data_science_models.models.icgm_sensor_generator import iCGMSensorGenerator
from tidepool_data_science_models.models.icgm_sensor_generator_functions import create_dataset


batch_size = 3
random_seed = 0

true_bg, results_df = create_dataset(
    kind="sine", N=2880, min_value=40, max_value=400, time_interval=5, oscillations=2, random_seed=random_seed
)


sensor_generator = iCGMSensorGenerator(
    batch_training_size=batch_size,
    use_g6_accuracy_in_loss=False,
    bias_type="percentage_of_value",
    bias_drift_type="random",
    random_seed=random_seed,
    verbose=False,
)

sensor_generator.fit(true_bg["value"].values)
save_path = os.path.join("..", "tests", "benchmark_results_with_new_dict_sensor_properties_2020_11_01.pkl")
with open(save_path, "wb") as f:
    pickle.dump(sensor_generator, f)
