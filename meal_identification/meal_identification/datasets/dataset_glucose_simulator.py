import pandas as pd
import os
from simglucose.simulation.user_interface import simulate
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.controller.basal_bolus_ctrller import BBController
from meal_identification.datasets.dataset_operations import get_root_dir
from meal_identification.datasets.dataset_data_obfuscator import start as obfuscate
from datetime import datetime, timedelta
import random
import matplotlib

# Set the backend to Agg to avoid displaying plots. I think the plot is blocking the thread.
matplotlib.use('Agg') 


def process_simulated_data(df):
    """
    Process individual patient's glucose data into project-specific format.
    CHO -> food_g
    CGM -> bgl
    BG -> bgl_real
    Time -> date

    Parameters
    ----------
    df (pd.DataFrame): Input DataFrame with simulation data

    Returns
    -------
    pd.DataFrame: Processed DataFrame with project-specific format
    """
    # Create a copy to avoid modifying the original
    processed_df = df.copy()

    # Create two separate dataframes for BG and CGM readings
    processed_df = processed_df[['Time', 'BG', 'CGM', 'CHO']].copy()

    # Add required columns
    processed_df['msg_type'] = ''
    processed_df['dose_units'] = ''

    # Map CGM to bgl column, Time to date and BG to bgl_real for reference only
    processed_df = processed_df.rename(columns={'CGM': 'bgl', 'BG': 'bgl_real', 'Time': 'date'})

    processed_df.loc[processed_df['CHO'] > 0, 'msg_type'] = 'ANNOUNCE_MEAL'

    # Rename CHO to food_g where it's > 0
    processed_df.loc[processed_df['CHO'] > 0, 'food_g'] = processed_df.loc[processed_df['CHO'] > 0, 'CHO']

    # Drop the CHO column
    processed_df = processed_df.drop(columns=['CHO'])

    # Truncate data to save some space
    processed_df['bgl_real'] = processed_df['bgl_real'].round(2)
    processed_df['bgl'] = processed_df['bgl'].round(2)
    processed_df['food_g'] = processed_df['food_g'].round(2)
    processed_df['date'] = pd.to_datetime(processed_df['date']).dt.strftime('%Y-%m-%d %H:%M')

    return processed_df


def run_glucose_simulation(
        start_time=None,
        simulation_days=7,
        patient_names=None,
        seeds=None,
        cgm_name="Dexcom",
        insulin_pump_name="Cozmo",
        parallel=True,
        data_dir=None
):
    if seeds and len(patient_names) != len(seeds):
        raise ValueError(
            f"Length mismatch: patient_names has {len(patient_names)} elements while seeds has {len(seeds)} elements. Both lists must have the same length.")
    if start_time is None:
        start_time = pd.Timestamp('2024-01-01 00:00:00')
    if patient_names is None:
        patient_names = ['adult#001']

    # Set up result directory
    result_dir = os.path.join(data_dir, 'sim')
    os.makedirs(result_dir, exist_ok=True)

    # Create a controller
    controller = BBController()

    # Set up simulation time
    sim_time = pd.Timedelta(days=simulation_days)

    # Generate a random seed for each patient for a better outcome
    # Tradeoff is that we can no longer parallelize the simulation process
    # but this is not intended to be run very regularly
    rand_seeds = []
    for idx, patient in enumerate(patient_names):
        if seeds is None:
            seed = random.randint(1, 1000)
            # Keep track of random seed for each patient
            rand_seeds.append(seed)
        else:
            # Use provided seeds for reproducibility
            seed = seeds[patient]

        scenario = RandomScenario(
            start_time=start_time,
            seed=seed
        )

        # Run simulation
        simulate(
            sim_time=sim_time,
            scenario=scenario,
            controller=controller,
            start_time=start_time,
            save_path=result_dir,
            cgm_name=cgm_name,
            cgm_seed=seed,
            insulin_pump_name=insulin_pump_name,
            animate=False,
            parallel=False,
            patient_names=[patient],
        )

        print(f"Simulation for {patient} completed!")

    if rand_seeds:
        print("Random seeds: ", rand_seeds)

    # Remove side products from the simulation
    for file in os.listdir(result_dir):
        if 'CVGA' in file or 'risk_trace' in file or 'performance' in file or file.endswith('.png'):
            file_path = os.path.join(result_dir, file)
            os.remove(file_path)

    return result_dir


def process_sim_data(
        simulation_days,
        naming,
        data_dir=None,
):
    """
    Process all patient CSV files in the data_dir/sim to data_dir/inter

    Returns:
    dict: Dictionary with patient IDs as keys and processed DataFrames as values
    """
    # Get the project root and construct sim directory path
    if data_dir is None:
        raise ValueError("data_dir is required")
    sim_dir = os.path.join(data_dir, 'sim')
    processed_dir = os.path.join(data_dir, 'inter')

    # Convert to Path object for easier handling
    csv_files = [f for f in os.listdir(sim_dir) if f.endswith('.csv')]

    # Dictionary to store processed data for each patient
    os.makedirs(processed_dir, exist_ok=True)

    for idx, file in enumerate(csv_files):
        # Skip CVGA_stats.csv and risk_trace.csv
        if ('CVGA' in file) or ('risk_trace' in file) or ('performance' in file):
            continue

        file_path = os.path.join(sim_dir, file)
        try:
            df = pd.read_csv(file_path)

            # Convert Time column to datetime
            df['date'] = pd.to_datetime(df['Time'])

            # Process the data
            processed_df = process_simulated_data(df)

            # Add id for each patient
            processed_df['id'] = idx

            # Create new filename (first 3 + last 3 characters before .csv)
            base_name = file.replace('.csv', '')
            short_name = f"{base_name[:3]}{base_name[-3:]}"
            now = datetime.today()
            to = now + timedelta(days=simulation_days)
            start_date = now.strftime('%Y-%m-%d')
            end_date = to.strftime('%Y-%m-%d')
            file = f"{short_name}_{naming['cgm_name']}_{naming['insulin_pump_name']}_{start_date}_{end_date}.csv"

            # Save the files
            output_file = os.path.join(processed_dir, file)
            processed_df.to_csv(output_file)
            print(f"Successfully processed and saved {file}")

        except Exception as e:
            print(f"Error processing {file}: {str(e)}")


def generate_simulated_data(
        start_time=None,
        simulation_days=7,
        patient_names=None,
        seeds=None,
        cgm_name="Dexcom",
        insulin_pump_name="Cozmo",
        parallel=True,
        data_dir=None
):
    """
    Run a glucose simulation with specified parameters and output to data/raw.
    General data flow:
        1. Generate simulate data to `data/sim`
        2. Process data in `data/sim` to `data/raw`

    Parameters
    ----------
    start_time (pd.Timestamp, optional): Start time for simulation. Defaults to '2024-01-01 00:00:00'.
    simulation_days (int, optional): Duration of simulation in days. Defaults to 7.
    cgm_name (str, optional): Name of the cgm device.
         - "Dexcom" | "GuardianRT" | "Navigator". Defaults to "Dexcom".
    insulin_pump_name (str, optional): Name of the insulin pump device.
         - "Cozmo" | "Insulet". Defaults to "Cozmo".
    parallel (bool, optional): Whether to run simulations in parallel. Defaults to True.
    patient_names (list, optional): List of patient IDs to simulate.
         - patient_names can be from adult#001 ~ adult#010, adolescent#001 ~ adolescent#010 and child#001 ~ child#010. Default to ["adult#001"].

    Returns
    -------
    str: Path to the directory containing simulation results.
    """
    if data_dir is None:
        raise ValueError("data_dir is required")
    
    run_glucose_simulation(
        start_time=start_time,
        simulation_days=simulation_days,
        patient_names=patient_names,
        seeds=seeds,
        cgm_name=cgm_name,
        insulin_pump_name=insulin_pump_name,
        parallel=parallel,
        data_dir=data_dir
    )
    process_sim_data(
        simulation_days=simulation_days,
        naming={'cgm_name': cgm_name, 'insulin_pump_name': insulin_pump_name},
        data_dir=data_dir
    )


if __name__ == '__main__':
    # Example usage
    default_patient_names = ['adult#001', 'adult#003']
    project_root = get_root_dir()
    start_time = datetime.now().isoformat()
    data_dir = os.path.join(project_root, 'meal_identification', 'data', 'simglucose', start_time)

    # Run the simulation and save the data to data_dir/sim and process it to data_dir/inter
    generate_simulated_data(
        patient_names=default_patient_names,
        data_dir=data_dir
    )

    # Obfuscate the data in data_dir/inter and save it to data_dir/obfuscated
    obfuscate(
        from_dir=os.path.join(data_dir, 'inter'),
        to_dir=os.path.join(data_dir, 'obfuscated')
    )

