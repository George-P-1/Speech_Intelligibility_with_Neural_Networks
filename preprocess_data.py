# Import libraries
import hydra
from omegaconf import DictConfig
from pathlib import Path
import json
import mystoi
import soundfile as sf
from scipy.signal import resample
import numpy as np
from datetime import datetime
import time

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Format: YYYY-MM-DD_HH-MM-SS

# Get the timestamp for the current run
timestamp = get_timestamp()

# SECTION - Main code here
@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    start_time = time.time()  # Start timing

    # NOTE - Select which data to preprocess
    # DATA_PART = 'Train'; cfg_data = cfg.train_path; cfg_results_file = None
    # DATA_PART = 'Test'; cfg_data = cfg.test_path; cfg_results_file = cfg.test_result_path.result_ref_file
    DATA_PART = 'Train_Independent'; cfg_data = cfg.train_indep_path; cfg_results_file = None
    # DATA_PART = 'Test_Independent'; cfg_data = cfg.test_indep_path;  cfg_results_file = cfg.test_result_path.result_indep_ref_file

    # NOTE -
    PREPROCESSED_DATASET_NAME = "d_matrices_2d_masks_correctness_audiograms"

    # NOTE - Largest d-matrix length in each data part:
    # Train: 277, Test: 263, Train Independent: 277, Test Independent: 263
    global_d_matrix_length = 277    # To ensure consistent input size for both train and test

    # Open training reference JSON file
    try:
        with open(cfg_data.ref_file, 'r') as ref_file:
            ref_json = json.load(ref_file)   # Load the JSON file
            print(f"Loaded {len(ref_json)} samples from {cfg_data.ref_file}\n")
            
            #SECTION - If test type data, then extract correctness from results JSON file
            # Load correctness JSON file (for test data)
            if DATA_PART == 'Test' or DATA_PART == 'Test_Independent':
                with open(cfg_results_file, 'r') as test_results_file:
                    correctness_data = json.load(test_results_file)
                    print(f"Loaded {len(correctness_data)} entries from {cfg_results_file}\n")
                    assert len(ref_json) == len(correctness_data), "Mismatch in number of entries between ref_file and correctness file!"
            #!SECTION - extract correctness from results JSON file

            #SECTION - Load listener audiogram JSON file
            with open(cfg_data.listeners_file, 'r') as listener_file:
                listener_data = json.load(listener_file)
            print(f"Loaded {len(listener_data)} listener entries from {cfg_data.listeners_file}\n")
            #!SECTION - Load listener audiogram JSON file

             # List to store d-matrix
            d_matrices = []
            correctness_scores = []  # Store correctness values
            audiograms = []         # Placeholder for audiogram data

            # SECTION - Iterate over each sample in the JSON file
            for scene_index, sample in enumerate(ref_json):
                # Path of audio files to open, spin and target (HA_Output and target_anechoic)
                spin_file_path = Path(cfg_data.spin_folder) / f"{ref_json[scene_index]['signal']}.wav"
                target_file_path = Path(cfg_data.scenes_folder) / f"{ref_json[scene_index]['scene']}_target_anechoic.wav"
                # Opening audio files using soundfile
                spin, spin_sr = sf.read(spin_file_path)
                target, target_sr = sf.read(target_file_path)
                # Resampling
                new_sr = cfg.sample_rate
                # REVIEW - Can use scipy (resample or decimate functions) or librosa library. Some issue with scipy using only frequency domain or something like that.
                spin = resample(spin, int(len(spin) * new_sr / spin_sr))   # current_no_of_samples / current_sampling_rate is the duration of audio signal
                target = resample(target, int(len(target) * new_sr / target_sr))
                # Padding to make both signals of same length in case of different lengths - Pad the shorter signal
                if len(spin) < len(target): # pad spin
                    spin = np.pad(spin, (0, len(target) - len(spin)))
                elif len(target) < len(spin):
                    raise Exception("Target signal is shorter than spin signal. This was not EXPECTED.")
                # In CPC1 data, in the case of shorter signals, spin is always the shorter one. So no need to check whether to pad target.
                # Convert to mono along the time axis - mean of two channels
                spin = spin.mean(axis=1)
                target = target.mean(axis=1)
                
                # Compute d-matrix from mystoi 
                d_matrix = mystoi.compute_stoi(target, spin, new_sr, return_d_matrix=True)
                # Shape is (_, 15, 30) where _ is the # of frames which is dependent on the length of the audio signal
                # and 15 is the number of frequency bands and 30 is the number of time frames.
                # and _ is global_d_matrix_length which is 277
            
                # Convert d-matrix to 2d by summing over the time axis of 30 elements
                d_matrix = np.sum(d_matrix, axis=2)
                # So now shape is (277, 15)

                # Store d-matrix to array
                d_matrices.append(d_matrix)

                # Store correctness value (depending on the data part)
                if DATA_PART == 'Train' or DATA_PART == 'Train_Independent':
                    correctness_scores.append(sample["correctness"])
                else:
                    # Get correctness value (directly from the corresponding entry)
                    correctness_scores.append(correctness_data[scene_index]["correctness"])

                # Extract and store listener audiogram data
                # NOTE - Left ear then right ear data is stored into a vector of length 16
                listener_id = sample["listener"]
                if listener_id in listener_data:
                    left_ear = listener_data[listener_id]["audiogram_levels_l"]
                    right_ear = listener_data[listener_id]["audiogram_levels_r"]
                    audiogram_vector = np.array(left_ear + right_ear, dtype=np.float32)  # Concatenate both ears
                    audiograms.append(audiogram_vector)
                else:
                    raise Exception(f"Listener ID {listener_id} not found in listener file!")

                # Debug print every 100 samples
                if scene_index % 200 == 0:
                    print(f"Processed {scene_index}/{len(ref_json)} samples...")
            #!SECTION - Iterate over each sample in the JSON file

            # SECTION - Create Masks - Track original sequence lengths before padding
            original_lengths = [np.array(d_matrix).shape[0] for d_matrix in d_matrices]

            # SECTION - Pad d-matrices to the same length
            d_matrices_padded = np.array([
                np.pad(d_matrix, ((0, global_d_matrix_length - np.array(d_matrix).shape[0]), (0, 0)), mode='constant')
                for d_matrix in d_matrices
            ], dtype=np.float32)
            print(f"Shape of d_matrices_padded: {d_matrices_padded.shape}")
            #!SECTION - Pad d-matrices to the same length

            # Create a mask where only padded regions are 0
            masks = np.array([
                np.pad(np.ones((length, 15), dtype=np.float32),  # Mask for original data
                    ((0, d_matrices_padded.shape[1] - length), (0, 0)),  # Pad zeros for added regions
                    mode='constant', constant_values=0)
                for length in original_lengths
            ])
            print(f"Sample Mask Shape: {masks.shape}")  # Should match d_matrices_padded shape
            #!SECTION - Create Masks

            # Convert correctness to a NumPy array
            correctness_array = np.array(correctness_scores, dtype=np.float32)
            print(f"Shape of correctness_array: {correctness_array.shape}")

            # Convert audiogram data to a NumPy array
            audiograms_array = np.array(audiograms, dtype=np.float32)
            print(f"Shape of audiograms_array: {audiograms_array.shape}")

            # SECTION - Save to a compressed NumPy file (.npz format)
            save_path = f"{PREPROCESSED_DATASET_NAME}_{DATA_PART}_{timestamp}.npz"
            np.savez_compressed(save_path, d_matrices=d_matrices_padded, masks=masks, correctness=correctness_array, audiograms=audiograms_array)
            print(f"Preprocessed data saved to {save_path}")
            #!SECTION - Save to a compressed NumPy file (.npz format)

        ref_file.close()

        # Record end time and compute duration
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nPreprocessing completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes).")
    except FileNotFoundError:
        print(f'File not found: {cfg_data.ref_file}')
        return None
    finally:
        print(f'Finished processing JSON file.')
# !SECTION - End of main code

if __name__ == '__main__':
    main()