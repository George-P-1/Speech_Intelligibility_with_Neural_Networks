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

# Get the timestamp for the current run
def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Format: YYYY-MM-DD_HH-MM-SS

timestamp = get_timestamp()

# SECTION - Main code here
@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    # Select which data to preprocess
    # data_part = 'Train'; cfg_data = cfg.train_path
    # data_part = 'Test'; cfg_data = cfg.test_path
    # data_part = 'Train_Independent'; cfg_data = cfg.train_indep_path
    data_part = 'Test_Independent'; cfg_data = cfg.test_indep_path

    # NOTE - Largest d-matrix length in each data part:
    # Train: 277, Test: 263, Train Independent: 277, Test Independent: 263
    global_d_matrix_length = 277

    # Open training reference JSON file
    try:
        with open(cfg_data.ref_file, 'r') as ref_file:
            ref_json = json.load(ref_file)   # Load the JSON file
            print(f"Loaded {len(ref_json)} samples from {cfg_data.ref_file}\n")
            
             # List to store d-matrix
            d_matrices = []
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
            
                # Store d-matrix to array
                d_matrices.append(d_matrix)

                # TODO - Store audiogram data

                # Debug print every 100 samples
                if scene_index % 200 == 0:
                    print(f"Processed {scene_index}/{len(ref_json)} samples...")
            #!SECTION - Iterate over each sample in the JSON file

            # SECTION - Pad d-matrices to the same length
            d_matrices_padded = np.array([
                np.pad(d_matrix, ((0, global_d_matrix_length - d_matrix.shape[0]), (0, 0), (0, 0)), mode='constant')
                for d_matrix in d_matrices
            ], dtype=np.float32)
            #!SECTION - Pad d-matrices to the same length

            print(f"Shape of d_matrices_padded: {d_matrices_padded.shape}")

            # SECTION - Save to a compressed NumPy file (.npz format)
            save_path = f"d_matrices_{data_part}_{timestamp}.npz"
            np.savez_compressed(save_path, d_matrices=d_matrices_padded)
            # np.savez_compressed(save_path, d_matrices=d_matrices_padded, audiograms=audiograms) # For saving audiograms too
            print(f"Preprocessed data saved to {save_path}")
            #!SECTION - Save to a compressed NumPy file (.npz format)

        ref_file.close()
    except FileNotFoundError:
        print(f'File not found: {cfg_data.ref_file}')
        return None
    finally:
        print(f'Finished processing JSON file.')
# !SECTION - End of main code

if __name__ == '__main__':
    main()