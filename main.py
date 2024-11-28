# Import libraries
import hydra
from omegaconf import DictConfig
from pathlib import Path
import json
import soundfile as sf


# SECTION Main code here

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(path_data: DictConfig) -> None:
    # print(path_data.test_path.ref_file) # REMOVE_LATER
    # Open reference JSON file
    try:
        with open(path_data.test_path.ref_file, 'r') as ref_file:
            ref_json = json.load(ref_file)
            
            # TODO - Make a automated way to open each spin and target anechoic scene using loop. Or maybe this script just needs to work on one prompt on each run.
            print(ref_json[0], "\n")    # REMOVE_LATER
            # TODO - NOTE - Path of audio files to open, HA_Output and target_anechoic
            spin_file_path = Path(path_data.test_path.spin_folder) / f"{ref_json[0]['signal']}.wav"
            print(spin_file_path, "\n")  # REMOVE_LATER
            target_file_path = Path(path_data.test_path.scenes_folder) / f"{ref_json[0]['scene']}_target_anechoic.wav"
            print(target_file_path, "\n")  # REMOVE_LATER
            # TODO - NOTE - Opening audio files using soundfile
            pass

        ref_file.close()
    except FileNotFoundError:
        print(f'File not found: {path_data.test_path.ref_file}')
        return None
    finally:
        print(f'Finished processing JSON file.')



# !SECTION

if __name__ == '__main__':
    main()