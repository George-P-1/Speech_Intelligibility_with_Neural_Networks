# Import libraries
import hydra
from omegaconf import DictConfig
from pathlib import Path
import json

# SECTION Main code here
@hydra.main(version_base=None, config_path=".", config_name="config")
def main(path_data: DictConfig) -> None:
    print(path_data.train_path.ref_file) # REMOVE_LATER
    
    # Open training reference JSON file
    try:
        with open(path_data.train_path.ref_file, 'r') as ref_file:
            ref_json = json.load(ref_file)   # Load the JSON file

            print(f'The loaded json data is of type: {type(ref_json)}'); print(f"Number of items in the JSON array: {len(ref_json)}"); print(ref_json[0], "\n")    # REMOVE_LATER
            # NOTE - Path of audio files to open, HA_Output and target_anechoic
            spin_file_path = Path(path_data.train_path.spin_folder) / f"{ref_json[0]['signal']}.wav"; print(spin_file_path, "\n")  # REMOVE_LATER
            target_file_path = Path(path_data.train_path.scenes_folder) / f"{ref_json[0]['scene']}_target_anechoic.wav"; print(target_file_path, "\n")  # REMOVE_LATER

        ref_file.close()
    except FileNotFoundError:
        print(f'File not found: {path_data.train_path.ref_file}')
        return None
    finally:
        print(f'Finished processing JSON file.')
# !SECTION

if __name__ == '__main__':
    main()