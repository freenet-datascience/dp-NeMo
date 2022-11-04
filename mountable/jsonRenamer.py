import argparse
import contextlib
import json
import os
from optparse import OptionParser
import csv
import re
from pathlib import Path

from tqdm.auto import tqdm



def main():
    parser = OptionParser()
    parser.add_option("-m", "--manifest", dest = "manifest", 
    help= "Path of the language model", default = None)
    parser.add_option("-c", "--combined", dest = "combined",
    help= "Path of the 'combined' folder containing the results, with each number corresponding to a mp3", default = None)
    (options, args) = parser.parse_args()
    audio_file_paths = []
    with open(options.manifest, 'r') as manifest_file:
        for idx, line in enumerate(tqdm(manifest_file, desc=f"Reading Manifest {options.manifest} ...", ncols=120)):
            
            data = json.loads(line)
            audio_file = Path(data['audio_filepath'])
            combined_path = os.path.join(options.combined, str(idx) + ".json")
            numbered_file_exists = os.path.exists(combined_path)
            new_base_name =  audio_file.stem
            new_path = os.path.join(options.combined, new_base_name + ".json")
            if numbered_file_exists:
                os.rename(combined_path, new_path)
            else:
                print("could not find " + combined_path + " which should be " + new_path)
            

if __name__ == '__main__':
    main()