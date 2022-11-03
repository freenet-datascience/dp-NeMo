
from optparse import OptionParser
import contextlib
import json
import os
from pathlib import Path





parser = OptionParser()
parser.add_option("-m", "--manifest", dest = "manifest", help= "Path of the language model", default = None)
parser.add_option("-c", "--combined", dest = "combined", help= "Path of the 'combined' folder containing the results, with each number corresponding to a mp3", default = None)

(options, args) = parser.parse_args()


# fwith open(args.input_manifest, 'r') as manifest_file:
audio_file_paths = []
for idx, line in enumerate(tqdm(manifest_file, desc=f"Reading Manifest {args.manifest} ...", ncols=120)):
  data = json.loads(line)
  audio_file = Path(data['audio_filepath'])
  combined_path = os.path.join(args.combined, str(idx), ".json")
  numbered_file_exists = os.path.exists(combined_path)
  if numbered_file_exists:
    new_base_name =  audio_file.stem()
    new_path = os.path.join(args.combined, new_name)
    print(new_path)
    # os.rename(combined_path, 'b.kml')
