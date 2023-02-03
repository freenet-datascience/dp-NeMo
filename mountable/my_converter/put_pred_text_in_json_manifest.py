import json
import os
from optparse import OptionParser
import re
from pathlib import Path

from tqdm.auto import tqdm

def main():
    parser = OptionParser()
    parser.add_option("-m", "--manifest", dest = "manifest", 
    help= "Path of the manifest containing names", default = None)
    parser.add_option("-p", "--preds", dest = "preds",
    help= "Path of the 'preds' file containing the results, with each number corresponding to a mp3", default = None)
    parser.add_option("-r", "--result", dest = "result",
    help= "Path of the 'result' file containing a changed manifest, but this time with text set to the preds", default = None)
    
    (options, args) = parser.parse_args()
    audio_file_paths = []
    predPath = options.preds
    predTxt = re.sub(r'-[0-9.]*\b', "", Path(predPath).read_text())
    predLines = [x for x in predTxt.split("\n") if x != ""]
    
    lineCountInManifest = 0
    if os.path.exists(options.result):
      os.remove(options.result) # we don't want to append the same results twice to the result file, so we start fresh
    
    with open(options.result, 'a') as result_file:
      with open(options.manifest, 'r') as manifest_file:
          for idx, line in enumerate(tqdm(manifest_file, desc=f"Reading Manifest {options.manifest} ...", ncols=120)):
              if idx > len(predLines):
                print("Error: our tsv with the preds has fewer lines than our manifest json")
              data = json.loads(line)
              data['text'] = predLines[idx]
              lineCountInManifest = idx
              output_string = json.dumps(data)
              result_file.write(output_string+"\n")
            
    if lineCountInManifest > len(predLines):
      print("Error: our tsv with the preds has fewer lines than our manifest json")

if __name__ == '__main__':
    main()
