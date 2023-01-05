import json
from pathlib import Path
import pprint
import Levenshtein
from optparse import OptionParser
import re
import os

# matches a good but simple transcript of words to a bad transcript with good timestamps

# rough approach (left is good, right is with timestamps
# Ich,Ich(0:00,0:05)
# kann,kan(0:10,0:15)
# Ihnen,Ihnen(0:20,0:30)
# gerne, helpen(0:35,0:40)
# helfen,

# we always put all good words in and as many good timestamps as possible
# sometimes it doesn't fit. Then we just reuse timestamps and put lower confidence for the word

# For "Ich" we try to find a good match in the first few words of the timestamp file, we find Ich(0:00,0:05) and put it in the final csv
# "Ich(0:00, 0:05)" does not need to be matched again, so we move the pointer in the csv (a variable called 'j') ahead
# we move on to "kann" and find "kan(0:10,0:15)" via its low Levenshtein ratio difference. We put "kann(0:10,0:15)" in the final csv.
# same with Ihnen
# for "gerne" we do not find close match. We just steal the next available timestamp and put "gerne(0:35,0:40)" without moving the pointer in the timestamp file ahead
# thus we can match "helfen" to "helpen" as well, and put "helfen(0:35,0:40)

# note that this algorithm is far from final. The example solution might get outdated 

parser = OptionParser()
parser.add_option("-g", "--good", dest = "good", help="a txt file in which the each line contains a good transcript of a audio file. Only first row used if combined with --timestamps. Alternatively: Same order as in --manifest please.")
parser.add_option("-t", "--timestamps", dest = "timestamps", help="a nemo json file which contains good timestamps, but occasionally bad words", default = None)
parser.add_option("-m", "--manifest", dest = "manifest", help="Alternative to --timestamps. The nemo manifest used to create the timestamps. Same order as in --good please. Combine with -f to pass the folder.", default = None)
parser.add_option("-f", "--folderJson", dest = "folderJson", help="Combine with --manifest to give the folder containing the timestamps. If used alone will assume index names like '1.json' for same order as in --good")
parser.add_option("-o", "--output", dest = "output", help = "path for the output files in the style of ibm")
parser.add_option("-l", "--limit", dest = "limit", help="in case we cannot find a good match, how many ill-fitting words should we go ahead to steal the timestamps?", default=1)


(options, args) = parser.parse_args()

goodPath = options.good
timestampPath = options.timestamps
manifestPath = options.manifest
outputPath = options.output
folderPath = options.folderJson
jOffsetLimit = int(options.limit)
# eval_beamsearch_ngram.txt
# nemo_voice.json

goodTxt = re.sub(r'-[0-9.]*\b', "", Path(goodPath).read_text())
goodChoices = [x for x in goodTxt.split("\n") if x != ""]

jsonPaths = []

if timestampPath == folderPath == None:
        print("Error. Use -t (--timestamps) OR -f (--folderJson), not neither.")
        goodChoice = [] # we do nothing.
elif timestampPath is not None and folderPath is not None:
        print("Error. Only use -t (--timestamps) OR -f (--folderJson), not both at the same time")
        goodChoice = [] # we do nothing.
elif timestampPath is not None:
        goodChoices = goodChoices[0]
        jsonPaths = [timestampPath]
        print("Taking the first row of -g (--good) and matching it to timestampPath")
elif manifestPath is not None:
        print("Good Choices count " + str(len(goodChoices)))
        manifestLines = Path(manifestPath).read_text().split("\n")

        wavPaths = [re.findall(r".*\.wav", x)[0].split('"')[-1] for x in manifestLines if re.match(r".*\.wav", x) is not None]
        print("Wav Paths in Manifest line count " + str(len(wavPaths)))
        jsonPaths = [os.path.join(folderPath, os.path.basename(x.replace(".wav", ".json"))) for x in wavPaths]

        #print(jsonPaths)
        print("Using the paths from -m (--manifest) and -f (--folderJson).")
elif manifestPath is None:
        print("Good Choices count " + str(len(goodChoices)))
        jsonPaths = [os.path.join(folderPath, str(i) + ".json") for i, _ in enumerate(goodChoices)]
        print("Using the paths from -f (--folderJson) but without using a manifest. Assuming files are numbered and in same order as in --good")
if not os.path.exists(outputPath):
        os.makedirs(outputPath)


for (lineIndex, goodChoice) in enumerate(goodChoices):
        goodWords = goodChoice.split(" ")

        # print("liney is " + str(len(goodChoices)) + " out of " + str(len(jsonPaths)))
        timestampPath = jsonPaths[lineIndex] # we use the matching line
        fJson = open(timestampPath)
        badWordObjs = json.load(fJson)['words']
        if (len(badWordObjs) == 0)
            print("No timestamped words found in " + timestampPath + " which is lineIndex " + lineIndex)
        j = 0
        jOffsetForMatchingInDoubt = 0
        lookAhead = 4
        lookBack = 0

        foundGoodMatchForTheseBadWords = [] # after matching a good word to a bad word, we don't match another good word to it
        # however: when we cannot find a match, we might reuse the timestamps of the bad word 
        lastAddition = {}

        compromiseSolution = []
        for i, goodWord in enumerate(goodWords):
                if i - j > lookAhead:
                        j = min(j + lookAhead, len(badWordObjs)-1) # we need to snap ahead in case we can't match a string of badWords
                        jOffsetForMatchingInDoubt = 0 # we snatch this back down, as we are unlikely to exhaust timestamp candidates right now
                potentialMatchJ = j
                bestJFit = None

                for potentialJ in range(potentialMatchJ - lookBack, min(potentialMatchJ + lookAhead, len(badWordObjs)-1)):
                        if potentialJ not in foundGoodMatchForTheseBadWords:
                                badWordObj = badWordObjs[potentialJ]
                                if (Levenshtein.ratio(goodWord, badWordObj['word'])) >= 0.7:  # TODO: use score_cutoff to write faster
                                        bestJFit = potentialJ
                                        break
                if (bestJFit is None):
                        
                        compromiseIdGuess = min(potentialMatchJ+jOffsetForMatchingInDoubt, len(badWordObjs) - 1)
                        compromise = badWordObjs[compromiseIdGuess].copy() # we copy the timestamps of a word that should be roughly around here
                        compromise['word'] = goodWord # TODO: avg timestamps for an unknown word
                        if jOffsetForMatchingInDoubt < jOffsetLimit:
                                jOffsetForMatchingInDoubt += 1
                                if (potentialMatchJ + jOffsetForMatchingInDoubt) >= len(badWordObjs) - 1:
                                        jOffsetForMatchingInDoubt -= 1
                        compromise['confidence'] = 0.99 # danger: we abuse confidence to write down whether we think this word has been said here
                        compromiseSolution.append(compromise)
                        lastAddition = compromise
                else:
                        compromise = badWordObjs[bestJFit] # we steal the timestamps of the matched word
                        compromise['word'] = goodWord
                        compromise['confidence'] = 1 # danger: we abuse confidence to write down whether we think this word has been said here
                        compromiseSolution.append(compromise)
                        lastAddition = compromise
                        foundGoodMatchForTheseBadWords.append(bestJFit)
                        j =  min(bestJFit + 1, len(badWordObjs)-1)
                        jOffsetForMatchingInDoubt = 0 # we snatch this back down, as we are no longer in doubt


        fJson.close()

        #pp = pprint.PrettyPrinter(indent=4)
        #pp.pprint(compromiseSolution) 


        ibm_ts_list = []
        for compromise in compromiseSolution:
                word = compromise['word']
                start = compromise['start_time']        
                end = compromise['end_time']
                package = [word,start,end]
                ibm_ts_list.append(package)     
        # pp.pprint(ibm_ts_list)

        ibm_conf_list = []
        for compromise in compromiseSolution:
                word = compromise['word']
                confidence = compromise['confidence']
                package = [word,confidence]
                ibm_conf_list.append(package)   
        # pp.pprint(ibm_conf_list)


        outputDictionary = [{
                "result_index" : 0,
                "results":
                {
                        "final": [True],
                        "alternatives": [{
                                "transcript": goodChoice,
                                "confidence": 1,
                                "timestamps": ibm_ts_list,
                                "word_confidence": ibm_conf_list
                        }],
                        "word_alternatives": [] # we do not provide alternatives, this is just for compatiblity with ibm
                }
        }]
        # pp.pprint(outputDictionary)


        outputFileForThis = os.path.basename(timestampPath)
        outputPathForThis = os.path.join(outputPath, outputFileForThis)

        with open(outputPathForThis, "w") as outfile:
                json.dump(outputDictionary, outfile)

print("Done! Check results in " + outputPath)
