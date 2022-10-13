import json
from pathlib import Path
import pprint
import Levenshtein
from optparse import OptionParser
import re

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
parser.add_option("-g", "--good", dest = "good", help="a txt file in which the first line contains a good transcript of the audio file")
parser.add_option("-t", "--timestamps", dest = "timestamps", help="a nemo json file which contains good timestamps, but occasionally bad words")
parser.add_option("-o", "--output", dest = "output", help = "path for the output file in the style of ibm")
parser.add_option("-l", "--limit", dest = "limit", help="in case we cannot find a good match, how many ill-fitting words should we go ahead to steal the timestamps?", default=1)

(options, args) = parser.parse_args()

goodPath = options.good
timestampPath = options.timestamps
outputPath = options.output
jOffsetLimit = options.limit
# eval_beamsearch_ngram.txt
# nemo_voice.json

goodTxt = re.sub(r'-[0-9.]*\b', "", Path(goodPath).read_text())
goodChoice = goodTxt.split("\n")[0]
goodWords = goodChoice.split(" ")

fJson = open(timestampPath)
badWordObjs = json.load(fJson)['words']
j = 0
jOffsetForMatchingInDoubt = 0
lookAhead = 4
lookBack = 0

foundGoodMatchForTheseBadWords = [] # after matching a good word to a bad word, we don't match another good word to it
# however: when we cannot find a match, we might reuse the timestamps of the bad word 

compromiseSolution = []
for i, goodWord in enumerate(goodWords):
	if i - j > lookAhead:
		j += lookAhead # we need to snap ahead in case we can't match a string of badWords
		jOffsetForMatchingInDoubt = 0 # we snatch this back down, as we are unlikely to exhaust timestamp candidates right now
	potentialMatchJ = j
	bestJFit = None
	for potentialJ in range(potentialMatchJ - lookBack, min(potentialMatchJ + lookAhead, len(badWordObjs))):
		if potentialJ not in foundGoodMatchForTheseBadWords:
			badWordObj = badWordObjs[potentialJ]
			if (Levenshtein.ratio(goodWord, badWordObj['word'])) > 0.8:  # TODO: use score_cutoff to write faster
				bestJFit = potentialJ
				break
	if (bestJFit is None):
		compromise = badWordObjs[potentialMatchJ+jOffsetForMatchingInDoubt] # we steal the timestamps of a word that should be roughly around here
		compromise['word'] = goodWord # TODO: avg timestamps for an unknown word
		if jOffsetForMatchingInDoubt < jOffsetLimit:
			jOffsetForMatchingInDoubt += 1
			if (potentialMatchJ + jOffsetForMatchingInDoubt) >= len(badWordObjs) - 1:
				jOffsetForMatchingInDoubt -= 1
		compromise['confidence'] = 0.99 # danger: we abuse confidence to write down whether we think this word has been said here
		compromiseSolution.append(compromise)
	else:
		compromise = badWordObjs[bestJFit] # we steal the timestamps of the matched word
		compromise['word'] = goodWord
		compromise['confidence'] = 1 # danger: we abuse confidence to write down whether we think this word has been said here
		compromiseSolution.append(compromise)
		foundGoodMatchForTheseBadWords.append(bestJFit)
		j = bestJFit + 1
		jOffsetForMatchingInDoubt = 0 # we snatch this back down, as we are no longer in doubt

fJson.close()

# pp = pprint.PrettyPrinter(indent=4)
# pp.pprint(compromiseSolution)	

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


outputDictionary = {
	"result_index" : 0,
	"results":
	 {
		"final": True,
		"alternatives": {
			"transcript": goodChoice,
			"confidence": 1,
			"timestamps": ibm_ts_list,
			"word_confidence": ibm_conf_list	
	 	}
	} 
}
# pp.pprint(outputDictionary)

with open(outputPath, "w") as outfile:
	json.dump(outputDictionary, outfile)

