import csv
import fileinput
import sys


fileToReplaceIn = sys.argv[1]
changeToSoundsLike = sys.argv[2] # use 'train' for training data. Words get replaced with their soundslike.
pathToSoundsLikeCsv = sys.argv[3]

# use 'result' for conversion from NeMo transcript to useable transcript for use
if changeToSoundsLike != "train" and changeToSoundsLike != "result":
    print("Second argument must be 'train' or 'result'!")
    exit()

with fileinput.FileInput(fileToReplaceIn, inplace=True, backup='.bak') as file:
    for line in file:
        with open(pathToSoundsLikeCsv) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    if row[0] != row[1]:
                        firstSoundsLike = row[1].split(',')[0]
                        if changeToSoundsLike == "train":
                            line = line.replace(row[0], firstSoundsLike)
                        elif changeToSoundsLike == "result":
                            line = line.replace(firstSoundsLike, row[0])
                        line_count += 1
                print(line, end='')  # this is vital: prints to the line
print("done!")
