
from optparse import OptionParser
import contextlib
import json
import os
from pickle import TRUE
import dill as pickle
import csv
import pandas
import torch
import numpy as np

# git clone --recursive https://github.com/parlance/ctcdecode.git
# cd ctcdecode && pip install .
from ctcdecode import CTCBeamDecoder

# code is heavily inspired by https://github.com/NVIDIA/NeMo/discussions/2577

parser = OptionParser()
parser.add_option("-t", "--tokenizer", dest = "tokenizer", help= "The pickle file containing the tokenizer", default = None)
parser.add_option("-v", "--vocab", dest = "vocab", help= "The pickle file containing the vocab.", default = None)
parser.add_option("-p", "--probs", dest = "probs", help= "The pickle file containing the probs.", default = None)
parser.add_option("-o", "--output", dest = "output", help= "Path of the output folder", default = None)
parser.add_option("-l", "--languageModel", dest = "languageModel", help= "Path of the language model", default = None)

(options, args) = parser.parse_args()


# fileTokenizer = open("/workspace/NeMo/mountable/dev_model/outProbsGymondoMini.txt_tokenizer.pickle", 'rb')
fileTokenizer = open(options.tokenizer, 'rb')
tokenizer = pickle.load(fileTokenizer)


# fileVocab = open("/workspace/NeMo/mountable/dev_model/outProbsGymondo.txt_vocab.pickle", 'rb')
fileVocab = open(options.vocab, 'rb')
parlanceDecodedPath = options.output

vocab = pickle.load(fileVocab)

print("Length of vocab " + str(len(vocab)))
print(" vocab " + str(vocab))
TOKEN_OFFSET = 100

# file = open("/workspace/NeMo/mountable/dev_model/outProbsGymondo.txt", 'rb')
file = open(options.probs, 'rb')

# take information of that file
allDataProbs = pickle.load(file)

# close the file
file.close()


lm_path = '/workspace/NeMo/mountable/dev_model/result_topchoice_frienett.bin'

labels = [chr(idx + TOKEN_OFFSET) for idx in range(len(vocab))]
labels.append("_")
blank_id = 129
print("labels " + str(labels))
decoder = CTCBeamDecoder(
    labels= labels,
    model_path=lm_path,
    alpha=1,
    beta=0.5,
    cutoff_top_n=40,
    cutoff_prob=1.0,
    beam_width=64,
    num_processes=max(os.cpu_count(), 1),
    blank_id=blank_id,
    log_probs_input=False
)

translated_labels = [tokenizer(idx) for idx in range(len(vocab))]
print("translated labels " + str(translated_labels))


SAMPLE_RATE = 16000
FRAME_LEN = 40.0
CHANNELS = 1
CHUNK_SIZE = int(FRAME_LEN*SAMPLE_RATE)
TOKEN_OFFSET = 100
TIME_STEP = 0.04
TIME_PAD = 1

# print(beam_results.size())

def beam_decoder(beam_results, beam_scores, timesteps, out_lens, chunk_pad):
    # probs = softmax(logits)
    # probs_seq = torch.FloatTensor(probs)
    
    # beam_results, beam_scores, timesteps, out_lens = self.beam_search_lm.decode(probs_seq.unsqueeze(0))
    
    
    beam_res = beam_results[0][0][:out_lens[0][0]].cpu().numpy()
    times = timesteps[0][0][:out_lens[0][0]].cpu().numpy()
    lens = out_lens[0][0]
    
    transcript = tokenizer(beam_res)
    # print(transcript)
    wordConcat = ""
    
    if len(times) > 0:
        if times[0] < TIME_PAD:
            start = chunk_pad
            end = (times[0] + TIME_PAD*2) * TIME_STEP + chunk_pad
        else:
            start = (times[0] - TIME_PAD) * TIME_STEP + chunk_pad
            end = (times[0] + TIME_PAD) * TIME_STEP + chunk_pad
            
        tocken_prev = labels[int(beam_res[0])]
        word = tocken_prev

        translated_word = tokenizer(int(beam_res[0]))
        
        

        result = []
        
        for n in range(1,lens):

            tocken = labels[int(beam_res[n])]
            
            # print(tocken + " is #" + str(int(beam_res[n])))
            if tocken[0] == "#":
                word = word + tocken[2:]
                print("token started with #")
                
            elif tocken[0] == "-" or tocken_prev[0] == "-":
                word = word + tocken
                print("token or previous token started with -")
            #elif int(beam_res[n]) in [0,1,77]:
            #    word = word + " "
            #    # print("token was mysterios 1")
            #   print("token was of id 1") # this means we cannot find a translated_word, but they don't align with whitespace. Weird.
            else:
                translated_word = tokenizer(int(beam_res[n]))
                if start > end:
                    result_word = { 'start': round(end, 3), 'end': round(start, 3), 'word': word, 'translatedWord': translated_word}
                else:
                    result_word = { 'start': round(start, 3), 'end': round(end, 3), 'word': word, 'translatedWord': translated_word}

                wordConcat += translated_word
                result.append(result_word)
                
                if times[n] < TIME_PAD:
                    start = chunk_pad
                else:
                    start = (times[n] - TIME_PAD) * TIME_STEP + chunk_pad

                word = tocken
                
                
            if times[n] < TIME_PAD:
                end = (times[n] + TIME_PAD*2) * TIME_STEP + chunk_pad
            else:
                end = (times[n] + TIME_PAD) * TIME_STEP + chunk_pad
            
            tocken_prev = tocken
            
            
        if start > end:
            result_word = { 'start': round(end, 3), 'end': round(start, 3), 'word': word, 'translatedWord': translated_word }
        else:
            result_word = { 'start': round(start, 3), 'end': round(end, 3), 'word': word, ' translatedWord': translated_word}
        
        wordConcat += translated_word
        result.append(result_word)
    else:
        # print("alt result")
        # print(transcript)
        result = []
    agreed_transcript = ""
    i_timestamped = 0

    # sometimes the transcript is bigger than the (timestamped) wordConcat. We get rid of extra letters, but keep the spaces
    for transcript_letter in transcript:
        if i_timestamped >= len(wordConcat):
            break
        timestamped_letter = wordConcat[i_timestamped]
        if transcript_letter == timestamped_letter:
            agreed_transcript += timestamped_letter
            i_timestamped += 1
        elif len(agreed_transcript) > 0 and transcript_letter == ' ' and agreed_transcript[-1] != ' ':
            agreed_transcript += ' ' # we do not want double white space, just because we delete a word

    return agreed_transcript, result



def blank_reinserter(transcript, result):
    # unlike the original author, we do not know the special symbols (he uses '#' and "-")
    # however, our tokenizer function does. So we take the blanks from the tokenizer's results

    i_in_results = 0
    word_in_progress = ""
    start_in_progress = 0.0
    end_in_progress = 0.0
    expecting_new_word = True

    skip_n_steps = 0

    
    result_filtered = [x for x in result if ('translatedWord' in x and len(x['translatedWord']) > 0)] # if we want to go by the letter in the transcript, we cannot have empty translatedWords
    #TODO: investigate why we ever get a empty translatedWord

    result_words = []
    # print(result_filtered)
    # print("let's go")
    # print(transcript)
    for letter in transcript:
        if skip_n_steps > 0:
            # print(letter + " skip")
            skip_n_steps -= 1
            continue
        
        if (i_in_results >= len(result_filtered)) or letter == ' ':
            result_good = {'start_time': start_in_progress, 'end_time': end_in_progress, 'word': word_in_progress}
            # print(word_in_progress)
            # print(letter)
            result_words.append(result_good)
            word_in_progress = ""
            expecting_new_word = True

        else:
            
            current_result = result_filtered[i_in_results]
            jump_size = max(0, len(current_result['translatedWord'])-1) # we might get "und" in the results, and thus we need to fit the words we found
            # print(letter + " vs " + current_result['translatedWord'] + " jump " + str(jump_size))
            word_in_progress += current_result['translatedWord']
            if expecting_new_word:
                start_in_progress = current_result['start'] # due to the frame system, we will get overlaps in the sub-timestep times
                expecting_new_word = False
            end_in_progress = current_result['end']
            i_in_results += 1
            skip_n_steps = jump_size
            

    # print("final result")
    # print(result_words)

    output_dict = {'words': result_words}
    return(output_dict)

for idx, dataProbs in enumerate(allDataProbs):
    df = pandas.DataFrame(dataProbs)
    # df.to_csv("hypotheses.csv")

    outputs = torch.from_numpy(np.expand_dims(dataProbs, axis=0))
    # print(outputs.shape)
    beam_results, beam_scores, timesteps, out_lens = decoder.decode(outputs)
    transcript, result = beam_decoder(beam_results, beam_scores, timesteps, out_lens, chunk_pad=0)

    resultDict = blank_reinserter(transcript, result)
    outPath = os.path.join(parlanceDecodedPath, (str(idx)+".json"))
    with open(outPath, "w") as outfile:
        json.dump(resultDict, outfile)
    

# import inspect
# lines = inspect.getsource(tokenizer)
# print(lines)
