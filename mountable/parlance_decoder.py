
import argparse
import contextlib
import json
import os
import dill as pickle
import csv
import pandas
import torch
import numpy as np

# git clone --recursive https://github.com/parlance/ctcdecode.git
# cd ctcdecode && pip install .
from ctcdecode import CTCBeamDecoder

# code is heavily inspired by https://github.com/NVIDIA/NeMo/discussions/2577

fileTokenizer = open("/workspace/NeMo/mountable/dev_model/outProbsGymondo.txt_tokenizer.pickle", 'rb')
tokenizer = pickle.load(fileTokenizer)


fileVocab = open("/workspace/NeMo/mountable/dev_model/outProbsGymondo.txt_vocab.pickle", 'rb')

vocab = pickle.load(fileVocab)
vocab.append("_")
print("Length of vocab " + str(len(vocab)))
print(" vocab " + str(vocab))
TOKEN_OFFSET = 100

file = open("/workspace/NeMo/mountable/dev_model/outProbsGymondo.txt", 'rb')

# take information of that file
dataProbs = pickle.load(file)[0]

# close the file
file.close()
df = pandas.DataFrame(dataProbs)
df.to_csv("hypotheses.csv")

outputs = torch.from_numpy(np.expand_dims(dataProbs, axis=0))
print(outputs.shape)

lm_path = '/workspace/NeMo/mountable/dev_model/result_topchoice_frienett.bin'

labels = [chr(idx + TOKEN_OFFSET) for idx in range(len(vocab))]
decoder = CTCBeamDecoder(
    labels= labels,
    model_path=lm_path,
    alpha=2,
    beta=1.5,
    cutoff_top_n=40,
    cutoff_prob=1.0,
    beam_width=16,
    num_processes=max(os.cpu_count(), 1),
    blank_id=129,
    log_probs_input=False
)

beam_results, beam_scores, timesteps, out_lens = decoder.decode(outputs)


SAMPLE_RATE = 16000
FRAME_LEN = 40.0
CHANNELS = 1
CHUNK_SIZE = int(FRAME_LEN*SAMPLE_RATE)
TOKEN_OFFSET = 100
TIME_STEP = 0.04
TIME_PAD = 1

def beam_decoder(beam_results, beam_scores, timesteps, out_lens, chunk_pad):
    # probs = softmax(logits)
    # probs_seq = torch.FloatTensor(probs)
    
    # beam_results, beam_scores, timesteps, out_lens = self.beam_search_lm.decode(probs_seq.unsqueeze(0))
    
    beam_res = beam_results[0][0][:out_lens[0][0]].cpu().numpy()
    times = timesteps[0][0][:out_lens[0][0]].cpu().numpy()
    lens = out_lens[0][0]
    
    transcript = tokenizer(beam_res)
    
    if len(times) > 0:
        if times[0] < TIME_PAD:
            start = chunk_pad
            end = (times[0] + TIME_PAD*2) * TIME_STEP + chunk_pad
        else:
            start = (times[0] - TIME_PAD) * TIME_STEP + chunk_pad
            end = (times[0] + TIME_PAD) * TIME_STEP + chunk_pad
            
        tocken_prev = vocab[int(beam_res[0])]
        word = tocken_prev
        
        result = []
        
        for n in range(1,lens):
            tocken = vocab[int(beam_res[n])]
            
            if tocken[0] == "#":
                word = word + tocken[2:]
                
            elif tocken[0] == "-" or tocken_prev[0] == "-":
                word = word + tocken
                
            else:
                if start > end:
                    result_word = { 'start': round(end, 3), 'end': round(start, 3), 'word': word}
                else:
                    result_word = { 'start': round(start, 3), 'end': round(end, 3), 'word': word}
                    
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
            result_word = { 'start': round(end, 3), 'end': round(start, 3), 'word': word}
        else:
            result_word = { 'start': round(start, 3), 'end': round(end, 3), 'word': word}
            
        result.append(result_word)
        
    else:
        print("alt result")
        print(transcript)
        result = []
        
    return transcript, result

transcript, result = beam_decoder(beam_results, beam_scores, timesteps, out_lens, chunk_pad=0)
print("final result")
print(transcript)
