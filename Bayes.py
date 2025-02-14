import os

import nltk
import pandas as pd
import math
import collections
from tqdm import tqdm
import numpy as np
nltk.download('punkt_tab')

#Import documents in bag-of-words format
tolstoy_file = pd.read_csv("BoW_Formatting/BoW_Tolstoy.csv", index_col=0)
tolstoy_file = tolstoy_file.loc[:, ~tolstoy_file.columns.str.contains('Unnamed')]
wells_file = pd.read_csv("BoW_Formatting/BoW_HGWells.csv", index_col=0)
wells_file = wells_file.loc[:, ~wells_file.columns.str.contains('Unnamed')]
# tolstoy_rows, tolstoy_cols = tolstoy_file.shape

#Initialize Variables
vocab = set()
class_dict = [collections.defaultdict(int), collections.defaultdict(int)]
total_tokens= [0, 0]

#Add all tolstoy tokens into tolstoy defaultdict with total counts initialize wells defaultdict token to 0 if new vocab entry
for token in tolstoy_file.index:
    token_sum = tolstoy_file.loc[token].sum()

    if token not in vocab:
        class_dict[1][token] = 0
        vocab.add(token)

    class_dict[0][token] = token_sum
    total_tokens[0] += token_sum

#Add all wells tokens into wells defaultdict with total counts initialize tolstoy defaultdict token to 0 if new vocab entry
for token in wells_file.index:
    token_sum = wells_file.loc[token].sum()

    if token not in vocab:
        class_dict[0][token] = 0
        vocab.add(token)

    class_dict[1][token] = wells_file.loc[token].sum()
    total_tokens[1] += token_sum


#Initialize Log-likelihood defaultdicts and vocab_count
log_likelihood = [collections.defaultdict(float), collections.defaultdict(float)]
vocab_count = len(vocab)

#Run through vocab
for token in vocab:
    #compute probability of word showing up in tolstoy and wells
    t_likelihood = math.log((class_dict[0][token] + 1) / (total_tokens[0] + vocab_count))
    w_likelihood = math.log((class_dict[1][token] + 1) / (total_tokens[1] + vocab_count))

    #subtract tolstoy log probability from wells log probability
    token_likelihood = t_likelihood - w_likelihood

    #if number is positive or 0, store number in tolstoy log-likelihood defaultdict
    if token_likelihood >= 0:
        log_likelihood[0][token] = token_likelihood
    else:
        #store absolute value of number in wells log-likelihood defaultdict
        #TODO: Ensure that the absolute value function works as intended on floats
        log_likelihood[1][token] = abs(token_likelihood)

#sort both defaultdicts descending
sorted_likelihood = []
sorted_likelihood.append(sorted(log_likelihood[0].items(), key=lambda item: item[1], reverse=True))
sorted_likelihood.append(sorted(log_likelihood[1].items(), key=lambda item: item[1], reverse=True))

#print top 10 words of each defaultdict
print("  |    Leo Tolstoy Values     |     H. G. Wells Values    |")
for x in range(10):
    print(f"{x+1} | {sorted_likelihood[0][x][0]}: {sorted_likelihood[0][x][1]} | {sorted_likelihood[1][x][0]}: {sorted_likelihood[1][x][1]} |")