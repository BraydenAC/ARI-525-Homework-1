import os
import pandas as pd

#Import documents in bag-of-words format
tolstoy_file = pd.read_csv("BoW_Formatting/BoW_Tolstoy.csv")
wells_file = pd.read_csv("BoW_Formatting/BoW_Wells.csv")

#Slit into train/dev/test through vertical slice

#Bayes Logic
#Initialize Variables
classes = ["HGWells", "Leo Tolstoy"]
#Vocab size initialize
#

