import json
import pickle
from typing import Union
import pandas as pd
from sklearn import preprocessing

def get_species_count(df):
    sum_ = sum(df['species_name'].value_counts())
    counts_ = df['species_name'].value_counts()
    percent_counts_ = df['species_name'].value_counts()/sum_
    return sum_, counts_, percent_counts_

def create_coarse_labels(df):
    df['coarse_species_name'] = np.where(df['species_name'] != "decoy", "non_decoy", "decoy")
    le = preprocessing.LabelEncoder()
    le.fit(df['coarse_species_name'].unique())
    y_index = le.transform(df['coarse_species_name'].values)
    df['labels'] = y_index
    print(f"Unique labels {len(df['coarse_species_name'].unique())}")
    return df, le

def create_label_df(df):
    # This is required for dataset function
    le = preprocessing.LabelEncoder()
    le.fit(df['species_name'].unique())
    y_index = le.transform(df['species_name'].values)
    df['labels'] = y_index
    print(f"Unique labels {len(df['species_name'].unique())}")

    return df, le

def read_canonical_kmer_dict(filepath="./training_data/6-mers.json"):
    # Load dictionary that maps k-mer to their corresponding index.
    # A k-mer and its reverse complement are mapped to the same index.
    # help to do label encoding on the reads values we used for ML training

    with open(filepath, 'r') as dict_file:
        return json.load(dict_file)

def create_sampling_idx(df, sample_num:int, replace:bool=False, random_state:int=40):
    return df.groupby('labels').sample(sample_num, replace=replace,random_state =random_state).index

def sequence_to_kmer_profile(sequence : str, k : int = 6):
    # We define a utility function here that turns sequences to their 6-mer profiles.
    # This is to allow the fasta file to be converted to data simialr npy file

    """
    Return the k-mer profile of the input sequence (string)
    """
    # Get canonical dict 
    canonical_kmer_dict = read_canonical_kmer_dict()
    res = np.zeros(len(set(canonical_kmer_dict.values()))) # values for the dict is the labels encoded for the reads
    for i in range(len(sequence) - k + 1): #iterate through all the kmers within the seqeunce
        k_mer = sequence[i:i + k]
        if k_mer in canonical_kmer_dict: # if found in the available dict
            res[canonical_kmer_dict[k_mer]] += 1 #index of the res will be the label encoded for the reads while the actual value will be the counts
        else:
            res[-1] += 1 #count those invalid reads

    res /= np.sum(res)
    return res

def save_file(transform_scale, filename:str, filepath:Union[str,None]):
    # saves transform_scale
    if ".pkl" not in filename:
        filename = f"{filename}.pkl"
    if filepath:
        filename = f"{filepath}/{filename}"
    pickle.dump(transform_scale, open(filename,"wb"))

def open_file(filename:str, filepath:Union[str,None]):
    if ".pkl" not in filename:
        filename = f"{filename}.pkl"
    if filepath:
        filename = f"{filepath}/{filename}"
    # Return transform_scale
    return pickle.load(open(filename,'rb'))