### This function is to mainly create datasets
import numpy as np
import pandas as pd

import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from .common import *

class CS4220Dataset(Dataset):
    def __init__(self, data_file, label_df=None, k=6, samples_index=None, kmer_profile_on_the_fly=False, dtype=np.float32, header_choice:str="first"):
        """
        Dataset class to load large CS4220 sequence database.

        Args:
            - data_file (`str`): Can either be a *.fasta file if the input is raw reads, or *.npy file
                                 if the input is k-mer profile.
            - label_df (`pd.DataFrame` or `None`): A dataframe with "labels" column indicating the label
                                                   of the data (must match with data_file), or `None` if there is
                                                   no label (in the case of test sets).
            - k (`int`): The lengt of k-mer. We use 6 in this project.
            - samples_index (`List` or `None`): list of indices of data we sample from the data file. You
                                                can use this if the dataset is very large and can't fit in memory.
                                                set this to `None` if you want to use all the data.
            - kmer_profile_on_the_fly (`bool`): If input data_file is raw reads and this set to `True`,
                                                we will build k-mer profile on the fly. This is helpful if you want to
                                                alter the input sequences during training, or the k-mer profile can't fit in memory.
                                                Otherwise, we build k-mer profile in advance, which will speed up the
                                                training process.
            - dtype: type to store the k-mer profile. You may use, for example, `np.float32` for better precision,
                     or `np.float16` for smaller memory usage. If loaded from ".npy" file, it is always `np.float16`.
        """
        self.data_file = data_file
        self.header_choice = header_choice

        # Getting the canononical data
        self.canonical_kmer_dict = read_canonical_kmer_dict()
        if ".fasta" in data_file or ".fa" in data_file or ".fna" in data_file:
            self.is_raw_reads = True
        elif ".npy" in data_file:
            self.is_raw_reads = False
        else:
            raise TypeError(f"The input file must be either a fasta file containing raw reads (.fasta, .fa, .fna) or a numpy file containing k-mer profiles (.npy).")


        self.label_df = label_df
        self.kmer_profile_otf = kmer_profile_on_the_fly

        # k-mer length, set to be 6.
        self.k = k

        # the samples we take from the read dataset
        self.samples_index = samples_index

        self.dtype = dtype

        # Load the data and store in self.reads and self.labels
        self.X = []
        self.Y = []
        self._read_labels()
        self._read_data()
        self.X_mapped = self.get_headers_mapped()

    def _sequence_to_kmer_profile(self, sequence : str, k : int = 6):
        """
        Return the k-mer profile of the input sequence (string)
        """
        res = np.zeros(len(set(self.canonical_kmer_dict.values()))) # values for the dict is the labels encoded for the reads
        for i in range(len(sequence) - k + 1): #iterate through all the kmers within the seqeunce
            k_mer = sequence[i:i + k]
            if k_mer in self.canonical_kmer_dict: #if found in the available dict
                res[self.canonical_kmer_dict[k_mer]] += 1 #index of the res will be the label encoded for the reads while the actual value will be the counts
            else:
                res[-1] += 1 #count those invalid reads

        res /= np.sum(res)
        return res  
    
    def get_headers_mapped(self):
        # Data self.X
        # Headers 
        headers = pd.DataFrame.from_dict(self.canonical_kmer_dict.items())
        if self.header_choice == "first":
            col_list = headers.groupby(1).first().reset_index(drop=True)[0].to_list()
        else:
            col_list = headers.groupby(1).tail(1).reset_index(drop=True)[0].to_list()
        return pd.DataFrame(data= self.X, columns=col_list)

    def _read_labels(self):
        """
        Read the labels and record them in self.labels.
        """
        if self.label_df is None:
            self.Y = None
        elif self.samples_index is None:
            # Load the whole dataset
            self.Y = list(self.label_df["labels"])
        else:
            # Load only the data corresponding to the sampled index
            self.Y = list(self.label_df.iloc[self.samples_index]["labels"])

    def _read_data(self):
        if self.is_raw_reads:
            # Read the fasta file
            with open(self.data_file, 'r') as fasta_file:
                lines = fasta_file.readlines()

            read_range = self.samples_index if self.samples_index is not None else range(int(len(lines) / 2))
            if not self.kmer_profile_otf:
                self.X = np.zeros(
                    (len(read_range), len(set(self.canonical_kmer_dict.values()))),
                    dtype=self.dtype
                )

            for i, j in enumerate(tqdm(read_range, desc=f"Parsing fasta file {self.data_file}")):
                read = lines[j * 2 + 1].strip()
                if self.kmer_profile_otf:
                    # If chose to do k-mer profiling on the fly, simply store the reads
                    self.X.append(read)
                else:
                    # Otherwise, do k-mer profiling during training/testing, cost more time during training/testing
                    self.X[i, :] = self._sequence_to_kmer_profile(read, self.k)
        else:
            # Read the .npy file, and load the numpy matrix
            # Each row corresponds to a read, and each column corresponds to a k-mer (see training_data/6-mers.txt).
            self.X = np.load(self.data_file)
            if self.samples_index is not None:
                self.X = self.X[self.samples_index, :]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        If you are using pytorch, this function helps taking data points during each epoch
        of your training.
        """
        x = self.X[idx]
        if self.kmer_profile_otf:
            read_tensor = torch.tensor(self._sequence_to_kmer_profile(x, self.k), dtype=self.dtype)
        else:
            read_tensor = torch.tensor(x)

        label = self.Y[idx] if self.Y is not None else None
        return read_tensor, label