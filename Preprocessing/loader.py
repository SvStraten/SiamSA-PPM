import io
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import utils
from sklearn import preprocessing 

class LogsDataLoader:
    def __init__(self, name, dir_path = "./datasets"):
        """Provides support for reading and 
            pre-processing examples from processed logs.
        Args:
            name: str: name of the dataset as used during processing raw logs
            dir_path: str: Path to dataset directory
        """
        self._dir_path = f"{dir_path}/{name}/processed"

    def prepare_data_next_activity(self, df, 
        x_word_dict, y_word_dict, 
        max_case_length, shuffle=True):
        
        x = df["prefix"].values
        y = df["next_act"].values
        if shuffle:
            x, y = utils.shuffle(x, y)

        token_x = list()
        for _x in x:
            token_x.append([x_word_dict[s] for s in _x.split()])
        # token_x = np.array(token_x, dtype = np.float32)

        token_y = list()
        for _y in y:
            token_y.append(y_word_dict[_y])
        # token_y = np.array(token_y, dtype = np.float32)

        token_x = tf.keras.preprocessing.sequence.pad_sequences(
            token_x, maxlen=max_case_length)

        token_x = np.array(token_x, dtype=np.float32)
        token_y = np.array(token_y, dtype=np.float32)

        return token_x, token_y
    
    def prepare_data_final_outcome(self, df, 
        x_word_dict, y_word_dict, 
        max_case_length, shuffle=True):
        
        x = df["prefix"].values
        y = df["final_outcome"].values
        if shuffle:
            x, y = utils.shuffle(x, y)

        token_x = list()
        for _x in x:
            token_x.append([x_word_dict[s] for s in _x.split()])
        # token_x = np.array(token_x, dtype = np.float32)

        token_y = list()
        for _y in y:
            token_y.append(y_word_dict[_y])
        # token_y = np.array(token_y, dtype = np.float32)

        token_x = tf.keras.preprocessing.sequence.pad_sequences(
            token_x, maxlen=max_case_length)

        token_x = np.array(token_x, dtype=np.float32)
        token_y = np.array(token_y, dtype=np.float32)

        return token_x, token_y

    def get_max_case_length(self, train_x):
        train_token_x = list()
        for _x in train_x:
            train_token_x.append(len(_x.split()))
        return max(train_token_x)

    def load_data(self, task):
        train_df = pd.read_csv(f"{self._dir_path}/{task}_train.csv")
        test_df = pd.read_csv(f"{self._dir_path}/{task}_test.csv")

        with open(f"{self._dir_path}/metadata.json", "r") as json_file:
            metadata = json.load(json_file)

        x_word_dict = metadata["x_word_dict"]
        y_word_dict = metadata["y_word_dict"]
        max_case_length = self.get_max_case_length(train_df["prefix"].values)
        vocab_size = len(x_word_dict) 
        total_classes = len(y_word_dict)

        return (train_df, test_df, 
            x_word_dict, y_word_dict, 
            max_case_length, vocab_size, 
            total_classes)
