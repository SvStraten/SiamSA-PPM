import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time
import warnings
import argparse

from Preprocessing.utils import data_loader_fop, preprocess_fop, check_gpu, evaluate_per_k_fop, evaluate_global_fop
from tensorflow.keras.models import load_model #type:ignore
from Model.model import TransformerBlock, TransformerEncoder, TokenAndPositionEmbedding

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Configure experiment settings.')
parser.add_argument('--dataName', type=str, default='sepsis', help='Name of the dataset')
parser.add_argument('--strategy', type=str, default='combi', help='Strategy to use')
args = parser.parse_args()

# Assign variables from arguments
dataName = args.dataName
repetitions = 1

print(f"\n📊 Evaluating dataset: {dataName}")

data = data_loader_fop(dataName)
train_df, test_df, x_word_dict, y_word_dict, \
max_case_length, vocab_size, num_output, \
train_token_x, train_token_y = preprocess_fop(data)

check_gpu()
all_k_results = []
all_results = []
method_summaries = []
dataset_summaries = []

for rep in range(repetitions):
    model_path = f"FOPModels/{dataName}_fop_{rep}.keras"
    model = load_model(
        model_path,
        compile=False,
        custom_objects={
            "TransformerEncoder": TransformerEncoder,
            "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
            "TransformerBlock": TransformerBlock,
        },
        safe_mode=False
    )
    
    # Evaluate per-k per-label accuracy/F1
    k_results = evaluate_per_k_fop(
        model, test_df, data,
        x_word_dict, y_word_dict,
        model.input_shape[1]
    )

    # Add method/repetition metadata
    for entry in k_results:
        entry["Repetition"] = rep

    all_k_results.extend(k_results)


    df = pd.DataFrame(all_k_results)

    summary = df.groupby(["Label", "Prefix Length (k)"]).agg({
        "Accuracy": "mean",
        "F-score": "mean"
    }).reset_index()

    method_summaries.append(summary)


    # Store results per dataset
    if method_summaries:
        combined_summary = pd.concat(method_summaries, ignore_index=True)
        combined_summary["Method"] = combined_summary["Method"].replace({"STRATEN": "SIMSA-PPM"})
        dataset_summaries[dataName] = combined_summary

        print(f"✅ Completed summary for dataset: {dataName}")