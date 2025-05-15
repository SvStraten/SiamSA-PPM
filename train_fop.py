import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from tensorflow.keras.preprocessing.sequence import pad_sequences #type:ignore  
from sklearn.model_selection import train_test_split
from tqdm import tqdm #type: ignore
import random
import argparse 
import os
from tensorflow.keras.models import load_model #type:ignore
from tensorflow.keras.layers import GlobalAveragePooling1D #type:ignore
from Augmentation.augmentation.get_patterns import get_patterns, map_patterns_to_tokens
from Augmentation.augmentation.get_replacement import get_xor_candidates, map_xor_candidates_to_tokens
from Augmentation.augmentation.easy_augmentors import RandomDeletion, RandomInsertion, RandomReplacement
from Augmentation.augmentation.easy_augmentors import StatisticalInsertion, StatisticalDeletion, StatisticalReplacement
from Preprocessing.utils import data_loader_fop, preprocess_fop, check_gpu, check_applicability
from Model.model import TransformerBlock, TransformerEncoder, TokenAndPositionEmbedding

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Configure experiment settings.')
parser.add_argument('--dataName', type=str, default='sepsis', help='Name of the dataset')
parser.add_argument('--strategy', type=str, default='combi', help='Strategy to use')
args = parser.parse_args()

# Assign variables from arguments
dataName = args.dataName
STRATEGY = args.strategy
BATCH_SIZE = 256
repetitions = 1

check_gpu()
    
# Load data
data = data_loader_fop(dataName)
train_df, test_df, x_word_dict, y_word_dict, max_case_length, vocab_size, num_output, train_token_x, train_token_y = preprocess_fop(data)
model_path = f"PreTrainedModels/{dataName}_pretrained.keras"

f_trained = load_model(
    model_path,
    compile=False,
    custom_objects={
        "TransformerEncoder": TransformerEncoder,
        "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
        "TransformerBlock": TransformerBlock,
    },
    safe_mode=False
)

# Find the first GlobalAveragePooling1D layer
gap_layer = None
for layer in f_trained.layers:
    if isinstance(layer, GlobalAveragePooling1D):
        gap_layer = layer
        break

target_len = f_trained.input_shape[1]

encoder_only = tf.keras.Model(inputs=f_trained.input, outputs=gap_layer.output)

for i in range(repetitions):
    train_x, val_x, train_y, val_y = train_test_split(
        train_token_x, train_token_y,
        test_size=0.15,
        random_state=42,
        shuffle=False
    )

    train_x_padded = pad_sequences(train_x, maxlen=target_len, padding='pre') 
    val_x_padded = pad_sequences(val_x, maxlen=target_len, padding='pre')

    training_ds = tf.data.Dataset.from_tensor_slices((train_x_padded, train_y)).batch(BATCH_SIZE)
    validation_ds = tf.data.Dataset.from_tensor_slices((val_x_padded, val_y)).batch(BATCH_SIZE)
        
    def get_linear_classifier(feature_backbone, trainable=False, num_classes=len(y_word_dict)):
        inputs = tf.keras.Input(shape=(target_len,), dtype=tf.int32)

        feature_backbone.trainable = trainable
        x = feature_backbone(inputs, training=False)

        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs)
        return model
        
    linear_model = get_linear_classifier(encoder_only, num_classes=len(y_word_dict))

    linear_model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    
    early_stopper = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    history = linear_model.fit(
        training_ds,
        validation_data=validation_ds,
        epochs=100,
        callbacks=[early_stopper]
    )

    linear_model.save(os.path.join(f"NAPModels/{dataName}_nap_{i}.keras"))
    print("Model saved.")