import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics 
from collections import Counter
from sklearn.model_selection import train_test_split
import argparse

from tqdm import tqdm
import random

from Augmentation.augmentation.get_patterns import get_patterns, map_patterns_to_tokens
from Augmentation.augmentation.get_replacement import get_xor_candidates, map_xor_candidates_to_tokens
from Augmentation.augmentation.easy_augmentors import RandomDeletion, RandomInsertion, RandomReplacement
from Augmentation.augmentation.easy_augmentors import StatisticalInsertion, StatisticalDeletion, StatisticalReplacement
from Preprocessing.utils import data_loader_nap, preprocess_nap, check_gpu, check_applicability, generate_augmented_views
from Model.model import get_encoder, get_predictor

parser = argparse.ArgumentParser(description='Configure experiment settings.')
parser.add_argument('--dataName', type=str, default='sepsis', help='Name of the dataset')
parser.add_argument('--strategy', type=str, default='combi', help='Strategy to use')
args = parser.parse_args()

# Assign variables from arguments
dataName = args.dataName
STRATEGY = args.strategy

ALPHA = 0.0001
BETA = 0.0001
GAMMA = 0.0001
DELTA = 0.0001
PATH_LENGTH = 4
BATCH_SIZE = 256
EPOCHS = 100
WARMUP_EPOCHS = 10
BASE_LR = 0.05 
WEIGHT_DECAY = 1e-5
TAU_BASE = 0.996

EMBED_DIM = 128 
NUM_HEADS = 4
FF_DIM = 256
LAYERS = 2
DROPOUT = 0.2
HIDDEN_DIM = 256
FEATURE_DIM = 256

data = data_loader_nap(dataName)
train_df, test_df, x_word_dict, y_word_dict, max_case_length, vocab_size, num_output, train_token_x, train_token_y = preprocess_nap(data)
available_tokens = list(x_word_dict.values())

patterns_df = get_patterns(f'datasets/{dataName}/processed/next_activity_train.csv', transition_threshold=BETA, path_threshold=GAMMA, max_path_length=PATH_LENGTH, activity_threshold=ALPHA)
patterns_token_df = map_patterns_to_tokens(patterns_df, x_word_dict)

xor_df = get_xor_candidates(f'datasets/{dataName}/processed/next_activity_train.csv', support_threshold=DELTA, max_path_length=PATH_LENGTH, activity_threshold=ALPHA)
xor_token_df = map_xor_candidates_to_tokens(xor_df, x_word_dict)

check_gpu()

if STRATEGY == 'random':
    main_augmentors = []

    fallback_augmentors = [
        RandomInsertion(available_tokens),
        RandomDeletion(),
        RandomReplacement(available_tokens)
    ]
    
elif STRATEGY == "combi":
    main_augmentors = [
        StatisticalInsertion(patterns_token_df),
        StatisticalDeletion(patterns_token_df),
        StatisticalReplacement(xor_token_df)
    ]

    fallback_augmentors = [
        RandomInsertion(available_tokens),
        RandomDeletion(),
        RandomReplacement(available_tokens)
    ]

applicability_info = check_applicability(
    train_token_x,
    main_augmentors,
    fallback_augmentors
)

# Flatten the list of augmentors
main_counts = Counter()
fallback_counts = Counter()

for item in applicability_info:
    main_counts.update(item["applicable_main"])
    fallback_counts.update(item["applicable_fallback"])

# Combine for full view
all_augmentors = set(main_counts.keys()) | set(fallback_counts.keys())

print("Augmentor availability across dataset:\n")
for aug in sorted(all_augmentors):
    main_count = main_counts.get(aug, 0)
    fallback_count = fallback_counts.get(aug, 0)
    total = main_count + fallback_count
    print(f"{aug}: {total} total (Main: {main_count}, Fallback: {fallback_count})")

# Run the augmentation
augmented_data, max_len, aug_type_counts = generate_augmented_views(
    train_token_x, applicability_info, main_augmentors, fallback_augmentors)

print(aug_type_counts)

# Preview samples
for i, ex in enumerate(augmented_data[:3]):
    print(f"Original:     {ex['original']}")
    print(f"Augmented 1:  {ex['augmented_1']} ({ex['augmentor_1']})")
    print(f"Augmented 2:  {ex['augmented_2']} ({ex['augmentor_2']})")
    print("─" * 60)

augmented_1_array = np.array([ex['augmented_1'] for ex in augmented_data])
augmented_2_array = np.array([ex['augmented_2'] for ex in augmented_data])

dataset_one = tf.data.Dataset.from_tensor_slices(augmented_1_array).batch(BATCH_SIZE).repeat()
dataset_two = tf.data.Dataset.from_tensor_slices(augmented_2_array).batch(BATCH_SIZE).repeat()

### === SETTINGS === ###
steps_per_epoch = max(1, len(augmented_1_array) // BATCH_SIZE)
total_steps = steps_per_epoch * EPOCHS

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, total_steps, warmup_steps):
        super().__init__()
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)

        # Define warmup learning rate
        warmup_lr = self.base_lr * (step / warmup_steps)

        # Define cosine decay learning rate
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        progress = tf.minimum(progress, 1.0)
        cosine_decay = 0.5 * (1 + tf.cos(np.pi * progress))
        decay_lr = self.base_lr * cosine_decay

        # Use tf.cond to choose between warmup and decay
        lr = tf.cond(step < warmup_steps, lambda: warmup_lr, lambda: decay_lr)

        return lr

lr_schedule = WarmupCosineDecay(base_lr=BASE_LR, total_steps=total_steps, warmup_steps=WARMUP_EPOCHS * steps_per_epoch)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)

def update_ema_weights(target_model, online_model, tau):
    for t, o in zip(target_model.weights, online_model.weights):
        t.assign(tau * t + (1. - tau) * o)

def compute_tau(base_tau, current_step, total_steps):
    cosine_decay = tf.cos(np.pi * current_step / total_steps)
    tau = 1 - (1 - base_tau) * (cosine_decay + 1) / 2
    return tau

def byol_loss(p, z):
    p = tf.math.l2_normalize(p, axis=1)
    z = tf.math.l2_normalize(z, axis=1)
    return 2 - 2 * tf.reduce_mean(tf.reduce_sum(p * z, axis=1))

@tf.function
def train_step(x1, x2, f_online, f_target, h_online, optimizer):
    with tf.GradientTape() as tape:
        z1 = f_online(x1, training=True)
        z2 = f_online(x2, training=True)
        p1 = h_online(z1, training=True)
        p2 = h_online(z2, training=True)

        t1 = tf.stop_gradient(f_target(x1, training=False))
        t2 = tf.stop_gradient(f_target(x2, training=False))

        loss = byol_loss(p1, t2) + byol_loss(p2, t1)

    variables = f_online.trainable_variables + h_online.trainable_variables
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(grads, variables))

    return loss

def train_byol(f_online, h_online, dataset_one, dataset_two, epochs):
    f_target = tf.keras.models.clone_model(f_online)
    f_target.set_weights(f_online.get_weights())

    step_wise_loss = []
    epoch_wise_loss = []
    
    steps_per_epoch = max(1, len(augmented_1_array) // BATCH_SIZE)
    total_steps = steps_per_epoch * epochs

    ds1_iter = iter(dataset_one)
    ds2_iter = iter(dataset_two)

    step = 0
    for epoch in range(epochs):
        for _ in range(steps_per_epoch):
            x1 = next(ds1_iter)
            x2 = next(ds2_iter)
            loss = train_step(x1, x2, f_online, f_target, h_online, optimizer)
            step_wise_loss.append(loss.numpy())

            tau = compute_tau(TAU_BASE, step, total_steps)
            update_ema_weights(f_target, f_online, tau)

            step += 1

        mean_loss = np.mean(step_wise_loss[-steps_per_epoch:])
        epoch_wise_loss.append(mean_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {mean_loss:.4f} - Tau: {tau:.5f}")

    return epoch_wise_loss, f_online, f_target

f_online = get_encoder(
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    ff_dim=FF_DIM,
    num_layers=LAYERS,
    dropout=DROPOUT,
    maxlen=max_len,
    vocab_size=len(x_word_dict),
    hidden_dim=HIDDEN_DIM,
    feature_dim=FEATURE_DIM
)

h_online = get_predictor(feature_dim=FEATURE_DIM)

# Train SimSA-PPM
device_name = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
print(f"Training on {device_name}")

with tf.device(device_name):
    epoch_losses, f_trained, f_target_trained = train_byol(
        f_online=f_online,
        h_online=h_online,
        dataset_one=dataset_one,
        dataset_two=dataset_two,
        epochs=EPOCHS
    )

# Saved the trained model
model_path = f"PreTrainedModels/{dataName}_pretrained.keras"
f_trained.save(model_path)
print(f"Model saved at {model_path}")