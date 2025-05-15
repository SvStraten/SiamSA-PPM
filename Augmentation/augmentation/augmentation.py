import random
import torch
from augmentation.transformations.augmentors import RandomReplacement, RandomInsertion, RandomDeletion, RandomSwap

import random

def augment_training_set(sequences, main_augmentors, x_word_dict):
    available_activities = list(x_word_dict.values())

    fallback_augmentors = {
        "UsefulInsertion": RandomInsertion(available_activities),
        "UsefulDeletion": RandomDeletion(),
        "UsefulReplacement": RandomReplacement(available_activities),
        "ParallelSwap": RandomSwap()
    }

    print(main_augmentors)

    original_list = []
    x1_list = []
    x2_list = []
    augmentor_usage_count = {aug.get_name(): 0 for aug in main_augmentors}
    for fallback in fallback_augmentors.values():
        augmentor_usage_count[fallback.get_name()] = 0
    augmentor_usage_count['No Augmentation'] = 0

    for sequence in sequences:
        original_sequence = sequence.copy()

        random.shuffle(main_augmentors)  # Shuffle before selecting augmentors
        x1_augmentor = random.choice(main_augmentors)
        x2_augmentor = random.choice([aug for aug in main_augmentors if aug != x1_augmentor])
        
        def try_augment(sequence, augmentor):
            try:
                augmented_sequence = augmentor.augment(sequence)
                augmentor_usage_count[augmentor.get_name()] += 1
                print(augmented_sequence, augmentor)
                return augmented_sequence
            except Exception:
                fallback = fallback_augmentors.get(augmentor.get_name())
                if fallback:
                    try:
                        augmented_sequence = fallback.augment(sequence)
                        augmentor_usage_count[fallback.get_name()] += 1
                        return augmented_sequence
                    except Exception:
                        pass
            augmentor_usage_count['No Augmentation'] += 1
            return sequence
        
        x1 = try_augment(sequence, x1_augmentor)
        x2 = try_augment(sequence, x2_augmentor)

        original_list.append(original_sequence)
        x1_list.append(x1)
        x2_list.append(x2)

    print("\nAugmentor usage count:")
    for augmentor, count in augmentor_usage_count.items():
        print(f"{augmentor}: {count}")

    return original_list, x1_list, x2_list













