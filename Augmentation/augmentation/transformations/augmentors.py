import dataclasses
import datetime
import math
import typing
import random
import pandas as pd

from abc import ABC

import pm4py #type: ignore
from pm4py.objects.log.obj import Trace, Event, EventLog #type: ignore
from augmentation.utils import event_log_utils #type: ignore 
from copy import deepcopy  # Import the deepcopy function


import datetime
import random
import typing
import torch
import pandas as pd
from copy import deepcopy
from abc import ABC

class BaseAugmentor(ABC):
    def augment(self, sequence: list) -> list:
        """Performs augmentation on a tokenized sequence."""
        raise NotImplementedError()

    def is_applicable(self, sequence: list) -> bool:
        """Checks if augmentation is applicable to a given sequence."""
        return len(sequence) > 2

    @staticmethod
    def get_name() -> str:
        """Returns the name of the augmentation method."""
        raise NotImplementedError()

class RandomInsertion(BaseAugmentor):
    """Inserts a random activity at a random position in the sequence."""
    
    def __init__(self, available_tokens: list):
        """Initialize with a list of valid activity tokens."""
        self.available_tokens = available_tokens

    def augment(self, sequence: list) -> list:
        """Inserts a random token at a random position in the sequence."""
        assert len(self.available_tokens) > 0, "No available activities to insert."

        augmented_sequence = deepcopy(sequence)
        position = random.randint(1, len(sequence) - 1)  # Avoid first position
        random_activity = random.choice(self.available_tokens)

        augmented_sequence.insert(position, random_activity)
        return augmented_sequence

    @staticmethod
    def get_name() -> str:
        return "RandomInsertion"


class RandomDeletion(BaseAugmentor):
    """Deletes a random activity from the sequence."""
    
    def augment(self, sequence: list) -> list:
        """Removes a random event from the sequence, ensuring it's not the first or last event."""
        assert len(sequence) > 2, "Sequence must have more than 2 events to perform deletion."

        augmented_sequence = deepcopy(sequence)
        position = random.randint(1, len(sequence) - 2)  # Avoid first and last positions

        del augmented_sequence[position]
        return augmented_sequence

    @staticmethod
    def get_name() -> str:
        return "RandomDeletion"


class RandomReplacement(BaseAugmentor):
    """Replaces a random activity in the sequence with another available activity."""
    
    def __init__(self, available_tokens: list):
        """Initialize with a list of valid activity tokens."""
        self.available_tokens = available_tokens

    def augment(self, sequence: list) -> list:
        """Replaces a random token in the sequence with another valid token."""
        assert len(sequence) > 2, "Sequence must have more than 2 events to perform replacement."
        assert len(self.available_tokens) > 0, "No available activities to replace."

        augmented_sequence = deepcopy(sequence)
        position = random.randint(0, len(sequence) - 2)  # Avoid first and last positions
        original_activity = sequence[position]
        random_activity = random.choice(self.available_tokens)

        while random_activity == original_activity:
            random_activity = random.choice(self.available_tokens)

        augmented_sequence[position] = random_activity
        return augmented_sequence

    @staticmethod
    def get_name() -> str:
        return "RandomReplacement"
    
class RandomSwap(BaseAugmentor):
    """Randomly swaps two activities in the sequence."""
    
    def augment(self, sequence: list) -> list:
        """Swaps two random events in the sequence, avoiding the first and last positions."""
        assert len(sequence) > 2, "Sequence must have more than 2 events to perform swapping."

        augmented_sequence = deepcopy(sequence)
        pos1, pos2 = random.sample(range(0, len(sequence) - 1), 2)  # Avoid first and last positions

        augmented_sequence[pos1], augmented_sequence[pos2] = augmented_sequence[pos2], augmented_sequence[pos1]
        return augmented_sequence

    @staticmethod
    def get_name() -> str:
        return "RandomSwap"


