import typing
import random

from abc import ABC
# from pm4py.objects.log.obj import Trace, Event, EventLog
import pandas as pd
from copy import deepcopy


# class BaseAugmentor(ABC):
#     def augment(self, trace: Trace) -> Trace:
#         raise NotImplementedError()

#     def is_applicable(self, task: str, trace: Trace) -> bool:
#         raise NotImplementedError()

#     def fit(self, event_log: EventLog):
#         raise NotImplementedError()

#     # noinspection PyMethodMayBeStatic
#     def check_order_of_trace(self, trace: Trace) -> bool:
#         previous_timestamp = trace[0]['time:timestamp']
#         for event in trace[1:]:
#             if event['time:timestamp'] < previous_timestamp:
#                 return False
#             previous_timestamp = event['time:timestamp']
#         return True

#     @staticmethod
#     def get_name() -> str:
#         raise NotImplementedError()

#     @staticmethod
#     def preserves_control_flow():
#         raise NotImplementedError()

#     def to_string(self):
#         raise NotImplementedError()


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
    def __init__(self, available_tokens: list):
        self.available_tokens = available_tokens

    def is_applicable(self, sequence: list) -> bool:
        return len([x for x in sequence if x != 0]) >= 2 and len(self.available_tokens) > 0

    def augment(self, sequence: list) -> list:
        clean_seq = [x for x in sequence if x != 0]
        position = random.randint(1, len(clean_seq) - 1)
        random_activity = random.choice(self.available_tokens)

        augmented = deepcopy(clean_seq)
        augmented.insert(position, random_activity)

        # Pad left to match original length
        return [0.0] * (len(sequence) - len(augmented)) + augmented

    @staticmethod
    def get_name() -> str:
        return "RandomInsertion"

class RandomDeletion(BaseAugmentor):
    def is_applicable(self, sequence: list) -> bool:
        return len([x for x in sequence if x != 0]) > 2

    def augment(self, sequence: list) -> list:
        clean_seq = [x for x in sequence if x != 0]
        position = random.randint(1, len(clean_seq) - 2)
        augmented = deepcopy(clean_seq)
        del augmented[position]

        return [0.0] * (len(sequence) - len(augmented)) + augmented

    @staticmethod
    def get_name() -> str:
        return "RandomDeletion"

class RandomReplacement(BaseAugmentor):
    def __init__(self, available_tokens: list):
        self.available_tokens = available_tokens

    def is_applicable(self, sequence: list) -> bool:
        clean_seq = [x for x in sequence if x != 0]
        return len(clean_seq) > 2 and len(self.available_tokens) > 0

    def augment(self, sequence: list) -> list:
        clean_seq = [x for x in sequence if x != 0]
        if not self.is_applicable(sequence):
            return sequence

        position = random.randint(1, len(clean_seq) - 2)
        replacement = random.choice(self.available_tokens)
        while replacement == clean_seq[position]:
            replacement = random.choice(self.available_tokens)

        augmented = deepcopy(clean_seq)
        augmented[position] = replacement

        # Pad left to match original length
        return [0.0] * (len(sequence) - len(augmented)) + augmented

    @staticmethod
    def get_name() -> str:
        return "RandomReplacement"




               
class StatisticalReplacement(BaseAugmentor):
    def __init__(self, patterns_df: pd.DataFrame):
        self.patterns_df = patterns_df
        self.alternative_cols = [col for col in patterns_df.columns if col.startswith("Alternative")]

    def is_applicable(self, sequence: list) -> bool:
        for _, row in self.patterns_df.iterrows():
            start = row['Start Activity']
            end = row['End Activity']
            alternatives = [row[col] for col in self.alternative_cols if pd.notna(row[col])]
            for i in range(len(sequence) - 2):
                if sequence[i] == start and sequence[i + 2] == end:
                    mid = sequence[i + 1]
                    if mid in alternatives and any(a != mid for a in alternatives):
                        return True
        return False

    def augment(self, sequence: list) -> list:
        valid_positions = []
        for _, row in self.patterns_df.iterrows():
            start = row['Start Activity']
            end = row['End Activity']
            alternatives = [row[col] for col in self.alternative_cols if pd.notna(row[col])]

            for i in range(len(sequence) - 2):
                if sequence[i] == start and sequence[i + 2] == end:
                    mid = sequence[i + 1]
                    if mid in alternatives:
                        alt_choices = [a for a in alternatives if a != mid]
                        if alt_choices:
                            valid_positions.append((i + 1, alt_choices))

        assert valid_positions, "No valid replacement positions found."

        pos, candidates = random.choice(valid_positions)
        replacement = random.choice(candidates)

        augmented = deepcopy(sequence)
        augmented[pos] = replacement
        return augmented

    @staticmethod
    def get_name() -> str:
        return 'StatisticalReplacement'

class StatisticalDeletion(BaseAugmentor):
    def __init__(self, patterns_df: pd.DataFrame):
        self.patterns_df = patterns_df
        self.intermediate_cols = [col for col in patterns_df.columns if col.startswith("Intermediate")]

    def is_applicable(self, sequence: list) -> bool:
        """
        Return True if the sequence contains (start, intermediate(s), end) subsequence.
        """
        for _, row in self.patterns_df.iterrows():
            start = row["Starting Activity"]
            end = row["Ending Activity"]
            intermediates = [int(row[col]) for col in self.intermediate_cols if pd.notna(row[col])]

            for i in range(len(sequence) - len(intermediates) - 1):
                window = sequence[i : i + len(intermediates) + 2]
                if (
                    window[0] == start
                    and window[-1] == end
                    and window[1:-1] == intermediates
                ):
                    return True
        return False

    def augment(self, sequence: list) -> list:
        valid_matches = []

        for _, row in self.patterns_df.iterrows():
            start = row["Starting Activity"]
            end = row["Ending Activity"]
            intermediates = [int(row[col]) for col in self.intermediate_cols if pd.notna(row[col])]

            for i in range(len(sequence) - len(intermediates) - 1):
                window = sequence[i : i + len(intermediates) + 2]
                if (
                    window[0] == start
                    and window[-1] == end
                    and window[1:-1] == intermediates
                ):
                    # collect the positions of intermediates to remove
                    valid_matches.append((i + 1, i + 1 + len(intermediates)))

        assert valid_matches, "No valid deletions found."

        start_del, end_del = random.choice(valid_matches)
        augmented_sequence = sequence[:start_del] + sequence[end_del:]

        return augmented_sequence

    @staticmethod
    def get_name() -> str:
        return 'StatisticalDeletion'

class StatisticalInsertion(BaseAugmentor):
    def __init__(self, patterns_df: pd.DataFrame):
        self.patterns_df = patterns_df
        self.intermediate_cols = [col for col in patterns_df.columns if col.startswith("Intermediate")]

    def is_applicable(self, sequence: list) -> bool:
        """
        Check if the sequence has at least one (start, end) pair directly next to each other.
        """
        for _, row in self.patterns_df.iterrows():
            start = row['Starting Activity']
            end = row['Ending Activity']
            for i in range(len(sequence) - 1):
                if sequence[i] == start and sequence[i + 1] == end:
                    return True
        return False

    def augment(self, sequence: list) -> list:
        """
        Insert intermediate tokens between matched (start, end) pairs.
        Picks one valid pattern at random.
        """
        valid_insertions = []

        for _, row in self.patterns_df.iterrows():
            start = row['Starting Activity']
            end = row['Ending Activity']
            intermediates = [int(row[col]) for col in self.intermediate_cols if pd.notna(row[col])]

            for i in range(len(sequence) - 1):
                if sequence[i] == start and sequence[i + 1] == end:
                    valid_insertions.append((i, intermediates))

        assert valid_insertions, "No valid (start → end) insertion points found."

        insert_pos, to_insert = random.choice(valid_insertions)

        augmented_sequence = sequence[:insert_pos + 1] + to_insert + sequence[insert_pos + 1:]

        return augmented_sequence

    @staticmethod
    def get_name() -> str:
        return 'StatisticalInsertion'



    
    


