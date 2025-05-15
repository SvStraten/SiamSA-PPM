from collections import Counter
from typing import List, Tuple
import pandas as pd
from collections import defaultdict

import pandas as pd
from collections import defaultdict
import numpy as np


def expand_prefix_csv_to_log(csv_path: str) -> pd.DataFrame:
    """
    Converts a CSV with columns [case_id, prefix, next_act] into flat event log format.
    """
    df = pd.read_csv(csv_path)
    records = []

    for _, row in df.iterrows():
        case_id = row["case_id"]
        activities = row["prefix"].split()

        for pos, act in enumerate(activities):
            records.append({
                "case:concept:name": case_id,
                "concept:name": act,
                "position": pos
            })

    event_log = pd.DataFrame(records)
    return event_log


def get_significant_activities(log: pd.DataFrame, threshold_ratio: float = 0.00001) -> pd.DataFrame:
    total_cases = log['case:concept:name'].nunique()
    activity_trace_counts = log.groupby('concept:name')['case:concept:name'].nunique()
    threshold = threshold_ratio * total_cases
    significant_activities = activity_trace_counts[activity_trace_counts >= threshold].index

    return log[log['concept:name'].isin(significant_activities)].copy()


def get_xor_candidates(csv_path: str, 
                                support_threshold: float = 0.01, 
                                max_path_length: int = 3, 
                                activity_threshold: float = 0.00001) -> pd.DataFrame:
    """
    Identifies XOR candidate paths in a CSV-based event log (e.g., next_activity_train.csv).

    Parameters:
    - csv_path (str): Path to CSV file with [case_id, prefix, next_act].
    - support_threshold (float): Minimum support threshold to consider a pattern.
    - max_path_length (int): Max path length (e.g., 3 means A → X → B).
    - activity_threshold (float): Minimum frequency ratio to keep an activity.

    Returns:
    - pd.DataFrame: DataFrame with XOR candidate patterns.
    """
    log = expand_prefix_csv_to_log(csv_path)
    log = get_significant_activities(log, threshold_ratio=activity_threshold)

    total_cases = log['case:concept:name'].nunique()
    xor_candidates = defaultdict(lambda: {"count": defaultdict(int), "total": 0})

    traces = log.groupby('case:concept:name')
    for case_id, trace in traces:
        events = trace.sort_values("position")["concept:name"].tolist()
        seen_triples = set()
        seen_pairs = set()

        for i in range(len(events) - max_path_length + 1):
            first = events[i]
            middle = events[i + 1]
            last = events[i + max_path_length - 1]
            triple = (first, middle, last)
            pair = (first, last)

            if pair not in seen_pairs:
                xor_candidates[pair]["total"] += 1
                seen_pairs.add(pair)

            if triple not in seen_triples:
                xor_candidates[pair]["count"][middle] += 1
                seen_triples.add(triple)

    max_alternatives = max((len(data["count"]) for data in xor_candidates.values()), default=0)
    xor_candidates_records = []

    for (start, end), data in xor_candidates.items():
        alts = data["count"]
        total_count = data["total"]

        if len(alts) > 1 and (total_count / total_cases) >= support_threshold:
            record = {
                'Start Activity': start,
                'End Activity': end,
                'Num Alternatives': len(alts),
            }
            for i, alt in enumerate(sorted(alts.keys())):
                record[f'Alternative {i + 1}'] = alt
            xor_candidates_records.append(record)

    xor_candidates_df = pd.DataFrame(xor_candidates_records)
    xor_candidates_df = xor_candidates_df.where(pd.notnull(xor_candidates_df), None)

    return xor_candidates_df

def map_xor_candidates_to_tokens(xor_df: pd.DataFrame, x_word_dict: dict) -> pd.DataFrame:
    """
    Converts XOR candidate activity names to their corresponding token values,
    and ensures all resulting values are of type np.float32.
    """
    token_df = xor_df.copy()
    activity_cols = ['Start Activity', 'End Activity'] + \
                    [col for col in token_df.columns if col.startswith('Alternative')]

    for col in activity_cols:
        token_df[col] = token_df[col].apply(
            lambda x: np.float32(x_word_dict[x]) if pd.notna(x) and x in x_word_dict else np.nan
        )

    return token_df

