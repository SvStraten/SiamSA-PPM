from collections import Counter
from typing import List, Tuple
import pandas as pd
from collections import defaultdict
import numpy as np



import pandas as pd
from collections import Counter, defaultdict


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


def get_significant_transitions(log: pd.DataFrame, threshold: float = 0.00001) -> pd.DataFrame:
    transition_counts = Counter()
    traces = log.groupby('case:concept:name')

    for _, trace in traces:
        events = trace.sort_values("position")["concept:name"].tolist()
        for i in range(len(events) - 1):
            transition_counts[(events[i], events[i + 1])] += 1

    total_transitions = sum(transition_counts.values())
    transition_probs = {
        (source, target): count / total_transitions
        for (source, target), count in transition_counts.items()
    }

    transition_df = pd.DataFrame([
        {'Source': source, 'Target': target, 'Probability': prob}
        for (source, target), prob in transition_probs.items()
    ])

    return transition_df[transition_df['Probability'] >= threshold]


def get_significant_paths(log: pd.DataFrame, threshold: float = 0.001, max_length: int = 3) -> pd.DataFrame:
    path_counts = defaultdict(int)
    traces = log.groupby('case:concept:name')

    for _, trace in traces:
        events = trace.sort_values("position")["concept:name"].tolist()
        for i in range(len(events)):
            for j in range(i + 1, min(i + max_length + 1, len(events) + 1)):
                path = tuple(events[i:j])
                path_counts[path] += 1

    total_paths = sum(path_counts.values())
    path_df = pd.DataFrame([
        {'Path': ' -> '.join(path), 'Frequency': count, 'Support': count / total_paths}
        for path, count in path_counts.items()
    ])

    return path_df[path_df['Support'] >= threshold]


def get_patterns(csv_path: str,
                          transition_threshold: float = 0.2,
                          path_threshold: float = 0.2,
                          max_path_length: int = 3,
                          activity_threshold: float = 0.00001) -> pd.DataFrame:
    """
    Main function to generate the patterns dataframe from a preprocessed CSV file.
    """
    log = expand_prefix_csv_to_log(csv_path)
    log = get_significant_activities(log, threshold_ratio=activity_threshold)

    transitions = get_significant_transitions(log, threshold=transition_threshold)
    paths = get_significant_paths(log, threshold=path_threshold, max_length=max_path_length)

    patterns = []

    for _, row in transitions.iterrows():
        patterns.append({
            'Pattern Type': 'Transition',
            'Starting Activity': row['Source'],
            'Ending Activity': row['Target'],
            'Intermediate Activities': [],
            'Probability': row['Probability']
        })

    max_intermediates = 0
    for _, row in paths.iterrows():
        path = row['Path'].split(' -> ')
        intermediates = path[1:-1] if len(path) > 2 else []
        max_intermediates = max(max_intermediates, len(intermediates))
        patterns.append({
            'Pattern Type': 'Path',
            'Starting Activity': path[0],
            'Ending Activity': path[-1],
            'Intermediate Activities': intermediates,
            'Support': row['Support']
        })

    # Convert to DataFrame with dynamic intermediate columns
    records = []
    for p in patterns:
        record = {
            'Starting Activity': p['Starting Activity'],
            'Ending Activity': p['Ending Activity'],
        }
        intermediates = p['Intermediate Activities']
        for i in range(max_intermediates):
            record[f'Intermediate {i + 1}'] = intermediates[i] if i < len(intermediates) else None
        records.append(record)

    patterns_df = pd.DataFrame(records)

    intermediate_cols = [col for col in patterns_df.columns if col.startswith('Intermediate')]
    patterns_df = patterns_df.dropna(subset=intermediate_cols, how='all')
    patterns_df = patterns_df.where(pd.notnull(patterns_df), None)

    return patterns_df.sort_values(['Starting Activity', 'Ending Activity'] + intermediate_cols)

def map_patterns_to_tokens(patterns_df: pd.DataFrame, x_word_dict: dict) -> pd.DataFrame:
    """
    Converts all activity names in a patterns_df to their corresponding token values,
    and ensures all resulting values are of type np.float32.
    """
    token_df = patterns_df.copy()
    activity_cols = ['Starting Activity', 'Ending Activity'] + \
                    [col for col in token_df.columns if col.startswith('Intermediate')]

    for col in activity_cols:
        token_df[col] = token_df[col].apply(
            lambda x: np.float32(x_word_dict[x]) if pd.notna(x) and x in x_word_dict else np.nan
        )

    return token_df



