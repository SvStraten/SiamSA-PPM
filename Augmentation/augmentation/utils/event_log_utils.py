import random
import typing

import numpy as np
import pandas.core.groupby as pd_groupby
from pm4py.objects.log.obj import EventLog #type: ignore
from pm4py.objects.log.util import dataframe_utils #type: ignore
from pm4py.objects.conversion.log import converter as log_converter #type: ignore
import pandas as pd
from datetime import timedelta


def df_number_of_traces(df) -> int:
    return df['case:concept:name'].nunique()

def df_average_trace_length(df):
    """Calculates the average trace length from a DataFrame."""
    trace_lengths = df.groupby('case:concept:name').size()
    return trace_lengths.mean()


def df_min_trace_length(df):
    """Finds the minimum trace length from a DataFrame."""
    trace_lengths = df.groupby('case:concept:name').size()
    return trace_lengths.min()


def df_max_trace_length(df):
    """Finds the maximum trace length from a DataFrame."""
    trace_lengths = df.groupby('case:concept:name').size()
    return trace_lengths.max()


def df_case_durations(df):
    """Calculates the durations of all cases."""
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
    start_times = df.groupby('case:concept:name')['time:timestamp'].min()
    end_times = df.groupby('case:concept:name')['time:timestamp'].max()
    durations = end_times - start_times
    return durations


def df_minimal_case_duration(df):
    """Finds the minimal duration of a case."""
    durations = df_case_durations(df)
    return durations.min()


def df_maximal_case_duration(df):
    """Finds the maximal duration of a case."""
    durations = df_case_durations(df)
    return durations.max()


def df_average_case_duration(df):
    """Calculates the average duration of a case."""
    durations = df_case_durations(df)
    return durations.mean()


def get_activity_resources(df: pd.DataFrame) -> typing.Dict[str, typing.List[str]]:
    # Ensure 'concept:name' and 'org:resource' columns exist
    if 'concept:name' not in df.columns or 'org:resource' not in df.columns:
        raise ValueError("DataFrame must contain 'concept:name' and 'org:resource' columns")

    # Initialize the dictionary with sets to prevent duplicate resources
    activities_resources = {activity: set() for activity in df['concept:name'].unique()}

    # Loop through each event (row) in the DataFrame
    for _, event in df.iterrows():
        activity = event['concept:name']
        resource = event['org:resource']
        activities_resources[activity].add(resource)

    # Convert the sets to lists
    return {activity: list(resources) for activity, resources in activities_resources.items()}


####
# Legacy Code
######

def convert_to_event_log(df):
    """Converts a DataFrame to an event log."""
    df = dataframe_utils.convert_timestamp_columns_in_df(df)
    parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case_id'}
    event_log = log_converter.apply(df, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
    return event_log


def average_trace_length(event_log):
    """Calculates the average trace length."""
    lengths = [len(trace) for trace in event_log]
    return sum(lengths) / len(lengths)


def min_trace_length(event_log):
    """Finds the minimum trace length."""
    return min(len(trace) for trace in event_log)


def max_trace_length(event_log):
    """Finds the maximum trace length."""
    return max(len(trace) for trace in event_log)


def case_durations(event_log):
    """Calculates the durations of all cases."""
    durations = []
    for trace in event_log:
        start_time = trace[0]["time:timestamp"]
        end_time = trace[-1]["time:timestamp"]
        duration = end_time - start_time
        durations.append(duration)
    return durations


def minimal_case_duration(event_log):
    """Finds the minimal duration of a case."""
    durations = case_durations(event_log)
    return min(durations)


def maximal_case_duration(event_log):
    """Finds the maximal duration of a case."""
    durations = case_durations(event_log)
    return max(durations)


def average_case_duration(event_log):
    """Calculates the average duration of a case."""
    durations = case_durations(event_log)
    total_duration = sum(durations, timedelta())
    return total_duration / len(durations)


#####


def get_case_ids(event_log: EventLog | pd.DataFrame) -> typing.Set:
    if isinstance(event_log, EventLog):
        return set([trace.attributes['concept:name'] for trace in event_log])
    else:
        return set([s for s in event_log['case:concept:name'].unique()])


def get_activities(event_log: pd.DataFrame) -> typing.Set[str]:
    return set([s for s in event_log['concept:name'].unique()])


def get_resources(event_log: pd.DataFrame) -> typing.Set[str]:
    if 'org:resource' in event_log.columns:
        return event_log['org:resource'].unique()
    else:
        return set()


def get_traces(event_log: pd.DataFrame) -> pd_groupby.DataFrameGroupBy:
    return event_log.sort_values(by='time:timestamp').groupby(by='case:concept:name')


def sample_traces(event_log: pd.DataFrame, num_samples: int, strategy: str) -> typing.List[pd.DataFrame]:
    assert strategy in ['uniform', 'rarity', 'first_n']

    cases = event_log['case:concept:name'].unique()
    if len(cases.tolist()) < num_samples:
        raise ValueError(f'Sampling {num_samples} traces, but log only contains {len(cases.tolist())}, '
                         f'returning only {num_samples} traces!')
    else:
        if strategy == 'uniform':
            cases = _sample_cases_uniformly(event_log, num_samples, cases)
        elif strategy == 'first_n':
            cases = _first_n(num_samples, cases)
        else:
            cases = _sample_cases_by_rarity(event_log, num_samples, cases)
    return [
        event_log[event_log['case:concept:name'] == c] for c in cases
    ]


def _first_n(num_samples: int, cases: typing.List[str]):
    return cases[:num_samples]


def _sample_cases_uniformly(_: pd.DataFrame, num_samples: int, cases: str) -> typing.Iterable[str]:
    return np.random.choice(cases, num_samples, replace=False)


def _sample_cases_by_rarity(event_log: pd.DataFrame, num_samples: int, cases: str) -> typing.Iterable[str]:
    trace_variants: typing.Dict[str, typing.List[str]] = {}
    for case in cases:
        trace = event_log[event_log['case:concept:name'] == case]
        trace_id = get_trace_flow(trace)
        if trace_id not in trace_variants:
            trace_variants[trace_id] = []
        trace_variants[trace_id].append(case)
    traces_count = [len(traces) for traces in trace_variants.values()]
    total_count = sum(traces_count)

    # invert the chance to draw this trace --> rare = high chance, common = low chance
    p = [total_count / v for v in traces_count]

    # rescale to sum(p) == 1.0
    total_count = sum(p)
    p = [v / total_count for v in p]

    selected_trace_variants = np.random.choice(list(trace_variants.keys()), num_samples, replace=True, p=p)
    return [random.choice(trace_variants[trace_variant_id]) for trace_variant_id in selected_trace_variants]


def get_trace_flow(trace: pd.DataFrame) -> str:
    return ' >> '.join([e for e in trace['concept:name']])


def trace_contains_activity(trace: pd.DataFrame, activity: str) -> bool:
    return activity in trace['concept:name'].unique()


def get_trace_prefix(trace: pd.DataFrame, prefix_length=2) -> pd.DataFrame:
    return trace.head(prefix_length)


def get_num_traces(event_log: pd.DataFrame) -> int:
    return len(event_log['case:concept:name'].unique().tolist())


def get_trace_length(trace: pd.DataFrame) -> int:
    return len(trace.index)


def get_longest_trace(event_log: pd.DataFrame) -> pd.DataFrame:
    longest_length_longest_trace = (0, pd.DataFrame())
    for case_id, trace in get_traces(event_log=event_log):
        current_length = get_trace_length(trace=trace)
        if longest_length_longest_trace[0] < current_length:
            longest_length_longest_trace = (current_length, trace)
    return longest_length_longest_trace[1]


def get_variants(event_log: pd.DataFrame):
    df = event_log.copy()
    df_trace_idx = df.groupby('case:concept:name').cumcount()
    df['@@index_in_trace'] = df_trace_idx

    variants = []
    df = df.sort_values(['case:concept:name', '@@index_in_trace']).groupby('case:concept:name')
    for groupName, df_group in df:
        variants.append(' '.join([event['concept:name'] + "|" for _, event in df_group.iterrows()]))

    return set(variants)


def remove_duplicates(variants: typing.List[typing.List[str]]) -> typing.List[typing.List[str]]:
    seen = set()
    unique_list_of_lists = []

    for sublist in variants:
        sublist_tuple = tuple(sublist)
        if sublist_tuple not in seen:
            seen.add(sublist_tuple)
            unique_list_of_lists.append(sublist)

    return unique_list_of_lists


def get_variants_as_list(event_log: pd.DataFrame):
    df = event_log.copy()
    df_trace_idx = df.groupby('case:concept:name').cumcount()
    df['@@index_in_trace'] = df_trace_idx

    variants = []
    df = df.sort_values(['case:concept:name', '@@index_in_trace']).groupby('case:concept:name')
    for groupName, df_group in df:
        variant = [event['concept:name'] for _, event in df_group.iterrows()]
        variants.append(variant)

    return remove_duplicates(variants)


def filter_by_cases(event_log: pd.DataFrame, case_ids: typing.List[str]):
    df = event_log.loc[event_log['case:concept:name'].isin(case_ids)]
    return df


def only_control_flow(trace) -> str:
    return ' '.join([event['concept:name'] + "|" for event in trace])
