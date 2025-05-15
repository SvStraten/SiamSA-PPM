import abc
import dataclasses
import json
import typing
import pandas as pd
import math
import random
import pm4py #type: ignore
import os

from augmentation.utils import event_log_utils
from pm4py.objects.log.obj import EventLog, Trace #type: ignore

from augmentation.pipelines.abstract_context import AbstractPipelineContext
from augmentation.pipelines.step import BasePipelineStep
from augmentation.transformations import augmentors

Augmentor = typing.TypeVar('Augmentor', bound=augmentors.BaseAugmentor)


class EdaAugmentation(abc.ABC):

    def __init__(self, activities: typing.Set[str], resources: typing.Set[str], event_log: pd.DataFrame, augmentors: typing.List[Augmentor],
                 augmentation_factor: float, allow_multiple: bool, record_augmentation: bool, dry_run: bool):
        self._activities = activities
        self._resources = resources
        self._event_log = event_log
        self._augmentors = augmentors
        self._augmentation_factor = augmentation_factor
        self._allow_multiple = allow_multiple
        self._record_augmentation = record_augmentation
        self._dry_run = dry_run

    @staticmethod
    def _get_trace_by_case_id(case_id: str, event_log: EventLog) -> typing.Union[Trace, None]:
        for trace in event_log:
            if trace.attributes['concept:name'] == case_id:
                return trace
        return None

    def augment(self) -> typing.Tuple[pd.DataFrame, typing.Dict, typing.Dict]:
        augmentation_count = {k.get_name(): 0 for k in self._augmentors}
        augmentation_record = {k.get_name(): [] for k in self._augmentors}

        if self._dry_run:
            return self._event_log, augmentation_count, augmentation_record

        augmented_event_log = self._event_log.__deepcopy__()
        augmented_event_log = pm4py.convert_to_event_log(augmented_event_log)

        number_of_traces_given = len(event_log_utils.get_case_ids(event_log=self._event_log))
        case_ids = list(event_log_utils.get_case_ids(event_log=self._event_log))
        traces_to_generate = math.ceil(self._augmentation_factor * number_of_traces_given) - number_of_traces_given

        i = 0
        while traces_to_generate > 0:
            random.shuffle(case_ids)
            current_case_index = number_of_traces_given - traces_to_generate + 1  # Track current case index
            total_cases = number_of_traces_given

            # Progress indicator
            print(f"Processing case {current_case_index}/{total_cases}...")

            if self._allow_multiple:
                trace = EdaAugmentation._get_trace_by_case_id(case_ids[0], augmented_event_log)
            else:
                trace = EdaAugmentation._get_trace_by_case_id(case_ids[0], pm4py.convert_to_event_log(self._event_log))

            # Select randomly an augmentation strategy for the selected trace
            random.shuffle(self._augmentors)

            if self._augmentors[0].is_applicable('', trace) is True:
                try:
                    new_id = f'{case_ids[0]}_{i}'
                    augmented_trace = self._augmentors[0].augment(trace)

                    augmented_event_log.append(Trace(augmented_trace[:], attributes={
                        'concept:name': new_id,
                        'creator': self._augmentors[0].get_name()
                    }))
                    # It is important that the following code is executed after calling augment, since in case of
                    # an exception it should not be executed!
                    if self._record_augmentation is True:
                        augmentation_count[self._augmentors[0].get_name()] = augmentation_count[
                                                                                self._augmentors[0].get_name()] + 1
                        augmentation_record[self._augmentors[0].get_name()].append(case_ids[0])

                    # Add trace to new event log (maybe also old)
                    if self._allow_multiple:
                        case_ids.append(new_id)

                    traces_to_generate = traces_to_generate - 1
                    i = i + 1
                except AssertionError:
                    print(f'WARNING: An assertion occurred (maybe the time order is violated due to low precision), '
                        f'we retry again with an other augmentor or an other trace.')
                    continue
            else:
                print(
                    'Augmentor could not applied to this trace. We try to augment an other '
                    'trace or with an other augmentor')

        if self._allow_multiple is False:
            for trace in self._event_log:
                augmented_event_log.append(trace)

        return pm4py.convert_to_dataframe(augmented_event_log.__deepcopy__()), augmentation_count, augmentation_record


@dataclasses.dataclass
class EdaAugmentationConfig:
    augmentation: typing.Type[EdaAugmentation]
    augmentors: typing.List[Augmentor]
    augmentation_factor: float
    allow_multiple: bool
    record_augmentation: bool


class EdaAugmenter(BasePipelineStep):

    def __init__(self, augmentations: typing.List[EdaAugmentationConfig], target_dir: str) -> None:
        super().__init__()
        self._augmentations = augmentations
        self._target_dir = target_dir

    def get_logging_content(self) -> typing.Dict | typing.List | str:
        aug_configs = []
        for aug in self._augmentations:
            aug_config_to_log = {
                'aug_name': aug.augmentation.__name__,
                'augmentors': [x.get_name() for x in aug.augmentors],
                'augmentation factor': aug.augmentation_factor,
                'allow_multiple': aug.allow_multiple,
                'record_augmentation': aug.record_augmentation
            }
            aug_configs.append(aug_config_to_log)

        return aug_configs

    def store_records(self, records: typing.List[dict]) -> None:
        print(records)
        with open(os.path.join(self._target_dir, 'record.json'), 'w', encoding='utf8') as f:
            json.dump(records, f, indent=4)

    def store_counts(self, counts: typing.List[dict]) -> None:
        print(counts)
        with open(os.path.join(self._target_dir, 'count.json'), 'w', encoding='utf8') as f:
            json.dump(counts, f, indent=4)

    def run(self, train_aug: typing.Union[typing.List[pd.DataFrame], pd.DataFrame], context: AbstractPipelineContext,
            dry_run: bool = False) -> typing.Union[typing.List[pd.DataFrame], pd.DataFrame]:
        all_activities, all_resources = EdaAugmenter._get_all_activities_and_resources_in_full_log(context)

        all_augmentations = []
        all_counts = []
        all_records = []
        for aug in self._augmentations:
            for augmenter in aug.augmentors:
                augmenter.fit(context.get_full_event_log())

            augmenter = aug.augmentation(activities=all_activities, resources=all_resources, event_log=train_aug,
                                         augmentors=aug.augmentors,
                                         augmentation_factor=aug.augmentation_factor, allow_multiple=aug.allow_multiple,
                                         record_augmentation=aug.record_augmentation, dry_run=dry_run)

            augmented_event_log, augmentation_count, augmentation_record = augmenter.augment()
            all_augmentations.append(augmented_event_log)
            all_counts.append(augmentation_count)
            all_records.append(augmentation_record)

        self.store_counts(all_counts)
        self.store_records(all_records)

        return all_augmentations

    @staticmethod
    def _get_all_activities_and_resources_in_full_log(context: AbstractPipelineContext) -> typing.Tuple[
        typing.Set[str], typing.Set[str]]:
        full_log = context.get_full_event_log()
        return event_log_utils.get_activities(full_log), event_log_utils.get_resources(full_log)

    @staticmethod
    def get_name() -> str:
        return 'Easy Data Augmentation Augmenter'

