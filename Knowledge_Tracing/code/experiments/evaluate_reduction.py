from time import time

from Knowledge_Tracing.code.experiments.experiment import experiment
from Knowledge_Tracing.code.experiments.utils import *
from Knowledge_Tracing.code.data_processing.data_processing import *
from Knowledge_Tracing.code.data_processing.import_files import *
from Knowledge_Tracing.code.visualization.histogram import *


class evaluate_reduction(experiment):
    def __init__(self):
        super().__init__("evaluate_reduction_on_different_datasets", "data quality and cleaning")

    def run(self):
        initial_t = round(time()/60.0, 2)
        # import text of poj (needed to import its interactions)
        datasets_texts = import_text()
        # import interaction datasets
        #assistment_dataset_npz = import_assistments_2012_npz()
        #assistment_dataset_2009 = import_assistments_2009()
        #assistment_dataset_2012 = import_assistments_2012()
        # POJ:
        poj_dataset = import_poj_interactions()
        # JUNYI:
        junyi_dataset = import_junyi_interactions()

        input_datasets = [poj_dataset, junyi_dataset]

        self.time_to_import = round((time())/60, 2)
        print('Time to import vocab: {} mins'.format(self.time_to_import - initial_t))
        processed_datasets = []
        for input_dataset in input_datasets:
            processed_datasets.append(process_dataset_text(input_dataset, datasets_texts, input_dataset.name, False))

        reduced_datasets = []
        for processed_dataset in processed_datasets:
            reduced_dataset = Dataset(name=processed_dataset.name+"_reduced", path=processed_dataset.path)
            problems, labels, lengths = remove_interactions_without_text(
                processed_dataset.interacted_with_text_problem_set, processed_dataset.problems, processed_dataset.labels,
                processed_dataset.lengths)
            reduced_dataset.set_interactions(problems, lengths, labels, 0.2, 0.2)
            reduced_dataset.set_texts(processed_dataset.texts_list, processed_dataset.problem_id_to_index)
            reduced_dataset.compute_intersection_texts_and_interactions()
            reduced_datasets.append(reduced_dataset)

        self.time_to_process = round(time()/60.0, 2)
        print('Time to process vocab: {} mins'.format(self.time_to_process - self.time_to_import))

        for reduced_dataset, processed_dataset in list(zip(reduced_datasets, processed_datasets)):
            processed_dataset.write_dataset_info()
            reduced_dataset.write_dataset_info()
        data = [x.lengths for x in processed_datasets]
        labels = [x.name for x in processed_datasets]
        bin_size = [5 for x in processed_datasets]
        comparison_histogram(data=data, labels=labels, bin_size=bin_size,
                             path="C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/logs/",
                             name="processed_lengths")

        data = [x.lengths for x in reduced_datasets]
        labels = [x.name for x in reduced_datasets]

        comparison_histogram(data=data, labels=labels, bin_size=bin_size,
                             path="C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/logs/",
                             name="reduced_lengths")

        data = [x.texts_lengths for x in processed_datasets]
        labels = [x.name for x in processed_datasets]
        bin_size = [5 for x in processed_datasets]
        comparison_histogram(data=data, labels=labels, bin_size=bin_size,
                             path="C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/logs/",
                             name="processed_texts_lengths")

        data = [x.texts_lengths for x in reduced_datasets]
        labels = [x.name for x in reduced_datasets]

        comparison_histogram(data=data, labels=labels, bin_size=bin_size,
                             path="C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/logs/",
                             name="reduced_texts_lengths")
