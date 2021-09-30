from time import time

import mlflow

from Knowledge_Tracing.code.experiments.working_experiments.experiment import experiment
from Knowledge_Tracing.code.experiments.utils import import_text, process_dataset_text
from Knowledge_Tracing.code.data_processing.data_processing import *
from Knowledge_Tracing.code.data_processing.import_files import *


class reduce_and_save(experiment):
    def __init__(self):
        super().__init__("clean_datasets", "data quality and cleaning")

    def run(self, current_experiment):
        initial_t = round(time() / 60.0, 2)
        # import texts
        datasets_texts = import_text()
        # import interaction datasets
        assistment_2012 = import_assistments_2012()
        # assistment_dataset_npz = import_assistments_2012_npz()
        # assistment_dataset_2009 = import_assistments_2009()
        # POJ:
        # poj_dataset = import_poj_interactions()
        # JUNYI:
        # junyi_dataset = import_junyi_interactions()

        input_datasets = [assistment_2012]

        mlflow.end_run()
        self.time_to_import = round((time()) / 60, 2)
        print('Time to import vocab: {} mins'.format(self.time_to_import - initial_t))
        processed_datasets = []
        with mlflow.start_run(experiment_id=current_experiment.experiment_id, run_name=current_experiment.name):
            for input_dataset in input_datasets:
                x = process_dataset_text(input_dataset, datasets_texts, input_dataset.name, False)
                processed_datasets.append(x)
            datasets_interactions_with_text = []
            for processed_dataset in processed_datasets:
                dataset_interactions_with_text = Dataset(name=processed_dataset.name + "_reduced",
                                                         path=processed_dataset.path, prefix=processed_dataset.prefix)
                problems, labels, lengths, timestamps = remove_interactions_without_text(
                    processed_dataset.interacted_with_text_problem_set, processed_dataset.problems,
                    processed_dataset.labels, processed_dataset.timestamps, processed_dataset.lengths)
                dataset_interactions_with_text.set_interactions(problems, lengths, labels, timestamps=timestamps,
                                                                standard_timestamps=True, validation_percentage=0.2,
                                                                test_percentage=0.2)
                dataset_interactions_with_text.set_texts(processed_dataset.texts_list, processed_dataset.problem_id_to_index)
                dataset_interactions_with_text.compute_intersection_texts_and_interactions()
                datasets_interactions_with_text.append(dataset_interactions_with_text)
            datasets_no_duplicated_interactions = []
            for d, d_reduced in list(zip(processed_datasets, datasets_interactions_with_text)):
                dataset_no_duplicated_interactions = Dataset(name=d.name, path=d.path, prefix="clean_datasets/")

                problems, labels, lengths, timestamps = remove_duplications(d_reduced.problems, d_reduced.labels,
                                                                            d_reduced.lengths, d_reduced.timestamps)
                dataset_no_duplicated_interactions.set_interactions(problems, lengths, labels, timestamps, 0.2, 0.2)
                dataset_no_duplicated_interactions.set_texts(d_reduced.texts_list, d_reduced.problem_id_to_index)
                dataset_no_duplicated_interactions.compute_intersection_texts_and_interactions()
                dataset_no_duplicated_interactions.save_interactions()
                dataset_no_duplicated_interactions.save_texts()
                dataset_no_duplicated_interactions.write_dataset_info()
                dataset_no_duplicated_interactions.draw_graphs()
                dataset_no_duplicated_interactions.log_all()
                datasets_no_duplicated_interactions.append(dataset_no_duplicated_interactions)
        self.time_to_process = round(time() / 60.0, 2)
        print('Time to process vocab: {} mins'.format(self.time_to_process - self.time_to_import))