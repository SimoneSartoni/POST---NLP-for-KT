from time import time

from Knowledge_Tracing.code.experiments.experiment import experiment
from Knowledge_Tracing.code.experiments.utils import *
from Knowledge_Tracing.code.data_processing.data_processing import *
from Knowledge_Tracing.code.data_processing.import_files import *


class TF_IDF_experiment(experiment):
    def __init__(self):
        super().__init__("TF_IDF", "encoding")

    def run(self):
        t = time()
        # import text of poj (needed to import its interactions)
        datasets_texts = import_text()
        # import interaction datasets
        # assistment_dataset_npz = import_assistments_2012_npz()
        # assistment_dataset_2009 = import_assistments_2009()
        # assistment_dataset_2012 = import_assistments_2012()

        # POJ:
        poj_dataset = import_poj_interactions()

        # JUNYI:
        junyi_dataset = import_junyi_interactions()

        input_datasets = [junyi_dataset]
        self.time_to_import = round((time()-t)/60, 2)
        print('Time to import vocab: {} mins'.format(self.time_to_import))
        processed_datasets = []
        for input_dataset in input_datasets:
            processed_datasets.append(process_dataset_text(input_dataset, datasets_texts, input_dataset.name, False))
        self.time_to_process = round(time()/60, 2) - self.time_to_import
        print('Time to process vocab: {} mins'.format(self.time_to_process))

        for processed_dataset in processed_datasets:
            models = add_tf_idf_model([], processed_dataset, True)

        self.time_to_train = round(time()/60, 2) - self.time_to_process
        print('Time to train vocab: {} mins'.format(self.time_to_train))

        metrics = add_balanced_accuracy([])
        evaluate(processed_datasets, models, metrics)
        for processed_dataset in processed_datasets:
            processed_dataset.write_dataset_info()

        self.time_to_evaluate = round(time()/60, 2) - self.time_to_train
        print('Time to process vocab: {} mins'.format(self.time_to_evaluate))
