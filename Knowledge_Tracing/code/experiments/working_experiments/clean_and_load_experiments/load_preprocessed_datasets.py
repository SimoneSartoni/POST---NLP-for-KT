from Knowledge_Tracing.code.experiments.working_experiments.experiment import experiment
from Knowledge_Tracing.code.data_processing.dataset import dataset


def load_dataset(name, path, prefix, standard_timestamps=True):
    loaded_dataset = dataset(name=name, path=path, prefix=prefix)
    loaded_dataset.load_interactions(standard_timestamps=standard_timestamps)
    loaded_dataset.load_saved_texts()
    loaded_dataset.compute_intersection_texts_and_interactions()
    return loaded_dataset


class load_preprocessed_datasets(experiment):
    def __init__(self):
        super().__init__("load_preprocessed_datasets", "work flow improvement / time reduction")

    def run(self, exp, poj=True, assist_12=True, assist_09=True, junyi=True):
        datasets = []
        if poj:
            datasets.append(load_dataset(name="poj",
                                         path="/Knowledge_Tracing/intermediate_files",
                                         prefix="clean_datasets/"))
        if assist_12:
            datasets.append(load_dataset(name="assistments_2012_npz",
                                         path="/Knowledge_Tracing/intermediate_files",
                                         prefix="clean_datasets/"))
        if assist_09:
            datasets.append(load_dataset(name="assistments_2009",
                                         path="/Knowledge_Tracing/intermediate_files",
                                         prefix="clean_datasets/", standard_timestamps=False))
        if junyi:
            datasets.append(load_dataset(name="junyi",
                                         path="/Knowledge_Tracing/intermediate_files/",
                                         prefix="clean_datasets/"))
        return datasets
