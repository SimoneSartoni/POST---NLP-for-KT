from time import time
from Knowledge_Tracing.code.experiments.experiment import experiment
from Knowledge_Tracing.code.data_processing.data_processing import *
from Knowledge_Tracing.code.data_processing.dataset import dataset
from Knowledge_Tracing.code.visualization.histogram import *


def load_dataset(name, path):
    loaded_dataset = dataset(name=name,path=path)
    loaded_dataset.load_interactions()
    loaded_dataset.load_saved_texts()
    return loaded_dataset


class load_preprocessed_datasets(experiment):
    def __init__(self):
        super().__init__("load_preprocessed_datasets", "work flow improvement / time reduction")

    def run(self):

        poj = load_dataset(name="poj", path="C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/intermediate_files")
        assistments_2012_npz = load_dataset(name="assistments_2012_npz", path="C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/intermediate_files")
        assistments_2009 = load_dataset(name="assistments_2009", path="C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/intermediate_files")
        assistments_2012 = load_dataset(name="assistments_2012", path="C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/intermediate_files")
        junyi = load_dataset(name="junyi", path="C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/intermediate_files")

        poj_reduced = load_dataset(name="poj_reduced",
                           path="C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/intermediate_files")
        assistments_2012_npz_reduced = load_dataset(name="assistments_2012_npz_reduced",
                                              path="C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/intermediate_files")
        assistments_2009_reduced = load_dataset(name="assistments_2009_reduced",
                                               path="C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/intermediate_files")
        assistments_2012_reduced = load_dataset(name="assistments_2012_reduced",
                                               path="C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/intermediate_files")
        junyi_reduced = load_dataset(name="junyi_reduced",
                                     path="C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/intermediate_files")
