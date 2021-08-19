from Knowledge_Tracing.code.data_processing.dataset import dataset
from Knowledge_Tracing.code.data_processing.import_files import import_questions_text

from Knowledge_Tracing.code.data_processing.data_processing import poj_process_bodies, assistments_process_bodies, \
    junyi_process_questions
from Knowledge_Tracing.code.models.models_creation import *


def import_text():
    assistment_df = dataset(name="assistments_texts",
                            path="C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/data/Assistments/problem_bodies"
                                 "/ASSISTments2012DataSet-ProblemBodies.csv")
    junyi_df = dataset(name="junyi_texts",
                       path="C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/data/Junyi/junyi_question_text"
                            ".txt")
    poj_df = dataset(name="poj_texts",
                     path="C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/data/poj/poj_question_text.txt")
    datasets_dict = [assistment_df, junyi_df, poj_df]
    datasets = import_questions_text(datasets_dict)
    return datasets


def process_dataset_text(target_dataset, dataset_texts, name, load_texts):
    if not load_texts:
        if name == "assistments_2012_npz" or name == "assistments_2012" or name == "assistments_2009":
            texts, problem_id_to_index = assistments_process_bodies(dataset_texts["assistments_texts"])
        elif name == "poj":
            texts, problem_id_to_index = poj_process_bodies(dataset_texts["poj_texts"])
        else:
            texts, problem_id_to_index = junyi_process_questions(dataset_texts["junyi_texts"])
        # produces set of problems according to data available
        target_dataset.set_texts(texts, problem_id_to_index)
    else:
        target_dataset.load_saved_texts(path="C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/results/")
    target_dataset.compute_intersection_texts_and_interactions()
    return target_dataset


def evaluate(input_datasets, models, metrics, predictors=[Predictor()]):
    labels = {}
    predictions = {}
    for input_dataset in input_datasets:
        for predictor in predictors:
            labels[predictor.name], predictions[predictor.name] = predictor.compute_predictions(dataset=input_dataset, models=models)
        evaluator = Evaluator("Evaluator", metrics)
        performances = evaluator.evaluate(labels, predictions, models, predictors)
        input_dataset.set_performances(performances)
    return input_datasets
