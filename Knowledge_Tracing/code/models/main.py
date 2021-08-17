from Knowledge_Tracing.code.data_processing.dataset import dataset
from Knowledge_Tracing.code.data_processing.import_files import import_questions_text, \
    import_poj_interactions, import_junyi_interactions

from Knowledge_Tracing.code.data_processing.data_processing import poj_process_bodies, assistments_process_bodies, \
    junyi_process_questions
from Knowledge_Tracing.code.models.TF_IDF.TF_IDF import TF_IDF
from Knowledge_Tracing.code.evaluation.predictor import predictor as Predictor
from Knowledge_Tracing.code.evaluation.evaluation import evaluator as Evaluator
from Knowledge_Tracing.code.evaluation.balanced_accuracy import balanced_accuracy
from Knowledge_Tracing.code.models.gensim_model.gensim_word2vec import world2vec
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
        if name == "assistments":
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


def evaluate(input_dataset, models, metrics):
    predictor = Predictor()
    labels, predictions = predictor.compute_predictions(dataset=input_dataset, models=models)
    evaluator = Evaluator("Evaluator", metrics)
    performances = evaluator.evaluate(labels, models, predictions)
    input_dataset.set_performances(performances)
    return input_dataset


def main():
    # import text of poj (needed to import its interactions)
    datasets = import_text()

    # import interaction datasets
    # assistment_dataset_npz = import_assistments_2009()


    # POJ:
    poj_dataset = import_poj_interactions()
    poj_dataset = process_dataset_text(poj_dataset, datasets, "poj", True)

    #JUNYI:
    #junyi_dataset = import_junyi_interactions()
    #junyi_dataset = process_dataset_text(junyi_dataset, datasets, "junyi", True)

    models = add_tf_idf_model([], poj_dataset, True)
    models = add_gensim_model(models, poj_dataset, True, vector_size=200, epochs= 20)
    metrics = add_balanced_accuracy([])
    evaluate(poj_dataset, models, metrics)
    poj_dataset.write_dataset_info()




main()
