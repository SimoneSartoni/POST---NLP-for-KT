from Knowledge_Tracing.code.data_processing.dataset import dataset
from Knowledge_Tracing.code.data_processing.import_files import import_questions_text, \
    import_poj_interactions

from Knowledge_Tracing.code.data_processing.data_processing import poj_process_bodies
from Knowledge_Tracing.code.models.TF_IDF.TF_IDF import TF_IDF
from Knowledge_Tracing.code.evaluation.predictor import predictor as Predictor
from Knowledge_Tracing.code.evaluation.evaluation import evaluator as Evaluator
from Knowledge_Tracing.code.evaluation.balanced_accuracy import balanced_accuracy
from Knowledge_Tracing.code.models.gensim_model.gensim_word2vec import world2vec


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


def tf_idf_evaluation(dataset, load = True):
    tf_idf = TF_IDF()
    tf_idf.fit(dataset.interacted_with_text_problem_set, dataset.problem_id_to_index, dataset.texts_list)
    if load:
        tf_idf.load_similarity_matrix(path="C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/results/")
    else:
        tf_idf.compute_similarity()
    gensim_model = world2vec(name="word2vec", class_of_method="NLP")
    path = "C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/logs/poj/"
    gensim_model.fit(dataset.texts_list,  path=path, name=dataset.name)
    gensim_model.encode_problems(dataset.problem_id_to_index, dataset.texts_list)
    models = [tf_idf, gensim_model]
    predictor = Predictor()
    labels, predictions = predictor.compute_predictions(dataset=dataset, models=models)
    metrics = [balanced_accuracy(name="balanced_accuracy")]
    evaluator = Evaluator("Evaluator", metrics)
    return evaluator.evaluate(labels, models, predictions)


def gensim_word2vec_optimization(dataset):
    models = []
    size_array = [60, 180, 300]
    epochs = [10, 20, 30]
    for size in size_array:
        for epoch in epochs:
            gensim_model = world2vec(name="word2vec_size" + str(size) + "_epoch" + str(epoch), class_of_method="NLP",
                                     min_count=2, window=5, vector_size=size, workers=3, sg=1)
            gensim_model.fit(dataset.texts_list, epochs=epoch)
            gensim_model.encode_problems(dataset.problem_id_to_index, dataset.texts_list)
            models.append(gensim_model)
    predictor = Predictor()
    labels, predictions = predictor.compute_predictions(dataset=dataset, models=models)
    metrics = [balanced_accuracy(name="balanced_accuracy")]
    evaluator = Evaluator("Evaluator", metrics)
    return evaluator.evaluate(labels, models, predictions)


def main():
    # import text of poj (needed to import its interactions)
    datasets = import_text()

    # import interaction datasets
    # assistment_dataset_npz = import_assistments_2009()

    # junyi_dataset = import_junyi_interactions()
    poj_dataset = import_poj_interactions()



    """# import assistment texts dataset
    texts, problem_id_to_index = assistments_process_bodies(datasets["assistments_texts"])
    # produces set of problems according to data available
    assistment_dataset_npz.set_texts(texts, problem_id_to_index)

        problems_with_text_set, problems_interacted_set, problems_text_and_interacted_set = \
        generate_text_and_interacted_sets(problem_ids, assistment_dataset_2012.users_interactions)
    assistment_dataset_2012.set_texts(texts, problem_id_to_index, problems_with_text_set, problems_interacted_set,
                                      problems_text_and_interacted_set)

    problems_with_text_set, problems_interacted_set, problems_text_and_interacted_set = \
        generate_text_and_interacted_sets(problem_ids, assistment_dataset_2009.users_interactions)
    assistment_dataset_2009.set_texts(texts, problem_id_to_index, problems_with_text_set, problems_interacted_set,
                                      problems_text_and_interacted_set)"""

    """print(len(assistment_dataset_2012.interacted_with_text_problem_set))
    print(len(assistment_dataset_2009.interacted_with_text_problem_set))"""

    """# import junyi texts dataset
    texts, problem_id_to_index = junyi_process_questions(datasets["junyi_texts"])
    problem_ids = problem_id_to_index.keys()    
    # produces set of problems according to data available
    problems_with_text_set, problems_interacted_set, problems_text_and_interacted_set = \
        generate_text_and_interacted_sets(problem_ids, junyi_dataset.problems)
    junyi_dataset.set_texts(texts, problem_id_to_index, problems_with_text_set, problems_interacted_set,
                            problems_text_and_interacted_set)
    print(len(junyi_dataset.problems_text_and_interacted_set))"""

    # import poj texts dataset
    # texts, problem_id_to_index = poj_process_bodies(datasets["poj_texts"])
    # produces set of problems according to data available
    # poj_dataset.set_texts(texts, problem_id_to_index)
    poj_dataset.load_saved_texts(path="C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/results/")
    poj_dataset.compute_intersection_texts_and_interactions()
    """
    assistment_dataset_npz.set_performances(tf_idf_evaluation(assistment_dataset_npz))
    assistment_dataset_2012.set_performances(tf_idf_evaluation(assistment_dataset_2012))
    assistment_dataset_2009.set_performances(tf_idf_evaluation(assistment_dataset_2009))
    """
    # junyi_dataset.set_performances(tf_idf_evaluation(junyi_dataset))
    poj_dataset.set_performances(tf_idf_evaluation(poj_dataset))

    # assistment_dataset_npz.write_dataset_info()
    # assistment_dataset_2012.write_dataset_info()
    # assistment_dataset_2009.write_dataset_info()
    # junyi_dataset.write_dataset_info()
    poj_dataset.write_dataset_info()

    """write_txt("C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/logs/problems_set", problems_text_and_interacted_set)
    write_txt("C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/logs/problems_texts", texts)
    write_txt("C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/logs/problems_ids", problem_ids)
    """


main()
