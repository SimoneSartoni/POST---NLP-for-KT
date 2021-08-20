from Knowledge_Tracing.code.models.TF_IDF.TF_IDF import TF_IDF
from Knowledge_Tracing.code.evaluation.balanced_accuracy import balanced_accuracy
from Knowledge_Tracing.code.models.gensim_model.gensim_word2vec import world2vec


def add_tf_idf_model(models, input_dataset, load=True):
    tf_idf = TF_IDF()
    tf_idf.fit(input_dataset.interacted_with_text_problem_set, input_dataset.problem_id_to_index, input_dataset.texts_list)
    if load:
        tf_idf.load_similarity_matrix(dataset_name=input_dataset.name)
    else:
        tf_idf.compute_similarity()
    models.append(tf_idf)
    return models


def add_gensim_model(models, input_dataset, load=True, min_count=2, window=5, vector_size=100, workers=3, sg=1, epochs=10):
    gensim_model = world2vec(name="word2vec_size" + str(vector_size) + "_epoch" + str(epochs), class_of_method="NLP",
                             min_count=2, window=5, vector_size=vector_size, workers=3, sg=1)
    if not load:
        gensim_model.fit(input_dataset.texts_list, epochs=epochs)
    else:
        gensim_model.load_model(epochs=epochs, path="C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/logs/", name=input_dataset.name)
        gensim_model.load_word_vectors(epochs=epochs, path="C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/logs/", name=input_dataset.name)
    gensim_model.encode_problems(input_dataset.problem_id_to_index, input_dataset.texts_list)
    models.append(gensim_model)
    return models


def add_balanced_accuracy(metrics):
    metrics.append(balanced_accuracy(name="balanced_accuracy"))
    return metrics

