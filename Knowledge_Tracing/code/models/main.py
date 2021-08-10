from Knowledge_Tracing.code.data_processing.dataset import dataset
from Knowledge_Tracing.code.data_processing.import_files import import_questions_text, import_assistments_interactions, \
    import_junyi_interactions, import_poj_interactions

from Knowledge_Tracing.code.data_processing.data_processing import assistments_process_bodies, poj_process_bodies, \
    junyi_process_questions, \
    generate_text_and_interacted_sets
from Knowledge_Tracing.code.evaluation import *
from Knowledge_Tracing.code.models.TF_IDF import TF_IDF
from Knowledge_Tracing.code.utils.utils import write_txt


def import_text():
    assistment_df = dataset(name="assistments_texts",
                            path="C:/Users/Simone Sartoni/Simone/Universita/5anno/thesis_2"
                                 "/TransformersForKnowledgeTracing/Knowledge_Tracing/data/Assistments/problem_bodies"
                                 "/ASSISTments2012DataSet-ProblemBodies.csv")
    junyi_df = dataset(name="junyi_texts",
                       path="C:/Users/Simone Sartoni/Simone/Universita/5anno/thesis_2"
                            "/TransformersForKnowledgeTracing/Knowledge_Tracing/data/Junyi/junyi_question_text"
                            ".txt")
    poj_df = dataset(name="poj_texts",
                     path="C:/Users/Simone Sartoni/Simone/Universita/5anno/thesis_2"
                          "/TransformersForKnowledgeTracing/Knowledge_Tracing/data/poj/poj_question_text.txt")
    datasets_dict = [assistment_df, junyi_df, poj_df]
    datasets = import_questions_text(datasets_dict)
    return datasets


def main():
    # import text of POJ (needed to import its interactions)
    datasets = import_text()
    number_to_index, texts = poj_process_bodies(datasets["poj_texts"])

    # import interaction datasets
    assistment_dataset = import_assistments_interactions()
    """junyi_dataset = import_junyi_interactions()
    poj_dataset = import_poj_interactions(number_to_index)"""

    # import assistment texts dataset
    problem_ids, texts = assistments_process_bodies(datasets["assistments_texts"])
    # produces set of problems according to data available
    problems_with_text_set, problems_interacted_set, problems_text_and_interacted_set = \
        generate_text_and_interacted_sets(problem_ids, assistment_dataset.problems)
    assistment_dataset.set_texts(texts, problem_ids, problems_with_text_set, problems_interacted_set,
                                 problems_text_and_interacted_set)

    """# import junyi texts dataset
    problem_ids, texts = junyi_process_questions(datasets["junyi_texts"])
    # produces set of problems according to data available
    problems_with_text_set, problems_interacted_set, problems_text_and_interacted_set = \
        generate_text_and_interacted_sets(problem_ids, junyi_dataset.problems)
    junyi_dataset.set_texts(texts, problem_ids, problems_with_text_set, problems_interacted_set,
                            problems_text_and_interacted_set)

    # import poj texts dataset
    number_to_index, texts = poj_process_bodies(datasets["poj_texts"])
    problem_ids = number_to_index.keys()
    print(problem_ids)
    # produces set of problems according to data available
    problems_with_text_set, problems_interacted_set, problems_text_and_interacted_set = \
        generate_text_and_interacted_sets(problem_ids, poj_dataset.problems)
    poj_dataset.set_texts(texts, problem_ids, problems_with_text_set, problems_interacted_set,
                          problems_text_and_interacted_set)"""

    assistment_dataset.write_dataset_info()
    """junyi_dataset.write_dataset_info()
    poj_dataset.write_dataset_info()"""

    """write_txt("C:/Users/Simone Sartoni/Simone/Universita/5anno/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/logs/problems_set", problems_text_and_interacted_set)
    write_txt("C:/Users/Simone Sartoni/Simone/Universita/5anno/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/logs/problems_texts", texts)
    write_txt("C:/Users/Simone Sartoni/Simone/Universita/5anno/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/logs/problems_ids", problem_ids)
    tf_idf = TF_IDF()
    tf_idf.fit(texts)
    tf_idf.compute_similarity()
    similarity_matrix = tf_idf.similarity_matrix"""


main()
