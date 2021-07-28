from Knowledge_Tracing.code.data_processing.dataset import dataset
from Knowledge_Tracing.code.data_processing.import_files import import_questions_text, import_assistments_interactions
from Knowledge_Tracing.code.data_processing.data_processing import assistments_process_bodies, \
    generate_text_and_interacted_sets
from Knowledge_Tracing.code.evaluation import *
from Knowledge_Tracing.code.models.TF_IDF import TF_IDF
from Knowledge_Tracing.code.utils.utils import write_txt


def main():
    assistment_df = dataset(name="assistments",
                            path="C:/Users/Simone Sartoni/Simone/Universita/5anno/thesis_2"
                                 "/TransformersForKnowledgeTracing/Knowledge_Tracing/data/Assistments/problem_bodies"
                                 "/ASSISTments2012DataSet-ProblemBodies.csv")
    datasets_dict = [assistment_df]
    problems, real_lens, corrects = import_assistments_interactions()

    datasets = import_questions_text(datasets_dict)
    dataset_assistment = datasets['assistments']
    problem_ids, texts = assistments_process_bodies(dataset_assistment[0:1000])
    problems_with_text_set, problems_interacted_set, problems_text_and_interacted_set = \
        generate_text_and_interacted_sets(problem_ids[0:1000], problems[0:1000])
    write_txt("C:/Users/Simone Sartoni/Simone/Universita/5anno/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/logs/problems_set", problems_text_and_interacted_set)
    write_txt("C:/Users/Simone Sartoni/Simone/Universita/5anno/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/logs/problems_texts", texts)
    write_txt("C:/Users/Simone Sartoni/Simone/Universita/5anno/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/logs/problems_ids", problem_ids)
    tf_idf = TF_IDF()
    tf_idf.fit(texts)
    tf_idf.compute_similarity()
    similarity_matrix = tf_idf.similarity_matrix

main()
