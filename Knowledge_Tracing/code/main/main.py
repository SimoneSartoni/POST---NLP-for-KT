# from Knowledge_Tracing.code.experiments.work_in_progress.optimize_word2vec import optimize_word2vec
from Knowledge_Tracing.code.experiments.working_experiments.clean_and_load_experiments.reduce_and_save import \
    reduce_and_save
"""from Knowledge_Tracing.code.experiments.working_experiments.evaluate_experiments.evaluate_count_vectorizer import \
    evaluate_count_vectorizer
from Knowledge_Tracing.code.experiments.working_experiments.optimization_experiments.optimize_count_vectorizer import \
    optimize_count_vectorizer"""
from Knowledge_Tracing.code.experiments.working_experiments.dataset_analysis.user_performances_analysis import \
    user_performances_analysis
"""from Knowledge_Tracing.code.experiments.working_experiments.evaluate_experiments.evaluate_doc2vec import \
    evaluate_doc2vec
from Knowledge_Tracing.code.experiments.working_experiments.evaluate_experiments.evaluate_word2vec import \
    evaluate_word2vec"""

from mlflow import *


def main():
    new_experiment = user_performances_analysis()
    set_tracking_uri("file:///C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/ml_experiments")
    exp = get_experiment_by_name(new_experiment.name + " " + new_experiment.class_of_method)
    if not exp:
        experiment_id = create_experiment(new_experiment.name + " " + new_experiment.class_of_method)
        exp = get_experiment(experiment_id)
    print("Name: {}".format(exp.name))
    print("Experiment_id: {}".format(exp.experiment_id))
    print("Artifact Location: {}".format(exp.artifact_location))
    print("Tags: {}".format(exp.tags))
    print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
    new_experiment.run(exp)


main()
