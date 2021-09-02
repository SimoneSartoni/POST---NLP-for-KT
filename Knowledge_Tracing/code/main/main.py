from Knowledge_Tracing.code.experiments.evaluate_gensim_tf_idf import *


def main():
    new_experiment = evaluate_gensim_tf_idf()
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
