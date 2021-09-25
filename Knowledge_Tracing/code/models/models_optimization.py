from Knowledge_Tracing.code.evaluation.predictors.predictor import predictor as Predictor
from Knowledge_Tracing.code.evaluation.evaluator import evaluator as Evaluator
from Knowledge_Tracing.code.evaluation.metrics.balanced_accuracy import balanced_accuracy
from Knowledge_Tracing.code.models.models_creation import add_gensim_model


def gensim_word2vec_optimization(models, input_dataset):
    size_array = [60, 180, 300]
    epochs = [10, 20, 30]
    for size in size_array:
        for epoch in epochs:
            add_gensim_model(models, input_dataset, load=False, vector_size=size, epochs=epoch)
    predictor = Predictor()
    labels, predictions = predictor.compute_predictions(dataset=input_dataset, models=models)
    metrics = [balanced_accuracy(name="balanced_accuracy")]
    evaluator = Evaluator("Evaluator", metrics)
    return evaluator.evaluate(labels, models, predictions)
