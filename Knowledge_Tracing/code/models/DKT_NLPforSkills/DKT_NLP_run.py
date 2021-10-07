import tensorflow as tf
from Knowledge_Tracing.code.models.DKT_NLPforSkills import NLP_deepkt, data_utils as dt_utils, metrics
from Knowledge_Tracing.code.models.DKT import data_utils as utils


def main():
    fn = "C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/data/assistments/2009_2010/skill_builder_data_corrected_collapsed.csv"  # Dataset path
    verbose = 1  # Verbose = {0,1,2}
    best_model_weights = "weights/bestmodel"  # File to save the model.
    log_dir = "logs"  # Path to save the logs.
    optimizer = "adam"  # Optimizer to use
    lstm_units = 100  # Number of LSTM units
    batch_size = 32  # Batch size
    epochs = 30  # Number of epochs to train
    dropout_rate = 0.3  # Dropout rate
    test_fraction = 0.2  # Portion of data to be used for testing
    validation_fraction = 0.2  # Portion of training data to be used for validation

    dataset, length, nb_features, encoding_depth = dt_utils.load_dataset_NLP_skills(fn=fn, batch_size=batch_size,
                                                                                    shuffle=True)

    train_set, test_set, val_set = utils.split_dataset(dataset=dataset, total_size=length, test_fraction=test_fraction,
                                                       val_fraction=validation_fraction)
    print(train_set)
    set_sz = length * batch_size
    test_set_sz = (set_sz * test_fraction)
    val_set_sz = (set_sz - test_set_sz) * validation_fraction
    train_set_sz = set_sz - test_set_sz - val_set_sz
    print("============= Data Summary =============")
    print("Total number of students: %d" % set_sz)
    print("Training set size: %d" % train_set_sz)
    print("Validation set size: %d" % val_set_sz)
    print("Testing set size: %d" % test_set_sz)
    print("Number of features: %d" % nb_features)
    print("========================================")

    student_model = NLP_deepkt.NLP_DKTModel(
        nb_features=nb_features,
        nb_encodings=encoding_depth,
        hidden_units=lstm_units,
        dropout_rate=dropout_rate)

    student_model.compile(
        optimizer=optimizer,
        metrics=[
            metrics.BinaryAccuracy(),
            metrics.AUC(),
            metrics.Precision(),
            metrics.Recall()
        ])

    student_model.summary()
    history = student_model.fit(dataset=train_set,
                                epochs=epochs,
                                verbose=verbose,
                                validation_data=val_set,
                                callbacks=[
                                    tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', mode='max',
                                                                     patience=5, restore_best_weights=True),
                                    tf.keras.callbacks.CSVLogger(f"{log_dir}/train.log"),
                                    tf.keras.callbacks.ModelCheckpoint(best_model_weights,
                                                                       save_best_only=True,
                                                                       save_weights_only=True),
                                    tf.keras.callbacks.TensorBoard(log_dir=log_dir)
                                ])

    student_model.load_weights(best_model_weights)

    result = student_model.custom_evaluate(test_set, verbose=verbose)

main()
