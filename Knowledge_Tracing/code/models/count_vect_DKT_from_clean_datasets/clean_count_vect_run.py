import tensorflow as tf
from Knowledge_Tracing.code.models.count_vect_DKT_from_clean_datasets import data_utils, count_vect_deepkt, metrics
from code.models.DKT.data_utils import split_dataset


def main():
    fn = "C:/thesis_2/TransformersForKnowledgeTracing/Knowledge_Tracing/intermediate_files/clean_datasets/assistments_2012/interactions.csv"  # Dataset path
    verbose = 1  # Verbose = {0,1,2}
    best_model_weights = "weights/bestmodel"  # File to save the model.
    log_dir = "logs"  # Path to save the logs.
    optimizer = "adam"  # Optimizer to use
    lstm_units = 100  # Number of LSTM units
    batch_size = 128  # Batch size
    epochs = 30  # Number of epochs to train
    dropout_rate = 0.3  # Dropout rate
    test_fraction = 0.2  # Portion of data to be used for testing
    validation_fraction = 0.2  # Portion of training data to be used for validation

    dataset, length, nb_encodings = data_utils.load_dataset_NLP_skills(fn=fn, batch_size=batch_size, shuffle=True)

    train_set, test_set, val_set = split_dataset(dataset=dataset, total_size=length, test_fraction=test_fraction,
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
    print("========================================")

    student_model = count_vect_deepkt.clean_count_vect_DKTModel(
        nb_encodings=nb_encodings,
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
                                                                     patience=10, restore_best_weights=True),
                                    tf.keras.callbacks.CSVLogger(f"{log_dir}/train.log"),
                                    tf.keras.callbacks.ModelCheckpoint(best_model_weights,
                                                                       save_best_only=True,
                                                                       save_weights_only=True),
                                    tf.keras.callbacks.TensorBoard(log_dir=log_dir)
                                ])

    student_model.load_weights(best_model_weights)

    result = student_model.custom_evaluate(test_set, verbose=verbose)

main()