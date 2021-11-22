import tensorflow as tf
from Knowledge_Tracing.code.models.DKT_models.count_vect_DKT.count_vect_DKT_no_encodings_outputs import \
    count_vect_deepkt, metrics
from Knowledge_Tracing.code.data_processing.load_preprocessed.get_DKT_dataloaders import get_DKT_dataloaders

fn = "../input/assistment-2012-processed/interactions_processed.csv"  # Dataset path
verbose = 1  # Verbose = {0,1,2}
best_model_weights = "weights/bestmodel"  # File to save the model.

batch_size = 128  # Batch size

test_fraction = 0.2  # Portion of data to be used for testing
validation_fraction = 0.2  # Portion of training data to be used for validation
texts_filepath = "../input/assistment-2012-processed/texts_processed.csv"
interactions_filepath = fn
results = {}
max_features = 1000
min_df = 3
inputs = {"question_id": False, "text_id": False, "skill": False,
          "label": False, "r_elapsed_time": False, "text_encoding": True}
outputs = {}
for max_df in [2e-4, 5e-4, 1e-3]:
    train_set, val_set, test_set, nb_encodings = get_DKT_dataloaders(batch_size=batch_size, shuffle=True,
                                                                     interactions_filepath=interactions_filepath,
                                                                     output_filepath='/kaggle/working/',
                                                                     texts_filepath=texts_filepath,
                                                                     interaction_sequence_len=25,
                                                                     text_encoding_model=None,
                                                                     negative_correctness=False,
                                                                     inputs=inputs, outputs=outputs)
    print(nb_encodings)
    log_dir = "logs"  # Path to save the logs.
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-3)  # Optimizer to use
    lstm_units = 100  # Number of LSTM units
    epochs = 60  # Number of epochs to train
    dropout_rate = 0.3  # Dropout rate
    for lr in [1e-4]:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)  # Optimizer to use
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
                                        tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                                                         patience=6, restore_best_weights=True),
                                        tf.keras.callbacks.CSVLogger(f"{log_dir}/train.log"),
                                        tf.keras.callbacks.ModelCheckpoint(best_model_weights,
                                                                           save_best_only=True,
                                                                           save_weights_only=True),
                                        tf.keras.callbacks.TensorBoard(log_dir=log_dir)
                                    ])

        student_model.load_weights(best_model_weights)

        result = student_model.custom_evaluate(test_set, verbose=verbose)
        print(result)
        results[min_df] = result
print(results)
