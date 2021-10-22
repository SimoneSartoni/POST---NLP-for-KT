import tensorflow as tf
from Knowledge_Tracing.code.models.DKT import deepkt, data_utils, metrics

fn = ""  # Dataset path
verbose = 1  # Verbose = {0,1,2}
best_model_weights = "weights/bestmodel"  # File to save the model.
log_dir = "logs"  # Path to save the logs.
optimizer = "adam"  # Optimizer to use
lstm_units = 100  # Number of LSTM units
batch_size = 32  # Batch size
epochs = 10  # Number of epochs to train
dropout_rate = 0.3  # Dropout rate
test_fraction = 0.2  # Portion of data to be used for testing
validation_fraction = 0.2  # Portion of training data to be used for validation

dataset, length, nb_features, nb_skills = data_utils.load_dataset(fn=fn,
                                                                  batch_size=batch_size,
                                                                  shuffle=True)

train_set, test_set, val_set = data_utils.split_dataset(dataset=dataset,
                                                        total_size=length,
                                                        test_fraction=test_fraction,
                                                        val_fraction=validation_fraction)

set_sz = length * batch_size
test_set_sz = (set_sz * test_fraction)
val_set_sz = (set_sz - test_set_sz) * validation_fraction
train_set_sz = set_sz - test_set_sz - val_set_sz
print("============= Data Summary =============")
print("Total number of students: %d" % set_sz)
print("Training set size: %d" % train_set_sz)
print("Validation set size: %d" % val_set_sz)
print("Testing set size: %d" % test_set_sz)
print("Number of skills: %d" % nb_skills)
print("Number of features in the input: %d" % nb_features)
print("========================================")

student_model = deepkt.DKTModel(
    nb_features=nb_features,
    nb_skills=nb_skills,
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
                                tf.keras.callbacks.CSVLogger(f"{log_dir}/train.log"),
                                tf.keras.callbacks.ModelCheckpoint(best_model_weights,
                                                                   save_best_only=True,
                                                                   save_weights_only=True),
                                tf.keras.callbacks.TensorBoard(log_dir=log_dir)
                            ])

student_model.load_weights(best_model_weights)

result = student_model.custom_evaluate(test_set, verbose=verbose)
