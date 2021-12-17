import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

from Knowledge_Tracing.code.data_processing.load_preprocessed.load_preprocessed_data import load_preprocessed_texts
from Knowledge_Tracing.code.models.encoding_models.sentence_transformers import sentence_transformer
from Knowledge_Tracing.code.data_processing.load_preprocessed.get_DKT_dataloaders import get_DKT_dataloaders

MASK_VALUE = -1.0  # The masking value cannot be zero.


def create_dataset(generator, encoding_depth, shuffle=True, batch_size=1024):
    input_types = {"text_encoding": tf.float32, "label": tf.float32, "target_text_encoding": tf.float32}
    output_types = {"target_label": tf.float32}

    input_shapes = {"text_encoding": [None, encoding_depth], "label": [None],
                    "target_text_encoding": [None, encoding_depth]}
    output_shapes = {"target_label": [None]}
    types = (input_types, output_types)
    shapes = (input_shapes, output_shapes)
    dataset = tf.data.Dataset.from_generator(
        generator=generator.generator,
        output_types=types,
        output_shapes=shapes
    )

    nb_users = generator.__len__()
    if shuffle:
        dataset = dataset.shuffle(buffer_size=nb_users, reshuffle_each_iteration=True)

    print(dataset)
    dataset = dataset.map(
        lambda inputs, outputs: (
            {"input_encoding": inputs['text_encoding'], "input_label": tf.expand_dims(inputs['label'], axis=-1),
             "target_encoding": inputs['target_text_encoding']},
            tf.expand_dims(outputs['target_label'], axis=-1)
        )
    )

    print(dataset)

    # Step 7 - Pad sequences per batch
    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padding_values=-1.0,
        drop_remainder=True
    )
    return dataset


def load_dataset(batch_size=32, shuffle=True,
                 interactions_filepath="../input/assistmentds-2012/2012-2013-data-with-predictions-4-final.csv",
                 save_filepath='/kaggle/working/', texts_filepath='../input/', encoding_model='all-mpnet-base-v2',
                 interaction_sequence_len=30, min_seq_len=5, encode_correct_in_encodings=False):
    inputs = {"question_id": False, "text_id": False, "skill": False,
              "label": False, "r_elapsed_time": False, 'text_encoding': True, "target_id": False,
              "target_text_id": False, "target_skill": False, 'target_label': False, 'target_text_encoding': True}
    outputs = {"question_id": False, "text_id": False, "skill": False,
               "label": False, "r_elapsed_time": False, "target_id": False,
               "target_text_id": False, "target_skill": False, 'target_label': True}

    text_df = load_preprocessed_texts(texts_filepath=texts_filepath, text_as_sentence=True)
    # Step 3.1 - Generate NLP extracted encoding for problems
    encode_model = sentence_transformer(encoding_model=encoding_model)
    encode_model.fit(text_df, save_filepath)

    train_gen, val_gen, test_gen, nb_questions, nb_skills = get_DKT_dataloaders(batch_size, shuffle,
                                                                                interactions_filepath,
                                                                                output_filepath='/kaggle/working/',
                                                                                interaction_sequence_len=interaction_sequence_len
                                                                                , min_seq_len=min_seq_len,
                                                                                text_encoding_model=encode_model,
                                                                                negative_correctness=False,
                                                                                inputs_dict=inputs,
                                                                                outputs_dict=outputs,
                                                                                encode_correct_in_encodings=encode_correct_in_encodings)

    encoding_depth = train_gen.encoding_depth
    train_loader = create_dataset(train_gen, encoding_depth, shuffle=shuffle, batch_size=batch_size)
    val_loader = create_dataset(val_gen, encoding_depth, shuffle=shuffle, batch_size=batch_size)
    test_loader = create_dataset(test_gen, encoding_depth, shuffle=shuffle, batch_size=batch_size)

    return train_loader, val_loader, test_loader, encoding_depth

