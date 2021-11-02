import gc

import tensorflow as tf
import numpy as np
from Knowledge_Tracing.code.models.encoding_models.count_vectorizer import count_vectorizer
from Knowledge_Tracing.code.models.encoding_models.sentence_transformers import sentence_transformer
from Knowledge_Tracing.code.data_processing.get_data_assistments_2012 import get_data_assistments_2012
from Knowledge_Tracing.code.data_processing.get_data_assistments_2009 import get_data_assistments_2009

MASK_VALUE = -1.0  # The masking value cannot be zero.


def load_dataset(batch_size=32, shuffle=True, dataset_name='assistment_2012',
                 interactions_filepath="../input/assistmentds-2012/2012-2013-data-with-predictions-4-final"
                                       ".csv",
                 save_filepath='/kaggle/working/', texts_filepath='../input/',
                 min_questions=2, max_questions=25, n_rows=None, n_texts=None,
                 personal_cleaning=True,
                 **encodings_kwargs):

    if dataset_name == 'assistment_2012':
        df, text_df = get_data_assistments_2012(min_questions=min_questions, max_questions=max_questions,
                                                interactions_filepath=interactions_filepath,
                                                texts_filepath=texts_filepath, n_rows=n_rows, n_texts=n_texts,
                                                make_sentences_flag=False, personal_cleaning=personal_cleaning)
    elif dataset_name == 'assistment_2009':
        df, text_df = get_data_assistments_2009(min_questions=min_questions, max_questions=max_questions,
                                                interactions_filepath=interactions_filepath,
                                                texts_filepath=texts_filepath, n_rows=n_rows, n_texts=n_texts,
                                                make_sentences_flag=False, personal_cleaning=personal_cleaning)

    coloumns = ['user_id', 'problem_id', 'correct']
    if encodings_kwargs['use_skills']:
        coloumns.append('skill')
        skill_depth = df['skill'].max() + 1
    df = df[coloumns]
    print(df)
    # Step 3.1 - Generate NLP extracted encoding for problems
    encode_models = []
    if encodings_kwargs['count_vect']:
        min_df, max_df, max_features = encodings_kwargs['min_df'], encodings_kwargs['max_df'], \
                                       encodings_kwargs['max_features']
        encode_model = count_vectorizer(min_df=min_df, max_df=max_df, binary=False, max_features=max_features)
        encode_model.fit(text_df, save_filepath)
        max_value = encode_model.words_num
        print("number of words is: " + str(max_value))
        encode_models.append(encode_model)
    if encodings_kwargs['sentence_encoder']:
        encoding_model = encodings_kwargs['sentence_encoder_model']
        encode_model = sentence_transformer(encoding_model=encoding_model)
        encode_model.fit(text_df, save_filepath)
        encode_models.append(encode_model)

    del text_df
    gc.collect()

    def generate_encodings():
        doc_to_encodings = {}
        i_doc_to_encodings = {}
        o_doc_to_encodings = {}

        for name, group in df.groupby('user_id'):
            labels = np.array([], dtype=np.int)
            skills = np.array([], dtype=np.int)
            for model in encode_models:
                doc_to_encodings[model.name] = []
            for problem, label, skill in list(zip(group['problem_id'].values, group['correct'].values, group['skill'].values)):
                for model in encode_models:
                    encoding = model.get_encoding(problem)
                    encoding = np.expand_dims(encoding, axis=0)
                    doc_to_encodings[model.name].append(encoding)
                labels = np.append(labels, label)
                if encodings_kwargs['use_skills']:
                    skills = np.append(skills, skill)
            for model in encode_models:
                doc_to_encodings[model.name] = np.concatenate(doc_to_encodings[model.name], axis=0)
                i_doc_to_encodings[model.name] = doc_to_encodings[model.name][:-1]
                o_doc_to_encodings[model.name] = doc_to_encodings[model.name][1:]
            i_label = labels[:-1]
            o_label = labels[1:]
            i_skill = skills[:-1]
            o_skill = skills[1:]
            inputs = []
            outputs = []
            for model in encode_models:
                inputs.append(i_doc_to_encodings[model.name])
                outputs.append(o_doc_to_encodings[model.name])
            if encodings_kwargs['use_skills']:
                inputs.append(i_skill)
                inputs.append(o_skill)
            inputs.append(i_label)
            inputs.append(o_label)
            yield inputs, outputs

    encoding_sizes = [model.vector_size for model in encode_models]

    input_types = [tf.float32 for x in encode_models]
    if encodings_kwargs['use_skills']:
        input_types.append(tf.int32)
    input_types.append(tf.float32)
    output_types = input_types
    types = (input_types, output_types)
    input_shapes = [[None, encode_model.vector_size] for encode_model in encode_models]
    if encodings_kwargs['use_skills']:
        input_shapes.append([None])
    input_shapes.append([None])
    output_shapes = input_shapes
    shapes = (input_shapes, output_shapes)
    print(types)
    print(shapes)
    # Step 5 - Get Tensorflow Dataset
    dataset = tf.data.Dataset.from_generator(
        generator=generate_encodings,
        output_types=types,
        output_shapes=shapes
    )


    nb_users = len(df.groupby('user_id'))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=nb_users)



    print(dataset)
    dataset = dataset.map(
        lambda inputs, outputs: (
            (inputs[0:-2], tf.one_hot(inputs[-2], depth=skill_depth), tf.expand_dims(inputs[-1], axis=-1)),
            tf.concat(values=[
                outputs[0:-2],
                tf.one_hot(outputs[-2], depth=skill_depth),
                tf.expand_dims(outputs[-1], axis=-1)],
                axis=-1)
        ) if encodings_kwargs['use_skills'] else
        (
            (inputs[0:-1], tf.expand_dims(inputs[-1], axis=-1)),
            tf.concat(values=[
                outputs[0:-1],
                tf.expand_dims(outputs[-1], axis=-1)],
                axis=-1)
        )
    )

    # Step 6 - Encode categorical features and merge skills with labels to compute target loss.
    # More info: https://github.com/tensorflow/tensorflow/issues/32142

    print(dataset)

    # Step 7 - Pad sequences per batch
    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padding_values=MASK_VALUE,
        drop_remainder=True
    )

    print(dataset)

    length = nb_users // batch_size
    if encodings_kwargs['use_skills']:
        return dataset, length, encoding_sizes, skill_depth
    return dataset, length, encoding_sizes


def get_target(y_true, y_pred, nb_encodings=300, nb_skills=300):
    # Get skills and labels from y_true

    mask = 1 - tf.cast(tf.equal(y_true, MASK_VALUE), y_true.dtype)
    y_true = y_true * mask
    encodings_true, y_true = tf.split(y_true, num_or_size_splits=[-1, 1], axis=-1)
    encodings_pred, y_pred = tf.split(y_pred, num_or_size_splits=[-1, 1], axis=-1)

    # Get predictions for each skill

    return y_true, y_pred
