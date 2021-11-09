import gc

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

from Knowledge_Tracing.code.models.encoding_models.count_vectorizer import count_vectorizer
from Knowledge_Tracing.code.models.encoding_models.sentence_transformers import sentence_transformer
from Knowledge_Tracing.code.models.encoding_models.gensim_model.gensim_pretrained_word2vec import pretrained_word2vec
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
        coloumns.append('skill_with_answer')
        skill_depth = df['skill'].max() + 1
        # Step 3 - Cross skill id with answer to form a synthetic feature
        df['skill_with_answer'] = df['skill'] * 2 + df['correct']
        df['skill_with_answer'] = df['skill_with_answer'].astype('int32')
        print(df['skill_with_answer'])
        features_depth = df['skill_with_answer'].max() + 1
    else:
        features_depth = 0
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
    if encodings_kwargs['pretrained_word2vec']:
        load, keyed_vectors = encodings_kwargs['pretrained_word2vec_load'], encodings_kwargs['pretrained_word2vec_keyed_vectors']
        encode_model = pretrained_word2vec(load=load, keyedvectors=keyed_vectors)
        encode_model.encode_problems()
        encode_model.fit(text_df)
        encode_models.append(encode_model)
    if encodings_kwargs['word2vec']:
        min_df, max_df, vector_size = encodings_kwargs['min_df'], encodings_kwargs['max_df'], \
                                       encodings_kwargs['vector_size']
        encode_model = pretrained_word2vec(min_count=min_df, vector_size=vector_size)
        encode_model.encode_problems(texts_df=text_df)
        encode_model.fit()
        encode_models.append(encode_model)

    del text_df
    gc.collect()
    print("number of words is: " + str(max_value))
    users = df['user_id'].unique()

    train_users, test_users = train_test_split(users, test_size=0.2)
    train_users, val_users = train_test_split(train_users, test_size=0.2)

    def generate_encodings_val():
        doc_to_encodings = {}
        i_doc_to_encodings = {}
        for name, group in df.loc[df['user_id'].isin(val_users)].groupby('user_id'):
            labels = np.array([], dtype=np.int)
            for model in encode_models:
                doc_to_encodings[model.name] = []
            for problem, label, feature in list(zip(group['problem_id'].values, group['correct'].values,
                                                    group['skills_with_answer'].values)):
                encoding = encode_model.get_encoding(problem)
                zeros = np.zeros(encoding.shape, dtype=np.float)
                if label:
                    encoding = np.concatenate([encoding, zeros])
                else:
                    encoding = np.concatenate([zeros, encoding])
                encoding = np.expand_dims(encoding, axis=0)
                doc_to_encodings[model.name].append(encoding)
                labels = np.append(labels, label)
                if encodings_kwargs['use_skills']:
                    features = np.append(features, feature)
            for model in encode_models:
                doc_to_encodings[model.name] = np.concatenate(doc_to_encodings[model.name], axis=0)
                i_doc_to_encodings[model.name] = doc_to_encodings[model.name][:-1]
            i_feature = features[:-1]
            o_label = labels[1:]
            inputs = []
            for model in encode_models:
                inputs.append(i_doc_to_encodings[model.name])
            if encodings_kwargs['use_skills']:
                inputs.append(i_feature)
            outputs = o_label
            yield inputs, outputs

    def generate_encodings_test():
        doc_to_encodings = {}
        i_doc_to_encodings = {}
        for name, group in df.loc[df['user_id'].isin(test_users)].groupby('user_id'):
            labels = np.array([], dtype=np.int)
            for model in encode_models:
                doc_to_encodings[model.name] = []
            for problem, label, feature in list(zip(group['problem_id'].values, group['correct'].values,
                                                    group['skills_with_answer'].values)):
                encoding = encode_model.get_encoding(problem)
                zeros = np.zeros(encoding.shape, dtype=np.float)
                if label:
                    encoding = np.concatenate([encoding, zeros])
                else:
                    encoding = np.concatenate([zeros, encoding])
                encoding = np.expand_dims(encoding, axis=0)
                doc_to_encodings[model.name].append(encoding)
                labels = np.append(labels, label)
                if encodings_kwargs['use_skills']:
                    features = np.append(features, feature)
            for model in encode_models:
                doc_to_encodings[model.name] = np.concatenate(doc_to_encodings[model.name], axis=0)
                i_doc_to_encodings[model.name] = doc_to_encodings[model.name][:-1]
            i_feature = features[:-1]
            o_label = labels[1:]
            inputs = []
            for model in encode_models:
                inputs.append(i_doc_to_encodings[model.name])
            if encodings_kwargs['use_skills']:
                inputs.append(i_feature)
            outputs = o_label
            yield inputs, outputs

    def generate_encodings_train():
        doc_to_encodings = {}
        i_doc_to_encodings = {}
        for name, group in df.loc[df['user_id'].isin(train_users)].groupby('user_id'):
            labels = np.array([], dtype=np.int)
            for model in encode_models:
                doc_to_encodings[model.name] = []
            for problem, label, feature in list(zip(group['problem_id'].values, group['correct'].values,
                                                    group['skills_with_answer'].values)):
                encoding = encode_model.get_encoding(problem)
                zeros = np.zeros(encoding.shape, dtype=np.float)
                if label:
                    encoding = np.concatenate([encoding, zeros])
                else:
                    encoding = np.concatenate([zeros, encoding])
                encoding = np.expand_dims(encoding, axis=0)
                doc_to_encodings[model.name].append(encoding)
                labels = np.append(labels, label)
                if encodings_kwargs['use_skills']:
                    features = np.append(features, feature)
            for model in encode_models:
                doc_to_encodings[model.name] = np.concatenate(doc_to_encodings[model.name], axis=0)
                i_doc_to_encodings[model.name] = doc_to_encodings[model.name][:-1]
            i_feature = features[:-1]
            o_label = labels[1:]
            inputs = []
            for model in encode_models:
                inputs.append(i_doc_to_encodings[model.name])
            if encodings_kwargs['use_skills']:
                inputs.append(i_feature)
            outputs = o_label
            yield inputs, outputs

    def create_dataset(generate_encodings, users, encode_models, features_depth):
        input_types = [tf.float32 for x in encode_models]
        if encodings_kwargs['use_skills']:
            input_types.append(tf.int32)
        output_types = tf.int32
        types = (input_types, output_types)
        input_shapes = [[None, encode_model.vector_size] for encode_model in encode_models]
        if encodings_kwargs['use_skills']:
            input_shapes.append([None])
        output_shapes = [None]
        shapes = (input_shapes, output_shapes)
        print(types)
        print(shapes)
        # Step 5 - Get Tensorflow Dataset
        dataset = tf.data.Dataset.from_generator(
            generator=generate_encodings,
            output_types=types,
            output_shapes=shapes
        )

        nb_users = len(users)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=nb_users)

        print(dataset)
        dataset = dataset.map(
            lambda inputs, outputs: (
                (inputs[0:-1], tf.one_hot(inputs[-1], depth=features_depth)),
                tf.expand_dims(outputs, axis=-1)
            ) if encodings_kwargs['use_skills'] else
            (
                inputs,
                tf.expand_dims(outputs, axis=-1)
            )
        )


        # Step 7 - Pad sequences per batch
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padding_values=MASK_VALUE,
            drop_remainder=True
        )
        return dataset

    encoding_sizes = [2*model.vector_size for model in encode_models]
    train_set = create_dataset(generate_encodings_train, train_users, encode_models, features_depth)
    val_set = create_dataset(generate_encodings_val, val_users, encode_models, features_depth)
    test_set = create_dataset(generate_encodings_test, test_users, encode_models, features_depth)

    if encodings_kwargs['use_skills']:
        return train_set, val_set, test_set, encoding_sizes, features_depth
    return train_set, val_set, test_set, encoding_sizes


def get_target(y_true, y_pred, nb_encodings=300, nb_skills=300):
    # Get skills and labels from y_true

    mask = 1 - tf.cast(tf.equal(y_true, MASK_VALUE), y_true.dtype)
    y_true = y_true * mask
    encodings_true, y_true = tf.split(y_true, num_or_size_splits=[-1, 1], axis=-1)
    encodings_pred, y_pred = tf.split(y_pred, num_or_size_splits=[-1, 1], axis=-1)

    # Get predictions for each skill

    return y_true, y_pred
