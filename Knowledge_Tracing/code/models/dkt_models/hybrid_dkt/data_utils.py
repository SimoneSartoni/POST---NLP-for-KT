import gc

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

from Knowledge_Tracing.code.models.nlp_models.count_vectorizer import count_vectorizer
# from Knowledge_Tracing.code.models.nlp_models.BERTopic_model import BERTopic_model
from Knowledge_Tracing.code.models.nlp_models.pretrained_distilbert import PretrainedDistilBERT
from Knowledge_Tracing.code.models.nlp_models.sentence_transformers import sentence_transformer

from Knowledge_Tracing.code.data_processing.load_preprocessed.load_preprocessed_data import load_preprocessed_texts
from Knowledge_Tracing.code.data_processing.load_preprocessed.get_hybrid_dkt_dataloaders import \
    get_hybrid_dkt_dataloaders
from Knowledge_Tracing.code.models.nlp_models.pretrained_distilbert_finetuned import PretrainedDistilBERTFinetuned

MASK_VALUE = -1.0  # The masking value cannot be zero.

"""
    possible_input_types = {"question_id": tf.float32, "text_id": tf.float32, "skill": tf.float32,
                            "label": tf.float32, "r_elapsed_time": tf.float32, "target_id": tf.float32,
                            "target_text_id": tf.float32, "target_skill": tf.float32,
                            'target_label': tf.float32}
    possible_input_shapes = {"question_id": [None, 1], "text_id": [None, 1], "skill": [None, 1],
                             "label": [None], "r_elapsed_time": [None], "target_id": [None],
                             "target_text_id": [None], "target_skill": [None],
                             'target_label': [None]}
"""


def create_dataset(generator, encode_models, encoding_depths, shuffle=True, batch_size=1024):
    input_types, output_types, input_shapes, output_shapes = {}, {}, {}, {}
    for encoding_depth, encode_model in list(zip(list(encoding_depths.values()), encode_models)):
        input_types[encode_model.name] = tf.float32
        input_types["target_" + encode_model.name] = tf.float32
        input_shapes[encode_model.name] = [None, encoding_depth]
        input_shapes["target_" + encode_model.name] = [None, encoding_depth]

    output_types = {"target_label": tf.float32}
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
            inputs,
            tf.expand_dims(outputs['target_label'], axis=-1)
        )
    )

    print(dataset)

    # Step 7 - Pad sequences per batch
    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padding_values=-1.0,
        padded_shapes=(input_shapes, [None, 1]),
        drop_remainder=True
    )
    return dataset


def load_dataset(batch_size=32, shuffle=True,
                 interactions_filepath="../input/assistmentds-2012/2012-2013-data-with-predictions-4-final.csv",
                 save_filepath='/kaggle/working/', texts_filepath='../input/', interaction_sequence_len=30,
                 min_seq_len=5, dictionary=None, **nlp_kwargs):
    inputs = {"question_id": False, "text_id": False, "skill": False,
              "label": False, "r_elapsed_time": False, 'text_encoding': True, "target_id": False,
              "target_text_id": False, "target_skill": False, 'target_label': False, 'target_text_encoding': True}
    outputs = {"question_id": False, "text_id": False, "skill": False,
               "label": False, "r_elapsed_time": False, 'text_encoding': False, "target_id": False,
               "target_text_id": False, "target_skill": False, 'target_label': True, 'target_text_encoding': False}

    text_df = load_preprocessed_texts(texts_filepath=texts_filepath)
    # Step 3.1 - Generate NLP extracted encoding for problems
    encode_models = []
    if 'count_vectorizer' in nlp_kwargs:
        count_vectorizer_args = nlp_kwargs['count_vectorizer']
        min_df, max_df, max_features = count_vectorizer_args['min_df'], count_vectorizer_args['max_df'], \
                                       count_vectorizer_args['max_features']
        encode_model = count_vectorizer(min_df=min_df, max_df=max_df, binary=False, max_features=max_features)
        encode_model.fit(text_df, save_filepath)
        encode_models.append(encode_model)

    if 'pretrained_distilBERT' in nlp_kwargs:
        pretrained_distilBERT_args = nlp_kwargs['pretrained_distilBERT']
        config_path, model_filepath = pretrained_distilBERT_args['config_path'], \
                                      pretrained_distilBERT_args['model_filepath']
        encode_model = PretrainedDistilBERT(config_path, model_filepath)
        encode_model.fit(text_df)
        encode_models.append(encode_model)

    if 'sentence_transformers' in nlp_kwargs:
        model_name, text_column = nlp_kwargs['sentence_transformers']['model_name'], \
                                   nlp_kwargs['sentence_transformers']['text_column']
        encode_model = sentence_transformer(encoding_model=model_name)
        encode_model.fit(text_df, text_column)
        encode_models.append(encode_model)

    if 'pretrained_distilbert_finetuned_on_CA' in nlp_kwargs:
        encode_model = PretrainedDistilBERTFinetuned()
        pretrained_distilbert_finetuned_args = nlp_kwargs['pretrained_distilbert_finetuned_on_CA']
        if 'load' in pretrained_distilbert_finetuned_args:
            print("load")
            config_path, model_filepath = pretrained_distilbert_finetuned_args['load']['config_path'], \
                                          pretrained_distilbert_finetuned_args['load']['model_filepath']
            encode_model.load(config_path=config_path, model_filepath=model_filepath)
        elif 'fit_on_custom' in pretrained_distilbert_finetuned_args:
            config_path, model_filepath = pretrained_distilbert_finetuned_args['fit_on_custom']['config_path'], \
                                          pretrained_distilbert_finetuned_args['fit_on_custom']['model_filepath']
            text_column, batch_size = pretrained_distilbert_finetuned_args['fit_on_custom']['text_column'], \
                pretrained_distilbert_finetuned_args['fit_on_custom']['text_column']
            encode_model.fit_on_custom(text_df, config_path=config_path, model_filepath=model_filepath,
                                       save_filepath=save_filepath, text_column=text_column, batch_size=batch_size)
        elif 'fit_on_nli' in pretrained_distilbert_finetuned_args:
            config_path, model_filepath = pretrained_distilbert_finetuned_args['fit_on_custom']['config_path'], \
                                          pretrained_distilbert_finetuned_args['fit_on_custom']['model_filepath']
            encode_model.fit_on_custom(text_df, config_path=config_path, model_filepath=model_filepath,
                                       save_filepath=save_filepath)
        text_column, batch_size = pretrained_distilbert_finetuned_args['transform']['text_column'], \
            pretrained_distilbert_finetuned_args['transform']['batch_size']
        encode_model.transform(text_df, text_column=text_column, batch_size=batch_size)
        encode_models.append(encode_model)

    train_gen, val_gen, test_gen, nb_questions, nb_skills = get_hybrid_dkt_dataloaders(
        batch_size, shuffle, interactions_filepath, output_filepath=save_filepath,
        interaction_sequence_len=interaction_sequence_len, min_seq_len=min_seq_len, text_encoding_models=encode_models,
        negative_correctness=False, inputs_dict=inputs, outputs_dict=outputs, encode_correct_in_encodings=True,
        encode_correct_in_skills=False, dictionary=dictionary)

    encoding_depths = train_gen.encoding_depths
    train_loader = create_dataset(train_gen, encode_models, encoding_depths, shuffle=shuffle, batch_size=batch_size)
    val_loader = create_dataset(val_gen, encode_models, encoding_depths, shuffle=shuffle, batch_size=batch_size)
    test_loader = create_dataset(test_gen, encode_models, encoding_depths, shuffle=shuffle, batch_size=batch_size)

    encode_names = [encode_model.name for encode_model in encode_models]
    return train_loader, val_loader, test_loader, encoding_depths, encode_names
