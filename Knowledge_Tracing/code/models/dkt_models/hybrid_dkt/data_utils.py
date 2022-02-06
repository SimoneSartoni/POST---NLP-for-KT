import gc

import tensorflow as tf

from Knowledge_Tracing.code.models.nlp_models.count_vectorizer import count_vectorizer
# from Knowledge_Tracing.code.models.nlp_models.BERTopic_model import BERTopic_model
from Knowledge_Tracing.code.models.nlp_models.pretrained_distilbert import PretrainedDistilBERT
from Knowledge_Tracing.code.models.nlp_models.sentence_transformers import sentence_transformer
from Knowledge_Tracing.code.models.nlp_models.bertopic_model import BERTopic_model
from Knowledge_Tracing.code.models.nlp_models.gensim_model.gensim_word2vec import word2vec

from Knowledge_Tracing.code.data_processing.load_preprocessed.load_preprocessed_data import load_preprocessed_texts
from Knowledge_Tracing.code.data_processing.load_preprocessed.get_hybrid_dkt_dataloaders import \
    get_hybrid_dkt_dataloaders

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
    encode_names = []
    parameters = []
    if 'count_vectorizer' in nlp_kwargs:
        count_vectorizer_args = nlp_kwargs['count_vectorizer']
        min_df, max_df, max_features = count_vectorizer_args['min_df'], count_vectorizer_args['max_df'], \
            count_vectorizer_args['max_features']
        encode_model = count_vectorizer(min_df=min_df, max_df=max_df, binary=False, max_features=max_features)
        encode_model.fit(text_df, save_filepath)
        encode_models.append(encode_model)
        parameters.append("_".join([str(min_df), str(max_df), str(max_features)]))
        encode_names.append(encode_model.name)

    if 'pretrained_distilbert' in nlp_kwargs:
        pretrained_distilbert_args = nlp_kwargs['pretrained_distilbert']
        config_path, model_filepath, fit_on_custom, fit_on_nli, text_column, batch_size = \
            pretrained_distilbert_args['config_path'], pretrained_distilbert_args['model_filepath'], \
            pretrained_distilbert_args['fit_on_custom'], pretrained_distilbert_args['fit_on_nli'], \
            pretrained_distilbert_args['text_column'], pretrained_distilbert_args['batch_size']
        encode_model = PretrainedDistilBERT(config_path, model_filepath)
        if fit_on_nli:
            encode_model.fit_on_nli(text_df)
        if fit_on_custom:
            batch_size = fit_on_custom['batch_size']
            encode_model.fit_on_custom(text_df, text_column=text_column, batch_size=batch_size)
        encode_model.transform(text_df, text_column, batch_size)
        encode_models.append(encode_model)
        parameters.append("_".join([str(fit_on_nli), str(fit_on_custom), str(text_column), str(text_column)]))
        encode_names.append(encode_model.name)

    if 'sentence_transformers' in nlp_kwargs:
        model_name, text_column, fit, load, save_filepath = nlp_kwargs['sentence_transformers']['model_name'], \
                                               nlp_kwargs['sentence_transformers']['text_column'], \
                                               nlp_kwargs['sentence_transformers']['fit'], \
                                               nlp_kwargs['sentence_transformers']['load'],\
                                               nlp_kwargs['sentence_transformers']['save_filepath'],

        encode_model = sentence_transformer(encoding_model=model_name)
        if load:
            encode_model.load_embeddings(load['load_path'])
        else:
            if fit:
                batch_size, fraction, epochs = fit['batch_size'], fit['fraction'], fit['epochs']
                encode_model.fit_on_custom(text_df, text_column=text_column, batch_size=batch_size, frac=fraction, epochs=epochs)
            encode_model.transform(text_df, text_column, save_filepath)
        encode_models.append(encode_model)
        parameters.append("_".join([str(model_name), str(fit), str(text_column)]))
        encode_names.append(encode_model.name)

    if 'bertopic' in nlp_kwargs:
        bertopic_args = nlp_kwargs['bertopic']
        pretrained, custom, text_column = bertopic_args['pretrained'], bertopic_args['custom'], \
                                          bertopic_args['text_column']
        nr_topics, calculate_probabilities, cluster_selection_method, output = bertopic_args['nr_topics'], \
            bertopic_args['calculate_probabilities'], bertopic_args['cluster_selection_method'], bertopic_args['output']
        encode_model = BERTopic_model(nr_topics, calculate_probabilities, cluster_selection_method, output)
        if pretrained:
            model_path_or_name = pretrained['model_path_or_name']
            encode_model.initialize_pretrained_bertopic(model_path_or_name)
        if custom:
            config_path, model_path_or_name, tokenizer_name, from_tf = custom['config_path'], \
                custom['model_path_or_name'], custom['tokenizer_name'], custom['from_tf']
            encode_model.initialize_custom_bertopic(config_path, model_path_or_name, from_tf)
        encode_model.fit(text_df)
        encode_model.transform(text_df, text_column)
        encode_models.append(encode_model)
        parameters.append("_".join([str(pretrained), str(custom), str(text_column), str(output), str(nr_topics)]))
        encode_names.append(encode_model.name)

    if 'word2vec' in nlp_kwargs:
        word2vec_args = nlp_kwargs['word2vec']
        text_column, epochs, save_filepath, save_name = word2vec_args['text_column'], word2vec_args['epochs'], \
            word2vec_args['save_filepath'], word2vec_args['save_name']
        min_count, window, vector_size, sg = word2vec_args['min_count'], word2vec_args['window'], \
            word2vec_args['vector_size'], word2vec_args['sg']
        encode_model = word2vec(min_count, window, vector_size)
        encode_model.fit(text_df, text_column, epochs, save_filepath, save_name)
        encode_model.transform(text_df, text_column)
        encode_models.append(encode_model)
        parameters.append("_".join([str(text_column), str(epochs), str(min_count), str(window), str(vector_size)]))
        encode_names.append(encode_model.name)

    train_gen, val_gen, test_gen, nb_questions, nb_skills = get_hybrid_dkt_dataloaders(
        batch_size, shuffle, interactions_filepath, output_filepath=save_filepath,
        interaction_sequence_len=interaction_sequence_len, min_seq_len=min_seq_len, text_encoding_models=encode_models,
        negative_correctness=False, inputs_dict=inputs, outputs_dict=outputs, encode_correct_in_encodings=True,
        encode_correct_in_skills=False, dictionary=dictionary)

    encoding_depths = train_gen.encoding_depths
    train_loader = create_dataset(train_gen, encode_models, encoding_depths, shuffle=shuffle, batch_size=batch_size)
    val_loader = create_dataset(val_gen, encode_models, encoding_depths, shuffle=shuffle, batch_size=batch_size)
    test_loader = create_dataset(test_gen, encode_models, encoding_depths, shuffle=shuffle, batch_size=batch_size)

    return train_loader, val_loader, test_loader, encoding_depths, encode_names, parameters
