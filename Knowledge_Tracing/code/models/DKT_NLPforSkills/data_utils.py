import pandas as pd
import tensorflow as tf
import numpy as np
from Knowledge_Tracing.code.data_processing.dataset import dataset as dt
from Knowledge_Tracing.code.evaluation.predictors.logistic_regression import logistic_regressor
from code.models.gensim_model.gensim_pretrained_word2vec import pretrained_word2vec

MASK_VALUE = -1.  # The masking value cannot be zero.


def load_dataset_NLP_skills(fn, batch_size=32, shuffle=True):
    df = pd.read_csv(fn, encoding="ISO-8859-1")

    if "skill_id" not in df.columns:
        raise KeyError(f"The column 'skill_id' was not found on {fn}")
    if "correct" not in df.columns:
        raise KeyError(f"The column 'correct' was not found on {fn}")
    if "user_id" not in df.columns:
        raise KeyError(f"The column 'user_id' was not found on {fn}")

    if not (df['correct'].isin([0, 1])).all():
        raise KeyError(f"The values of the column 'correct' must be 0 or 1.")

    # Step 1.1 - Remove questions without skill
    df.dropna(subset=['skill_id'], inplace=True)

    # Step 1.2 - Remove users with a single answer
    df = df.groupby('user_id').filter(lambda q: len(q) > 1).copy()

    # Step 2 - Enumerate skill id
    df['skill'], _ = pd.factorize(df['skill_id'], sort=True)
    print(df['skill'])

    df['skill_with_answer'] = df['skill'] * 2 + df['correct']

    # Step 3.1 - Generate NLP extracted encoding for problems
    loaded_dataset = dt(name="assistments_2009", path="/Knowledge_Tracing/intermediate_files", prefix="clean_datasets/")
    loaded_dataset.load_saved_texts()
    encode_model = pretrained_word2vec()
    encode_model.fit()
    encode_model.encode_problems(loaded_dataset.problem_id_to_index, loaded_dataset.texts_list)

    # Step 3.2 - Remove problems without encoding (because we do not have text)
    df = df.loc[df['problem_id'].isin(loaded_dataset.problem_id_to_index)]

    df['NLP_skills'] = [tf.constant(encode_model.get_encoding(problem), shape=encode_model.vector_size,
                                    dtype=tf.float32) for problem in df['problem_id']]
    print(df['NLP_skills'])

    encoding_depth = encode_model.vector_size
    skill_depth = df['skill'].max() + 1
    # Step 4 - Convert to a sequence per user id and shift features 1 timestep
    seq = df.groupby('user_id').apply(
        lambda r: (
            r['correct'].to_numpy()[:-1],
            r['NLP_skills'].to_numpy()[:-1],
            tf.one_hot(r['skill'].to_numpy()[:-1], depth=skill_depth),
            r['correct'].to_numpy()[1:],
            r['NLP_skills'].to_numpy()[1:],
            tf.one_hot(r['skill'].to_numpy()[1:], depth=skill_depth)
        )
    )
    nb_users = len(seq)


    # Step 5 - Get Tensorflow Dataset
    dataset = tf.data.Dataset.from_generator(
        generator=lambda: seq,
        output_types=({'input_corrects': tf.int32, 'input_encodings': tf.float32, 'input_skills': tf.int32,
                      'output_corrects': tf.int32, 'output_encodings': tf.float32, 'output_skills': tf.int32}),
        output_shapes=({'input_corrects': [None, 1], 'input_encodings': [None, encode_model.vector_size],
                        'input_skills': [None, skill_depth],
                        'output_corrects': [None, 1], 'output_encodings': [None, encode_model.vector_size],
                        'output_skills': [None, skill_depth]})
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=nb_users)

    print(dataset)
    dataset = dataset.map(
        lambda dictionary: (
            {"input_correct": dictionary["input_corrects"],
             "input_encoding": dictionary["input_encodings"],
             "input_skill": dictionary["input_skills"]
             },
            {"output_correct": dictionary["output_corrects"],
             "output_encoding": dictionary["output_encodings"], "output_skill": dictionary["output_skills"]}
        )
    )

    # Step 6 - Encode categorical features and merge skills with labels to compute target loss.
    # More info: https://github.com/tensorflow/tensorflow/issues/32142

    print(dataset)

    # Step 7 - Pad sequences per batch
    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padding_values=({'input_correct': -1, 'input_encoding': -1.0, 'input_skill': -1},
                        {"output_correct": -1, "output_encoding": -1.0, "output_skill": -1}),
        drop_remainder=True
    )
    """padded_shapes=({'input_correct': [None], 'input_skill': [None, None],
                            'input_encoding': [None, encoding_depth]},
                           {'output_correct': [None], 'output_skill': [None, None],
                            'output_encoding': [None, encoding_depth]}),"""
    print(dataset)

    length = nb_users // batch_size
    return dataset, length, skill_depth, encoding_depth
