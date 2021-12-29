import tensorflow as tf
from sklearn.model_selection import train_test_split

from Knowledge_Tracing.code.data_processing.load_preprocessed.get_dkt_dataloaders import get_DKT_dataloaders

MASK_VALUE = -1.  # The masking value cannot be zero.


def create_dataset(generator, features_depth, skill_depth, shuffle=True, batch_size=1024):
    input_types = {"feature": tf.float32, "target_skill": tf.int32}
    output_types = {"target_label": tf.float32}

    input_shapes = {"feature": [None, features_depth],  "target_skill": [None]}
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
            {"input_feature": inputs['feature'], "target_skill": tf.one_hot(inputs['target_skill'], depth=skill_depth)},
            tf.expand_dims(outputs['target_label'], -1)
        )
    )

    print(dataset)

    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padded_shapes=({"input_feature": [None, features_depth], "target_skill": [None, skill_depth]}, [None, 1]),
        padding_values=({"input_feature": MASK_VALUE, "target_skill": MASK_VALUE}, MASK_VALUE),
        drop_remainder=True
    )
    return dataset


def load_dataset(batch_size=32, shuffle=True,
                 interactions_filepath="../input/assistmentds-2012/2012-2013-data-with-predictions-4-final.csv",
                 save_filepath='/kaggle/working/', texts_filepath='../input/',
                 interaction_sequence_len=30, min_seq_len=5, dictionary=None):
    inputs = {"question_id": False, "text_id": False, "skill": False,
              "label": False, "r_elapsed_time": False, 'text_encoding': False, "target_id": False, "feature": True,
              "target_text_id": False, "target_skill": True, 'target_label': False, 'target_text_encoding': False,
              "target_feature": False}
    outputs = {"question_id": False, "text_id": False, "skill": False,
               "label": False, "r_elapsed_time": False, 'text_encoding': False, "target_id": False, "feature": False,
               "target_text_id": False, "target_skill": False, 'target_label': True, 'target_text_encoding': False,
               "target_feature": False}

    train_gen, val_gen, test_gen, \
        nb_questions, nb_skills = get_DKT_dataloaders(batch_size, shuffle, interactions_filepath,
                                                      output_filepath='/kaggle/working/',
                                                      interaction_sequence_len=interaction_sequence_len
                                                      , min_seq_len=min_seq_len,
                                                      text_encoding_model=None,
                                                      negative_correctness=False,
                                                      inputs_dict=inputs, outputs_dict=outputs,
                                                      encode_correct_in_encodings=False,
                                                      encode_correct_in_skills=True,
                                                      dictionary=dictionary)
    features_depth = 2*nb_skills
    skill_depth = nb_skills
    train_loader = create_dataset(train_gen, features_depth, skill_depth, shuffle=shuffle, batch_size=batch_size)
    val_loader = create_dataset(val_gen, features_depth, skill_depth, shuffle=shuffle, batch_size=batch_size)
    test_loader = create_dataset(test_gen, features_depth, skill_depth, shuffle=shuffle, batch_size=batch_size)

    return train_loader, val_loader, test_loader, features_depth, skill_depth
