import numpy as np


def encode_correctness_in_encodings(text_encoding_model, text_id, correctness):
    encoding = text_encoding_model.get_encoding(text_id)
    zeros = np.zeros(encoding.shape, dtype=np.float)
    target_encoding = np.concatenate([encoding, encoding])
    if correctness:
        encoding = np.concatenate([encoding, zeros])
    else:
        encoding = np.concatenate([zeros, encoding])
    return encoding, target_encoding


def encode_correctness_in_skills(skill, correctness, nb_skills):
    zeros = np.zeros(nb_skills, dtype=np.int)
    skill_one_hot_encoding = zeros
    skill_one_hot_encoding[skill] = 1
    target_features = np.concatenate([skill_one_hot_encoding, skill_one_hot_encoding])
    if correctness:
        features = np.concatenate([skill_one_hot_encoding, zeros])
    else:
        features = np.concatenate([zeros, skill_one_hot_encoding])
    return features, target_features


def encode_correctness_in_id(question_id, correctness, nb_questions):
    zeros = np.zeros(nb_questions, dtype=np.int)
    id_one_hot_encoding = zeros
    id_one_hot_encoding[question_id] = 1
    target_feature_ids = np.concatenate([id_one_hot_encoding, id_one_hot_encoding])
    if correctness:
        feature_ids = np.concatenate([id_one_hot_encoding, zeros])
    else:
        feature_ids = np.concatenate([zeros, id_one_hot_encoding])
    return feature_ids, target_feature_ids


class hybrid_dkt_dataset:
    def __init__(self, grouped_df, text_encoding_models={}, max_seq=100, negative_correctness=False,
                 encode_correct_in_encodings=True, encode_correct_in_skills=True, encode_correct_in_id=False,
                 inputs_dict={}, outputs_dict={}, nb_skills=300, nb_questions=10000):
        self.max_seq = max_seq
        self.data = grouped_df
        self.encode_correct_in_encodings = encode_correct_in_encodings
        self.encode_correct_in_skills = encode_correct_in_skills
        self.encode_correct_in_id = encode_correct_in_id
        self.negative_correctness = negative_correctness
        self.text_encoding_models = text_encoding_models
        self.inputs_dict = inputs_dict
        self.outputs_dict = outputs_dict
        self.nb_skills = nb_skills
        self.nb_questions = nb_questions

        self.encoding_depths = {}
        if self.text_encoding_models:
            for encoding_model in self.text_encoding_models:
                if self.encode_correct_in_encodings:
                    self.encoding_depths[encoding_model.name] = 2 * encoding_model.vector_size
                else:
                    self.encoding_depths[encoding_model.name] = encoding_model.vector_size
        else:
            self.encoding_depths = 0

    def __len__(self):
        return len(self.data)

    def generator(self):
        for user_id, unique_question_id, text_ids, answered_correctly, response_elapsed_time, exe_skill in self.data:

            if self.negative_correctness:
                ans = [1.0 if x == 1.0 else -1.0 for x in answered_correctly]
            else:
                ans = answered_correctly

            input_ids = unique_question_id[:-1]
            input_text_ids = text_ids[:-1]
            input_skill = exe_skill[:-1]
            text_encodings = {}
            target_text_encodings = {}
            if self.text_encoding_models:
                for encoding_model in self.text_encoding_models:
                    encodings = []
                    target_encodings = []
                    if self.encode_correct_in_encodings:
                        for text_id, correct in list(zip(text_ids, answered_correctly)):
                            encoding, target_encoding = encode_correctness_in_encodings(encoding_model, text_id,
                                                                                        correct)
                            encodings.append(encoding)
                            target_encodings.append(target_encoding)
                        text_encodings[encoding_model.name] = encodings[:-1]
                        target_text_encodings[encoding_model.name] = target_encodings[1:]
                    else:
                        all_text_encodings = [encoding_model.get_encoding(text_id) for text_id in text_ids]
                        target_text_encodings[encoding_model.name] = all_text_encodings[1:]
                        text_encodings[encoding_model.name] = all_text_encodings[:-1]

            input_r_elapsed_time = response_elapsed_time[:-1].copy().astype(np.int)
            input_label = ans[:-1]

            target_ids = unique_question_id[1:]
            target_text_ids = text_ids[1:]
            target_skill = exe_skill[1:]
            target_label = ans[1:]
            features = []
            target_features = []
            if self.encode_correct_in_skills:
                for skill, correct in list(zip(exe_skill, answered_correctly)):
                    feature, skill_repeated = encode_correctness_in_skills(skill, correct, self.nb_skills)
                    features.append(feature)
                    target_features.append(skill_repeated)
                input_features = features[:-1]
                target_features = target_features[1:]

            feature_ids = []
            target_feature_ids = []
            if self.encode_correct_in_id:
                for question_id, correct in list(zip(unique_question_id, answered_correctly)):
                    feature_id, target_feature_id = encode_correctness_in_id(question_id, correct, self.nb_questions)
                    feature_ids.append(feature_id)
                    target_feature_ids.append(target_feature_id)
                input_feature_ids = feature_ids[:-1]
                target_feature_ids = target_feature_ids[1:]

            possible_inputs = {"question_id": input_ids, "text_id": input_text_ids, "skill": input_skill,
                               "label": input_label, "r_elapsed_time": input_r_elapsed_time,
                               "target_id": target_ids, "target_text_id": target_text_ids, "target_skill": target_skill,
                               'target_label': target_label}
            if self.encode_correct_in_skills:
                possible_inputs["feature"] = input_features
                possible_inputs["target_feature"] = target_features
            if self.encode_correct_in_id:
                possible_inputs["feature_id"] = input_feature_ids
                possible_inputs["target_feature_id"] = target_feature_ids
            possible_outputs = possible_inputs
            inputs = {}
            for key in possible_inputs.keys():
                if self.inputs_dict[key]:
                    inputs[key] = possible_inputs[key]
            if self.text_encoding_models:
                for encoding_model, encoding, target_encoding in list(zip(self.text_encoding_models, text_encodings,
                                                                          target_text_encodings)):
                    inputs[encoding_model.name] = text_encodings[encoding]
                    inputs["target_" + encoding_model.name] = target_text_encodings[target_encoding]
            outputs = {}
            for key in possible_outputs.keys():
                if self.outputs_dict[key]:
                    outputs[key] = possible_outputs[key]
            yield inputs, outputs
