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


class DKT_Dataset:
    def __init__(self, grouped_df, text_encoding_model=None, max_seq=100, negative_correctness=False,
                 encode_correct_in_encodings=True, encode_correct_in_skills=True, inputs_dict={}, outputs_dict={},
                 nb_skills=300):
        self.max_seq = max_seq
        self.data = grouped_df
        self.encode_correct_in_encodings = encode_correct_in_encodings
        self.encode_correct_in_skills = encode_correct_in_skills
        self.negative_correctness = negative_correctness
        self.text_encoding_model = text_encoding_model
        self.inputs_dict = inputs_dict
        self.outputs_dict = outputs_dict
        self.nb_skills = nb_skills
        if self.text_encoding_model:
            if self.encode_correct_in_encodings:
                self.encoding_depth = 2 * self.text_encoding_model.vector_size
            else:
                self.encoding_depth = self.text_encoding_model.vector_size
        else:
            self.encoding_depth = 0

    def __len__(self):
        return len(self.data)

    def generator(self):
        for user_id, unique_question_id, text_ids, answered_correctly, response_elapsed_time, exe_skill in self.data:

            if self.negative_correctness:
                ans = [1.0 if x == 1.0 else -1.0 for x in answered_correctly]
            else:
                ans = answered_correctly

            input_ids = unique_question_id
            input_text_ids = text_ids
            input_skill = exe_skill[:-1]
            text_encodings = []
            target_text_encodings = []
            if self.text_encoding_model:
                if self.encode_correct_in_encodings:
                    for text_id, correct in list(zip(text_ids, answered_correctly)):
                        text_encoding, target_encoding = encode_correctness_in_encodings(self.text_encoding_model, text_id, correct)
                        text_encodings.append(text_encoding)
                        target_text_encodings.append(target_encoding)
                    text_encodings = text_encodings[:-1]
                    target_text_encodings = target_text_encodings[1:]
                else:
                    all_text_encodings = [self.text_encoding_model.get_encoding(text_id) for text_id in text_ids]
                    target_text_encodings = all_text_encodings[1:]
                    text_encodings = all_text_encodings[:-1]


            input_r_elapsed_time = response_elapsed_time[:-1].copy().astype(np.int)
            input_label = ans[:-1]

            target_ids = input_ids[1:]
            target_text_ids = input_text_ids[1:]
            target_skill = input_skill[1:]
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

            possible_inputs = {"question_id": input_ids, "text_id": input_text_ids, "skill": input_skill,
                               "label": input_label, "r_elapsed_time": input_r_elapsed_time,
                               "target_id": target_ids, "target_text_id": target_text_ids, "target_skill": target_skill,
                               'target_label': target_label}
            if self.text_encoding_model:
                possible_inputs["text_encoding"] = text_encodings
                possible_inputs["target_text_encoding"] = target_text_encodings
            if self.encode_correct_in_skills:
                possible_inputs["feature"] = input_features
                possible_inputs["target_feature"] = target_features
            possible_outputs = possible_inputs
            inputs = {}
            for key in possible_inputs.keys():
                if self.inputs_dict[key]:
                    inputs[key] = possible_inputs[key]

            outputs = {}
            for key in possible_outputs.keys():
                if self.outputs_dict[key]:
                    outputs[key] = possible_outputs[key]
            yield inputs, outputs