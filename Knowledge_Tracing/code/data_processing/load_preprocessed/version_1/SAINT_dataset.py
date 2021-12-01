import numpy as np
from torch.utils.data import Dataset, DataLoader


def encode_correctness_in_encodings(text_encoding_model, text_ids, corrects, max_seq, encode_correct_in_encodings):
    i = 0
    if encode_correct_in_encodings:
        input_text_encoding = np.zeros((max_seq, 2*text_encoding_model.vector_size), dtype=int)
    else:
        input_text_encoding = np.zeros((max_seq, text_encoding_model.vector_size), dtype=int)
    for text_id, correctness in list(zip(*(text_ids, corrects))):
        encoding = text_encoding_model.get_encoding(text_id)
        if encode_correct_in_encodings:
            zeros = np.zeros(encoding.shape, dtype=np.float)
            if correctness:
                encoding = np.concatenate([encoding, zeros])
            else:
                encoding = np.concatenate([zeros, encoding])
        input_text_encoding[i] = encoding
    return input_text_encoding

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


class SAINT_Dataset(Dataset):
    def __init__(self, grouped_df, text_encoding_model=None, max_seq=100, negative_correctness=False,
                 encode_correct_in_encodings=True, encoder_inputs_dict={}, decoder_inputs_dict={}, outputs_dict={}):
        self.max_seq = max_seq
        self.data = grouped_df
        self.encode_correct_in_encodings = encode_correct_in_encodings
        self.negative_correctness = negative_correctness
        self.text_encoding_model = text_encoding_model
        if text_encoding_model:
            if encode_correct_in_encodings:
                self.encoding_depth = 2*self.text_encoding_model.vector_size
            else:
                self.encoding_depth = self.text_encoding_model.vector_size
        else:
            self.encoding_depth = 0
        self.encoder_inputs_dict = encoder_inputs_dict
        self.decoder_inputs_dict = decoder_inputs_dict
        self.outputs_dict = outputs_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user_id, unique_question_id, text_id, answered_correctly, response_elapsed_time, exe_skill = self.data[idx]
        seq_len = len(unique_question_id)

        q_id = np.zeros(self.max_seq, dtype=int)
        text_id = np.zeros(self.max_seq, dtype=int)
        ans = np.zeros(self.max_seq, dtype=int)
        r_elapsed_time = np.zeros(self.max_seq, dtype=int)
        skill = np.zeros(self.max_seq, dtype=int)

        q_id[-seq_len:] = unique_question_id
        text_id[-seq_len:] = text_id
        if self.negative_correctness:
            ans[-seq_len:] = [1.0 if x == 1.0 else -1.0 for x in answered_correctly]
        else:
            ans[-seq_len:] = [1.0 if x == 1.0 else 0.0 for x in answered_correctly]
        r_elapsed_time[-seq_len:] = response_elapsed_time
        skill[-seq_len:] = exe_skill


        if self.text_encoding_model:
            input_text_encoding = encode_correctness_in_encodings(self.text_encoding_model, text_id,
                                                                  answered_correctly, self.max_seq,
                                                                  self.encode_correct_in_encodings)

        input_id = np.zeros(self.max_seq, dtype=int)
        input_r_elapsed_time = np.zeros(self.max_seq, dtype=int)
        input_label = np.zeros(self.max_seq, dtype=int)
        input_text_id = np.zeros(self.max_seq, dtype=int)
        input_skill = np.zeros(self.max_seq, dtype=int)

        input_id[1:] = q_id[:-1]
        input_text_id[1:] = text_id[:-1]
        input_r_elapsed_time[1:] = r_elapsed_time[:-1].copy().astype(np.int)
        input_skill[1:] = skill
        input_label[1:] = ans[:-1]

        target_id = np.zeros(self.max_seq, dtype=int)
        target_r_elapsed_time = np.zeros(self.max_seq, dtype=int)
        target_label = np.zeros(self.max_seq, dtype=int)
        target_text_id = np.zeros(self.max_seq, dtype=int)
        target_skill = np.zeros(self.max_seq, dtype=int)

        target_id[1:] = input_id[1:]
        target_text_id[1:] = input_text_id[1:]
        target_r_elapsed_time[1:] = r_elapsed_time[1:].copy().astype(np.int)
        target_skill[1:] = input_skill[1:]
        target_label[1:] = ans[1:]

        possible_inputs = {"question_id": question_id, "text_id": text_id, "skill": skill,
                           "label": label, "r_elapsed_time": r_elapsed_time,
                           "input_question_id": input_id, "input_text_id": input_text_id, "input_skill": input_skill,
                           "input_label": input_label, "input_r_elapsed_time": input_r_elapsed_time,
                           "target_id": target_ids,  "target_text_id": target_text_ids, "target_skill": target_skill,
                           "target_r_elapsed_time": target_r_elapsed_time, 'target_label': target_label}
        if self.text_encoding_model:
            possible_inputs["text_encoding"] = text_encoding
            possible_inputs["input_text_encoding"] = input_text_encoding
            possible_inputs["target_text_encoding"] = input_text_encoding[1:]
        possible_outputs = possible_inputs
        encoder_inputs = {}
        for key in possible_inputs.keys():
            if self.encoder_inputs_dict[key]:
                encoder_inputs[key] = possible_inputs[key]

        decoder_inputs = {}
        for key in possible_inputs.keys():
            if self.decoder_inputs_dict[key]:
                decoder_inputs[key] = possible_inputs[key]

        outputs = {}
        for key in possible_outputs.keys():
            if self.outputs_dict[key]:
                outputs[key] = possible_outputs[key]
        inputs = {"decoder": decoder_inputs, "encoder": encoder_inputs}
        return inputs, outputs
