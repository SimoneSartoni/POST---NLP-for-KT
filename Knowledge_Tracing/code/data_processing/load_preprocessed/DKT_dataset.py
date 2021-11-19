import numpy as np
from torch.utils.data import Dataset, DataLoader


def encode_correctness_in_encodings(text_encoding_model, text_id, correctness):
    encoding = text_encoding_model.get_encoding(text_id)
    zeros = np.zeros(encoding.shape, dtype=np.float)
    if correctness:
        encoding = np.concatenate([encoding, zeros])
    else:
        encoding = np.concatenate([zeros, encoding])
    return encoding


class DKT_Dataset(Dataset):
    def __init__(self, grouped_df, text_encoding_model=None, max_seq=100, negative_correctness=False,
                 encode_correct_in_encodings=True, **inputs_outputs):
        self.max_seq = max_seq
        self.data = grouped_df
        self.encode_correct_in_encodings = encode_correct_in_encodings
        self.negative_correctness = negative_correctness
        self.text_encoding_model = text_encoding_model
        self.inputs_outputs = inputs_outputs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user_id, unique_question_id, text_id, answered_correctly, response_elapsed_time, exe_skill = self.data[idx]
        seq_len = len(unique_question_id)

        q_ids = np.zeros(self.max_seq, dtype=int)
        text_ids = np.zeros(self.max_seq, dtype=int)
        ans = np.zeros(self.max_seq, dtype=int)
        r_elapsed_time = np.zeros(self.max_seq, dtype=int)
        skill = np.zeros(self.max_seq, dtype=int)
        input_label = np.zeros(self.max_seq, dtype=int)
        input_r_elapsed_time = np.zeros(self.max_seq, dtype=int)

        q_ids[-seq_len:] = unique_question_id
        text_ids[-seq_len:] = text_id
        if self.negative_correctness:
            ans[-seq_len:] = [1.0 if x == 1.0 else -1.0 for x in answered_correctly]
        else:
            ans[-seq_len:] = answered_correctly
        r_elapsed_time[-seq_len:] = response_elapsed_time
        skill[-seq_len:] = exe_skill

        input_ids = q_ids
        input_text_ids = text_ids
        if self.text_encoding_model:
            if self.encode_correct_in_encodings:
                input_text_encodings = [encode_correctness_in_encodings(self.text_encoding_model, text_id, correct)
                                        for text_id, correct in list(zip(text_ids, answered_correctly))]
            else:
                input_text_encodings = [self.text_encoding_model.get_encoding(text_id) for text_id in text_ids]
        input_r_elapsed_time[1:] = r_elapsed_time[:-1].copy().astype(np.int)
        input_skill = skill
        input_label[1:] = ans[:-1]

        target_ids = input_ids[:]
        target_text_ids = input_text_ids
        target_skill = input_skill[:]
        target_label = ans
        possible_inputs = {"question_id": input_ids, "text_id": input_text_ids, "skill": input_skill,
                           "label": input_label, "r_elapsed_time": input_r_elapsed_time}
        if self.text_encoding_model:
            possible_inputs["text_encoding"] = input_text_encodings
        possible_outputs = {"target_id": target_ids, "target_text_id": target_text_ids, "target_skill": target_skill,
                           'target_label': target_label}
        inputs = {}
        for key in possible_inputs.keys():
            if self.encodings_inputs[key]:
                inputs[key] = possible_inputs[key]
        outputs = {}
        for key in possible_outputs.keys():
            if self.encodings_inputs[key]:
                outputs[key] = possible_outputs[key]
        return inputs, outputs
