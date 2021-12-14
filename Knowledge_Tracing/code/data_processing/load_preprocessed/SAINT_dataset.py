import numpy as np
from torch.utils.data import Dataset, DataLoader


def encode_correctness_in_encodings(text_encoding_model, text_ids, max_seq, mask_value):
    i = 0
    text_encoding = np.full((max_seq, text_encoding_model.vector_size), fill_value=mask_value, dtype=double)
    for text_id in text_ids:
        if text_id != mask_value:
            text_encoding[i] = text_encoding_model.get_encoding(text_id)
        i += 1
    input_text_encoding = text_encoding[-1]
    target_text_encoding = text_encoding[1:]
    return text_encoding, input_text_encoding, target_text_encoding


class SAINT_Dataset(Dataset):
    def __init__(self, grouped_df, text_encoding_model=None, max_seq=100, negative_value=0.0,
                 inputs_output_dict={}, mask_value=0.0):
        self.encoder_inputs_dict = inputs_output_dict['encoder']
        self.decoder_inputs_dict = inputs_output_dict['decoder']
        self.outputs_dict = inputs_output_dict['output']
        self.max_seq = max_seq
        self.data = grouped_df
        self.mask_value = mask_value
        self.negative_value = negative_value
        self.text_encoding_model = text_encoding_model
        if text_encoding_model:
            self.encoding_depth = self.text_encoding_model.vector_size
        else:
            self.encoding_depth = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user_id, unique_question_id, unique_text_id, answered_correctly, response_elapsed_time, exe_skill = self.data[idx]
        seq_len = len(unique_question_id)

        question_id = np.full(self.max_seq, fill_value=self.mask_value, dtype=double)
        text_id = np.full(self.max_seq, fill_value=self.mask_value, dtype=double)
        label = np.full(self.max_seq, fill_value=self.mask_value, dtype=double)
        r_elapsed_time = np.full(self.max_seq, fill_value=self.mask_value, dtype=double)
        skill = np.full(self.max_seq, fill_value=self.mask_value, dtype=double)

        question_id[-seq_len:] = unique_question_id
        text_id[-seq_len:] = unique_text_id
        label[-seq_len:] = [1.0 if x == 1.0 else self.negative_value for x in answered_correctly]

        r_elapsed_time[-seq_len:] = response_elapsed_time
        skill[-seq_len:] = exe_skill

        input_id = np.full(self.max_seq, fill_value=self.mask_value, dtype=double)
        input_r_elapsed_time = np.full(self.max_seq, fill_value=self.mask_value, dtype=double)
        input_label = np.full(self.max_seq, fill_value=self.mask_value, dtype=double)
        input_text_id = np.full(self.max_seq, fill_value=self.mask_value, dtype=double)
        input_skill = np.full(self.max_seq, fill_value=self.mask_value, dtype=double)

        input_id[1:] = question_id[:-1]
        input_text_id[1:] = text_id[:-1]
        input_r_elapsed_time[1:] = r_elapsed_time[:-1].copy().astype(np.double)
        input_skill[1:] = skill[:-1]
        input_label[1:] = label[:-1]

        target_id = np.full(self.max_seq, fill_value=self.mask_value, dtype=double)
        target_r_elapsed_time = np.full(self.max_seq, fill_value=self.mask_value, dtype=double)
        target_label = np.full(self.max_seq, fill_value=self.mask_value, dtype=double)
        target_text_id = np.full(self.max_seq, fill_value=self.mask_value, dtype=double)
        target_skill = np.full(self.max_seq, fill_value=self.mask_value, dtype=double)

        target_id[1:] = question_id[1:]
        target_text_id[1:] = text_id[1:]
        target_r_elapsed_time[1:] = r_elapsed_time[1:].copy().astype(np.int)
        target_skill[1:] = skill[1:]
        target_label[1:] = label[1:]

        if self.text_encoding_model:
            text_encoding, input_text_encoding, target_text_encoding = encode_correctness_in_encodings(
                self.text_encoding_model, text_id, self.max_seq, mask_value=self.mask_value)
        else:
            text_encoding, input_text_encoding, target_text_encoding = None, None, None

        possible_inputs = {"question_id": question_id, "text_id": text_id, "skill": skill, "label": label,
                           "r_elapsed_time": r_elapsed_time, "input_question_id": input_id,
                           "input_text_id": input_text_id, "input_skill": input_skill, "input_label": input_label,
                           "input_r_elapsed_time": input_r_elapsed_time, "target_id": target_id,
                           "target_text_id": target_text_id, "target_skill": target_skill,
                           "target_r_elapsed_time": target_r_elapsed_time, 'target_label': target_label,
                           "text_encoding": text_encoding, "input_text_encoding": input_text_encoding,
                           "target_text_encoding": target_text_encoding}
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
