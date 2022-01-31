import gc
import numpy as np
import torch.nn.functional as F
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertModel, TFDistilBertModel
import torch
from torch.utils.data import Dataset, DataLoader
import math
import datasets


def identity_tokenizer(text):
    return text


# define mean pooling function
def mean_pool(token_embeds, attention_mask, dim=1):
    # reshape attention_mask to cover 768-dimension embeddings
    in_mask = attention_mask.unsqueeze(-1).expand(
        token_embeds.size()
    ).float()
    # perform mean-pooling but exclude padding tokens (specified by in_mask)
    pool = torch.sum(token_embeds * in_mask, dim) / torch.clamp(in_mask.sum(dim), min=1e-9)
    return pool


def mean_pool_np(token_embeddings, attention_mask):
    input_mask_expanded = np.broadcast_to(np.expand_dims(attention_mask, -1), token_embeddings.shape)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, 0)
    sum_mask = np.clip(np.sum(input_mask_expanded, 0), 1e-9, 1000)
    return sum_embeddings / sum_mask


class SentenceSimilarityDataset(Dataset):
    def __init__(self, texts_df, text_column, tokenizer, batch_size=16):
        self.texts_df = texts_df
        self.texts_df_2 = texts_df.sample(frac=1)
        print(self.texts_df)
        print(self.texts_df_2)
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.batch_size = batch_size

    def __len__(self):
        return len(self.texts_df)//self.batch_size

    def __getitem__(self, idx):
        start = idx * self.batch_size
        texts = list(self.texts_df[self.text_column].values)[start: start+self.batch_size]
        texts_2 = list(self.texts_df_2[self.text_column].values)[start: start+self.batch_size]
        batch_encoding = self.tokenizer(texts, max_length=128, padding='max_length', truncation=True)
        anchor_ids, anchor_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
        batch_encoding = self.tokenizer(texts_2, max_length=128, padding='max_length', truncation=True)
        positive_ids, positive_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
        anchor_ids = torch.from_numpy(np.array(anchor_ids))
        anchor_mask = torch.from_numpy(np.array(anchor_mask))
        positive_ids = torch.from_numpy(np.array(positive_ids))
        positive_mask = torch.from_numpy(np.array(positive_mask))
        list_a = list(self.texts_df['list_of_words'].values)[start: start+self.batch_size]
        list_b = list(self.texts_df_2['list_of_words'].values)[start: start+self.batch_size]

        def counter_cosine_similarity(c1, c2):
            common = set(c1).intersection(set(c2))
            dot_product = len(common)
            mag_a = math.sqrt(float(len(set(c1))))
            mag_b = math.sqrt(float(len(set(c2))))
            if mag_a * mag_b != 0:
                return float(dot_product) / float(mag_a * mag_b)
            else:
                return 0.0
        cos_sim = [[counter_cosine_similarity(a, b) for a in list_a] for b in list_b]

        inputs = {"anchor_ids": anchor_ids, "anchor_mask": anchor_mask,
                  "positive_ids": positive_ids, "positive_mask": positive_mask,
                  "cos_sim": cos_sim}
        return inputs


class PretrainedDistilBERT():
    def __init__(self, config_path="/content/drive/MyDrive/simone sartoni - text enhanced deep knowledge tracing/"
                                   "pretrained_distilbert_base_uncased_24_epochs/config.json",
                 model_filepath="/content/drive/MyDrive/simone sartoni - text enhanced deep knowledge tracing/"
                                "pretrained_distilbert_base_uncased_24_epochs/tf_model.h5",
                 ):
        self.name = "pretrained_distilbert"
        self.method = "nlp_model"
        self.config = DistilBertConfig.from_json_file(config_path)
        self.model = DistilBertModel.from_pretrained(model_filepath, config=self.config, from_tf=True)
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.encodings = {}
        self.pro_num = None
        self.words_num = None
        self.texts_df = None
        self.vector_size = 0

    def fit_on_custom(self, texts_df, save_filepath='/content/', text_column="sentence",
                      batch_size=64):
        self.texts_df = texts_df
        print(texts_df)
        dataset = SentenceSimilarityDataset(texts_df=texts_df, text_column=text_column, tokenizer=self.tokenizer,
                                            batch_size=batch_size)
        batch_size = 1

        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        # set device and move model there
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        print(f'moved to {device}')
        # define layers to be used in multiple-negatives-ranking
        cos_sim = torch.nn.CosineSimilarity()
        loss_func = torch.nn.CrossEntropyLoss()
        scale = 20.0  # we multiply similarity score by this scale value
        # move layers to device
        cos_sim.to(device)
        loss_func.to(device)
        from transformers.optimization import get_linear_schedule_with_warmup
        # initialize Adam optimizer
        optim = torch.optim.Adam(self.model.parameters(), lr=2e-5)
        # setup warmup for first ~10% of steps
        total_steps = int(len(dataset) / batch_size)
        warmup_steps = int(0.1 * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optim, num_warmup_steps=warmup_steps,
            num_training_steps=total_steps - warmup_steps
        )
        from tqdm.auto import tqdm
        epochs = 1
        # 1 epoch should be enough, increase if wanted
        for epoch in range(epochs):
            self.model.train()  # make sure model is in training mode
            # initialize the dataloader loop with tqdm (tqdm == progress bar)
            loop = tqdm(loader, leave=True)
            for batch in loop:
                # zero all gradients on each new step
                optim.zero_grad()
                # prepare batches and more all to the active device
                anchor_ids = torch.squeeze(batch['anchor_ids'], axis=0).to(device)
                anchor_mask = torch.squeeze(batch['anchor_mask'], axis=0).to(device)
                pos_ids = torch.squeeze(batch['positive_ids'], axis=0).to(device)
                pos_mask = torch.squeeze(batch['positive_mask'], axis=0).to(device)
                cos_sim_list = batch['cos_sim']
                a = self.model(input_ids=anchor_ids, attention_mask=anchor_mask, output_attentions=False)[0]
                p = self.model(input_ids=pos_ids, attention_mask=pos_mask, output_attentions=False)[0]
                # get the mean pooled vectors
                a = mean_pool(a, anchor_mask)
                p = mean_pool(p, pos_mask)
                # calculate the cosine similarities

                scores = torch.stack([cos_sim(a_i.reshape(1, a_i.shape[0]), p) for a_i in a])
                # get label(s) - we could define this before if confident of consistent batch sizes
                labels = torch.stack([torch.Tensor(sim_list) for sim_list in cos_sim_list]).cuda()
                # and now calculate the loss
                loss = loss_func(scores * scale, labels)
                # using loss, calculate gradients and then optimize
                loss.backward()
                optim.step()
                # update learning rate scheduler
                scheduler.step()
                # update the TDQM progress bar
                loop.set_description(f'Epoch {epoch}')
                loop.set_postfix(loss=loss.item())

        import os
        model_path = save_filepath + '/sbert_trained_on_CA/'
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        self.model.save_pretrained(model_path)

    def fit_on_nli(self, save_filepath='/content/', ):
        snli = datasets.load_dataset('snli', split='train')
        mnli = datasets.load_dataset('glue', 'mnli', split='train')
        mnli = mnli.remove_columns("idx")
        print(mnli)
        print(snli)
        snli = snli.cast(mnli.features)
        dataset = datasets.concatenate_datasets([snli, mnli])
        del snli, mnli

        print(f"before: {len(dataset)} rows")
        dataset = dataset.filter(
            lambda x: True if x['label'] == 0 else False
        )
        print(f"after: {len(dataset)} rows")

        dataset = dataset.map(
            lambda x: self.tokenizer(
                x['premise'], max_length=128, padding='max_length',
                truncation=True
            ), batched=True
        )
        dataset = dataset.rename_column('input_ids', 'anchor_ids')
        dataset = dataset.rename_column('attention_mask', 'anchor_mask')
        dataset = dataset.map(
            lambda x: self.tokenizer(
                x['hypothesis'], max_length=128, padding='max_length',
                truncation=True
            ), batched=True
        )
        dataset = dataset.rename_column('input_ids', 'positive_ids')
        dataset = dataset.rename_column('attention_mask', 'positive_mask')
        dataset = dataset.remove_columns(['premise', 'hypothesis', 'label'])
        dataset.set_format(type='torch', output_all_columns=True)
        batch_size = 32

        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        # set device and move model there
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        print(f'moved to {device}')
        # define layers to be used in multiple-negatives-ranking
        cos_sim = torch.nn.CosineSimilarity()
        loss_func = torch.nn.CrossEntropyLoss()
        scale = 20.0  # we multiply similarity score by this scale value
        # move layers to device
        cos_sim.to(device)
        loss_func.to(device)
        from transformers.optimization import get_linear_schedule_with_warmup
        # initialize Adam optimizer
        optim = torch.optim.Adam(self.model.parameters(), lr=2e-5)
        # setup warmup for first ~10% of steps
        total_steps = int(len(dataset) / batch_size)
        warmup_steps = int(0.1 * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optim, num_warmup_steps=warmup_steps,
            num_training_steps=total_steps - warmup_steps
        )
        from tqdm.auto import tqdm
        epochs = 1
        # 1 epoch should be enough, increase if wanted
        for epoch in range(epochs):
            self.model.train()  # make sure model is in training mode
            # initialize the dataloader loop with tqdm (tqdm == progress bar)
            loop = tqdm(loader, leave=True)
            for batch in loop:
                # zero all gradients on each new step
                optim.zero_grad()
                # prepare batches and more all to the active device
                anchor_ids = batch['anchor_ids'].to(device)
                anchor_mask = batch['anchor_mask'].to(device)
                pos_ids = batch['positive_ids'].to(device)
                pos_mask = batch['positive_mask'].to(device)
                # extract token embeddings from BERT
                a = self.model(
                    anchor_ids, attention_mask=anchor_mask
                )[0]  # all token embeddings
                p = self.model(pos_ids, attention_mask=pos_mask)[0]
                # get the mean pooled vectors
                a = mean_pool(a, anchor_mask)
                p = mean_pool(p, pos_mask)
                # calculate the cosine similarities
                scores = torch.stack([
                    cos_sim(a_i.reshape(1, a_i.shape[0]), p) for a_i in a])
                # get label(s) - we could define this before if confident of consistent batch sizes
                labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)
                # and now calculate the loss
                loss = loss_func(scores * scale, labels)
                # using loss, calculate gradients and then optimize
                loss.backward()
                optim.step()
                # update learning rate scheduler
                scheduler.step()
                # update the TDQM progress bar
                loop.set_description(f'Epoch {epoch}')
                loop.set_postfix(loss=loss.item())

        import os
        model_path = save_filepath + 'pretrained_sbert_mnr'
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        self.model.save_pretrained(model_path)

    def transform(self, texts_df, text_column="sentence", save_filepath='./'):
        self.texts_df = texts_df
        start = 0
        batch_size = 100
        while start < len(texts_df.index):
            end = start + batch_size
            inputs = self.tokenizer(list(self.texts_df[text_column].values)[start:end], truncation=True,
                                    return_tensors="pt",
                                    padding=True)
            ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
            output = self.model(input_ids=ids, attention_mask=attention_mask, output_attentions=False)
            encoding = output.to_tuple()[0]
            for problem_id, enc, attention in list(zip(self.texts_df['problem_id'].values[start:end],
                                                       encoding, attention_mask)):
                self.encodings[problem_id] = F.normalize(mean_pool(enc, attention, dim=0), p=2, dim=0).detach().cpu().numpy()
            start = start + batch_size
            del [ids, attention_mask, output, encoding]
            gc.collect()
            print(end)
        print(len(list(self.encodings.keys())))
        print("pretrainedBERT model created")

        self.words_num = list(self.encodings.values())[0].shape[0]
        print("vector_size: " + str(self.words_num))
        self.vector_size = self.words_num
        self.pro_num = len(self.texts_df.index)

    def get_encoding(self, problem_id):
        encodings = np.array(self.encodings[problem_id])
        return encodings
