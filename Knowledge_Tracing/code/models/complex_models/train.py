import config
from Knowledge_Tracing.code.models.complex_models.models.ltmti import LTMTI
from Knowledge_Tracing.code.models.complex_models.models.utmti import UTMTI
from Knowledge_Tracing.code.models.complex_models.models.saint_models.saint_on_skills import SAINT_on_skills
from Knowledge_Tracing.code.models.complex_models.models.saint_models.saint_on_questions import SAINT_on_questions
from Knowledge_Tracing.code.models.complex_models.models.ssakt import SSAKT
from Knowledge_Tracing.code.models.complex_models.dataset import get_dataloaders

import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


class AttentionBasedModel(pl.LightningModule):

    def __init__(self, model_args, model="saint_on_skills"):
        super().__init__()
        if model == "ltmti":
            self.model = LTMTI(**model_args)
        elif model == "utmti":
            self.model = UTMTI(**model_args)
        elif model == "ssakt":
            self.model = SSAKT(**model_args)
        elif model == "saint_on_skills":
            self.model = SAINT_on_skills(**model_args)
        elif model == "saint_on_questions":
            self.model = SAINT_on_questions(**model_args)
        if config.device == 'cuda':
            self.model = self.model.cuda()

    def forward(self, exercise, category, response, etime):
        return self.model(exercise, category, response, etime)

    def configure_optimizers(self, lr=1e-3):
        return torch.optim.Adam(self.parameters(), lr=lr)

    def training_step(self, batch, batch_idx):
        inputs, decoder_targets = batch
        target_ids = decoder_targets['target_ids']
        target = decoder_targets['target_label']
        if config.device == 'cuda':
            inputs = inputs.cuda()
            target_ids = target_ids.cuda()
            target = target.cuda()
        output = self(inputs, decoder_targets)
        target_mask = (target_ids != 0)
        output = torch.masked_select(output.squeeze(), target_mask)
        target = torch.masked_select(target, target_mask)
        loss = nn.BCEWithLogitsLoss()(output.float(), target.float())
        return {"loss": loss, "output": output, "target": target}

    def validation_step(self, batch, batch_idx):
        inputs, decoder_targets = batch
        target_ids = decoder_targets['target_ids']
        target = decoder_targets['target_label']
        if config.device == 'cuda':
            inputs = inputs.cuda()
            target_ids = target_ids.cuda()
            target = target.cuda()
        output = self(inputs, decoder_targets)
        target_mask = (target_ids != 0)
        output = torch.masked_select(output.squeeze(), target_mask)
        target = torch.masked_select(target, target_mask)
        loss = nn.BCEWithLogitsLoss()(output.float(), target.float())
        return {"val_loss": loss, "output": output, "target": target}


train_loader, val_loader, test_loader, config.NB_QUESTIONS, config.NB_SKILL = get_dataloaders(nrows=None)

ARGS = {"n_dims": config.EMBED_DIMS,
        'n_encoder': config.NUM_ENCODER,
        'n_decoder': config.NUM_DECODER,
        'enc_heads': config.ENC_HEADS,
        'dec_heads': config.DEC_HEADS,
        'total_ex': config.NB_QUESTIONS,
        'total_cat': config.NB_SKILL,
        'total_responses': config.NB_RESPONSES,
        'seq_len': config.MAX_SEQ}

########### TRAINING AND SAVING MODEL #######
checkpoint = ModelCheckpoint(filename="{epoch}_model",
                             verbose=True,
                             save_top_k=1,
                             monitor="val_loss")


saint_on_skills_model = AttentionBasedModel(model="saint_on_skills", model_args=ARGS)
if config.device == 'cuda':
    saint_on_skills_model.cuda()
trainer = pl.Trainer(progress_bar_refresh_rate=21,
                     max_epochs=1, callbacks=[checkpoint])
trainer.fit(model=saint_on_skills_model,
            train_dataloader=train_loader, val_dataloaders=val_loader)
trainer.save_checkpoint("model_saint_skills.pt")

saint_on_questions_model = AttentionBasedModel(model="saint_on_questions", model_args=ARGS)
if config.device == 'cuda':
    saint_on_questions_model.cuda()
trainer = pl.Trainer(progress_bar_refresh_rate=21,
                     max_epochs=1, callbacks=[checkpoint])
trainer.fit(model=saint_on_questions_model,
            train_dataloader=train_loader, val_dataloaders=val_loader)
trainer.save_checkpoint("model_saint_questions.pt")
