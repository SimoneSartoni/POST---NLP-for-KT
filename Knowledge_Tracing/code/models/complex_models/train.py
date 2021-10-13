import config
from Knowledge_Tracing.code.models.complex_models.models.ltmti import LTMTI
from Knowledge_Tracing.code.models.complex_models.models.utmti import UTMTI
from Knowledge_Tracing.code.models.complex_models.models.saint import SAINT
from Knowledge_Tracing.code.models.complex_models.models.ssakt import SSAKT
from Knowledge_Tracing.code.models.complex_models.dataset import get_dataloaders

import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


class SAKTModel(pl.LightningModule):

    def __init__(self, model_args, model="saint"):
        super().__init__()
        if model == "ltmti":
            self.model = LTMTI(**model_args)
        elif model == "utmti":
            self.model = UTMTI(**model_args)
        elif model == "ssakt":
            self.model = SSAKT(**model_args)
        elif model == "saint":
            self.model = SAINT(**model_args)
        if config.device == 'cuda':
            self.model = self.model.cuda()

    def forward(self, exercise, category, response, etime):
        return self.model(exercise, category, response, etime)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        inputs, target_ids, target = batch
        if config.device == 'cuda':
            inputs = inputs.cuda()
            target_ids = target_ids.cuda()
            target = target.cuda()
        output = self(inputs["input_ids"], inputs["input_skills"], target_ids, inputs["input_rtime"])
        target_mask = (target_ids != 0)
        output = torch.masked_select(output.squeeze(), target_mask)
        target = torch.masked_select(target, target_mask)
        loss = nn.BCEWithLogitsLoss()(output.float(), target.float())
        return {"loss": loss, "output": output, "target": target}

    def validation_step(self, batch, batch_idx):
        inputs, target_ids, target = batch
        if config.device == 'cuda':
            inputs = inputs.cuda()
            target_ids = target_ids.cuda()
            target = target.cuda()
        output = self(inputs["input_ids"], inputs["input_skills"], target_ids, inputs["input_rtime"])
        target_mask = (target_ids != 0)
        output = torch.masked_select(output.squeeze(), target_mask)
        target = torch.masked_select(target, target_mask)
        loss = nn.BCEWithLogitsLoss()(output.float(), target.float())

        return {"val_loss": loss, "output": output, "target": target}


train_loader, val_loader = get_dataloaders()

ARGS = {"n_dims": config.EMBED_DIMS,
        'n_encoder': config.NUM_ENCODER,
        'n_decoder': config.NUM_DECODER,
        'enc_heads': config.ENC_HEADS,
        'dec_heads': config.DEC_HEADS,
        'total_ex': config.TOTAL_EXE,
        'total_cat': config.TOTAL_CAT,
        'total_responses': config.TOTAL_EXE,
        'seq_len': config.MAX_SEQ}

########### TRAINING AND SAVING MODEL #######
checkpoint = ModelCheckpoint(filename="{epoch}_model",
                             verbose=True,
                             save_top_k=1,
                             monitor="val_loss")


sakt_model = SAKTModel(model="ltmti", model_args=ARGS)
if config.device == 'cuda':
    sakt_model.cuda()
trainer = pl.Trainer(progress_bar_refresh_rate=21,
                     max_epochs=1, callbacks=[checkpoint])
trainer.fit(model=sakt_model,
            train_dataloader=train_loader, val_dataloaders=val_loader)
trainer.save_checkpoint("model_sakt.pt")
