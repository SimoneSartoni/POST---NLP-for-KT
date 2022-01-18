from Knowledge_Tracing.code.models.complex_models.multihead_ffn import MultiHeadWithFFN
from Knowledge_Tracing.code.models.complex_models.utils import pos_encode, get_clones, ut_mask
from Knowledge_Tracing.code.models.complex_models import config

import torch
from torch import nn


class Dakt(nn.Module):
    def __init__(self, n_encoder, n_decoder, n_predictors, enc_heads, dec_heads, n_dims, nb_questions, nb_skills, nb_responses,

                 seq_len):
        super(Dakt, self).__init__()
        self.n_encoder = n_encoder
        self.n_decoder = n_decoder
        self.embedding = EmbeddingBlock(n_dims, nb_questions, nb_skills, nb_responses,  seq_len)
        self.content_encoder = get_clones(EncoderBlock(enc_heads, n_dims, nb_questions, nb_skills, seq_len), n_encoder)
        self.results_decoder = get_clones(DecoderBlock(dec_heads, n_dims, nb_responses, seq_len), n_decoder)
        self.prediction_decoder = get_clones(PredictionBlock(dec_heads, n_dims, nb_responses, seq_len), n_predictors)
        self.fc = nn.Linear(n_dims, 1)

    def forward(self, inputs, decoder_targets):
        encoder_inputs, decoder_inputs = inputs['encoder'], inputs['decoder']
        in_exercise, in_skill, in_response = encoder_inputs['input_question_id'], encoder_inputs['input_skill'], \
                                             decoder_inputs['input_label']
        in_elapsed_time = decoder_inputs['input_r_elapsed_time'].float().unsqueeze(-1)

        out_exercise, out_skill = decoder_targets['target_id'], decoder_targets['target_skill']

        in_content_encoder, in_results_decoder, in_prediction_decoder = self.embedding(in_exercise, in_skill,
                                                                                       in_response, in_elapsed_time,
                                                                                       out_exercise, out_skill)
        for n in range(self.n_encoder):
            in_content_encoder = self.content_encoder[n](in_content_encoder)
        out_encoder = in_content_encoder
        first_block = True
        for n in range(self.n_decoder):
            if n >= 1:
                first_block = False
            output_results_decoder = self.results_decoder[n](in_results_decoder, out_encoder, first_block)
            out_encoder = output_results_decoder
            in_results_decoder = output_results_decoder
        for n in range(self.n_encoder):
            if n == 1:
                out_prediction_decoder = self.prediction_decoder[n](in_prediction_decoder, output_results_decoder)
            else:
                out_prediction_decoder = self.prediction_decoder[n](out_prediction_decoder, out_prediction_decoder)
        out_linear_layer = self.fc(out_prediction_decoder)
        return torch.sigmoid(out_linear_layer)


class EmbeddingBlock(nn.Module):
    def __init__(self, n_dims, nb_questions, nb_skills, nb_responses,  seq_len):
        super(EmbeddingBlock, self).__init__()
        self.seq_len = seq_len
        self.exercise_embed = nn.Embedding(nb_questions, n_dims)
        self.skill_embed = nn.Embedding(nb_skills, n_dims)
        self.response_embed = nn.Embedding(nb_responses, n_dims)
        self.elapsed_time = nn.Linear(1, n_dims)
        self.position_embed = nn.Embedding(seq_len, n_dims)

    def forward(self, input_exercise, input_skill, input_r, in_elapsed_time, out_exercise, out_skill):
        position_encoded = pos_encode(self.seq_len).cuda()
        _pos = self.position_embed(position_encoded)
        _exe = self.exercise_embed(input_exercise)
        _skill = self.skill_embed(input_skill)
        _response = self.response_embed(input_r)
        _elapsed_time = self.elapsed_time(in_elapsed_time)
        _out_exe = self.exercise_embed(out_exercise)
        _out_skill = self.skill_embed(out_skill)
        input_encoder = _exe + _skill + _pos
        input_decoder = _response + _elapsed_time + _pos
        output = _out_exe + _out_skill
        return input_encoder, input_decoder, output


class EncoderBlock(nn.Module):
    def __init__(self, n_heads, n_dims, nb_questions, nb_skills, seq_len):
        super(EncoderBlock, self).__init__()
        self.seq_len = seq_len
        self.layer_norm = nn.LayerNorm(n_dims)
        self.multihead = MultiHeadWithFFN(n_heads=n_heads,
                                          n_dims=n_dims)

    def forward(self, input_embedding):
        output = self.multihead(q_input=input_embedding, kv_input=input_embedding)
        return output


class DecoderBlock(nn.Module):
    def __init__(self, n_heads, n_dims, nb_responses, seq_len):
        super(DecoderBlock, self).__init__()
        self.seq_len = seq_len
        self.layer_norm = nn.LayerNorm(n_dims)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=n_dims,
                                                         num_heads=n_heads,
                                                         dropout=0.2)
        self.multihead = MultiHeadWithFFN(n_heads=n_heads,
                                          n_dims=n_dims)

    def forward(self, input_embedding, encoder_output, first_block=True):
        input_embedding = input_embedding.permute(1, 0, 2)
        # assert out_embed.size(0)==n_dims, "input dimension should be (seq_len, batch_size, dims)"
        out_norm = self.layer_norm(input_embedding)
        mask = ut_mask(out_norm.size(0))
        if config.device == "cuda":
            mask = mask.cuda()
        out_attention, weights_attention = self.multihead_attention(query=out_norm,
                                                                    key=out_norm,
                                                                    value=out_norm,
                                                                    attn_mask=mask)
        out_attention += out_norm
        out_attention = out_attention.permute(1, 0, 2)
        output = self.multihead(q_input=out_attention, kv_input=encoder_output)
        return output


class PredictionBlock(nn.Module):
    def __init__(self, n_heads, n_dims, nb_responses, seq_len):
        super(PredictionBlock, self).__init__()
        self.seq_len = seq_len
        self.layer_norm = nn.LayerNorm(n_dims)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=n_dims,
                                                         num_heads=n_heads,
                                                         dropout=0.2)
        self.multihead = MultiHeadWithFFN(n_heads=n_heads,
                                          n_dims=n_dims)

    def forward(self, input_embedding, encoder_output):
        input_embedding = input_embedding.permute(1, 0, 2)
        # assert out_embed.size(0)==n_dims, "input dimension should be (seq_len, batch_size, dims)"
        out_norm = self.layer_norm(input_embedding)
        mask = ut_mask(out_norm.size(0))
        if config.device == "cuda":
            mask = mask.cuda()
        out_attention, weights_attention = self.multihead_attention(query=out_norm,
                                                                    key=out_norm,
                                                                    value=out_norm,
                                                                    attn_mask=mask)
        out_attention += out_norm
        out_attention = out_attention.permute(1, 0, 2)
        output = self.multihead(q_input=out_attention, kv_input=encoder_output)
        return output
