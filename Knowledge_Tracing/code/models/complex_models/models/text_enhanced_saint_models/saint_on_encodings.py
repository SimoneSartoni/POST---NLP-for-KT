from Knowledge_Tracing.code.models.complex_models.multihead_ffn import MultiHeadWithFFN
from Knowledge_Tracing.code.models.complex_models.utils import pos_encode, get_clones, ut_mask
from Knowledge_Tracing.code.models.complex_models import config

import torch
from torch import nn


class saint_on_encodings(nn.Module):
    def __init__(self, n_encoder, n_decoder, enc_heads, dec_heads, n_dims, nb_questions, encoding_size, nb_skills,
                 nb_responses, seq_len):
        super(saint_on_encodings, self).__init__()
        self.n_encoder = n_encoder
        self.n_decoder = n_decoder
        self.encoder = get_clones(EncoderBlock(enc_heads, n_dims, nb_questions, encoding_size, nb_skills, seq_len),
                                  n_encoder)
        self.decoder = get_clones(DecoderBlock(dec_heads, n_dims, nb_responses, encoding_size, seq_len), n_decoder)
        self.fc = nn.Linear(encoding_size, 1)

    def forward(self, inputs, decoder_targets):
        first_block = True
        encoder_inputs, decoder_inputs = inputs['encoder'], inputs['decoder']
        in_exercise, in_skill, in_response, text_encoding = encoder_inputs['question_id'], encoder_inputs['skill'], \
                                                            decoder_inputs['label'], encoder_inputs['text_encoding']
        for n in range(self.n_encoder):
            if n >= 1:
                first_block = False

            enc = self.encoder[n](in_exercise, text_encoding, in_skill, first_block=first_block)
            in_exercise = enc
            text_encoding = enc
            in_skill = enc

        first_block = True
        for n in range(self.n_decoder):
            if n >= 1:
                first_block = False
            dec = self.decoder[n](in_response, encoder_output=in_exercise, first_block=first_block)
            in_exercise = dec
            in_response = dec

        return torch.sigmoid(self.fc(dec))


class EncoderBlock(nn.Module):
    def __init__(self, n_heads, n_dims, nb_questions, encoding_size, nb_skills, seq_len):
        super(EncoderBlock, self).__init__()
        self.seq_len = seq_len
        self.total_dim = encoding_size
        self.position_embed = nn.Embedding(seq_len, encoding_size)
        self.layer_norm = nn.LayerNorm(self.total_dim)

        self.multihead = MultiHeadWithFFN(n_heads=n_heads, n_dims=self.total_dim)

    def forward(self, input_exercise, input_encoding, input_skill, first_block=True):
        if first_block:
            # _exe = self.exercise_embed(input_exercise)
            # _skill = self.skill_embed(input_skill)
            position_encoded = pos_encode(self.seq_len).cuda()
            _pos = self.position_embed(position_encoded)
            # out_skill = _skill + _pos
            # out_ex = _exe + _pos
            out = input_encoding + _pos
        else:
            out = input_exercise
        output = self.multihead(q_input=out, kv_input=out)
        return output


class DecoderBlock(nn.Module):
    def __init__(self, n_heads, n_dims, nb_responses, encoding_size, seq_len):
        super(DecoderBlock, self).__init__()
        self.seq_len = seq_len
        self.total_dim = encoding_size
        self.response_embed = nn.Embedding(nb_responses, encoding_size)
        self.position_embed = nn.Embedding(seq_len, encoding_size)
        self.layer_norm = nn.LayerNorm(self.total_dim)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=self.total_dim,
                                                         num_heads=n_heads,
                                                         dropout=0.2)
        self.multihead = MultiHeadWithFFN(n_heads=n_heads,
                                          n_dims=self.total_dim)

    def forward(self, input_r, encoder_output, first_block=True):
        if first_block:
            _response = self.response_embed(input_r)
            position_encoded = pos_encode(self.seq_len)
            _pos = self.position_embed(position_encoded.cuda())
            out = _response + _pos
        else:
            out = input_r
        out = out.permute(1, 0, 2)
        # assert out_embed.size(0)==n_dims, "input dimension should be (seq_len, batch_size, dims)"
        out_norm = self.layer_norm(out)
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
