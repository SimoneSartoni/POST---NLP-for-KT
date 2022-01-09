from Knowledge_Tracing.code.models.complex_models.multihead_ffn import MultiHeadWithFFN
from Knowledge_Tracing.code.models.complex_models.utils import pos_encode, get_clones, ut_mask
from Knowledge_Tracing.code.models.complex_models import config

import torch
from torch import nn


class SaintEncodings(nn.Module):
    def __init__(self, n_encoder, n_decoder, enc_heads, dec_heads, n_dims, nb_questions, nb_skills, nb_responses,
                 seq_len):
        super(SaintEncodings, self).__init__()
        self.n_encoder = n_encoder
        self.n_decoder = n_decoder
        self.embedding = EmbeddingBlock(n_dims, nb_questions, nb_skills, nb_responses, seq_len)
        self.encoder = get_clones(EncoderBlock(enc_heads, n_dims, nb_questions, nb_skills, seq_len), n_encoder)
        self.decoder = get_clones(DecoderBlock(dec_heads, n_dims, nb_responses, seq_len), n_decoder)

        self.multiply = MultiplyBlock()
        self.fc = nn.Linear(n_dims, 1)

    def forward(self, inputs, decoder_targets):
        first_block = True
        encoder_inputs, decoder_inputs = inputs['encoder'], inputs['decoder']
        input_nlp_embedding, in_exercise, in_skill, in_response = encoder_inputs['input_text_encoding'], \
                                                                  encoder_inputs['input_question_id'], \
                                                                  encoder_inputs['input_skill'], \
                                                                  decoder_inputs['input_label']
        in_elapsed_time = decoder_inputs['input_r_elapsed_time'].float().unsqueeze(-1)

        out_exercise, out_skill, output_nlp_embedding = decoder_targets['target_id'], decoder_targets['target_skill'],\
            decoder_targets['target_text_encoding']

        in_encoder, in_decoder, target_output = self.embedding(input_nlp_embedding, in_response,
                                                               in_elapsed_time, output_nlp_embedding)
        for n in range(self.n_encoder):
            in_encoder = self.encoder[n](in_encoder)
        out_encoder = in_encoder
        first_block = True
        for n in range(self.n_decoder):
            if n >= 1:
                first_block = False
            output_decoder = self.decoder[n](in_decoder, out_encoder, first_block)
            out_encoder = output_decoder
            in_decoder = output_decoder
        multiply_output = self.multiply(output_decoder, target_output)
        return torch.sigmoid(self.fc(multiply_output))


class EmbeddingBlock(nn.Module):
    def __init__(self, n_dims, nb_questions, nb_skills, nb_responses,  seq_len):
        super(EmbeddingBlock, self).__init__()
        self.seq_len = seq_len
        self.response_embed = nn.Embedding(nb_responses, n_dims)
        self.elapsed_time = nn.Linear(1, n_dims)
        self.position_embed = nn.Embedding(seq_len, n_dims)

    def forward(self, input_nlp_embedding, input_r, in_elapsed_time, output_nlp_embedding):
        position_encoded = pos_encode(self.seq_len).cuda()
        _pos = self.position_embed(position_encoded)
        _response = self.response_embed(input_r)
        _elapsed_time = self.elapsed_time(in_elapsed_time)
        input_encoder = input_nlp_embedding + _pos
        input_decoder = _response + _elapsed_time + _pos
        output = output_nlp_embedding
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


class MultiplyBlock(nn.Module):
    def __init__(self):
        super(MultiplyBlock, self).__init__()

    def forward(self, input_embedding, target_embedding):
        output = input_embedding * target_embedding
        return output
