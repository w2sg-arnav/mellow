
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from transformers import AutoModelForCausalLM
from typing import Optional

def get_decoder(name: str):
    if name == "Decoder":
        return DecoderModel
    else:
        raise Exception('The decoder model {} is incorrect or not supported'.format(name))
    
def downsample(x):
    clip_latent = x[:,0,:].unsqueeze(1)
    pooled = nnf.avg_pool2d(x[:,1:,:], kernel_size=(8,1))
    x = torch.concat((clip_latent,pooled),axis=1)
    return x

class DecoderModel(nn.Module):
    def __init__(self, text_decoder: str, prefix_length: int,):
        super(DecoderModel, self).__init__()
        self.prefix_length = prefix_length
        self.text_decoder = text_decoder.lower()
        self.lm = AutoModelForCausalLM.from_pretrained(text_decoder)
        if "gpt2" in self.text_decoder:
            self.lm_embedding_size = self.lm.transformer.wte.weight.shape[1]
        elif "smollm2" in self.text_decoder:
            self.lm_embedding_size = self.lm.model.embed_tokens.weight.shape[1]
        else:
            raise ValueError(f"text decoder {self.text_decoder} not supported")

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)
    
    def generate_prefix_inference(self, daudio1, daudio2, texts_enc):
        audio_projections1 = downsample(daudio1).contiguous()
        audio_projections2 = downsample(daudio2).contiguous()

        # separate token between two audios'
        if "gpt" in self.text_decoder:
            dtext = self.lm.transformer.wte(texts_enc['input_ids'])
            dtext = dtext.contiguous()
            sep_token = torch.tensor([50256]).to(dtext.device)
            sep_embed = self.lm.transformer.wte(sep_token).unsqueeze(0).repeat(dtext.shape[0],1,1)
        elif "smollm2" in self.text_decoder:
            dtext = self.lm.model.embed_tokens(texts_enc['input_ids'])
            dtext = dtext.contiguous()
            sep_token = torch.tensor([0]).to(dtext.device)
            sep_embed = self.lm.model.embed_tokens(sep_token).unsqueeze(0).repeat(dtext.shape[0],1,1)
        else:
            raise ValueError(f"text decoder {self.text_decoder} not supported")
        
        prefix = torch.cat((audio_projections1, sep_embed, audio_projections2, sep_embed, dtext), dim=1)
        return prefix

    def forward(self, daudio1: torch.Tensor, daudio2: torch.Tensor, texts_enc: torch.Tensor, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):

        if "gpt2" in self.text_decoder:
            # input prompt
            dtext = self.lm.transformer.wte(texts_enc['input_ids'])
            dtext = dtext.contiguous()
            # output labels
            embedding_text = self.lm.transformer.wte(tokens['input_ids'])
            # separator
            sep_token = torch.tensor([50256]).to(dtext.device)
            sep_embed = self.lm.transformer.wte(sep_token).unsqueeze(0).repeat(dtext.shape[0],1,1)
        elif "smollm2" in self.text_decoder:
            # input prompt
            dtext = self.lm.model.embed_tokens(texts_enc['input_ids'])
            dtext = dtext.contiguous()
            # output labels
            embedding_text = self.lm.model.embed_tokens(tokens['input_ids'])
            # separator
            sep_token = torch.tensor([0]).to(dtext.device)
            sep_embed = self.lm.model.embed_tokens(sep_token).unsqueeze(0).repeat(dtext.shape[0],1,1)
        else:
            raise ValueError(f"text decoder {self.text_decoder} not supported")
        
        audio_projections1 = downsample(daudio1).contiguous()
        audio_projections2 = downsample(daudio2).contiguous()
        
        prefix = torch.cat((audio_projections1, sep_embed, audio_projections2, sep_embed, dtext), dim=1)
        embedding_cat = torch.cat((prefix, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens['input_ids'].shape[0], tokens['input_ids'].device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.lm(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out