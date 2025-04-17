import os
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from transformers import AutoTokenizer
import os
import torch
from collections import OrderedDict
from importlib_resources import files
import yaml
import argparse
import torchaudio
import torchaudio.transforms as T
import collections
import random
from .model.model import get_model_class
from tqdm import tqdm
import torch.nn.functional as F
from pathlib import Path
import math
from huggingface_hub.file_download import hf_hub_download

class MellowWrapper():
    """
    A class for interfacing mellow model
    """
    model_repo = "soham97/mellow"
    model_name = {
        'v0': 'v0.ckpt',
        'v0_s': 'v0_s.ckpt'
    }

    def __init__(self, config, model, device, use_cuda=True):
        # Check if version is supported
        self.supported_versions = self.model_name.keys()
        if model not in self.supported_versions:
            raise ValueError(f"The model {model} is not supported. The supported versions are {str(self.supported_versions)}")
        
        self.model_path = hf_hub_download(self.model_repo, self.model_name[model])
        counter_path = hf_hub_download(self.model_repo, "config.json")
        self.parent_path = Path(os.path.realpath(__file__)).parent
        self.config_path = os.path.join(self.parent_path, "config", config + ".yaml")
        self.use_cuda = use_cuda
        self.device = device

        self.model, self.tokenizer, self.args = self.get_model_and_tokenizer(config_path=self.config_path)
        self.model.eval()

    def read_config_as_args(self,config_path):
        return_dict = {}
        with open(config_path, "r") as f:
            yml_config = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in yml_config.items():
            return_dict[k] = v
        return argparse.Namespace(**return_dict)

    def get_model_and_tokenizer(self, config_path):
        r"""Load Adiff with args from config file"""
        args = self.read_config_as_args(config_path)
        args.model["decoder"]["prefix_dim"] = args.model["encoder"]["d_proj"]
        model_path = self.model_path.split(os.path.sep)[-1]
        config_path = self.config_path.split(os.path.sep)[-1]

        Model = get_model_class(model_type=args.model['model_type'])
        model = Model(
                audioenc_name = args.model['encoder']['audioenc_name'],
                d_in = args.model['encoder']['out_emb'],
                text_decoder = args.model['decoder']['text_decoder'],
                prefix_length = args.model['decoder']['prefix_length'],
                d_out = args.model['encoder']['d_proj'],
                )
        model_state_dict = torch.load(self.model_path, map_location=torch.device('cpu'))
        try:
            model.load_state_dict(model_state_dict)
        except:
            new_state_dict = OrderedDict()
            for k, v in model_state_dict.items():
                name = k[7:] # remove 'module.'
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

        tokenizer = AutoTokenizer.from_pretrained(args.model["decoder"]["text_decoder"])
        tokenizer.add_special_tokens({'pad_token': '!'})

        if self.use_cuda and torch.cuda.is_available():
            model = model.to(f"cuda:{self.device}")

        params = 0
        for p in model.parameters():
            params += math.prod(p.size())
        print(f"model {model_path}, {config_path}, parameter count: {params}")
        
        return model, tokenizer, args

    def default_collate(self, batch):
        r"""Puts each data field into a tensor with outer dimension batch size"""
        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
                # array of string classes and object
                if self.np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(
                        self.default_collate_err_msg_format.format(elem.dtype))

                return self.default_collate([torch.as_tensor(b) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, collections.abc.Mapping):
            return {key: self.default_collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(self.default_collate(samples) for samples in zip(*batch)))
        elif isinstance(elem, collections.abc.Sequence):
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError(
                    'each element in list of batch should be of equal size')
            transposed = zip(*batch)
            return [self.default_collate(samples) for samples in transposed]

        raise TypeError(self.default_collate_err_msg_format.format(elem_type))

    def load_audio_into_tensor(self, audio_path, audio_duration, resample=True):
        r"""Loads audio file and returns raw audio."""
        # Randomly sample a segment of audio_duration from the clip or pad to match duration
        audio_time_series, sample_rate = torchaudio.load(audio_path)
        resample_rate = self.args.data["sampling_rate"]
        if resample and resample_rate != sample_rate:
            resampler = T.Resample(sample_rate, resample_rate)
            audio_time_series = resampler(audio_time_series)
        audio_time_series = audio_time_series.reshape(-1)
        sample_rate = resample_rate

        # audio_time_series is shorter than predefined audio duration,
        # so audio_time_series is extended
        if audio_duration*sample_rate >= audio_time_series.shape[0]:
            repeat_factor = int(np.ceil((audio_duration*sample_rate) /
                                        audio_time_series.shape[0]))
            # Repeat audio_time_series by repeat_factor to match audio_duration
            audio_time_series = audio_time_series.repeat(repeat_factor)
            # remove excess part of audio_time_series
            audio_time_series = audio_time_series[0:audio_duration*sample_rate]
        else:
            # audio_time_series is longer than predefined audio duration,
            # so audio_time_series is trimmed
            start_index = random.randrange(
                audio_time_series.shape[0] - audio_duration*sample_rate)
            audio_time_series = audio_time_series[start_index:start_index +
                                                  audio_duration*sample_rate]
        return torch.FloatTensor(audio_time_series)

    def preprocess_audio(self, audio_files, resample):
        r"""Load list of audio files and return raw audio"""
        audio_tensors = []
        for audio_file in audio_files:
            audio_tensor = self.load_audio_into_tensor(
                audio_file, self.args.data["segment_seconds"], resample)
            audio_tensor = audio_tensor.reshape(
                1, -1).to(f"cuda:{self.device}") if self.use_cuda and torch.cuda.is_available() else audio_tensor.reshape(1, -1)
            audio_tensors.append(audio_tensor)
        return self.default_collate(audio_tensors)

    def preprocess_text(self, prompts):
        r"""Load list of prompts and return tokenized text"""
        tokenized_texts = []
        for ttext in prompts:
            ttext = ttext + ' <|endoftext|>' if 'gpt' in self.args.model["decoder"]["text_decoder"] else ttext
            tok = self.tokenizer.encode_plus(
                        text=ttext, add_special_tokens=True,\
                        truncation=True,
                        max_length=self.args.data["text_tokenization_len"], 
                        pad_to_max_length=True, return_tensors="pt")
                
            for key in tok.keys():
                tok[key] = tok[key].reshape(-1).to(f"cuda:{self.device}") if self.use_cuda and torch.cuda.is_available() else tok[key].reshape(-1)
            tokenized_texts.append(tok)
        return self.default_collate(tokenized_texts)

    def _generate_batch(
            self,
            embed=None,
            entry_length=300,  # maximum number of words
            top_p=0.8,
            temperature=1.,
            stop_token: str = '<|endoftext|>',
        ):
        self.model.eval()
        tokens = None
        generated_num = 0
        stop_token_index = self.tokenizer.encode(stop_token)[0]
        filter_value = -float("Inf")
        device = next(self.model.parameters()).device

        with torch.no_grad():
            if embed is not None:
                generated = embed

            for i in tqdm(range(entry_length)):
                outputs = self.model.caption_decoder.lm(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                for k in range(len(sorted_indices_to_remove)):
                    indices_to_remove = sorted_indices[k][sorted_indices_to_remove[k]]
                    logits[k, indices_to_remove] = filter_value

                next_token = torch.argmax(logits, -1).unsqueeze(1)

                if "gpt2" in self.model.caption_decoder.text_decoder:
                    next_token_embed = self.model.caption_decoder.lm.transformer.wte(next_token)
                elif "smollm2" in self.model.caption_decoder.text_decoder:
                    next_token_embed = self.model.caption_decoder.lm.model.embed_tokens(next_token)
                else:
                    raise ValueError(f"text decoder { self.model.caption_decoder.text_decoder} not supported")

                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)

                condition = (tokens == stop_token_index).sum(dim=-1)
                if (condition> 0).all():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            if output_list[0].ndim == 0:
                output_list = [output_list]
            generated_list = [self.tokenizer.decode(x).split("<|endoftext|>")[0] for x in output_list]

        return generated_list
    
    def generate(self, examples, max_len, top_p, temperature, stop_token='<|endoftext|>', audio_resample=True):
        r"""Produces text response for the given audio file and text prompts
        examples: (list<list>) List of examples. Each example is a list containing three entries [audio path 1, audio path 2, text prompt]
        max_len: (int) maximum length for text generation. Necessary to stop generation if LM gets "stuck" producing same token
        temperature: (float) top-p parameter for LM sampling
        temperature: (float) temperature parameter for LM sampling
        stop_token: (str) token used to stop text generation 
        audio_resample (bool) True for resampling audio. The model support only 32 kHz
        """
        preds = []
        audio_paths1 = []
        audio_paths2 = []
        text_prompts = []
        for example in examples:
            ap1, ap2, tp = example
            audio_paths1.append(ap1)
            audio_paths2.append(ap2)
            text_prompts.append(tp)
        
        audio1_embed = self.preprocess_audio(audio_paths1, resample=audio_resample).squeeze(1)
        audio2_embed = self.preprocess_audio(audio_paths2, resample=audio_resample).squeeze(1)
        text_embed = self.preprocess_text(text_prompts)
        d = {
                "audio1": audio1_embed,
                "audio2": audio2_embed,
                "input": text_embed,
            }
        prefix, _, _ = self.model.generate_prefix_inference(d)
        preds = self._generate_batch(embed=prefix, top_p=top_p, temperature=temperature, stop_token=stop_token, entry_length=max_len)
        return preds