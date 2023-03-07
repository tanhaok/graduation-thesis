"""
    - Summary: This module used to encode text use T5 model
    - T5 is an encoder-decoder model and converts all NLP problems into text to text format
"""

import torch
import transformers
from transformers import T5Tokenizer, T5EncoderModel, T5Config

transformers.logging.set_verbosity_error()

# config
MAX_LENGTH = 256
DEFAULT_T5_NAME = 'google/t5-v1_1-base'
T5_CONFIGS = {}


def get_model_and_tokenizer(name):
    """Get model and tokenizer

    Args:
        name (_type_): _description_

    Returns:
        T5EncoderModel : A model
        T5Tokenizer    : A token
    """
    global T5_CONFIGS

    if name not in T5_CONFIGS:
        T5_CONFIGS[name] = {}

    if "model" not in T5_CONFIGS[name]:
        T5_CONFIGS[name]["model"] = T5EncoderModel.from_pretrained(name)

    if "tokenizer" not in T5_CONFIGS[name]:
        T5_CONFIGS[name]["tokenizer"] = T5Tokenizer.from_pretrained(name)

    return T5_CONFIGS[name]['model'], T5_CONFIGS[name]['tokenizer']


def get_encoded_dim(name):
    """Func used to get dimensions of model

    Args:
        name (str): key

    Raises:
        ValueError: when key not in config

    Returns:
        Dimensions of model
    """

    if name not in T5_CONFIGS:
        config = T5Config.from_pretrained(name)
        T5_CONFIGS[name] = {config:config}
    elif "config" in T5_CONFIGS[name]:
        config = T5_CONFIGS[name]['config']
    elif 'model' in T5_CONFIGS[name]:
        config = T5_CONFIGS[name]['model'].config
    else:
        raise ValueError(f'Invalid T5 name {name}')

    return config


# Encode text with T5


def t5_encoder(texts, name=DEFAULT_T5_NAME, output_device=None):
    """T5 encoder for text

    Args:
        texts (_type_): text to encode
        name (_type_, optional): model name. Defaults to DEFAULT_T5_NAME.
        output_device (_type_, optional): cpu | gpu. Defaults to None.

    Returns:
        tensor: text encoded
    """
    t5_model, tokenizer = get_model_and_tokenizer(name)

    # get device -> gpu = cuda,
    if torch.cuda.is_available():
        t5_model = t5_model.cuda()

    device = next(t5_model.parameters()).device

    encoded = tokenizer.batch_encode_plus(texts,
                                          return_tensors='pt',
                                          padding='longest',
                                          max_length=MAX_LENGTH,
                                          truncation=True)

    input_ids = encoded.input_ids.to(device)
    attn_mask = encoded.attention_mask.to(device)

    t5_model.eval()

    with torch.no_grad():
        output = t5_model(input_ids=input_ids, attention_mask=attn_mask)
        encoded_text = output.last_hidden_state.detach()

    attn_mask = attn_mask[..., None].bool()

    if output_device is None:
        encoded_text = encoded_text.masked_fill(~attn_mask, 0.)
        return encoded_text

    encoded_text = encoded_text.to(output_device)
    attn_mask = attn_mask.to(output_device)

    encoded_text = encoded_text.masked_fill(~attn_mask, 0.)
    return encoded_text
