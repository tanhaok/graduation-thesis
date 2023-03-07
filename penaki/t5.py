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
        T5_CONFIGS[name] = dict()
    
    if "model" not in T5_CONFIGS[name]:
        T5_CONFIGS[name]["model"] = T5EncoderModel.from_pretrained(name)
    
    if "tokenizer" not in T5_CONFIGS[name]:
        T5_CONFIGS[name]["tokenizer"] = T5Tokenizer.from_pretrained(name);
        
    return T5_CONFIGS[name]['model'], T5_CONFIGS[name]['tokenizer']

def get_encoded_dim (name):
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
        T5_CONFIGS[name] = dict(config = config)
    elif "config" in T5_CONFIGS[name]:
        config = T5_CONFIGS[name]['config']
    elif 'model' in T5_CONFIGS[name]:
        config = T5_CONFIGS[name]['model'].config
    else:
        raise ValueError(f'Invalid T5 name {name}')

    return config

# Encode text with T5

def t5_encoder (texts, name = DEFAULT_T5_NAME, output_device = None):
    t5, tokenizer = get_model_and_tokenizer(name)
   
    # get device -> gpu = cuda, 
    if torch.cuda.is_available():
        t5 = t5.cuda()
    
    device = next(t5.parameters()).device
    
    encoded = tokenizer.batch_encode_plus(texts,return_tensors = 'pt', padding = 'longest', max_length = MAX_LENGTH, truncation = True)

    input_ids = encoded.input_ids.to(device)
    attn_mask = encoded.attention_mask.to(device)

    t5.eval()

    with torch.no_grad():
        output = t5(input_ids = input_ids, attention_mask = attn_mask)
        encoded_text = output.last_hidden_state.detach()

    attn_mask = attn_mask[..., None].bool()

    if output_device is None:
        encoded_text = encoded_text.masked_fill(~attn_mask, 0.)
        return encoded_text

    encoded_text = encoded_text.to(output_device)
    attn_mask = attn_mask.to(output_device)

    encoded_text = encoded_text.masked_fill(~attn_mask, 0.)
    return encoded_text 
    