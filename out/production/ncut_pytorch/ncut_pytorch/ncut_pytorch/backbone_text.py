# %%
# 
from typing import List, Union
import torch
import os
from torch import nn
from typing import Optional, Tuple

from functools import partial

MODEL_DICT = {}
LAYER_DICT = {}

class Llama(nn.Module):
    def __init__(self, model_id="meta-llama/Meta-Llama-3.1-8B"):
        super().__init__()
        
        try: 
            import transformers
        except ImportError:
            raise ImportError("Please install transformers package: \n pip install transformers==4.44.2")
        
        access_token = os.getenv("HF_ACCESS_TOKEN")
        if access_token is None:
            raise ValueError("HF_ACCESS_TOKEN environment variable must be set")
        
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            token=access_token,
            device='cpu',
        )
        
        tokenizer = pipeline.tokenizer
        model = pipeline.model
        
        def new_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
            **kwargs,
        ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
            residual = hidden_states

            hidden_states = self.input_layernorm(hidden_states)

            # Self Attention
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            
            self.attn_output = hidden_states.clone()
            
            hidden_states = residual + hidden_states

            # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            
            self.mlp_output = hidden_states.clone()
            
            hidden_states = residual + hidden_states
            
            self.block_output = hidden_states.clone()

            outputs = (hidden_states,)

            if output_attentions:
                outputs += (self_attn_weights,)

            if use_cache:
                outputs += (present_key_value,)

            return outputs
        
        setattr(model.model.layers[0].__class__, "forward", new_forward)
        setattr(model.model.layers[0].__class__, "__call__", new_forward)
        
        self.model = model
        self.tokenizer = tokenizer
    
    @torch.no_grad()
    def forward(self, text: str):
        encoded_input = self.tokenizer(text, return_tensors='pt')
        device = next(self.model.parameters()).device
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        output = self.model(**encoded_input, output_hidden_states=True)
    
        attn_outputs, mlp_outputs, block_outputs = [], [], []
        for i, blk in enumerate(self.model.model.layers):
            attn_outputs.append(blk.attn_output)
            mlp_outputs.append(blk.mlp_output)
            block_outputs.append(blk.block_output)
        
        token_ids = encoded_input['input_ids']
        token_texts = [self.tokenizer.decode([token_id]) for token_id in token_ids[0]]
        
        return {"attn": attn_outputs, "mlp": mlp_outputs, "block": block_outputs, "token_texts": token_texts}

MODEL_DICT["meta-llama/Meta-Llama-3.1-8B"] = partial(Llama, model_id="meta-llama/Meta-Llama-3.1-8B")
LAYER_DICT["meta-llama/Meta-Llama-3.1-8B"] = 32
MODEL_DICT["meta-llama/Meta-Llama-3-8B"] = partial(Llama, model_id="meta-llama/Meta-Llama-3-8B")
LAYER_DICT["meta-llama/Meta-Llama-3-8B"] = 32

class GPT2(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            from transformers import GPT2Tokenizer, GPT2Model
        except ImportError:
            raise ImportError("Please install transformers package: pip install transformers")
        
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2Model.from_pretrained('gpt2')
        
        def new_forward(
            self,
            hidden_states: Optional[Tuple[torch.FloatTensor]],
            layer_past: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
        ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
            residual = hidden_states
            hidden_states = self.ln_1(hidden_states)
            attn_outputs = self.attn(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
            outputs = attn_outputs[1:]
            # residual connection
            self.attn_output = attn_output.clone()
            hidden_states = attn_output + residual

            if encoder_hidden_states is not None:
                # add one self-attention block for cross-attention
                if not hasattr(self, "crossattention"):
                    raise ValueError(
                        f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                        "cross-attention layers by setting `config.add_cross_attention=True`"
                    )
                residual = hidden_states
                hidden_states = self.ln_cross_attn(hidden_states)
                cross_attn_outputs = self.crossattention(
                    hidden_states,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    output_attentions=output_attentions,
                )
                attn_output = cross_attn_outputs[0]
                # residual connection
                hidden_states = residual + attn_output
                outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

            residual = hidden_states
            hidden_states = self.ln_2(hidden_states)
            feed_forward_hidden_states = self.mlp(hidden_states)
            # residual connection
            self.mlp_output = feed_forward_hidden_states.clone()
            hidden_states = residual + feed_forward_hidden_states

            if use_cache:
                outputs = (hidden_states,) + outputs
            else:
                outputs = (hidden_states,) + outputs[1:]

            self.block_output = hidden_states.clone()
            return outputs  # hidden_states, present, (attentions, cross_attentions)
        
        setattr(model.h[0].__class__, "forward", new_forward)
        
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def forward(self, text: str):
        encoded_input = self.tokenizer(text, return_tensors='pt')
        device = next(self.model.parameters()).device
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        output = self.model(**encoded_input, output_hidden_states=True)
    
        attn_outputs, mlp_outputs, block_outputs = [], [], []
        for i, blk in enumerate(self.model.h):
            attn_outputs.append(blk.attn_output)
            mlp_outputs.append(blk.mlp_output)
            block_outputs.append(blk.block_output)
            
        token_ids = encoded_input['input_ids']
        token_texts = [self.tokenizer.decode([token_id]) for token_id in token_ids[0]]
        
        return {"attn": attn_outputs, "mlp": mlp_outputs, "block": block_outputs, "token_texts": token_texts}
    
MODEL_DICT["gpt2"] = GPT2
LAYER_DICT["gpt2"] = 12

def get_demo_model_names():
    return list(MODEL_DICT.keys())

def download_all_models():
    for model_name in MODEL_DICT:
        print(f"Downloading {model_name}")
        try:
            model = MODEL_DICT[model_name]()
        except Exception as e:
            print(f"Error downloading {model_name}: {e}")
            continue
        
def load_text_model(model_name: str):
    return MODEL_DICT[model_name]()

if __name__ == '__main__':

    model = MODEL_DICT["meta-llama/Meta-Llama-3-8B"]()
    # model = MODEL_DICT["gpt2"]()
    text = """
    1. The majestic giraffe, with its towering height and distinctive long neck, roams the savannas of Africa. These gentle giants use their elongated tongues to pluck leaves from the tallest trees, making them well-adapted to their environment. Their unique coat patterns, much like human fingerprints, are unique to each individual.
    """
    model = model.cuda()
    output = model(text)
    print(output["block"][1].shape)
    print(output["token_texts"])
