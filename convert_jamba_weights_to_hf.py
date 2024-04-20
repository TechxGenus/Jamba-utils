import argparse
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, JambaConfig, JambaForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="Path to the input model")
parser.add_argument("--output_path", type=str, help="Path to save the output model")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
old_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True).to_dict()

del old_config["auto_map"]
del old_config["calc_logits_for_entire_prompt"]
del old_config["mamba_inner_layernorms"]
old_config["max_position_embeddings"] = old_config["n_ctx"]
del old_config["n_ctx"]
old_config["num_logits_to_keep"] = 1

old_model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)

def name_mapping(name):
    if "b_layernorm" in name:
        return name.replace("b_layernorm", "B_layernorm")
    elif "c_layernorm" in name:
        return name.replace("c_layernorm", "C_layernorm")
    elif "pre_ff_layernorm" in name:
        return name.replace("pre_ff_layernorm", "pre_moe_layernorm")
    elif "feed_forward.experts" in name or "feed_forward.router" in name:
        return name.replace("feed_forward", "moe")
    elif "feed_forward" in name:
        return name.replace("feed_forward", "moe.experts.0")
    return name

new_config = JambaConfig(**dict(old_config))
new_model = JambaForCausalLM(new_config)

param_dict = dict(old_model.named_parameters())

for name, param in new_model.named_parameters():
    param.data.copy_(param_dict[name_mapping(name)].data)

tokenizer.save_pretrained(args.output_path)
new_model.save_pretrained(args.output_path)
