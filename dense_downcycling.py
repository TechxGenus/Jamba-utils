import shutil
import argparse
import torch

from Jamba.configuration_jamba import JambaConfig
from Jamba.modeling_jamba import JambaForCausalLM
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="Path to the input model")
parser.add_argument("--output_path", type=str, help="Path to save the output model")
parser.add_argument("--expert_ids", type=int, nargs='+', help="Expert IDs to use")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

sparse_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True).to_dict()

# Modify the config to set dense model
sparse_config["num_experts"] = 1
sparse_config["num_experts_per_tok"] = 1

def name_mapping(name):
    """
    Maps the given name to a list of names based on certain conditions.

    Args:
        name (str): The name to be mapped.

    Returns:
        tuple: A tuple containing a list of mapped names and a boolean value indicating whether the name contains "down_proj" or not.
    """
    if "experts" in name and (int(name.split(".")[2]) - sparse_config["expert_layer_offset"]) % sparse_config["expert_layer_period"] == 0:
        if "down_proj" in name:
            return [name.replace("experts.0", f"experts.{i}") for i in args.expert_ids], True
        return [name.replace("experts.0", f"experts.{i}") for i in args.expert_ids], False
    return [name], False

sparse_model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)

dense_config = JambaConfig(**dict(sparse_config))
dense_model = JambaForCausalLM(dense_config)

param_dict = dict(sparse_model.named_parameters())

for name, param in dense_model.named_parameters():
    names, scale = name_mapping(name)
    if scale:
        # Scale down_proj for numerical stability
        param.data.copy_(torch.stack([param_dict[name].data for name in names]).mean(dim=0) * 0.045)
    else:
        param.data.copy_(torch.stack([param_dict[name].data for name in names]).mean(dim=0))

tokenizer.save_pretrained(args.output_path)
dense_model.save_pretrained(args.output_path)
shutil.copy("Jamba/configuration_jamba.py", args.output_path)
shutil.copy("Jamba/modeling_jamba.py", args.output_path)
