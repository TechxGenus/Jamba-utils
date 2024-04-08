# Jamba-utils
Utils about Jamba model.

Make a dense model based on Jamba-v0.1 weights without using MoE:
```sh
python dense_downcycling.py --model_path ai21labs/Jamba-v0.1 --output_path test --expert_ids 0
```
