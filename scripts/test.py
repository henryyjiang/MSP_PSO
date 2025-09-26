import torch
from collections import OrderedDict
from mattersim.forcefield.m3gnet import m3gnet

state_dict = torch.load("MatterSim-v1.0.0-5M.pth", map_location="cpu")
model = m3gnet.M3Gnet()
ckpt = torch.load("MatterSim-v1.0.0-5M.pth", map_location="cpu")
print("checkpoint top-level keys:", list(ckpt.keys()))

# Heuristics to find the real parameter dict inside the checkpoint:
candidates = []
if isinstance(ckpt, dict):
    # common places
    for key in ("state_dict", "model_state_dict", "model", "model_state"):
        if key in ckpt:
            candidates.append((key, ckpt[key]))
    # EMA shadow weights (sometimes used for inference)
    if "ema" in ckpt and isinstance(ckpt["ema"], dict):
        # common naming: ckpt["ema"]["shadow"] or ckpt["ema"]["state_dict"]
        if "shadow" in ckpt["ema"]:
            candidates.append(("ema.shadow", ckpt["ema"]["shadow"]))
        elif "state_dict" in ckpt["ema"]:
            candidates.append(("ema.state_dict", ckpt["ema"]["state_dict"]))
    # if none of the above, maybe the checkpoint *is* the state dict
    if not candidates:
        candidates.append(("root", ckpt))

# pick the first candidate that *looks* like a state_dict (mapping of tensors)
state_dict = None
for name, cand in candidates:
    if isinstance(cand, dict) and any(isinstance(v, torch.Tensor) for v in cand.values()):
        state_dict = cand
        print("Using candidate:", name)
        break

if state_dict is None:
    raise RuntimeError("Couldn't find a parameter dict in checkpoint. Candidates were: "
                       + ", ".join([c[0] for c in candidates]))

# Strip common prefixes like 'module.' (DDP) or 'model.' (wrapped)
clean_state = OrderedDict()
for k, v in state_dict.items():
    new_k = k
    # remove DDP prefix
    if new_k.startswith("module."):
        new_k = new_k[len("module."):]
    # remove wrapper prefix
    if new_k.startswith("model."):
        new_k = new_k[len("model."):]
    clean_state[new_k] = v

# (Optional) If checkpoint contains model_args you can instantiate the exact arch:
if "model_args" in ckpt:
    print("Found model_args in checkpoint; use these to construct the model:", ckpt["model_args"])
    # e.g. model = M3Gnet(**ckpt["model_args"])   <-- replace with the actual constructor call
else:
    print("No model_args found; ensure you construct M3Gnet with the SAME hyperparameters")

# Now load (use strict=False to see what remains mismatched)
res = model.load_state_dict(clean_state, strict=False)
print("missing keys (params your model expected but were not in the checkpoint):")
print(res.missing_keys)
print("unexpected keys (keys in checkpoint not used by model):")
print(res.unexpected_keys)
