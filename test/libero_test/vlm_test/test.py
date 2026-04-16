import sys, torch
sys.path.insert(0, "/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/InternVLA-M1")

from InternVLA.model.framework.M1 import InternVLA_M1

model = InternVLA_M1.from_pretrained(
    "/home/A.CARDAMONE7/checkpoints/InternVLA-M1-LIBERO-Goal/checkpoints/steps_30000_pytorch_model.pt"
)

# ─── Safe introspection — never raises AttributeError ───────────────────────
print("=== TOP-LEVEL SUBMODULES ===")
for name, mod in model._modules.items():
    print(f"  model.{name}  →  {type(mod).__name__}")

print("\n=== INSTANCE ATTRIBUTES (non-modules) ===")
for k, v in model.__dict__.items():
    if not k.startswith("_") and k not in model._modules:
        print(f"  model.{k}  =  {type(v).__name__}  /  {repr(v)[:80]}")
