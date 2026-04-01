import sys
sys.path.insert(0, "..")
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from InternVLA.model.framework.M1 import InternVLA_M1

MODEL_PATH = "/home/A.CARDAMONE7/checkpoints/InternVLA-M1-LIBERO-Goal/checkpoints/steps_30000_pytorch_model.pt"
FRAME_PATH = "/home/A.CARDAMONE7/repo/VLA-Bench/robosuite_test/TinyVLA/test/vqa_test/first_frame.png"

model = InternVLA_M1.from_pretrained(MODEL_PATH).to("cuda").eval()

# ── Trova il processor dentro qwen_vl_interface ───────────────────────────────
print("=== qwen_vl_interface attrs ===")
for k, v in model.qwen_vl_interface.__dict__.items():
    if not k.startswith("_"):
        print(f"  .{k}  =  {type(v).__name__}")

processor = None
for candidate in ["processor", "tokenizer", "image_processor", "vlm_processor"]:
    if hasattr(model.qwen_vl_interface, candidate):
        processor = getattr(model.qwen_vl_interface, candidate)
        print(f"\n✅ Processor trovato: qwen_vl_interface.{candidate}")
        break

if processor is None:
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    print("\n⚠️  Processor caricato da HF (fallback)")

vlm = model.qwen_vl_interface
frame = Image.open(FRAME_PATH)

# ── Test 1: generazione testo ─────────────────────────────────────────────────
print("\n=== TEST 1: Text generation ===")
for q in ["What object is in front of the stove?", "List all objects on the table."]:
    msg = [{"role": "user", "content": [
        {"type": "image", "image": frame},
        {"type": "text",  "text": q}
    ]}]
    text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    if isinstance(text, list): text = text[0]
    image_inputs, _ = process_vision_info(msg)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = vlm.model.generate(**inputs, max_new_tokens=64, do_sample=False)
    answer = processor.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"Q: {q}\nA: {answer}\n")

# ── Test 2: cosine similarity L0 vs L3 ───────────────────────────────────────
print("=== TEST 2: Embedding cosine similarity ===")

def get_last_hidden(instruction):
    msg = [{"role": "user", "content": [
        {"type": "image", "image": frame},
        {"type": "text",  "text": instruction}
    ]}]
    text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    if isinstance(text, list): text = text[0]
    image_inputs, _ = process_vision_info(msg)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = vlm.model(**inputs, output_hidden_states=True)
    return out.hidden_states[-1].mean(dim=1).squeeze().cpu().float()

l0  = get_last_hidden("Put the cream cheese in the bowl")
l3  = get_last_hidden("Put the object in front of the stove on the bowl")
cot = get_last_hidden("Put the object in front of the stove on the bowl. Locate the key object needed.")

cos = torch.nn.functional.cosine_similarity
print(f"cos(L0, L3)     = {cos(l0.unsqueeze(0), l3.unsqueeze(0)).item():.4f}")
print(f"cos(L0, CoT-L3) = {cos(l0.unsqueeze(0), cot.unsqueeze(0)).item():.4f}")
print(f"cos(L3, CoT-L3) = {cos(l3.unsqueeze(0), cot.unsqueeze(0)).item():.4f}")
