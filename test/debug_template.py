from PIL import Image
import numpy as np
from InternVLA.model.framework.M1 import InternVLA_M1

model_path = '/home/A.CARDAMONE7/checkpoints/InternVLA-M1-LIBERO-Goal/checkpoints/steps_30000_pytorch_model.pt'
model = InternVLA_M1.from_pretrained(model_path)
proc  = model.qwen_vl_interface.processor

img = Image.fromarray(np.zeros((224,224,3), dtype=np.uint8))
msg = [{'role': 'user', 'content': [
    {'type': 'image', 'image': img},
    {'type': 'image', 'image': img},
    {'type': 'text',  'text': 'open the drawer'},
]}]

text = proc.apply_chat_template([msg], tokenize=False, add_generation_prompt=True)
print('=== TEMPLATE OUTPUT ===')
print(text[:800])
print()
print('vision_start present:', '<|vision_start|>' in text)
print('image_pad present:',    '<|image_pad|>'    in text)

# Conta token visivi
from qwen_vl_utils import process_vision_info
img_inputs, _ = process_vision_info([msg])
print(f'image_inputs restituiti: {len(img_inputs) if img_inputs else 0}')
print(f'image sizes: {[i.size for i in img_inputs] if img_inputs else []}')