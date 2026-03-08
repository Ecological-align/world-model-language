# add_clip_text.py  — run once, ~2 min
import sys; sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, torch, json, os
from transformers import CLIPProcessor, CLIPModel
from phrase_bank import PHRASE_BANK

DEVICE = torch.device("cuda")
DATA_DIR = "lm_output/phrase_level"

with open(os.path.join(DATA_DIR, "event_index.json")) as f:
    events = json.load(f)["events"]

clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_model.eval()

phrases = [e["phrase"] for e in events]
all_text_embeds = []

with torch.no_grad():
    for i in range(0, len(phrases), 32):
        batch = phrases[i:i+32]
        inputs = clip_proc(text=batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=77).to(DEVICE)
        text_out = clip_model.text_model(**inputs)
        projected = clip_model.text_projection(text_out.pooler_output)
        all_text_embeds.append(projected.float().cpu().numpy())
        print(f"  {min(i+32, len(phrases))}/{len(phrases)}")

clip_text = np.vstack(all_text_embeds)
np.save(os.path.join(DATA_DIR, "clip_text_hiddens_phrase.npy"), clip_text)
print(f"Saved: {clip_text.shape}")
