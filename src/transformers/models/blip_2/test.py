import requests
import torch
from PIL import Image

from transformers import Blip2ForConditionalGeneration, Blip2Processor


device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl", device_map="auto", load_in_8bit=True
)
# model.to(device)

url = "https://huggingface.co/spaces/Salesforce/BLIP2/resolve/main/house.png"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

prompt = "Question: How could someone get out of the house? Answer:"
# prompt = ""
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

for k, v in inputs.items():
    print(k, v.shape, v.device)

generated_ids = model.generate(**inputs)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)
