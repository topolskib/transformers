import requests
from PIL import Image

from transformers import BlipImageProcessor, T5Tokenizer
from transformers.models.instructblip.processing_instructblip import InstructBlipProcessor


image_processor = BlipImageProcessor()
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

processor = InstructBlipProcessor(image_processor=image_processor, tokenizer=tokenizer)

url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
prompt = "What is unusual about this image?"
inputs = processor(images=image, text=prompt, return_tensors="pt")

for k, v in inputs.items():
    print(k, v.shape)
