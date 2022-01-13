from transformers import MarkupLMTokenizerFast
from transformers.tokenization_utils import AddedToken

added_tokens = [AddedToken("howaboutthat", lstrip=True)]

tokenizer = MarkupLMTokenizerFast.from_pretrained("microsoft/markuplm-base", additional_special_tokens=added_tokens)

print(tokenizer.special_tokens_map)

text = "<html> hello this is howaboutthat a special token </html>"

encoding = tokenizer(text, return_tensors="pt")

for id in encoding.input_ids.squeeze().tolist():
    print(id, tokenizer.decode([id]))