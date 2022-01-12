from transformers import MarkupLMTokenizer

tokenizer = MarkupLMTokenizer.from_pretrained("microsoft/markuplm-base")

empty_tokens = tokenizer("", padding=True, pad_to_multiple_of=8)

for k,v in empty_tokens.items():
    print(k, len(v))