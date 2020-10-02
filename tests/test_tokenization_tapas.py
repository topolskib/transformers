import os
import pandas as pd
import unittest

from transformers.tokenization_tapas import TapasTokenizer


SAMPLE_VOCAB = r"C:\Users\niels.rogge\Documents\Python projecten\tapas_tokenizer\vocab.txt"

tokenizer = TapasTokenizer(vocab_file=SAMPLE_VOCAB)

data = {'Actors': ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], 
        'Age': ["56", "45", "59"],
        'Number of movies': ["87", "53", "69"]}
queries = ["What is the name of the third actor?", "What is his age?", "What's the number of movies Brad Pitt has played in?"]
table = pd.DataFrame.from_dict(data)

print("Tokenized table:")
print(tokenizer._tokenize_table(table))

num_rows = tokenizer._get_num_rows(table, tokenizer.drop_rows_to_fit)
num_columns = tokenizer._get_num_columns(table)


print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")


for position, question in enumerate(queries):
    text_tokens = tokenizer.tokenize(question)
    tokenized_table = tokenizer._tokenize_table(table)

    feature_dict = tokenizer._to_trimmed_features(
            question=question,
            table=table,
            question_tokens=text_tokens,
            tokenized_table=tokenized_table,
            num_columns=num_columns,
            num_rows=num_rows,
            drop_rows_to_fit=tokenizer.drop_rows_to_fit)

    print(feature_dict)


