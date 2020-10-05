import os
import pandas as pd
import unittest

from transformers.tokenization_tapas import TapasTokenizer


SAMPLE_VOCAB = r"C:\Users\niels.rogge\Documents\Python projecten\tapas_tokenizer\vocab.txt"

tokenizer = TapasTokenizer(vocab_file=SAMPLE_VOCAB, model_max_length=100)

data = {'Actors': ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], 
        'Age': ["56", "45", "59"],
        'Number of movies': ["87", "53", "69"],
        'Date of birth': ["7 february 1967", "10 june 1996", "28 november 1967"]}
queries = ["What is the name of the third actor?", "What is his age?", "Which actor is 45 years old?", "Which actors are older than 50?"]
answer_coordinates = [[(2,0)], [(2,1)], [(1,0)], [(0,0), (2,0)]]
answer_texts = [["George Clooney"], ["59"], ["Leonardo Di Caprio"], ["Brad Pitt", "George Clooney"]]
table = pd.DataFrame.from_dict(data)

print("Model max length:")
print(tokenizer.model_max_length)

print("Tokenized table:")
print(tokenizer._tokenize_table(table))

num_rows = tokenizer._get_num_rows(table, tokenizer.drop_rows_to_fit)
num_columns = tokenizer._get_num_columns(table)


print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")


# for position, question in enumerate(queries):
#     text_tokens = tokenizer.tokenize(question)
#     tokenized_table = tokenizer._tokenize_table(table)

#     serialized_example, features = tokenizer._to_trimmed_features(
#                                                 question=question,
#                                                 table=table,
#                                                 question_tokens=text_tokens,
#                                                 tokenized_table=tokenized_table,
#                                                 num_columns=num_columns,
#                                                 num_rows=num_rows,
#                                                 drop_rows_to_fit=tokenizer.drop_rows_to_fit)
#     print(features)
# #     for token, rel in zip(serialized_example.tokens, features['numeric_relations']):
# #              if rel != 0:
# #                 print(token, rel)


encoded_inputs = tokenizer.batch_encode_plus(table=table, queries=queries, answer_coordinates=answer_coordinates, answer_texts=answer_texts, return_tensors="pt")
print(encoded_inputs)


# for id, label_id in zip(encoded_inputs["input_ids"][2], encoded_inputs["label_ids"][2]):
#         print(tokenizer.decode([id]), label_id.item())


