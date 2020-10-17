TAPAS-----------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The TAPAS model was proposed in `TAPAS: Weakly Supervised Table Parsing via Pre-training
<https://arxiv.org/abs/2004.02349>`__ by Jonathan Herzig, Paweł Krzysztof Nowak, Thomas Müller, Francesco Piccinno and 
Julian Martin Eisenschlos.
It's a BERT-based model specifically designed (and pre-trained) for answering questions about tabular data. 
Compared to BERT, TAPAS has 7 token type ids that encode tabular structure. TAPAS is pre-trained on the masked language
modeling (MLM) objective on a large dataset comprising millions of tables from English Wikipedia and corresponding
texts. For question answering, TAPAS has 2 heads on top: a cell selection head and an aggregation head, for (optionally) 
performing aggregations (such as counting or summing) among selected cells. TAPAS has been fine-tuned on several datasets: 
SQA (Sequential Question Answering by Microsoft), WTQ (WikiTable Questions by Stanford University) and WikiSQL (by Salesforce). 
It achieves state-of-the-art on the former, while having comparable performance to SOTA algorithms for both WTQ and WikiSQL, 
while having a much simpler architecture.

The abstract from the paper is the following:

*Answering natural language questions over tables is usually seen as a semantic parsing task. 
To alleviate the collection cost of full logical forms, one popular approach focuses on weak 
supervision consisting of denotations instead of logical forms. However, training semantic parsers 
from weak supervision poses difficulties, and in addition, the generated logical forms are only used
as an intermediate step prior to retrieving the denotation. In this paper, we present TAPAS, an 
approach to question answering over tables without generating logical forms. TAPAS trains from weak 
supervision, and predicts the denotation by selecting table cells and optionally applying a corresponding 
aggregation operator to such selection. TAPAS extends BERT's architecture to encode tables as input, 
initializes from an effective joint pre-training of text segments and tables crawled from Wikipedia, 
and is trained end-to-end. We experiment with three different semantic parsing datasets, and find 
that TAPAS outperforms or rivals semantic parsing models by improving state-of-the-art accuracy on 
SQA from 55.1 to 67.2 and performing on par with the state-of-the-art on WIKISQL and WIKITQ, but 
with a simpler model architecture. We additionally find that transfer learning, which is trivial 
in our setting, from WIKISQL to WIKITQ, yields 48.7 accuracy, 4.2 points above the state-of-the-art.*

In addition, the authors have further pre-trained TAPAS to recognize table entailment, by creating a balanced dataset of millions 
of automatically created training examples which are learned in an intermediate step prior to fine-tuning. The authors of TAPAS 
call this further pre-training intermediate pre-training (since TAPAS is first pre-trained on MLM, and then on another dataset). 
They found that intermediate pre-training further improves performance on SQA, achieving a new state-of-the-art as well as 
state-of-the-art on TabFact, a large-scale dataset with 16k Wikipedia tables for table entailment (a binary classification task).
For more details, see their new paper: `Understanding tables with intermediate pre-training <https://arxiv.org/abs/2010.00571>`__ 
by Julian Martin Eisenschlos, Syrine Krichene and Thomas Müller.

Tips:

- TAPAS is a model that uses relative position embeddings by default (restarting the position embeddings at every cell). According to
  the authors, this usually results in a slightly better performance, and allows you to encode longer sequences without running out 
  of embeddings.
  If you don't want this, set the `reset_position_index_per_cell` parameter of :class:`~transformers.TapasConfig` to False, and make 
  sure you're loading the weights of a model that was pretrained with absolute position embeddings.
- TAPAS is similar to BERT and therefore relies on the masked language modeling (MLM) objective.
  It is therefore efficient at predicting masked tokens and at NLU in general, but is not optimal for
  text generation. Models trained with a causal language modeling (CLM) objective are better in that regard.

TapasConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TapasConfig
    :members:


TapasTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TapasTokenizer
    :members: build_inputs_with_special_tokens, get_special_tokens_mask,
        create_token_type_ids_from_sequences, save_vocabulary


TapasModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TapasModel
    :members:


TapasForMaskedLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TapasForMaskedLM
    :members:


TapasForQuestionAnswering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TapasForQuestionAnswering
    :members:
