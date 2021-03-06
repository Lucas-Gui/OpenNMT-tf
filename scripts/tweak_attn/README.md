# Attention extraction and injection in OpenNMT-tf self-attention models

This code has been written (or modified, for the OpenNMT-tf folders) during my internship in the summer 2020.

It can be used to :
1. Identify self-attention heads of a Transformer or GPT-2 model according to their ability to focus on syntactic dependencies,
as in  [Voita et al.](https://arxiv.org/pdf/1905.09418.pdf)
2. Modify attention between tokens on a given set of heads, in order to modify or inject syntactic information inside
 the model.


Contact : [lucas.guirardel@polytechnique.edu](mailto:lucas.guirardel@polytechnique.edu)

### General observations:

This has been designed for french -> german translation.
To adjust it to other languages pairs, one should mind the different tagging schemes in Spacy, or use another tagging 
tool (such as Stanford Dependencies ?)

All mentions of attention refer to _dot-product attention_, before the application of softmax.

### Required libraries:

- Fasttext (for language identification by _find_fr.py_)
    - Download the language identification pretrained model from 
    [here](https://fasttext.cc/docs/en/language-identification.html).  
- Spacy with necessary models (as is, fr_core_news_md, de_core_news_md) 
- OpenNMT 2.11.1 modified version (this package) 
- Tensorflow 
- Matplotlib for _process_result_attention.py_ 

## Attention measurement: process from a corpus without syntactical labels

- Clean raw text
    - Remove sentences from other languages with _find_fr.py_
    - Remove non-standard Unicode characters, especially spaces. (Not strictly necessary, but makes the rest easier)
        - _remove_spaces.py_ replaces non-standard spaces.

- Prepare syntactic labels with _make_UD.py_
    - Can separate sentences from the corpus if needed (flag `-S`)
    - Specify studied categories : e.g. `--cats gen nb`, to save labels of gender and number for each sentence. 

- Tokenize (do according to the model used).
    - If applicable, tokenize the sentences file made by _make_UD.py_ .
    - You might want to filter sentences above a certain length (for exemple with _reduce.py_).
     It should not be necessary if the corpus has been separated into sentences.
    - Avoid the addition of \<UNK\> tokens as noise : they cannot be aligned with the sentence.
    Any sentence containing some will be ignored by _align_pos.py_. 
     
- Align tokens and words : _align_pos.py_
    - Give the result of _make_UD.py_ and the tokens file as arguments
 
- Determine the frequency of tokens with _rarity.py_. 
Required to study the focus of attention on rarest tokens.

- Study attention : _analyze_attn_dep.py_
    - Give tokens, heads, dependencies, and labels files, aligned by align_pos.py, as arguments.
    - Parameter `--dep`: analyze the focus of attention on these dependencies 
    (e.g. nsubj obj amod advmod to replicae Voita's experiments.)
    - Parameter `-R`: tokens rarity file, as given by rarity.py
    - The script saves a pickled AttnResult object.

- Display the results : _process_results_attn.py_
    - Take a path to a pickled AttnResult object as argument
    - Creates different diagrams, and a file score.txt which indicate the best heads for each dependency.
    - Might need manual edition to suit your needs. 

 
## Attention injection

- Statistical measurements of attention: _measure_attn.py_ :
    - For each head, computes statistical values on attention values (mean, std, first and last deciles.)
    - Makes a folder containing saved numpy arrays.
    - **Warning**: This script is very memory-consuming, since it keeps every value taken by attention over the corpus.
    You should either give it only a small extract of the corpus, or implement a low-memory quantile approximation method.

- Study different ways of injecting attention with _inject_tasks.py_
    - Uses aligned dependencies, heads and labels, and the folder created by _measure_attn.py_.
    - Is able to perform a number of "tasks", i.e. to inject attention in different ways given the sentence.
    You can easily create tasks according to your needs, e.g. :
        - Inject random gaussian noise on attention from verb to subject or subject to verb
        - Inject low attention between verb and subject
        - attempt to exchange the subjects of two verbs 
        - attempt to exchange subject and object of a verb
        - attempt to change a grammatical property of a word (e.g. put a verb to passive voice)
    - Very slow, probably because it does not use batches for inference.

- Evaluate the impact of the injection tasks with _eval_inject.py_ (quite basic) : 
    - Counts and save lines that pass different tests.
        - Are translations different ?
        - Do they differ on words involved in a given dependency ?
    - Note : for compatibility with Spacy, _eval_inject.py_ replaces numerals placeholder with numbers in (1980, 2004)
    - The other solution is to evaluate by hand...

## Modified version of OpenNMT


### Usage : extraction
Return the values of self-attention computed over the input.
Only for GPT-2's decoder and Transformer encoder.

Once the model is loaded:
```python 
tokens, attn = model(data, return_attn = True, training = False)
```
`attn` is a numpy.ndarray of shape (L,H,S,S) where L and H are the number of layers and heads, and S the sentence's size.
 `attn[l,h,i,j]` it the attention of token i over token j.
 
 Note: the model only carries out encoding, and not decoding. The output tokens are those from the input.

### Usage : injection
The user can choose attention values as a function of the input sentence, inject them in the dot-product self-attention of 
chosen heads, and compare the resulting translation with the original one. 
However, as of now, it is not possible to use the attention values as computed by the model to chose the new attention values.

This only compatible with Transformer models. Injection is applied only to the encoder.

For example, one could decide to always put an attention of 0 between verbs and subject, but not to double the attention
of nouns on their adjectives.

Let L the number of encoder layers, H the number of heads per layer, and S the number of tokens in the sentence.
Let `inj_val` a `np.ndarray` of shape (L,H,S,S) containing the values to inject, and
`inj_mask` a boolean `np.ndarray` of the same shape, 
containing `True` where attention is to be injected, and `False` elsewhere.

Then with :
```python
_, predict = model(data, training=False, inject=(inj_val, inj_mask))
```
`predict` is the modified result. 

