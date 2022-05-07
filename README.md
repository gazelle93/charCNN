# Overview
- Character-CNN (charCNN) is a word embedding method that is applied in various language models such as ELMo and characterBERT. Since charCNN utilizes characters instead of words or sub-words, it allows limiting the size of the vocabulary of the model. Moreover, charCNN is able to handle Out Of Vocabulary (OOV) issue when the new words that appeared at the test step are not foreshadowed at the training step and do not exist in the vocabulary. This is a crucial problem to the systems that leverage word embeddings because OOV words cannot be converted to a real-value vector as a representation of a token in the vector space. In ELMo and characterBERT utilized highway networks where it does a gated combination of a linear transformation and a non-linear transformation of its input, so it allows for smoother information transfer through the input. This project aims to implement the charCNN word embedding method that is leveraged in ELMo and characterBERT.

# Brief description
- text_processing.py
> Output format
> - output: Tokenized result of a given text. (list)
- charCNN.py
> Output format
> - output: List of tensor of input tokens.

# Prerequisites
- argparse
- stanza
- spacy
- nltk
- gensim
- torch

# Parameters
- nlp_pipeline(str, defaults to "spacy"): Tokenization method (spacy, stanza, nltk, gensim).
- num_channels(int, defaults to 10): The number of channels (filter map).
- output_dim(int, defaults to 128): The size of word embedding.
- max_characters_per_token(int, defaults to 50): The maximum size of characters of token.
- n_highway(int, defaults to 2): The number of highway layers.

# References
- Stanza: Qi, P., Zhang, Y., Zhang, Y., Bolton, J., & Manning, C. D. (2020). Stanza: A Python natural language processing toolkit for many human languages. arXiv preprint arXiv:2003.07082.
- Spacy: Matthew Honnibal and Ines Montani. 2017. spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing. To appear (2017).
- NLTK: Bird, Steven, Edward Loper and Ewan Klein (2009). Natural Language Processing with Python.  O'Reilly Media Inc.
- Gensim: Rehurek, R., & Sojka, P. (2010). Software framework for topic modelling with large corpora. In In Proceedings of the LREC 2010 workshop on new challenges for NLP frameworks.
- Word2vec: Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781.
- Character-CNN: Kim, Y., Jernite, Y., Sontag, D., & Rush, A. M. (2016, March). Character-aware neural language models. In Thirtieth AAAI conference on artificial intelligence.
- Character-CNN + LSTM: Jozefowicz, R., Vinyals, O., Schuster, M., Shazeer, N., & Wu, Y. (2016). Exploring the limits of language modeling. arXiv preprint arXiv:1602.02410.
- ELMo: Sarzynska-Wawer, J., Wawer, A., Pawlak, A., Szymanowska, J., Stefaniak, I., Jarkiewicz, M., & Okruszek, L. (2021). Detecting formal thought disorder by deep contextualized word representations. Psychiatry Research, 304, 114135.
- CharacterBERT: Boukkouri, H. E., Ferret, O., Lavergne, T., Noji, H., Zweigenbaum, P., & Tsujii, J. (2020). CharacterBERT: Reconciling ELMo and BERT for word-level open-vocabulary representations from characters. arXiv preprint arXiv:2010.10392.
- Highway Networks: Srivastava, R. K., Greff, K., & Schmidhuber, J. (2015). Highway networks. arXiv preprint arXiv:1505.00387.
