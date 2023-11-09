# CS 195: Natural Language Processing
**Fall 2023**

## Instructor

Eric Manley

 :speech_balloon: [Microsoft Teams](https://teams.microsoft.com/l/chat/0/0?users=eric.manley@drake.edu)

:email: eric.manley@drake.edu

:office: Collier-Scripps 327


### Office Hours

Schedule in [Starfish](https://drake.starfishsolutions.com/starfish-ops/dl/instructor/serviceCatalog.html?bookmark=connection/8352/schedule) the day before or drop in
* M 10:00am-1:00pm
* T 10:00am-12:00pm


---

## Fortnight 1

### 8/29 Prologue: Adventure Awaits
* [Syllabus](F0_0_Syllabus.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ericmanley/f23-CS195NLP/blob/main/F0_0_Syllabus.ipynb)


### 8/31 Using the Transformers Library
* [Introduction to Hugging Face Transformers Library](F1_1_HuggingFace.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ericmanley/f23-CS195NLP/blob/main/F1_1_HuggingFace.ipynb)
* Further Reading
    - [Hugging Face *Quicktour*](https://huggingface.co/docs/transformers/quicktour)
    - [Hugging Face *Run Inference with Pipelines tutorial*](https://huggingface.co/docs/transformers/pipeline_tutorial)
    - [Hugging Face *NLP Course, Chapter 2*](https://huggingface.co/learn/nlp-course/chapter2/1)

### 9/5 Text Classification Data and Evaluation
* [Loading Data and Evaluating Classification Models](F1_2_DataEvaluation.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ericmanley/f23-CS195NLP/blob/main/F1_2_DataEvaluation.ipynb)
* Further Reading
    - [Hugging Face Load a dataset from the Hub tutorial](https://huggingface.co/docs/datasets/load_hub)
    - [scikit-learn Classification Metrics User's Guide](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)

### 9/7 ROUGE and Summarization
* [ROUGE and Summarization](F1_3_RougeSummarization.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ericmanley/f23-CS195NLP/blob/main/F1_3_RougeSummarization.ipynb)
* Further Reading
    - [*Two minutes NLP — Learn the ROUGE metric* by examples](https://medium.com/nlplanet/two-minutes-nlp-learn-the-rouge-metric-by-examples-f179cc285499) by Fabio Chiusan 
    - [Google's implementation of rouge_score](https://github.com/google-research/google-research/tree/master/rouge)
    - [Hugging Face's wrapper for Google's implementation](https://huggingface.co/spaces/evaluate-metric/rouge)
    - [Hugging Face Task Guide on Summarization](https://huggingface.co/docs/transformers/tasks/summarization)

---

## Fortnight 2

### 9/12 Demo Day (and dealing with issues using datasets)
* Demo Day
    - 5-min demo of creative synthesis project or completed applied exploration (or core practice if that's what you have)
    - Write down the names of the people you presented to
    - Nominate a cool project to show off to everyone
* [More on Dataset Organization](F2_1_MoreOnDatasets.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ericmanley/f23-CS195NLP/blob/main/F2_1_MoreOnDatasets.ipynb)
* Further Reading
    - [Hugging Face Load a dataset from the Hub tutorial](https://huggingface.co/docs/datasets/load_hub)
    - [Hugging Face dataset features documentation](https://huggingface.co/docs/datasets/about_dataset_features)

### 9/14 Summarization, Translation, and Question Answering (if time)
* [Summarization, Translation, and Question Answering](F2_2_SummarizationTranslationQuestionAnswering.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ericmanley/f23-CS195NLP/blob/main/F2_2_SummarizationTranslationQuestionAnswering.ipynb)
* Further Reading
    - [Hugging Face Task Guide on Summarization](https://huggingface.co/docs/transformers/tasks/summarization)
    - [Hugging Face Task Guide on Translation](https://huggingface.co/docs/transformers/tasks/translation)
    - [Hugging Face Task Guide on Question Answering](https://huggingface.co/docs/transformers/tasks/question_answering)

### 9/19 Question Answering
* [Question Answering](F2_3_QuestionAnswering.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ericmanley/f23-CS195NLP/blob/main/F2_3_QuestionAnswering.ipynb)
* Further Reading
    - [Hugging Face Task Guide on Question Answering](https://huggingface.co/docs/transformers/tasks/question_answering)

### 9/21 Markov Models
* [Markov Models](F2_4_MarkovModels.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ericmanley/f23-CS195NLP/blob/main/F2_4_MarkovModels.ipynb)
* Further Reading
    - [Markov chain on Wikipedia](https://en.wikipedia.org/wiki/Markov_chain)
    - [NLTK Book Chapter 2: Accessing Text Corpora and Lexical Resources](https://www.nltk.org/book/ch02.html)
    - [What is ChatGPT Doing and Why Does it Work? By Stephen Wolfram](https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/)
        * This is a fascinating article that covers a lot of NLP topics. The opening motivates text generation with Markov-like descriptions.

---

## Fortnight 3

### 9/26 Demo Day and Tokenization
* Demo Day
    - 5-min demo of creative synthesis project or completed applied exploration (or core practice if that's what you have)
    - Write down the names of the people you presented to
    - Nominate a cool project to show off to everyone
* [Tokenization](F3_1_Tokenization.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ericmanley/f23-CS195NLP/blob/main/F3_1_Tokenization.ipynb)
* Further Reading:
    - [Python `requests` library quickstart](https://requests.readthedocs.io/en/latest/user/quickstart/)
    - [GPT Tokenizer Illustration](https://platform.openai.com/tokenizer)
    - [Python `split` method](https://docs.python.org/3/library/stdtypes.html#str.split)

### 9/28 Automatic Tokenization
* [Automatic Tokenization](F3_2_AutoTokenization.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ericmanley/f23-CS195NLP/blob/main/F3_2_AutoTokenization.ipynb)
* Further Reading:
    - [Beautiful Soup documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
    - [Hugging Face Byte-Pair Encoding tokenization](https://huggingface.co/learn/nlp-course/chapter6/5?fw=pt)
    - [Hugging Face WordPiece tokenization](https://huggingface.co/learn/nlp-course/chapter6/6?fw=pt)

### 10/3 Hidden Markov Models and Part-of-Speech Tagging
* [Hidden Markov Models and Part-of-Speech Tagging](F3_3_HMMPOS.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ericmanley/f23-CS195NLP/blob/main/F3_3_HMMPOS.ipynb)
* Further Reading:
    - [Wisdom ML's *Hidden Markov Model (HMM) in NLP: Complete Implementation in Python*](https://wisdomml.in/hidden-markov-model-hmm-in-nlp-python/)
    - [Great Learning's *Part of Speech (POS) tagging with Hidden Markov Model*](https://www.mygreatlearning.com/blog/pos-tagging/)
    - [Hidden Markov Model on Wikipedia](https://en.wikipedia.org/wiki/Hidden_Markov_model)
    - [Viterbi Algorithm on Wikipedia](https://en.wikipedia.org/wiki/Viterbi_algorithm)

### 10/5 Semantic Folding with Saul Varshavsky
* [Workshop Slides](F3_4_SaulVarshavsky_Semantic%20Folding_Workshop.pptx) [![Open in Google Slides](https://img.shields.io/badge/Open%20in-Google%20Slides-blue?logo=google%20slides&style=flat-square&link=https://docs.google.com/presentation/d/1k4svXVYZ8M1gLDFD0MMz0y3jIdbIfjOs/edit?usp=sharing&ouid=104219288264290628620&rtpof=true&sd=true)](https://docs.google.com/presentation/d/1k4svXVYZ8M1gLDFD0MMz0y3jIdbIfjOs/edit?usp=sharing&ouid=104219288264290628620&rtpof=true&sd=true)
* [Worksheet](F3_4_SaulVarshavsky_Semantic%20Folding_Worksheet.docx) [![Open in Google Docs](https://img.shields.io/badge/Open%20in-Google%20Docs-blue?logo=google%20docs&style=flat-square&link=https://docs.google.com/document/d/1GjZOk9b-VdRRVOl2JfSCFZREXBg2KHzo/edit?usp=sharing&ouid=104219288264290628620&rtpof=true&sd=true)](https://docs.google.com/document/d/1GjZOk9b-VdRRVOl2JfSCFZREXBg2KHzo/edit?usp=sharing&ouid=104219288264290628620&rtpof=true&sd=true)
* [Semantic Folding Code](F3_4_SaulVarshavsky_Semantic%20Folding_Code.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ericmanley/F23-CS195NLP/blob/main/F3_4_SaulVarshavsky_SemanticFolding_Code.ipynb)
* [SemanticFoldingData.csv](data/SemanticFoldingData.csv)


---

## Fortnight 4

### 10/10 Demo Day and WordNet
* Example Portfolio
    - Here's an example of a Jupyter Notebook being used to format the [main page of the portfolio](https://drive.google.com/file/d/1dDvHaw2nlPS37Ex2Qze61FRCUA4QMUbo/view?usp=drive_link)
    - [Here's a link to the Google Drive folder](https://drive.google.com/drive/folders/1No-v1NK2qCKnSgW0otKKmGz7XmEnhyNX?usp=drive_link) - you can see it is just a place to put all of the files you worked on.
* Demo Day
    - 5-min demo of creative synthesis project or completed applied exploration (or core practice if that's what you have)
    - Write down the names of the people you presented to
    - Nominate a cool project to show off to everyone
* [WordNet](F4_1_WordNet.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ericmanley/f23-CS195NLP/blob/main/F4_1_WordNet.ipynb)
* Further Reading:
    - [Sample usage for WordNet](https://www.nltk.org/howto/wordnet.html)
    - [WordNet documentation](https://www.nltk.org/api/nltk.corpus.reader.wordnet.html)
    - [NLTK Book Chapter 2 (see Section 5)](https://www.nltk.org/book/ch02.html)
    - [Dive into WordNet with NLTK by Norbert Kozlowski](https://medium.com/@don_khozzy/dive-into-wordnet-with-nltk-b313c480e788)
    - [Getting started with nltk-wordnet in Python](https://www.section.io/engineering-education/getting-started-with-nltk-wordnet-in-python/)

### 10/12 Word Sense Disambiguation
* [Word Sense Disambiguation](F4_2_WordSenseDisambiguation.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ericmanley/f23-CS195NLP/blob/main/F4_2_WordSenseDisambiguation.ipynb)
* Further Reading:
    - [Word Senses and WordNet, Chapter 23 of *Speech and Language Processing* by Daniel Jurafsky & James H. Martin](https://web.stanford.edu/~jurafsky/slp3/23.pdf)
    - [WordNet documentation](https://www.nltk.org/api/nltk.corpus.reader.wordnet.html)
    - [SemCor Corpus Module documentation](https://www.nltk.org/api/nltk.corpus.reader.semcor.html)
    - [NLTK Stopwords](https://pythonspot.com/nltk-stop-words/)
    - [Lemmatization with NLTK](https://www.geeksforgeeks.org/python-lemmatization-with-nltk/)


### 10/19 Context-Free Grammars
* [Context-Free Grammars](F4_3_ContextFreeGrammars.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ericmanley/f23-CS195NLP/blob/main/F4_3_ContextFreeGrammars.ipynb)
* Further Reading:
    - [Context-Free Grammars and Constituency Parsing, Chapter 17 of *Speech and Language Processing* by Daniel Jurafsky & James H. Martin](https://web.stanford.edu/~jurafsky/slp3/17.pdf)
    - [NLTK Book Chapter 8: Analyzing Sentence Structure](https://www.nltk.org/book/ch08.html)


---

## Fortnight 5

### 10/24 Demo Day and Parsing
* Demo Day
    - 5-min demo of creative synthesis project or completed applied exploration (or core practice if that's what you have)
    - Write down the names of the people you presented to
    - Nominate a cool project to show off to everyone
* [Midterm Course Feedback](https://forms.gle/4Qfi5oBHd5Pvvumk6)
* [Parsing](F5_1_Parsing.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ericmanley/f23-CS195NLP/blob/main/F5_1_Parsing.ipynb)
* Further Reading:
    - [Context-Free Grammars and Constituency Parsing, Chapter 17 of *Speech and Language Processing* by Daniel Jurafsky & James H. Martin](https://web.stanford.edu/~jurafsky/slp3/17.pdf)
    - [NLTK Book Chapter 8: Analyzing Sentence Structure]( https://www.nltk.org/book/ch08.html)
    - [Wikipedia article on CYK Parsing](https://en.wikipedia.org/wiki/CYK_algorithm)

### 10/26 Machine Learning with Text Data
* [Machine Learning](F5_2_MachineLearning.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ericmanley/f23-CS195NLP/blob/main/F5_2_MachineLearning.ipynb)
* Further Reading:
    - [Vector Semantics and Embeddings, Chapter 6 of Speech and Language Processing by Daniel Jurafsky & James H. Martin](https://web.stanford.edu/~jurafsky/slp3/6.pdf)
    - [scikit-learn API reference](https://scikit-learn.org/stable/modules/classes.html)
    
### 10/31 Neural Networks
* [Neural Networks](F5_3_NeuralNetworks.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ericmanley/f23-CS195NLP/blob/main/F5_3_NeuralNetworks.ipynb)
* Further Reading:
    - [Neural Networks and Neural Language Models, Chapter 7 of Speech and Language Processing by Daniel Jurafsky & James H. Martin](https://web.stanford.edu/~jurafsky/slp3/7.pdf)
    - [Artificial Neural Networks, Chapter 4 of Machine Learning by Tom M. Mitchell](http://www.cs.cmu.edu/~tom/files/MachineLearningTomMitchell.pdf)
    - [Sequential Model from Keras Developer Guide](https://keras.io/guides/sequential_model/)

### 11/2 Embeddings
* [Embeddings](F5_4_Embeddings.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ericmanley/f23-CS195NLP/blob/main/F5_4_Embeddings.ipynb)
* Further Reading:
    - [Word2Vec Tutorial - The Skip-Gram Model by Chris McCormick](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
    - [Word2Vec - Negative Sampling made easy by Munesh Lakhey](https://medium.com/@mnshonco/word2vec-negative-sampling-made-easy-9a587cb4695f)
    - [Keras Embedding Layer](https://keras.io/api/layers/core_layers/embedding/)
    - [Keras Tokenizer](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer)

---

## Fortnight 6

### 11/7 Demo Day and Neural Language Models
* Demo Day
    - 5-min demo of creative synthesis project or completed applied exploration (or core practice if that's what you have)
    - Write down the names of the people you presented to
    - Nominate a cool project to show off to everyone
* [Neural Language Modeling](F6_1_NeuralLanguageModeling.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ericmanley/f23-CS195NLP/blob/main/F6_1_NeuralLanguageModeling.ipynb)
* Further Reading:
    - [Neural Networks and Neural Language Models, Chapter 7 of *Speech and Language Processing* by Daniel Jurafsky & James H. Martin](https://web.stanford.edu/~jurafsky/slp3/7.pdf)

### 11/9 Recurrent Neural Networks
* [Recurrent Neural Networks](F6_2_RecurrentNeuralNetworks.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ericmanley/f23-CS195NLP/blob/main/F6_2_RecurrentNeuralNetworks.ipynb)
* Further Reading:
   - [RNNs and LSTMs, Chapter 9 of Speech and Language Processing by Daniel Jurafsky & James H. Martin](https://web.stanford.edu/~jurafsky/slp3/9.pdf)
    - [Keras documentation for SimpleRNN Layer](https://keras.io/api/layers/recurrent_layers/simple_rnn/)
    