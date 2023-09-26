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
* [Syllabus](F0_0_Syllabus.ipynb) [<img src="https://colab.research.google.com/assets/colab-badge.svg" width="70px">](https://colab.research.google.com/github/ericmanley/f23-CS195NLP/blob/main/F0_0_Syllabus.ipynb)


### 8/31 Using the Transformers Library
* [Introduction to Hugging Face Transformers Library](F1_1_HuggingFace.ipynb) [<img src="https://colab.research.google.com/assets/colab-badge.svg" width="70px">](https://colab.research.google.com/github/ericmanley/f23-CS195NLP/blob/main/F1_1_HuggingFace.ipynb)
* Further Reading
    - [Hugging Face *Quicktour*](https://huggingface.co/docs/transformers/quicktour)
    - [Hugging Face *Run Inference with Pipelines tutorial*](https://huggingface.co/docs/transformers/pipeline_tutorial)
    - [Hugging Face *NLP Course, Chapter 2*](https://huggingface.co/learn/nlp-course/chapter2/1)

### 9/5 Text Classification Data and Evaluation
* [Loading Data and Evaluating Classification Models](F1_2_DataEvaluation.ipynb) [<img src="https://colab.research.google.com/assets/colab-badge.svg" width="70px">](https://colab.research.google.com/github/ericmanley/f23-CS195NLP/blob/main/F1_2_DataEvaluation.ipynb)
* Further Reading
    - [Hugging Face Load a dataset from the Hub tutorial](https://huggingface.co/docs/datasets/load_hub)
    - [scikit-learn Classification Metrics User's Guide](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)

### 9/7 ROUGE and Summarization
* [ROUGE and Summarization](F1_3_RougeSummarization.ipynb) [<img src="https://colab.research.google.com/assets/colab-badge.svg" width="70px">](https://colab.research.google.com/github/ericmanley/f23-CS195NLP/blob/main/F1_3_RougeSummarization.ipynb)
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
* [More on Dataset Organization](F2_1_MoreOnDatasets.ipynb) [<img src="https://colab.research.google.com/assets/colab-badge.svg" width="70px">](https://colab.research.google.com/github/ericmanley/f23-CS195NLP/blob/main/F2_1_MoreOnDatasets.ipynb)
* Further Reading
    - [Hugging Face Load a dataset from the Hub tutorial](https://huggingface.co/docs/datasets/load_hub)
    - [Hugging Face dataset features documentation](https://huggingface.co/docs/datasets/about_dataset_features)

### 9/14 Summarization, Translation, and Question Answering (if time)
* [Summarization, Translation, and Question Answering](F2_2_SummarizationTranslationQuestionAnswering.ipynb) [<img src="https://colab.research.google.com/assets/colab-badge.svg" width="70px">](https://colab.research.google.com/github/ericmanley/f23-CS195NLP/blob/main/F2_2_SummarizationTranslationQuestionAnswering.ipynb)
* Further Reading
    - [Hugging Face Task Guide on Summarization](https://huggingface.co/docs/transformers/tasks/summarization)
    - [Hugging Face Task Guide on Translation](https://huggingface.co/docs/transformers/tasks/translation)
    - [Hugging Face Task Guide on Question Answering](https://huggingface.co/docs/transformers/tasks/question_answering)

### 9/19 Question Answering
* [Question Answering](F2_3_QuestionAnswering.ipynb) [<img src="https://colab.research.google.com/assets/colab-badge.svg" width="70px">](https://colab.research.google.com/github/ericmanley/f23-CS195NLP/blob/main/F2_3_QuestionAnswering.ipynb)
* Further Reading
    - [Hugging Face Task Guide on Question Answering](https://huggingface.co/docs/transformers/tasks/question_answering)

### 9/21 Markov Models
* [Markov Models](F2_4_MarkovModels.ipynb) [<img src="https://colab.research.google.com/assets/colab-badge.svg" width="70px">](https://colab.research.google.com/github/ericmanley/f23-CS195NLP/blob/main/F2_4_MarkovModels.ipynb)
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
* [Tokenization](F3_1_Tokenization.ipynb) [<img src="https://colab.research.google.com/assets/colab-badge.svg" width="70px">](https://colab.research.google.com/github/ericmanley/f23-CS195NLP/blob/main/F3_1_Tokenization.ipynb)
* Further Reading:
    - [Python `requests` library quickstart](https://requests.readthedocs.io/en/latest/user/quickstart/)
    - [Beautiful Soup documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
    - [GPT Tokenizer Illustration](https://platform.openai.com/tokenizer)
    - [Python `split` method](https://docs.python.org/3/library/stdtypes.html#str.split)
    - [Hugging Face Byte-Pair Encoding tokenization](https://huggingface.co/learn/nlp-course/chapter6/5?fw=pt)
    - [Hugging Face WordPiece tokenization](https://huggingface.co/learn/nlp-course/chapter6/6?fw=pt)