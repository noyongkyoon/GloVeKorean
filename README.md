# GloVe-Korean
word embeddings for the Korean language modeled after GloVe

## GloVe is a set of algorithms for representing words with a vector of numbers.
Pennington, Socher, and Manning (2014) proposed an unsupervised learning
algorithm for representing words in terms of vectors of numbers. Called GloVe,
this algorithm is trained on word-word cooccurrence statistics from a corpus
and is similar to Word2Vec, of Mikolov (2013), in that its representations encode interesting
linear substructures of the word vector space.

## NLP in the other languages is lagging behind. Due to what?
While the state-of-the-art in NLP (and in AI in general) is quite remarkable and
gives one an impression that it matches the ordinary man's competence, this is
the case when the language in question is nothing other than the English
language.

The reason why the various algorithms (like RNN's and Transformers) can be easily trained on virtually raw corpora when the texts are in English is that texts in this language happen to be written in
accordance with a strict but reasonable convention: each word is written with spaces at
both ends, thus demarcating the word boundary. The convention has been
established by collective wisdom accummulated for several centuries.

The word as the smallest unit of the syntactic component has not been made
concrete enough to the NLP community. Many meaningful sequences that are 
to be recognized as words on scientific examination are treated no differently
than units smaller than words, in most natural languages. The notion of
the syntactic word is presumably very abstract and practicing linguists
continuously have difficulty classifying each meaningful unit as a word or as an affix in a
language which hasn't been subjected to anything near a thorough scientific analysis.

Korean is just one example. Orthography of the
language does not require every word-boundary to be separated by a space. It
even prohibits some words from being separated, thus giving rise to many
sequences with two or more words in them. (In this respect, Korean is not very different from
Chinese or Japanese, despite its obligatory use of some interword spaces.)

## The corpus used to train the algorithm has been word-segmented beforehand.
There do exist a few segmenters for the Korean language. These are variously
called ``morpheme analyzer'', ``morphological analyzer'', and ``tokenizer''.
It is this component that is error-prone and trouble-ridden. Traditional
linguists working on the language would have different opinions on the status
of many meaningful units. A certain morpheme would be a word according to one
linguist while it is a unit smaller than word to another.

The corpus used for this work is ["문서요약 말뭉치" (Corpus for Document
Summarization)] (corpus.korean.go.kr)
distributed by Korean Institute of the National Language. We put the texts through a word-segmenter,
a version of which has been deployed as a python package,
[_hangul-korean_](pypi.org/project/hangul-korean/), and got a useful, yet not
one hundred percent correct, string of segmented words.

The corpus has about one hundred million words. Roughly three fourths of the
texts are newspaper articles and the rest are journal essays and commentaries.
A small portion, about 6 percent, of the texts are courtroom verdicts. 

## The python code used to train the model
We used [the python code provided by Peng
(2013)](https://github.com/pengyan510/nlp-paper-implementation/tree/master/glove) to train the model.

## Some parameters for this work
Forty epochs were run. The most frequent fifty thousand words were represented
each as a vector of dimension 100. Word-word cooccurrence statistics were
gathered within windows of size 10.

## Prospects
A corpus of one hundred million words is quite small. (Be reminded that [the pretrained English version of 
Pennington, Socher, and Manning (2014)] (https://nlp.stanford.edu/projects/glove/) was obtained from the corpus sizes of six
billion, forty two billion, or eight hundred forty billion tokens.)
The usefulness of this model would certainly fall short of expectations, as there would be
aspects of word meanings that, due to the small size of the training corpus, are not 
reflected in the representations. A good quality word-segmenter is a precondition for a truly useful model. The main culprit behind this suboptimal situation is the absence of a near perfect word segmenter.
