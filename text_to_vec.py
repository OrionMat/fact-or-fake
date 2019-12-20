import string
from word2number import w2n
import nltk
from nltk.corpus import words as corpus_words
from nltk.corpus import wordnet 
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nlp = spacy.load("en_core_web_sm")
sentiment_analyzer = SentimentIntensityAnalyzer()
word_ID_counter = 0
english_words = set(corpus_words.words())
word_ID_dict = {}
POS_tag_dict = {"NOUN" : 1, "PROPN" : 1, "PRON" : 1,
                "VERB" : 2, "ADV" : 2, "AUX" : 2,
                "ADJ" : 3, "NUM" : 4, "DET" : 5, "ADP" : 6,
                "CONJ" : 7, "CCONJ" : 7, "SCONJ" : 7, 
                "INTJ" : 8, "SYM" : 9}
entity_type_dict = {"DATE" : 1, "TIME" : 1, 
                    "PERCENT" : 2, "MONEY" : 2, "QUANTITY" : 2, 
                    "ORDINAL" : 2, "CARDINAL" : 2}
dependency_dict = {"nsubj" : 1, "pobj" : 2, "amod" : 3, 
                   "det" : 4, "ROOT" : 5, "prep" : 6, 
                   "advmod" : 7, "advcl" : 8}
negation_prefixes = ["a", "dis", "il", "im", "in", "ir", "non", 
                     "un", "mis", "mal", "anti", "de", "under",
                     "semi", "mini", "ex", "sub", "infra"] 

def remove_stop_words(wordList):
    """ takes a word list (str) and removes any stop words """
    with open('stop_words.txt') as f:
        stopWords = [line.rstrip('\n') for line in f]
    cleanedList = [word for word in wordList if word not in stopWords]
    return cleanedList

def to_words_lemmas(doc):
    """ takes a spaCy (object) and returns a list of lemma words (str list) minus punctuation """
    wordList = []
    for token in doc:
        if token.lemma_ not in string.punctuation:
            wordList += [token.lemma_.strip('\n')]
    return wordList

def jaccard_simularity(A, B): 
    """ computes Jaccard Similarity between word groups A and B """
    try:
        setA = set(A)
        setB = set(B)
        return float(len(setA & setB) / len(setA | setB))
    except:
        return 0.0

def check_prefix_negation(word, word_sentiment):
    """ checks internal negation from prefixes """
    for prefix in negation_prefixes:
        if word.startswith(prefix):
            fix_word = word[len(prefix):]
            if fix_word in english_words:
                # antonym check
                for synset in wordnet.synsets(fix_word):
                    for lemma in synset.lemmas():
                        if lemma.antonyms():
                            antonym = lemma.antonyms()[0].name()
                            if antonym == word:
                                return True
                # sentiment check
                sentiment_dict = sentiment_analyzer.polarity_scores(fix_word)
                fix_sentiment = sentiment_dict['compound']
                if word_sentiment != 0 and fix_sentiment != 0:
                    if word_sentiment * fix_sentiment < 0.0:
                        return True
    return False

def to_base_vector(token, word_ID_dict, external_negation):
    """ inputs are: 
        * spaCy token (object)
        * word IDs (dict)
        * external neation (boolean)
    returns a list of elements to form a vector which represents the token/word:
        * word ID (int)
        * POS tag (int)
        * entity type (int)
        * dependency (int)
        * sentiment (float)
        * is negated (Boolean)
    """

    global word_ID_counter
    is_negated = None
    word_ID = None
    POS_ID = None
    entity_ID = None
    dep_ID = None
    sentiment = None
    lemma = token.lemma_.casefold()
    POS_tag = token.pos_
    entity_type = token.ent_type_
    dependency = token.dep_

    # handle word ID
    if lemma in word_ID_dict.keys():
        word_ID = word_ID_dict[lemma]
    elif POS_tag == "NUM":
        num = w2n.word_to_num(lemma)
        num_str = str(num)
        if num_str in word_ID_dict.keys():
            word_ID = word_ID_dict[num_str]
        else:
            word_ID = word_ID_counter
            word_ID_dict[str(num)] = word_ID
            word_ID_counter += 1
    else:
        word_ID = word_ID_counter
        word_ID_dict[lemma] = word_ID
        word_ID_counter += 1

    # handle POS ID
    if POS_tag in POS_tag_dict.keys():
        POS_ID = POS_tag_dict[POS_tag]
    else:
        POS_ID = 10

    # handle entity ID
    if entity_type in entity_type_dict.keys():
        entity_ID = entity_type_dict[entity_type]
    else:
        entity_ID = 3

    # handle dependency
    if dependency in dependency_dict.keys():
        dep_ID = dependency_dict[dependency]
    else:
        dep_ID = 9

    # handle sentiment
    sentiment_dict = sentiment_analyzer.polarity_scores(lemma)
    sentiment = sentiment_dict['compound']

    # handle negation
    if external_negation: 
        is_negated = True
    else:
        is_negated = check_prefix_negation(lemma, sentiment)

    return [word_ID, POS_ID, entity_ID, dep_ID], sentiment, is_negated

doc = nlp(""" unimportant important agree disagree comfort discomfort legal illegal legible illegible mobile immobile moral immoral
        3 killed in car crash 
        Three men were killed in a horrific car crash early saturday morning.
        Three army soldiers have been killed in a highway accident early saturday morning on highway 24 in San Francisco when their red 2009 Nissan Versa slammed into a tree, according to the California Highway Patrol.""")
for token in doc:
    print("{:-<10} {}, {}".format(token.lemma_, " --> ", to_base_vector(token, word_ID_dict, False)))
