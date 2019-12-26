import string
import numpy as np
from word2number import w2n
import nltk
from nltk.corpus import words as corpus_words
from nltk.corpus import wordnet 
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nlp = spacy.load('en_core_web_lg')
sentiment_analyzer = SentimentIntensityAnalyzer()
max_words_in_sentence = 50
word_ID_counter = 0
english_words = set(corpus_words.words())
word_ID_dict = {}
POS_tag_dict = {'NOUN' : 1, 'PROPN' : 1, 'PRON' : 1,
                'VERB' : 2, 'ADV' : 2, 'AUX' : 2,
                'ADJ' : 3, 'NUM' : 4, 'DET' : 5, 'ADP' : 6,
                'CONJ' : 7, 'CCONJ' : 7, 'SCONJ' : 7, 
                'INTJ' : 8, 'SYM' : 9}
entity_type_dict = {'DATE' : 1, 'TIME' : 1, 
                    'PERCENT' : 2, 'MONEY' : 2, 'QUANTITY' : 2, 
                    'ORDINAL' : 2, 'CARDINAL' : 2}
dependency_dict = {'nsubj' : 1, 'pobj' : 2, 'amod' : 3, 
                   'det' : 4, 'ROOT' : 5, 'prep' : 6, 
                   'advmod' : 7, 'advcl' : 8}
negation_prefixes = ['a', 'ab', 'an', 'anti', 'de', 'dis', 'ex', 
                     'ig', 'il', 'im', 'in', 'infra', 'ir', 'mal', 
                     'mini', 'mis', 'non', 'semi', 'sub', 'un', 'under'] 
negation_words = {'aint', 'arent', 'barely', 'cannot', 'cant', 'couldnt', 'darent', 
                  'despite', 'didnt', 'doesnt', 'dont', 'hadnt', 'hardly', 'hasnt', 
                  'havent', 'isnt', 'mightnt', 'mustnt', 'neednt', 
                  'neither', 'never', 'nil', 'no', 'nobody', 'non', 'none', 
                  'nope', 'nor', 'not', 'nothing', 'nowhere', 'oughtnt', 
                  'rarely', 'refuse', 'reject', 'retract', 'scarcely', 'seldom', 'shant', 
                  'shouldnt', 'subside', 'uh-uh', 'uhuh', 'wasnt', 'werent', 
                  'without', 'withoutdeny', 'wont', 'wouldnt'}

def load_text_data(statement_file, article_file):
    """ reads statement and article into strings from their text files """
    with open(statement_file, encoding='utf-8') as f:
        statement = f.read()
    with open(article_file, encoding='utf-8') as f:
        article = f.read()
    return statement, article

def remove_stop_words(word_list):
    """ takes a word list (str) and removes any stop words """
    with open('stop_words.txt') as f:
        stop_words = [line.rstrip('\n') for line in f]
    cleaned_list = [word for word in word_list if word not in stop_words]
    return cleaned_list

def jaccard_simularity(A, B): 
    """ computes Jaccard Similarity between word groups A and B """
    try:
        setA = set(A)
        setB = set(B)
        return float(len(setA & setB) / len(setA | setB))
    except:
        return 0.0

def to_word_lemmas(sentence):
    """ takes a spaCy sentence (object) and returns a list of lemma words (str list)
        * list contains no punctuation or new lines """
    word_list = []
    for token in sentence:
        if token.lemma_ not in string.punctuation:
            word_list += [token.lemma_.strip('\n')]
            word_list = list(filter(None, word_list))
    return word_list

def get_similar_sentences(doc_statement, doc_article):
    """ gets articles five most similar sentences to statement 
        1: lemmatized -> jaccard score
                      -> remove stop words -> jaccard score
        2: Universal Sentence Encoder similarity
        weights similar sentences to get a compound similarity score"""

    statement_words = to_word_lemmas(doc_statement)
    statement_no_stops = remove_stop_words(statement_words)
    article_lemma_sentences = []
    jaccard_score_list = []
    jaccard_no_stops_list = []

    for sentence in doc_article.sents:
        article_words = to_word_lemmas(sentence)
        article_no_Stops = remove_stop_words(article_words)
        article_lemma_sentences += [article_words]
        # computes jaccard score for word lemmas
        jaccard_score = jaccard_simularity(statement_words, article_words)
        jaccard_score_list += [jaccard_score]
        # computes jaccard score for word lemmas with no stop words
        jaccard_no_stops = jaccard_simularity(statement_no_stops, article_no_Stops)
        jaccard_no_stops_list += [jaccard_no_stops]

    five_most_similar_raw = sorted(zip(jaccard_score_list, article_lemma_sentences), reverse=True)[:5]
    five_most_similar_no_stops = sorted(zip(jaccard_no_stops_list, article_lemma_sentences), reverse=True)[:5]

    print("jaccard similar sentences:")
    for idx in range(5):
        print("raw ", idx, ": ", five_most_similar_raw[idx])
        print("no stops ", idx, ": ", five_most_similar_no_stops[idx])



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
                                return 1
                # sentiment check
                sentiment_dict = sentiment_analyzer.polarity_scores(fix_word)
                fix_sentiment = sentiment_dict['compound']
                if word_sentiment != 0 and fix_sentiment != 0:
                    if word_sentiment * fix_sentiment < 0.0:
                        return 1
    return 0

def word_to_vector(token, word_ID_dict):
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
    self_negation = None
    is_negation_word = None
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

    # handle internal negation
    self_negation = check_prefix_negation(lemma, sentiment)

    # handel case where word is nation word
    if lemma in negation_words:
        is_negation_word = 1
    else:
        is_negation_word = 0

    base_array = np.array([word_ID, POS_ID, entity_ID, dep_ID, sentiment, self_negation, is_negation_word])
    standard_array = token.vector
    word_array = np.concatenate((base_array, standard_array), axis=0)
    return word_array

# get statement and article and process them with spacy
statement, article = load_text_data('test_statement.txt', 'test_article.txt')
doc_statement = nlp(statement)
doc_article = nlp(article)

#  get article's five most similar sentences to statement
five_similar_sentences = get_similar_sentences(doc_statement, doc_article)

# doc = nlp(""" Scarcely Barely barely kjashd
#         couldn't can't aren't ain't isn't didn't won't 'wouldn't
#         unimportant important agree disagree comfort discomfort legal illegal legible illegible mobile immobile moral immoral
#         3 killed in car crash 
#         Three men were killed in a horrific car crash early saturday morning.
#         Three army soldiers have been killed in a highway accident early saturday morning on highway 24 in San Francisco when their red 2009 Nissan Versa slammed into a tree, according to the California Highway Patrol.""")
# for token in doc:
#     vector = word_to_vector(token, word_ID_dict)
#     print("{:-<10} {} {} {} {}".format(token.lemma_, " --> ", vector[0:3], " shape: ", vector.shape))
