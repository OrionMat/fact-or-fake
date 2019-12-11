import string
from word2number import w2n
import spacy
nlp = spacy.load("en_core_web_sm")



def removeStopWords(wordList):
    """ takes a word list (str) and removes any stop words """
    with open('stop_words.txt') as f:
        stopWords = [line.rstrip('\n') for line in f]
    cleanedList = [word for word in wordList if word not in stopWords]
    return cleanedList

def sentenceToWords(sentence):
    """ takes a spaCy sentence (object) and returns a list of lemma words (str list) """
    wordList = []
    for token in sentence:
        if token.lemma_ not in string.punctuation:
            wordList += [token.lemma_.strip('\n')]
    return wordList

def getJaccardSim(A, B): 
    """ computes Jaccard Similarity between word groups A and B """
    try:
        setA = set(A)
        setB = set(B)
        return float(len(setA & setB) / len(setA | setB))
    except:
        return 0.0

def statementToPOS(statement):
    """ takes a spaCy doc object and returns:
        * list of POS_ID words (str list) 
        * dict of Lemma : POS_ID
    """
    POSdict = {}
    POSList = []
    nounID = 0
    verbID = 0
    adjID = 0 
    numID = 0 
    detID = 0
    adpID = 0
    conjID = 0
    intjID = 0
    symID = 0
    xID = 0
    for token in statement:
        if token.lemma_ in POSdict.keys():
            POSList += [POSdict[token.lemma_]]
        elif token.pos_ in {"NOUN", "PROPN", "PRON"}:
            POSList += ["NOUN" + "_" + str(nounID)]
            POSdict[token.lemma_] = "NOUN" + "_" + str(nounID)
            nounID += 1
        elif token.pos_ in {"VERB", "ADV", "AUX"}:
            POSList += ["VERB" + "_" + str(verbID)]
            POSdict[token.lemma_] = "VERB" + "_" + str(verbID)
            verbID += 1
        elif token.pos_ in {"ADJ"}:
            POSList += ["ADJ" + "_" + str(adjID)]
            POSdict[token.lemma_] = "ADJ" + "_" + str(adjID)
            adjID += 1
        elif token.pos_ in {"NUM"}:
            num = w2n.word_to_num(token.lemma_)
            if str(num) in POSdict.keys():
                POSList += [POSdict[str(num)]]
            else:
                POSList += ["NUM" + "_" + str(numID)]
                POSdict[str(num)] = "NUM" + "_" + str(numID)
                numID += 1
        elif token.pos_ in {"DET"}:
            POSList += ["DET" + "_" + str(detID)]
            POSdict[token.lemma_] = "DET" + "_" + str(detID)
            detID += 1
        elif token.pos_ in {"ADP"}:
            POSList += ["ADP" + "_" + str(adpID)]
            POSdict[token.lemma_] = "ADP" + "_" + str(adpID)
            adpID += 1
        elif token.pos_ in {"CONJ", "CCONJ", "SCONJ"}:
            POSList += ["CONJ" + "_" + str(conjID)]
            POSdict[token.lemma_] = "CONJ" + "_" + str(conjID)
            conjID += 1
        elif token.pos_ in {"INTJ"}:
            POSList += ["INTJ" + "_" + str(intjID)]
            POSdict[token.lemma_] = "INTJ" + "_" + str(intjID)
            intjID += 1
        elif token.pos_ in {"SYM"}:
            POSList += ["SYM" + "_" + str(symID)]
            POSdict[token.lemma_] = "SYM" + "_" + str(symID)
            symID += 1
        else:
            POSList += ["X" + "_" + str(xID)]
            POSdict[token.lemma_] = "X" + "_" + str(xID)
            xID += 1
    return POSList, POSdict


    
# gets statement and article as strings
with open('test_statement.txt', encoding='utf-8') as f:
    statement = f.read()
with open('test_article.txt', encoding='utf-8') as f:
    article = f.read()

# process statement and article with spaCy
docStatement = nlp(statement)
docArticle = nlp(article)

# # gets articles three most similar sentences to statment (lemmatized -> jaccard score)
# statementWords = [token.lemma_ for token in docStatement if token.lemma_ not in string.punctuation]
# statementSubjs = [token.lemma_ for token in docStatement if token.dep_ == "nsubj"]
# print("statement subjs: ", statementSubjs)
# statementNoStops = removeStopWords(statementWords)
# artlemmaList = []
# jaccardScoreList = []
# jaccardNoStopsList = []
# jaccardSubjsList = []
# for sentence in docArticle.sents:
#     # computes jaccard score for word lemmas
#     artWords = sentenceToWords(sentence)
#     jaccardScore = getJaccardSim(statementWords, artWords)
#     jaccardScoreList += [jaccardScore]
#     # computes jaccard score for word lemmas with no stop words
#     articleNoStops = removeStopWords(artWords)
#     jaccardNoStops = getJaccardSim(statementNoStops, articleNoStops)
#     jaccardNoStopsList += [jaccardNoStops]
#     # computes jaccard score for subject
#     jaccardSubjs = getJaccardSim(statementSubjs, artWords)
#     jaccardSubjsList += [jaccardSubjs]
#     # stores the lemma sentence
#     artlemmaList += [artWords]

    

# threeMostSimilarSents = sorted(zip(jaccardScoreList, artlemmaList), reverse=True)[:3]
# threeMostSimilarNoStops = sorted(zip(jaccardNoStopsList, artlemmaList), reverse=True)[:3]
# threeMostSimilarSubjs = sorted(zip(jaccardSubjsList, artlemmaList), reverse=True)[:3]


# print("straight:\n", threeMostSimilarSents, "\n")
# print("No Stops:\n", threeMostSimilarNoStops, "\n")
# print("Subject:\n", threeMostSimilarSubjs, "\n")


print(statementToPOS(docStatement))