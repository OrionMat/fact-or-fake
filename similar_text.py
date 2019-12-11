import string
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
        

    
# gets statement and article as strings
with open('test_statement.txt', encoding='utf-8') as f:
    statement = f.read()
with open('test_article.txt', encoding='utf-8') as f:
    article = f.read()

# process statement and article with spaCy
docStatement = nlp(statement)
docArticle = nlp(article)

# gets articles three most similar sentences to statment (lemmatized -> jaccard score)
statementWords = [token.lemma_ for token in docStatement if token.lemma_ not in string.punctuation]
statementSubjs = [token.lemma_ for token in docStatement if token.dep_ == "nsubj"]
print("statement subjs: ", statementSubjs)
statementNoStops = removeStopWords(statementWords)
artlemmaList = []
jaccardScoreList = []
jaccardNoStopsList = []
jaccardSubjsList = []
for sentence in docArticle.sents:
    # computes jaccard score for word lemmas
    artWords = sentenceToWords(sentence)
    jaccardScore = getJaccardSim(statementWords, artWords)
    jaccardScoreList += [jaccardScore]
    # computes jaccard score for word lemmas with no stop words
    articleNoStops = removeStopWords(artWords)
    jaccardNoStops = getJaccardSim(statementNoStops, articleNoStops)
    jaccardNoStopsList += [jaccardNoStops]
    # computes jaccard score for subject
    jaccardSubjs = getJaccardSim(statementSubjs, artWords)
    jaccardSubjsList += [jaccardSubjs]
    # stores the lemma sentence
    artlemmaList += [artWords]

    

threeMostSimilarSents = sorted(zip(jaccardScoreList, artlemmaList), reverse=True)[:3]
threeMostSimilarNoStops = sorted(zip(jaccardNoStopsList, artlemmaList), reverse=True)[:3]
threeMostSimilarSubjs = sorted(zip(jaccardSubjsList, artlemmaList), reverse=True)[:3]


print("straight:\n", threeMostSimilarSents, "\n")
print("No Stops:\n", threeMostSimilarNoStops, "\n")
print("Subject:\n", threeMostSimilarSubjs, "\n")