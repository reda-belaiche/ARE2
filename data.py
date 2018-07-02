# -*- coding: utf-8 -*-
import re
import sys
from os import environ, path
from string import punctuation

from watson import get_categories, get_concepts

try:
    from nltk import pos_tag
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords


except ImportError as e:
    print("Can not import NLTK Modules", e)

try:
    environ.update(
            {'SPARK_HOME': '/home/reda/Desktop/spark/spark-2.0.1-bin-hadoop2.7/'})
    spark_home = environ.get('SPARK_HOME')
    sys.path.insert(0, spark_home + "/python")
    sys.path.insert(0, path.join(spark_home, 'python/lib/py4j-0.10.3-src.zip'))

    from pyspark.mllib.regression import LabeledPoint
    from pyspark import SparkContext
    from pyspark.sql import SQLContext
    from pyspark.mllib.feature import HashingTF, IDF

    print("Successfully imported Spark Modules data")

except ImportError as e:
    print("Can not import Spark Modules", e)
# Module-level global variables for the `tokenize` function below
PUNCTUATION = set(punctuation)
STOPWORDS = set(stopwords.words('english'))

sc = SparkContext()
sqlContext = SQLContext(sc)

def get_data():
    """
    Returns a DataFrame containing only the rows containing the description
    :return: DataFrame
    """
    lines = sc.textFile("/home/reda/Desktop/spark/Are2_Reda/data_set/udemy.csv")
    
    header = lines.first()

    lines = lines.filter(lambda line: line != header)

    lines_splitted = lines.map(lambda line: re.sub(r'(?!,\s)(?=,)', '|', line.rstrip()))
    lines_splitted = lines_splitted.map(lambda line: re.sub(r'\|\,', '|', line.rstrip()))
    lines_splitted = lines_splitted.map(lambda line: line.split("|"))
    print lines_splitted.collect()
    lines_splitted = lines_splitted.filter(lambda row: row[6] is not None and row[6] != 'null' and row[6] != u'')

    lines_preprocessed = lines_splitted.map(lambda row: row + preprocess_description(row[6],row[3]))

    print lines_preprocessed.collect()
    # df = lines_preprocessed.toDF(header.split(',') + ['features'])
    # df.show()
    return lines_preprocessed


def get_data2():
    """
    Returns a DataFrame containing only the rows containing the description
    :return: DataFrame
    """
    lines = sc.textFile("somefile.csv")
    header = lines.first()

    lines = lines.filter(lambda line: line != header)

    lines_splitted = lines.map(lambda line: re.sub(r'(?!,\s)(?=,)', ';', line.rstrip()))
    lines_splitted = lines_splitted.map(lambda line: re.sub(r'\;\,', ';', line.rstrip()))
    lines_splitted = lines_splitted.map(lambda line: line.split(";"))
    print lines_splitted.collect()
    lines_splitted = lines_splitted.filter(lambda row: row[0] is not None and row[0] != 'null' and row[0] != u'')

    lines_preprocessed = lines_splitted.map(lambda row: row + preprocess_description(row[0],row[3]))

    print lines_preprocessed.collect()
    # df = lines_preprocessed.toDF(header.split(',') + ['features'])
    # df.show()
    return lines_preprocessed


def tokenize(text):
    tokens = word_tokenize(text)
    return tokens


def remove_punctuation(tokens):
    lowercased = [t.lower() for t in tokens]
    no_punctuation = []
    for word in lowercased:
        punct_removed = ''.join(
                [letter for letter in word if letter not in PUNCTUATION])
        no_punctuation.append(punct_removed)
    return no_punctuation


def remove_stop_words(tokens):
    return [w for w in tokens if w not in STOPWORDS]


def preprocess_description(text,rating):
    categories = get_categories(text)
    concepts = get_concepts(text)
    #write_to_file(text,rating)
    tokens = tokenize(text)
    tokens = tokens + categories + concepts
    no_ponctuation = remove_punctuation(tokens)
    no_stop_words = remove_stop_words(no_ponctuation)
    return [no_stop_words]


def tf_idf(documents):
    hashingTF = HashingTF(100000)
    htf = documents.map(lambda doc: hashingTF.transform(doc[-1]))
    try:
    	labels = documents.map(lambda doc: float(doc[3]))
    except:
	labels = documents.map(lambda doc: float(doc[2]))

    # While applying HashingTF only needs a single pass to the data, applying IDF needs two passes:
    # First to compute the IDF vector and second to scale the term frequencies by IDF.
    idf = IDF().fit(htf)
    tfidf = idf.transform(htf)
    return tfidf.zip(labels)


def get_features():
    features = tf_idf(get_data())
    features = features.map(lambda (feature, label): LabeledPoint(label, feature))
    train, test = features.randomSplit([.7, .3])
    return train, test


def write_to_file(text,rating):
    conc = ""
    for x in get_concepts(text):
	conc += x+"_"
    cat = ""
    for x in get_categories(text):
	cat += x+"_"
    res = text + ';' + str(conc) + ';' + str(cat) +";"+str(rating)
    with open('somefile.csv', 'a') as the_file:
        the_file.write(res + "\n")

if __name__ == '__main__':
    print tf_idf(get_data()).collect()




