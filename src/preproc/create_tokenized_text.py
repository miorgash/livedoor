import sys
sys.path.append('/tmp/work/livedoor/src')
import os
import pandas as pd
from const import *
from datetime import datetime
from tqdm import tqdm
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from mecab_trial import tokenize

if __name__ == '__main__':
    splits = ['train', 'test']

    for split in splits:
        print(datetime.now(), f'{split} start')
        # output
        file_o = os.path.join(DIR_BIN, f'corpus.{split}.tokenized.pkl')
        if os.path.isfile(file_o):
            print(datetime.now(), f'{split} file exists: {file_o}')
        else:

            # load
            file_i = os.path.join(DIR_DATA, f'{split}.csv')
            # corpus = pd.read_csv(file_i)#.head()

            # print(corpus.head())

            # initialize spark session
            # conf = SparkConf().setMaster('local').setAppName('MyApp')
            # sc = SparkContext.getOrCreate(conf = conf)
            # sc.addFile('/tmp/work/livedoor/src/mecab_trial.py')

            # tokenize
            # rdd = sc.parallelize(corpus.values)
            # rdd = rdd.map(lambda x: (x[0], [t for t in tokenize(x[1])]))
            # print(rdd.first())

            # initialize spark
            spark = SparkSession.builder \
                .master('local') \
                .appName('myApp') \
                .getOrCreate()
            spark.sparkContext.addPyFile('/tmp/work/livedoor/src/mecab_trial.py')
            print('spark init done')

            # load
            df_corpus = spark.read.csv(file_i, header = True)
            print(df_corpus.count())

            # tokenize
            df_corpus2 = df_corpus \
                .rdd \
                .map(lambda x: (x[0], [t for t in tokenize(x[1])])) \
                .toDF(df_corpus.columns)
            print(df_corpus2.count())
            df_corpus2.show()
            # persistance
            # rdd.saveAsTextFile('hoge.txt')

            print(datetime.now(), f'{split} tokenizing done')

        print(datetime.now(), f'{split} done')