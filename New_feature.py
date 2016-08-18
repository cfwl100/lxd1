#!/usr/bin/pythonw
# -*- coding: utf-8 -*-

# 1. read CSV file(wordcount), get the matrix
# 2. compute TF-IDF
# 3. sort by TF-IDF to get the feature of each Question
# 4. integrate all words of feature
# 5. count the word_freq

# input : csv(wordcount)
# output: csv(New_feature)

import csv
from numpy import array
from sklearn.feature_extraction.text import TfidfTransformer

#input file
filename = "wordcount_simplified_utf-8_1.csv"
osencoding = "utf-8"
# osencoding = "sjis"
# osencoding = "gb18030"
sort_num = 20 # 取排序后的前20个
threshold_set = 0.1 # 设定输出TF-IDF阈值
def main():
    list_row=[]
    head = []
    with open(filename, newline='', encoding=osencoding) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        i = 0
        for line in spamreader:
            if i == 0:
                head.append(line)
                i += 1
            else:
                for index, item in enumerate(line):
                    line[index] = eval(item)
                list_row.append(line)
        tf_cor = array(list_row)
    print('computing tfidf value...')
    tfidf = tfidfcount(tf_cor)
    print('sorting...')
    sortedTF = sort_TF(head[0],tfidf)
    print('After sorting!')
    print('saving to csv!')
    savetocsv(head[0],sortedTF)
    print('After saving to csv')

def tfidfcount(tf_cor):
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(tf_cor)
    tfidf = tfidf.toarray()
    return tfidf

def sort_TF(word,weight):
    tfidfDict = {}
    sortTF = []
    for i in range(len(weight)):
        for j in range(len(word)):
            getWord = word[j]
            getValue = weight[i][j]
            tfidfDict.update({getWord: getValue})
        sorted_tfidf = sorted(tfidfDict.items(),key=lambda e: e[1], reverse=True)
        sorted_tfidf = sorted_tfidf[0:sort_num]
        sortTF.append(sorted_tfidf)
    sortTF = array(sortTF)
    return sortTF

def savetocsv(word_list,vectors):
    matrix = []
    for i in range(len(vectors)):
        line = []
        for word in word_list:
            flag = 0
            for j in range(len(vectors[i])):
                vector = vectors[i][j]
                value = float(vector[1])
                if word in vector:
                    if value >= threshold_set:
                        line.append(value)
                        flag = 1
                        break
            if flag == 0:
                line.append(0)
        matrix.append(array(line))
    matrix = array(matrix)

    with open("wordcount_TF_IDF" + osencoding + ".csv", "w", newline="", encoding=osencoding) as datacsv:
        csvwriter = csv.writer(datacsv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(word_list)
        for vector in matrix:
            csvwriter.writerow(list(vector))

if __name__ == '__main__':
    main()
