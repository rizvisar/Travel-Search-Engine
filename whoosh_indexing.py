import os
import pandas as pd
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
import sys
from whoosh.qparser import QueryParser
from whoosh import scoring
from whoosh.index import open_dir
from operator import itemgetter
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re
from nltk.stem.snowball import SnowballStemmer
#nltk.download('punkt')
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

root = sys.argv[1]
indexoppath = sys.argv[2]
urllistpath = sys.argv[3]
query_str = sys.argv[4]


URLMap , inlinkmap , outlinkMap , pagerankmap, hitsscore = {}, {} , {} , {}, {}

def createURLMap(urllistfile):
    df = pd.read_csv(urllistfile)
    for i in range(0, len(df)):
        url,docid, parentidlist = df.iloc[i]['URL'],  str(df.iloc[i]['DOCID']) ,  eval(str(df.iloc[i]['PARENTID']).replace('{', '[').replace('}', ']'))
        if  docid not in URLMap:
            URLMap[docid] = {'URL': url, 'ParentList': parentidlist}

def createInlinkMap():
    for key in URLMap.keys():
        inlinkmap[key] = URLMap[key]['ParentList']

def createOutlinkMap():
    for docID in URLMap.keys():
        parentIDlist = URLMap[docID]['ParentList']
        for parent in parentIDlist:
            if parent not in outlinkMap:
                childList = set()
                childList.add(docID)
                outlinkMap[parent] = childList
            else:
                childList = outlinkMap[parent]
                childList.add(docID)
                outlinkMap[parent] = childList


def createSearchableData(root):
    schema = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT, URL=TEXT(stored=True) , textdata=TEXT(stored=True))
    if not os.path.exists(indexoppath):
        os.mkdir(indexoppath)
    ix = create_in(indexoppath, schema)
    writer = ix.writer()
    filepaths = [os.path.join(root, i) for i in os.listdir(root)]
    for path in filepaths:
        fp = open(path, 'r')
        text = fp.read()
        doc_title = path.split("/")[-1]
        if doc_title in URLMap.keys():
            url = str(URLMap[doc_title]['URL'])
        else:
            print(doc_title + ' not in url map')
            url = 'default'
        writer.add_document(title=doc_title, path=path,content=text, URL=url ,  textdata=text)
        fp.close()
    writer.commit()


def searchIndex(queryStr, index, topN  ):
    resultdict = {}
    ix = open_dir(index)
    with ix.searcher(weighting=scoring.Frequency) as searcher:
        query = QueryParser("content", ix.schema).parse(queryStr)
        results = searcher.search(query, limit=topN)
        for i in range(0, topN):
            resultdocid, resulturl = results[i]['title'], results[i]['URL']
            resultdict[resultdocid] = resulturl
    return resultdict


def searchIndexedData(queryStr, index, topN  ):
    resultURLList = []
    resultdict = searchIndex(queryStr, index, topN  )
    for key , entry in resultdict.items():
        resultURLList.append(entry)
    return  resultURLList


def calculatePageRank(inlinkMap, outlinkMap):
        pagerankmap = {}
        for doc in inlinkMap.keys():
            pagerankmap[doc] = 0.15
            for inlink in inlinkMap[doc]:
                pagerankmap[inlink] = 0.15
        iter = 50
        while (iter > 0):
            for doc in inlinkMap.keys():
                pr = 0
                if len(inlinkMap[doc]) > 0:
                    for inlink in inlinkMap[doc]:
                        if inlink > 0 and inlink in outlinkMap.keys():
                            pr_inlink = pagerankmap[inlink]
                            c_inlink = len(outlinkMap[inlink])
                            pr += (pr_inlink / c_inlink)
                    pr *= 0.85
                    pagerankmap[doc] = 0.15 + pr
                else:
                    pagerankmap[doc] = 1 / len(inlinkMap.keys())
            iter -= 1
        return pagerankmap

def initializehubscores(resultMap):
    for docID in resultMap.keys():
        hitsscore[docID] = {'hubscore': 1.0, 'authscore': 1.0}
        inlinklist = inlinkmap[docID]
        if len(inlinklist) > 0 and inlinklist[0] != -1:
            for inlink in inlinklist:
                if inlink!= -1:
                    hitsscore[str(inlink)] = {'hubscore': 1.0, 'authscore': 1.0}


def calchubscores():
    for k in range(0, 25):
        for docid in hitsscore.keys():
            authscore, hubscore = 1.0, 1.0
            inlinklist = inlinkmap[docid] # this list can be empty of hold just 0
            outlinklist = []
            if len(inlinklist):
                for inlink in inlinklist:
                    if inlink != -1:
                        authscore += hitsscore[str(inlink)]['hubscore']
            if docid in outlinkMap:
                outlinklist = outlinkMap[docid]
                for outlink in outlinklist:
                    hubscore += hitsscore[outlink]['authscore']
            hitsscore[docid]['hubscore'] = hubscore
            hitsscore[docid]['authscore'] = authscore
        normalizehitscores()
        k += 1


def normalizehitscores():
    minhub = min(hitsscore, key=lambda x: float(hitsscore[x]['hubscore']))
    maxhub = max(hitsscore, key=lambda x: float(hitsscore[x]['hubscore']))
    minauth = min(hitsscore, key=lambda x: float(hitsscore[x]['authscore']))
    maxauth = max(hitsscore, key=lambda x: float(hitsscore[x]['authscore']))
    for docid in hitsscore.keys():
        if maxhub!=minhub:
            normalizedhubscore = (float(hitsscore[docid]['hubscore']) - float(minhub)) / (float(maxhub) - float(minhub))
            hitsscore[docid]['hubscore'] = normalizedhubscore
            normalizedauthscore = (float(hitsscore[docid]['authscore']) - float(minauth)) / (float(maxauth) - float(minauth))
            hitsscore[docid]['authscore'] = normalizedauthscore

def getResultsByPagerank(query_str, indexoppath, topN):
    resultdict = searchIndex(query_str, indexoppath, topN)
    resultpagerank = {}
    resultURL = []
    for key in resultdict.keys():
        resultpagerank[key] = pagerankmap[key]
    for docid, value  in sorted(resultpagerank.items(), key=itemgetter(1), reverse=True):
        resultURL.append(resultdict[docid])
    return resultURL


def getResultsByHitScore(query_str, indexoppath, topN):
    resultdict = searchIndex(query_str, indexoppath, topN)
    resulthubscore = {}
    resultauthorityscore = {}
    resultURLhubscore, resultURLauthscore = [], []
    initializehubscores(resultdict)
    calchubscores()
    for key, entry in hitsscore.items():
        resulthubscore[key] = entry['hubscore']
        resultauthorityscore[key] = entry['authscore']
    for docid, value in sorted(resulthubscore.items(), key=itemgetter(1), reverse=True):
        resultURLhubscore.append(URLMap[docid]['URL'])
    for docid, value in sorted(resultauthorityscore.items(), key=itemgetter(1), reverse=True):
        resultURLauthscore.append(URLMap[docid]['URL'])

    return resultURLhubscore, resultURLauthscore

createURLMap(urllistpath)
createInlinkMap()
createOutlinkMap()

pagerankmap = calculatePageRank(inlinkmap, outlinkMap)

#createSearchableData(root)

result = searchIndexedData(query_str, indexoppath, 10)
print(result)

resultPagerank = getResultsByPagerank(query_str, indexoppath, 10)
print(resultPagerank)

resulthub, resultauth = getResultsByHitScore(query_str, indexoppath, 10)
print(resulthub)
print(resultauth)
