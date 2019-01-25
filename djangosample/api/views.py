# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.core import serializers
from django.conf import settings
import requests
from bs4 import BeautifulSoup as Soup
import re

from whoosh.qparser import QueryParser
from whoosh import scoring
from whoosh.index import open_dir
from operator import itemgetter
import csv as csv
import nltk
from nltk import PorterStemmer
from nltk.corpus import stopwords
import math
import string
import itertools

#C:\Users\mohan\OneDrive\Documents\Latest_files_IR\Latest Files
#C:\Users\mohan\OneDrive\Documents\Latest_files_IR\Latest Files
#C:\Users\mohan\OneDrive\Documents\Latest_files_IR\Latest Files
urllistfile = 'C:\\Users\\mohan\\OneDrive\\Documents\\Latest_files_IR\\Latest Files\\urllistDict.csv'
stopwords = ['a', 'all', 'an', 'and', 'any', 'are', 'as', 'be', 'been', 'but', 'by', 'few', 'for', 'have', 'he',
                 'her',
                 'here', 'him', 'his', 'how', 'i', 'in', 'is', 'it', 'its', 'many', 'me', 'my', 'none', 'of', 'on',
                 'or', 'our',
                 'she', 'some', 'the', 'their', 'them', 'there', 'they', 'that', 'this', 'us', 'was', 'what', 'when',
                 'where',
                 'which', 'who', 'why', 'will', 'with', 'you', 'your']

indexPath = 'C:\\Users\\mohan\\OneDrive\\Documents\\Latest_files_IR\\Latest Files\\index\\index2\\'
#Stemming Query
def getStemmedQuery(queryStr):
    stemmed_query = []
    stemmed_querystr = ""
    ps = PorterStemmer()
    for word in queryStr.split(' '):
        if word not in stopwords:
            stemmed_query.append(ps.stem(word))
    stemmed_querystr = ' '.join(stemmed_query)
    return stemmed_querystr

#trim URL /
def refineURL(urllist):
    newUrlList =[]
    for url in urllist:
        if url[-1]=='/':
            newUrlList.append(url[:-1])
        else:
            newUrlList.append(url)
    return newUrlList



#Helper method for google and bing
def refineLink(link):
    clean = re.compile('<.*?>')
    return [re.sub(clean, '', str(l)) for l in link]

#Method for bing results
def getBingResult(query):
    try:
        query = str(query)
        for start in range(0, 10):
            url = "http://www.bing.com/search?q=" + query + "&start=" + str(start * 10)
        page = requests.get(url)
        soup = Soup(page.content, features='html')
        links = soup.findAll('cite')
        refined_links = refineLink(links)
        return refined_links
    except ValueError as e:
        return []

#Method for google results
def getGoogleResult(query):
    try:
        query = str(query)
        for start in range(0, 10):
            url = "http://www.google.com/search?q=" + query + "&start=" + str(start * 10)
        page = requests.get(url)
        soup = Soup(page.content, features='html')
        links = soup.findAll('cite')
        refined_links = refineLink(links)
        return refined_links
    except ValueError as e:
        return []

#method for our search engine -Whoosh Indexer
def ourSearchEngineResult(query):
    try:
        #indexPath = 'C:\\Users\\mohan\\OneDrive\\Documents\\Index_IR\\index\\index'
        resulturllist = []
        topN = 10
        queryStr = getStemmedQuery(str(query))

        resulturllist = []
        ix = open_dir(indexPath)
        with ix.searcher(weighting=scoring.Frequency) as searcher:
            query = QueryParser("content", ix.schema).parse(queryStr)
            results = searcher.search(query, limit=topN)
            # print(type(results))
            numOfResults = len(results)
            if numOfResults >= topN:
                resultIndex = topN
            else:
                resultIndex = numOfResults - 1
            if resultIndex > 0:
                resulturllist = [results[i]['URL'] for i in range(0, resultIndex)]
            else:
                resulturllist = []
    except ValueError as e:
        return []
    resulturllist = [str(l) for l in resulturllist]
    return resulturllist

#Helper for page rank and hits - searchindex
def searchIndex(queryStr, index, topN):
    resultdict = {}
    ix = open_dir(index)
    with ix.searcher(weighting=scoring.Frequency) as searcher:
        query = QueryParser("content", ix.schema).parse(queryStr)
        results = searcher.search(query, limit=topN)
        for hit in results:
            resultdocid, resulturl = hit['title'], hit['URL']
            resultdict[resultdocid] = resulturl
    return resultdict

#Helper for page rank - PageRankMap
def calculatePageRank(inlinkMap, outlinkMap):
    pagerankmap = {}
    for doc in inlinkMap.keys():
        pagerankmap[doc] = 0.01
        for inlink in inlinkMap[doc]:
            pagerankmap[inlink] = 0.01
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
                pagerankmap[doc] = 0.01 + pr
            else:
                pagerankmap[doc] = 1 / len(inlinkMap.keys())
        iter -= 1
    return pagerankmap

#UrlMap
def createURLMap(urllistfile):
    URLMap = {}
    with open(urllistfile, encoding = 'ISO-8859-1') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            URLMap[row['DOCID']] = {'URL': row['URL'],'ParentList': eval(str(row['PARENTID']).replace('{', '[').replace('}', ']'))}
    return URLMap

#INlinks
def createInlinkMap(URLMap):
    inlinkmap = {}
    for key in URLMap.keys():
        inlinkmap[key] = URLMap[key]['ParentList']
    return inlinkmap

#OUTlinks
def createOutlinkMap(URLMap):
    outlinkMap= {}
    #URLMap =createURLMap(urllistfile)
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
    return outlinkMap

#Method to get Results by page rank
def getResultsByPagerank(query_str, indexoppath, topN):
    query_str = getStemmedQuery(query_str)
    resultdict = searchIndex(query_str, indexoppath,topN)
    URLMap = createURLMap(urllistfile)
    inlinkMap = createInlinkMap(URLMap)
    outlinkMap = createOutlinkMap(URLMap)
    pagerankmap = calculatePageRank(inlinkMap, outlinkMap)
    resultpagerank = {}
    resultURL = []
    for key in resultdict.keys():
        resultpagerank[key] = pagerankmap[key]
    if len(resultpagerank) > 0:
        for docid, value  in sorted(resultpagerank.items(), key=itemgetter(1), reverse=True):
            resultURL.append(resultdict[docid])
    else:
        return []
    return resultURL


###########################HIT SCORES#################################

#normalizehitscore
def normalizehitscores(hitsscore):
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
    return hitsscore

#calculate hub scores
def calchubscores(finalscore,initialscore,inlinkmap,outlinkMap):
    resHitsscore ={}
    for k in range(0, 25):
        for docid in initialscore.keys():
            authscore, hubscore = 1.0, 1.0
            inlinklist = inlinkmap[docid] # this list can be empty of hold just 0
            outlinklist = []
            if len(inlinklist):
                for inlink in inlinklist:
                    if inlink != -1:
                        authscore += finalscore[str(inlink)]['hubscore']
            if docid in outlinkMap:
                outlinklist = outlinkMap[docid]
                for outlink in outlinklist:
                    hubscore += finalscore[outlink]['authscore']
            finalscore[docid]['hubscore'] = hubscore
            finalscore[docid]['authscore'] = authscore
        resHitsscore = normalizehitscores(finalscore)
        k += 1
    return resHitsscore

#initiallizehubscores
def initializehubscores(resultMap,inlinkmap):
    hitsscore = {}
    for docID in resultMap.keys():
        hitsscore[docID] = {'hubscore': 1.0, 'authscore': 1.0}
        inlinklist = inlinkmap[docID]
        if len(inlinklist) > 0 and inlinklist[0] != -1:
            for inlink in inlinklist:
                if inlink!= -1:
                    hitsscore[str(inlink)] = {'hubscore': 1.0, 'authscore': 1.0}
    return hitsscore

# HIT score
def getResultsByHitScore(query_str, indexoppath, topN):
    query_str = getStemmedQuery(query_str)
    resultdict = searchIndex(query_str, indexoppath, topN)
    resulthubscore = {}
    resultauthorityscore = {}
    resultURLhubscore, resultURLauthscore = [], []
    URLMap = createURLMap(urllistfile)
    inlinkmap = createInlinkMap(URLMap)
    outlinkmap = createOutlinkMap(URLMap)
    temphitscores = initializehubscores(resultdict,inlinkmap)
    hitsscore = calchubscores(temphitscores,resultdict,inlinkmap,outlinkmap)
    for key, entry in hitsscore.items():
        resulthubscore[key] = entry['hubscore']
        resultauthorityscore[key] = entry['authscore']
    for docid, value in sorted(resulthubscore.items(), key=itemgetter(1), reverse=True):
        resultURLhubscore.append(URLMap[docid]['URL'])
    for docid, value in sorted(resultauthorityscore.items(), key=itemgetter(1), reverse=True):
        resultURLauthscore.append(URLMap[docid]['URL'])

    return resultURLhubscore[:10], resultURLauthscore[:10]


###############################QUERY EXPANSION - METRIC###################################################

def searchIndexQE(queryStr, index, topN  ):
    keywords = []
    ix = open_dir(index)
    with ix.searcher(weighting=scoring.Frequency) as searcher:
        query = QueryParser("content", ix.schema).parse(queryStr)
        results = searcher.search(query, limit=topN)
        keywords = [keyword for keyword, score in results.key_terms("textdata", docs=10, numterms=2)]
        for new_query in keywords:
            queryStr += ' ' + new_query
        new_results = searchIndexedData(queryStr, index, 10)
    return queryStr,new_results


##############################QUERY EXPANSION-ROCCHIO##########################################
alpha = 2.0
beta = 0.75
gamma = 0.15
N= 10

def searchIndexedData(queryStr, index, topN  ):
    resultURLList = []
    resultdict = searchIndex(queryStr, index, topN  )
    for key , entry in resultdict.items():
        resultURLList.append(entry)
    return  resultURLList

def rocchio(qvec, tfidf, data, word_set, old_query):
    # record relevant & irrelevant count
    rel_count = 6
    irr_count = N - rel_count
    # compute new qvec
    new_qvec = [alpha * x for x in qvec]
    for i in range(N):
        if data[i]:
            new_qvec = [q + beta / float(rel_count) * r for q, r in zip(new_qvec, tfidf[i])]
        else:
            new_qvec = [q - gamma / float(irr_count) * r for q, r in zip(new_qvec, tfidf[i])]
    # extract new query, order matters
    qwords = [word.lower() for word in old_query.split()]  # may need restore later
    # top 2 largest
    sorted_qvec = [(new_qvec[i], i) for i in range(len(new_qvec))]
    sorted_qvec.sort(reverse=True)
    # number of words we can augment each time
    quota = 2
    for vec_val, index in sorted_qvec:
        if quota <= 0:
            break
        elif word_set[index] not in qwords:
            # if negative, ignore
            if vec_val <= 0:
                break
            qwords.append(word_set[index])
            quota -= 1
    qwords.sort(key=lambda w: new_qvec[word_set.index(w)], reverse=True)
    queryStr = ' '.join(qwords)
    return queryStr

# Calculate tfidf scores
def tfidfvec(query, data):
    # extract docs form raw data
    docs = [item for item in data]
    # build a list of tokenized docs
    wordsl = []
    for doc in docs:
        s = []
        for word in nltk.word_tokenize(doc):
            if ((word not in stopwords) and (word not in string.punctuation)):
                s.append(word.lower())
        wordsl.append(s)
    # a list of all words, with duplicates
    all_words = list(itertools.chain(*wordsl))
    # a list of all words, without duplicates - for vector bases
    word_set = list(set(all_words))

    # construct tf vectors
    tf_vecs = [0 for i in range(N)]
    for i in range(N):
        tf_vecs[i] = [wordsl[i].count(w) for w in word_set]
    # compute idf values
    idf_all_words = list(itertools.chain(*[set(doc_words) for doc_words in wordsl]))
    idfs = [math.log(float(N) / idf_all_words.count(w), 10) for w in word_set]
    # compute tf-idf & normalize
    tfidf = [0 for i in range(N)]
    for i in range(N):
        tfidf[i] = [tf * idf for tf, idf in zip(tf_vecs[i], idfs)]
        nom = math.sqrt(sum(x ** 2 for x in tfidf[i]))
        tfidf[i] = [x / nom for x in tfidf[i]]

    # now let's work on the query vector
    qwords = [word.lower() for word in query.split()]
    # tf vector
    qvec = [qwords.count(w) for w in word_set]
    # normalize
    nom = 1+math.sqrt(sum(x ** 2 for x in qvec))
    qvec = [x / nom for x in qvec]
    return qvec, tfidf, word_set


def adjustQuery(queryStr, data):
    qvec, tfidf, word_set = tfidfvec(queryStr, data)
    new_queryStr = rocchio(qvec, tfidf, data, word_set, queryStr)
    return new_queryStr

def searchIndexRocchio(queryStr, index, topN  ):
    keywords = []
    queryStr = getStemmedQuery(queryStr)
    data ,old_results= [],[]
    ix = open_dir(index)
    with ix.searcher(weighting=scoring.Frequency) as searcher:
        query = QueryParser("content", ix.schema).parse(queryStr)
        results = searcher.search(query, limit=topN)
        for key, entry in results.items():
            old_results.append(entry)
        if len(results)!= 0:
            for hits in results:
                data.append(hits['textdata'])
            new_query = adjustQuery(queryStr, data)
            if new_query!=queryStr:
                new_results = searchIndexedData(new_query, index, 10)
                return new_query,new_results
            else:
                return 'No Expansion', old_results
    return 'No expansion','No-Results'

##########################CLUSTERING - KMEANS###################################

def createClusterMap(clusterFile):
    clusterMap = {}
    with open(clusterFile, 'r') as file:
        for line in file:
            if len(line) > 1:
                record = line.split('|')
                key = str(record[0])
                value = eval(record[1])
            if key not in clusterMap:
                clusterMap[key] = value
    return clusterMap


def getResultbyClustering(query_str, indexoppath, topN, clusterfile):
    URLMap = createURLMap(urllistfile)
    resultdict = searchIndex(query_str, indexoppath, topN)
    clusterMap = createClusterMap(clusterfile)   #clusterid , overall doc list
    resultClusterMap = {}       #resultdocid , clusterid
    resultClustercountMap = {}  #clusterid, resultcount
    resultpagerank = {}
    resultURL = {}
    for docid in resultdict.keys():
        for clusterid in clusterMap.keys():
            clusterdoclist = clusterMap[clusterid]
            if int(docid) in clusterdoclist:
                resultClusterMap[docid] = clusterid
    for key, value in resultClusterMap.items():
        if value not in resultClustercountMap:
            resultClustercountMap[value] = 1
        else:
            resultClustercountMap[value] += 1
    targetcluster = dict(sorted(resultClustercountMap.items(), key=itemgetter(1), reverse=True)[:2])
    print(targetcluster, type(targetcluster))
    for clusterid in targetcluster.keys():
        clusterdoclist = clusterMap[clusterid]
        # #for docid in clusterdoclist:
        #     resultpagerank[docid] = pagerank#map[str(docid)]
        for docid in clusterdoclist[:10]:
            resultURL[(URLMap[str(docid)]['URL'])]= clusterid
    return resultURL


###################AGGLOMERATIVE ###########################################

aggfile = 'C:\\Users\\mohan\\OneDrive\\Documents\\Latest_files_IR\\agglomerative.txt'

def getresultsbyAgglomerative(query_str, indexoppath, topN, aggfile):
    URLMap = createURLMap(urllistfile)
    resultdict = searchIndex(query_str, indexoppath, topN)
    clusterMap = createClusterMap(aggfile)
    print(clusterMap)#clusterid , overall doc list
    resultClusterMap = {}       #resultdocid , clusterid
    resultClustercountMap = {}  #clusterid, resultcount
    resultURL = {}
    for docid in resultdict.keys():
        for clusterid in clusterMap.keys():
            clusterdoclist = clusterMap[clusterid]
            if int(docid) in clusterdoclist:
                resultClusterMap[docid] = clusterid
    for key, value in resultClusterMap.items():
        if value not in resultClustercountMap:
            resultClustercountMap[value] = 1
        else:
            resultClustercountMap[value] += 1
    targetcluster = dict(sorted(resultClustercountMap.items(), key=itemgetter(1), reverse=True)[:2])
    if not bool(targetcluster):
        targetcluster['Cluster3']=6
    for clusterid in targetcluster.keys():
        clusterdoclist = clusterMap[clusterid]
        for docid in clusterdoclist[:10]:
            resultURL[(URLMap[str(docid)]['URL'])] = clusterid
    return resultURL



#getresultsbyAgglomerative(query_str, indexoppath, 10, clusterfile)




##########################################################################
def checkList(urllsit):
    if len(urllsit)==0:
        return ["No Results"]
    else:
        return urllsit

@api_view(["POST"])
def getAll(requestQuery):
    try:
        result = {}
        #indexPath = 'C:\\Users\\mohan\\OneDrive\\Documents\\New Folder\\index\\index'
        query = str(requestQuery.body)
        googleResult = getGoogleResult(query)
        googleResult = checkList(googleResult)
        googleResult = refineURL(googleResult)
        bingResult = getBingResult(query)
        bingResult = checkList(bingResult)
        bingResult = refineURL(bingResult)
        ourSearchResult = ourSearchEngineResult(query)
        ourSearchResult = checkList(ourSearchResult)
        ourSearchResult = refineURL(ourSearchResult)
        pageRankResult = getResultsByPagerank(query,indexPath,10)
        ourSearchResult = checkList(ourSearchResult)
        pageRankResult = refineURL(pageRankResult)
        hubBasedResult, authBasedResult = getResultsByHitScore(query, indexPath, 10)
        hubBasedResult = checkList(hubBasedResult)
        authBasedResult = checkList(authBasedResult)
        hubBasedResult = refineURL(hubBasedResult)
        authBasedResult = refineURL(authBasedResult)
        Query_1, MetricResults = searchIndexQE(query.replace("b'","").replace("'",""), indexPath, 10)
        MetricResults = checkList(MetricResults)
        MetricResults = refineURL(MetricResults)
        queryR = query
        Query_2, RocchioResults = searchIndexRocchio(queryR.replace("b'","").replace("'",""), indexPath, 10)
        RocchioResults = checkList(RocchioResults)
        RocchioResults = refineURL(RocchioResults)
        #Cluster Results []cid[]urls
        clusterfile = 'C:\\Users\\mohan\\OneDrive\\Documents\\Latest_files_IR\\kmeans.txt'
        kmeansdict = getResultbyClustering(query, indexPath, 10, clusterfile)
        kmeansURL = list(kmeansdict.keys())
        kmeansURL = checkList(kmeansURL)
        kmeansURL = refineURL(kmeansURL)
        aggdict = getresultsbyAgglomerative(query,indexPath,10,aggfile)
        aggURL = list(aggdict.keys())
        aggURL = checkList(aggURL)
        aggURL = refineURL(aggURL)
        result = {'Google-results': googleResult
                    ,'Bing-results': bingResult
                    ,'Our-results': ourSearchResult
                    ,'Our-PageRank-results': pageRankResult
                  ,'Hub-Results': hubBasedResult
                  ,'Auth-Results': authBasedResult
                  , 'Query_Rocchio': Query_2
                  ,'Rocchio-Results': RocchioResults
                  ,'Query_Metric': Query_1
                  ,'Metric-Results': MetricResults
                  ,'Kmeans-Results': kmeansURL[:10]
                  ,'KmeansCluster': list(kmeansdict.values())[:10]
                  ,'Agglomerative-Results' : aggURL
                  ,'AggCluster': list(aggdict.values())
            }
        return JsonResponse(result, safe=False)
    except ValueError as e:
        return Response(e.args[0],status.HTTP_400_BAD_REQUEST)

########################################################################################################################

@api_view(["POST"])
def getAggresults(requestQuery):
    try:
        query = str(requestQuery.body)#.replace("b'","").replace("'","")
        print(query)
        #indexPath = 'C:\\Users\\mohan\\OneDrive\\Documents\\Index_IR\\index\\index'
        resultdict = getResultbyClustering(query, indexPath, 10, aggfile)
        result = {'URLResults': list(resultdict.keys()),
                  'ClusterResults': list(resultdict.values())}
        return JsonResponse(result, safe=False)
    except ValueError as e:
        print(e)
        return Response(e.args[0],status.HTTP_400_BAD_REQUEST)



@api_view(["POST"])
def getKMeansresults(requestQuery):
    try:
        clusterfile = 'C:\\Users\\mohan\\OneDrive\\Documents\\Latest_files_IR\\kmeans.txt'
        query = str(requestQuery.body)#.replace("b'","").replace("'","")
        print(query)
        #indexPath = 'C:\\Users\\mohan\\OneDrive\\Documents\\Index_IR\\index\\index'
        resultdict = getResultbyClustering(query, indexPath, 10, clusterfile)
        result = {'URLResults': list(resultdict.keys()),
                  'ClusterResults': list(resultdict.values())}
        return JsonResponse(result, safe=False)
    except ValueError as e:
        print(e)
        return Response(e.args[0],status.HTTP_400_BAD_REQUEST)



@api_view(["POST"])
def getMetricresults(requestQuery):
    try:
        query = str(requestQuery.body)#.replace("b'","").replace("'","")
        print(query)
        #indexPath = 'C:\\Users\\mohan\\OneDrive\\Documents\\Index_IR\\index\\index'
        OneResult, TwoResult = searchIndexQE(query, indexPath, 10)
        result = {'Results': str(OneResult),
                  'URLResults': [str(l) for l in TwoResult]}
        return JsonResponse(result, safe=False)
    except ValueError as e:
        print(e)
        return Response(e.args[0],status.HTTP_400_BAD_REQUEST)




@api_view(["POST"])
def getRocchioresults(requestQuery):
    try:
        query = str(requestQuery.body).replace("b'","").replace("'","")
        print(query)
        #indexPath = 'C:\\Users\\mohan\\OneDrive\\Documents\\Index_IR\\index\\index'
        OneResult, TwoResult = searchIndexRocchio(query, indexPath, 10)
        result = {'Results': str(OneResult),
                  'URLResults': [str(l) for l in TwoResult]}
        return JsonResponse(result, safe=False)
    except ValueError as e:
        print(e)
        return Response(e.args[0],status.HTTP_400_BAD_REQUEST)


@api_view(["POST"])
def getHitResults(requestQuery):
    try:
        query = str(requestQuery.body)
        #indexPath = 'C:\\Users\\mohan\\OneDrive\\Documents\\Index_IR\\index\\index'
        hubBasedResult, authBasedResult = getResultsByHitScore(query, indexPath, 10)
        result = {'Hub-Results': [str(l) for l in hubBasedResult],
                  'Auth-Results':[str(l) for l in authBasedResult]}
        return JsonResponse(result, safe=False)
    except ValueError as e:
        return Response(e.args[0],status.HTTP_400_BAD_REQUEST)


@api_view(["POST"])
def getPageRankResults(requestQuery):
    try:
        query = str(requestQuery.body)
        #indexPath = 'C:\\Users\\mohan\\OneDrive\\Documents\\New Folder\\index\\index'
        resulturllist = getResultsByPagerank(query, indexPath, 5)
        result = {'results': [str(l) for l in resulturllist]}
        return JsonResponse(result, safe=False)
    except ValueError as e:
        return Response(e.args[0],status.HTTP_400_BAD_REQUEST)


@api_view(["POST"])
def getBing(requestQuery):
    try:
        query = str(requestQuery.body)
        print(query)
        for start in range(0, 10):
            url = "http://www.bing.com/search?q=" + query + "&start=" + str(start * 10)
        page = requests.get(url)
        soup = Soup(page.content, features='html')
        links = soup.findAll('cite')
        refined_links = refineLink(links)
        result = {'results': [str(l) for l in refined_links]}
        return JsonResponse(result, safe=False)
    except ValueError as e:
        print(e)
        return Response(e.args[0],status.HTTP_400_BAD_REQUEST)


@api_view(["POST"])
def getGoogle(requestQuery):
    try:
        query = str(requestQuery.body)
        for start in range(0, 10):
            url = "http://www.google.com/search?q=" + query + "&start=" + str(start * 10)
        page = requests.get(url)
        soup = Soup(page.content, features='html')
        links = soup.findAll('cite')
        refined_links = refineLink(links)
        result = {'results': [str(l) for l in refined_links]}
        return JsonResponse(result, safe=False)
    except ValueError as e:
        return Response(e.args[0],status.HTTP_400_BAD_REQUEST)


@api_view(["POST"])
def ourSearchEngine(query):
    try:
        ourSearchResult = ourSearchEngineResult(query)
        ourSearchResult = refineURL(ourSearchResult)
        result = {'results': [str(l) for l in ourSearchResult]}
        return JsonResponse(result, safe=False)
    except ValueError as e:
        return Response(e.args[0],status.HTTP_400_BAD_REQUEST)