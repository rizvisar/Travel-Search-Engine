from django.http import HttpResponse
import requests
import sys
#import lxml
import re
sys.path.append("./BeautifulSoup")
from bs4 import BeautifulSoup as Soup


def index(request):
    return HttpResponse("Hello, world. This is my evil empire")

def results(request, search_query):
    result = 'Hi'
    return HttpResponse("You've requested Yama for %s" %result)




'''
def resultPage(request):
    # it is a list of around 10 urls
    googleResults = getGoogle()
    #It is a list of 10 Urls
    bingResults= getBing()
    # it is a list od 10 urls
    OurSearchResults = getResults()

    return googleResults,bingResults,OurSearchResults

def getGoogle():
    '''