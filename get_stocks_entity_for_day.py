# -*- encoding:utf8 -*-
import codecs
import json

def get_news_lists():
    newsLists = []
    for i in range(num_of_days):
        newsList = []
        m = i + 1
        f = open('newsList/%d.txt' % m, 'r')
        for line in f.readlines():
            newsList.append(int(line))
        newsLists.append(newsList)
        f.close()
    return newsLists

def get_stocks_entity():
    fout = open('stocks_for_day/%d.json' % day, 'w')
    stock_dict = {}
    for ele in newsList:
        f = codecs.open('../news_data/%d.json' % ele, encoding='utf-8')
        f_d = json.load(f)
        stocks_list = f_d["refStocks"]
        stocks = []
        for _ in stocks_list:
            stocks.append(_["name"])
        stock_dict[ele] = stocks
    strObj = json.dumps(stock_dict)
    fout.write(strObj)
    fout.close()

if __name__ == '__main__':
    num_of_days = 302
    newsLists = get_news_lists()
    for i in range(num_of_days):
        day = i + 1
        newsList = newsLists[i]
        get_stocks_entity()