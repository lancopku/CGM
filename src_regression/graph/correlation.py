import os
import csv
import scipy as sp
import numpy as np
from graph.file import read_price
import json


def find_correlation(c1, c2):
    seq_1 = []
    seq_2 = []
    for date in set(c1.keys()).intersection(set(c2.keys())):
        # use movement percent by now
        seq_1.append(c1[date][0])
        seq_2.append(c2[date][0])
    assert len(seq_1) == len(seq_2)
    r, p = sp.stats.pearsonr(seq_1, seq_2)
    return r


def build_graph():
    price_dir = os.path.join('../stock-dataset', 'price', 'preprocessed')
    company_names = os.listdir(price_dir)
    prices = {}
    for company_file in company_names:
        company_name = os.path.splitext(company_file)[0]
        prices[company_name] = read_price(os.path.join(price_dir, company_file))
    graph = {}
    edge_num = 0
    for c_1 in prices:
        graph[c_1] = {}
        for c_2 in prices:
            if c_1 == c_2:
                continue
            pearson = find_correlation(prices[c_1], prices[c_2])
            if abs(pearson) > 0.5:
                edge_num += 1
                graph[c_1][c_2] = pearson
    print('edge number', edge_num)
    # json.dump(graph, open('pearsonsR.json', 'w'))
    return graph


def map_company_name(name):
    return name.replace('/', '-').split(' ')[0]


def read_bloomberg_correlation(fname):
    graph = {}
    edge_num = 0
    with open(fname) as csvfile:
        reader = csv.reader(csvfile)
        companies = next(reader)
        companies = [map_company_name(name) for name in companies[1:]]
        for row in reader:
            name = map_company_name(row[0])
            graph[name] = {}
            for i in range(len(companies)):
                num = row[i + 1]
                if num == '-':
                    num = 0
                num = float(num)
                if abs(num) > 0.6:
                    graph[name][companies[i]] = num
                    edge_num += 1
    json.dump(graph, open('bloomberg.json', 'w'))
    print('edge number', edge_num)
    return graph


if __name__ == '__main__':
    read_bloomberg_correlation('corr_table.csv')
    # graph = build_graph()
    # page_rank_graph(build_graph())
