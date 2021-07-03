import csv
import json
import os
import copy
import numpy as np
import utils
import datetime
import pickle
import math
import sys
from nltk.tokenize import word_tokenize


class News:
    bert_dict = {}

    def __init__(self, id, headline, body, topic, related_companies, time):
        self.id = id
        self.headline = headline
        self.body = body
        self.time = time
        self.topic = topic
        self.related_companies = related_companies

    def extend(self, headline, body, topic):
        self.headline += headline
        self.body += body
        self.topic += topic

    @classmethod
    def load_bert_file(cls, filename='extracted_bert.json'):
        """

        :param filename: the name of the bert file, which is jsonl
        :return:
        """
        cls.bert_dict = json.load(open(filename))

    def get_body_fixed_vec(self):
        """

        :param bert_file: pre-train feature extracted with BERT model
        :param id: the news id identifying the position in BERT
        :return:
        """
        return self.bert_dict[self.id]


class Company:
    def __init__(self, name):
        """
        News, prices and features are all dicts with date as the keys
        :param name:
        """
        self.name = name
        self.news = {}
        self.features = {}
        self.prices = {}
        self.movement_mean = 0
        self.movement_std = 0

    def set_prices(self, price):
        self.prices = price

    def set_news(self, news):  
        self.news = {datetime.datetime.strptime(date, '%Y-%m-%d'): news[date] for date in news}

    def set_movement_std(self, std):
        self.movement_std = std

    def set_movement_mean(self, mean):
        self.movement_mean = mean

    def judge_label(self, movement, isLabel = False):
        # using this setting because of the weight in loss function
        # print(movement, self.std)
        # from ipdb import set_trace; set_trace()
        # print(movement)
        movement = movement - self.movement_mean 
        if movement < -0.5 * self.movement_std:
            label =  0
        elif movement > 0.5 * self.movement_std:
            label =  1
        else:
            label =  2
        # print(movement, self.movement_std, label)
        # if(isLabel): print(movement, self.movement_mean, self.movement_std, label)
        # if(isLabel): print("Label " + str(label) + '\n')
        return label


class GraphSnap:
    def __init__(self, companies, date, node_num, time='open'):
        self.companies = companies
        self.nodes = [i for i in range(node_num)]
        self.node_headline = [[] for _ in range(node_num)]
        # self.node_body_bert = [np.zeros(768) for _ in range(node_num)]
        self.node_features = [[0 for _ in range(8)] for _ in range(node_num)]
        self.date = date
        self.time = time
        self.has_news = False
        self.movement = [2 for _ in range(node_num)]

    def add_node_text(self, node_id, news_list):
        if self.has_news == False: self.has_news = True
        for news in news_list:
            # id, headline, body, topic, rics, time
            # TODO: 
            # out irrelevant news, for example, keep only ``price'' related news
            n = News(news[0], news[1], news[2], news[3], news[4], news[5])
            # if 'price' in n.headline:
            self.node_headline[node_id].append(n.headline)   #
            # self.node_body_bert[node_id] += n.get_body_fixed_vec()
 
    def add_node_movement(self, node_id, movement):
        # self.movement[node_id] = self.companies[node_id].judge_label(movement, True)
        # print(movement)
        self.movement[node_id] = movement

    def add_node_feature(self, node_id, feature, feature_scale):
        def scale(p, high, low):
            return (p - low) / float(high - low)

        feature[1] = scale(feature[1], feature_scale[0], feature_scale[1])  # open
        feature[2] = scale(feature[2], feature_scale[0], feature_scale[1])  # high
        feature[3] = scale(feature[3], feature_scale[0], feature_scale[1])  # low
        feature[4] = scale(feature[4], feature_scale[0], feature_scale[1])  # close
        # feature[6] = feature[5] / feature[6]  # total_volume -> volume_percentage
        feature[6] = math.log(feature[6])
        feature[5] = scale(feature[5], feature_scale[2], feature_scale[3])  # first_hour volume
        feature.append(feature[6])  # label of movement
        # assert len(feature) == 7, feature
        self.node_features[node_id] = feature

    def get_movement_mask(self):
        movement_mask = []
        for i in range(len(self.movement)):
            # along the number of nodes
            if self.movement[i] != 2:
                movement_mask.append(1)
            else:
                movement_mask.append(0)
        return movement_mask

class GraphSpan:
    def __init__(self, span):
        # each date in the span consists of (date, Snap)
        span = [snap for snap in span if snap is not None]
        self.data = span
        # need to exclude the last hour
        self.size = len(self.data) - 1
        # last_movement, open_price, high, low, close, volume, close-open, high-low
        self.span_features = self.convert_feature([d.node_features for d in self.data[:-1]])
        self.span_nodes = self.data[-1].nodes
        self.span_news_mask = self.get_span_news_mask(self.data[-1].node_headline)
        self.has_news = True if self.data[-1].has_news else False
        self.span_text = self.get_span_text(self.data[-1].node_headline)
        # self.bert_vec = [d.node_body_bert for d in self.data[:-1]]
        self.span_movement = self.data[-1].movement
        self.span_movement_mask = self.data[-1].get_movement_mask()
        self.movement_num = sum(self.span_movement_mask)
        # self.valid_node = [d[1].valid_node for d in self.data]

    @staticmethod
    def get_span_news_mask(node_headline_list):
        # node_headline_list: time, node, news_list
        node_mask = []  # mask for each node
        for node in range(len(node_headline_list)):
            # along the node
            node_mask.append(any([len(text) > 1 for text in node_headline_list[node]]))
        # assert len(node_mask) == 498, len(node_mask)
        return node_mask

    @staticmethod
    def get_span_text(node_headline_list):
        result = [['[START]'] for _ in range(len(node_headline_list))]
        for node_id in range(len(node_headline_list)):
            headline_set = set()
            for text in node_headline_list[node_id]:
                if text not in headline_set:
                    headline_set.add(text)
            for text in headline_set:
                result[node_id].extend(word_tokenize(text))
        return result

    def pad_text(self, word_vocab, max_len=None):
        '''
        self.span_text:     (node, seq)
        :param word_vocab:
        :return:
        '''
        mlen = max([len(node_text) for node_text in self.span_text])
        if max_len is not None:
            max_len = min(max_len, mlen)
        else:
            max_len = mlen
        result = []
        span_mask = []
        for node_text in self.span_text:
            words = [word_vocab.word2id(word) for word in node_text][:max_len]
            mask = [1 for _ in words]
            result.append(words + [word_vocab.PAD_token] * (max_len - len(words)))
            span_mask.append(mask + [0] * (max_len - len(mask)))
            assert len(result) == len(span_mask)
        return result, span_mask

    # TODO
    @staticmethod
    def convert_feature(span_features):
        '''
        features: movement, open, high, low, close, volume
        :param span_features:  (span, node, features)
        :return:
        '''
        features = []
        for i in range(len(span_features)):
            node_features = []
            # assert len(span_features[i]) == 498, len(span_features[i])  # 95 companies
            for node_id in range(len(span_features[i])):
                # from ipdb import set_trace; set_trace()
                assert len(span_features[i][node_id]) == 8, len(span_features[i][node_id])  # 8 source features
                # 0 is the digit number of last movement,
                node_features.append([span_features[i][node_id][1]] + [span_features[i][node_id][2]] + \
                                     [span_features[i][node_id][3]] + [span_features[i][node_id][4]] + \
                                     [span_features[i][node_id][4] - span_features[i][node_id][1]] + \
                                     [span_features[i][node_id][2] - span_features[i][node_id][3]] + \
                                     [span_features[i][node_id][5]] + [span_features[i][node_id][6]])
                                     # o, h, l, c,  o - c, h - l,  v, v_percentage,
            features.append(node_features)
        return features


class Graph:
    def __init__(self, companies, config, high_price, low_price, high_vol, low_vol):
        self.config = config
        self.companies = companies
        self.company_list = []
        self.node_num = len(self.companies)
        self.feature_scale = (high_price, low_price, high_vol, low_vol)
        assert self.node_num == config.node_num, self.node_num
        self.com2id = {}
        self.id2com = []
        self.set_mapping()
        # News.load_bert_file()

    def set_mapping(self):
        for company in self.companies:
            self.com2id[company] = len(self.id2com)
            self.id2com.append(company)
        for id in range(len(self.id2com)):
            self.company_list.append(self.companies[self.id2com[id]])

    def load_adjacency(self):
        # usage 
        # correlation_table: 'data/corr_table.json,data/vol_corr_table.json'  each json file split with comma
        # relation_num: 2  means positive and negative are different, if relation_num is 1, means only use positive correlation.
        correlation_files = self.config.correlation_table.split(',')
        adj = [[[0 for _ in range(self.node_num)] for _ in range(self.node_num)] for _ in range(self.config.relation_num * len(correlation_files))]
        for idx, cur_file in enumerate(correlation_files):
            adj_dict = json.load(open(cur_file))
            for c1 in adj_dict:
                if c1 in self.com2id: 
                    for c2 in adj_dict[c1]:
                        if c2 in self.com2id:
                            score = float(adj_dict[c1][c2])
                            if self.config.relation_num == 1:
                                if abs(score) >= self.config.edge_threshold:
                                    adj[0+idx*2][self.com2id[c1]][self.com2id[c2]] = abs(score)
                            else:
                                if score >= self.config.edge_threshold:
                                    adj[0+idx*2][self.com2id[c1]][self.com2id[c2]] = score
                                elif score <= -self.config.edge_threshold:
                                    adj[1+idx*2][self.com2id[c1]][self.com2id[c2]] = abs(score)
        # print(adj)
        return adj

    def extract_data_span(self):
        dates = {}  # dict of GraphSnaps

        for company in self.companies:
            for prices_idx, date in enumerate(self.companies[company].prices):
                # find the previous day
                if prices_idx < 20:
                    continue
                if judge_dataset(date, self.config) >= 0:  

                    # ----get previouse 20 day. feature data --------
                    previous_day_list = []
                    date_delta = 1
                    while len(previous_day_list) < 20 and date_delta < 50:
                        prev_date = date - datetime.timedelta(days=date_delta)
                        date_delta += 1
                        if prev_date not in self.companies[company].prices:  continue
                        previous_day_list.append(prev_date)
                    if len(previous_day_list) < 20: continue

                    #NOTE --- get previouse one day news
                    prev_date = date - datetime.timedelta(days=1)
                    limit_day_num = 0
                    while prev_date not in self.companies[company].prices:  # if there is no data about previouse two data, then  we skip this data sample.
                        prev_date = prev_date - datetime.timedelta(days=1)
                        limit_day_num += 1
                        if limit_day_num == 2:
                            break
                    if limit_day_num == 2:
                        # there is no previous day, maybe this is the first day in the dataset
                        continue
                    #NOTE --- there are 21 snaps in previous 20 day, 0~20 are the price features from the previous 20 day in terms of 'open' , 21 is the last day 'close' feature, 22 is the label
                    if date not in dates:
                        dates[date] = [None for _ in range(22)]

                    # hours from the previous day, but the features are used for today, all the hours including the 14:00
                    # hour_number = len(self.companies[company].prices[prev_date][0])  # 
                    sample_idx  = 0 
                    for day in previous_day_list[::-1]:  # From far to near
                        # for hour_time in ['open', 'close']:
                        # if hour_time == 'close': hour = hour_number - 1
                        if dates[date][sample_idx] is None:
                            snap = GraphSnap(self.company_list, day, self.config.node_num, 'open')  # build a graph snap
                        else:
                            snap = dates[date][sample_idx]
                        # index 0 is movement, which is calculated as (close-open) of the previous hour
                        # prev_movement = 0 if hour == 0 else self.companies[company].prices[day][0][hour - 1]
                        # 1 ~ 7 are the indices of features excluding movement
                        # 7 is the feature of movement which we don't use in volume prediction
                        # movement, o, h, l, c, v, close, total_v
                        feature = [0] + [self.companies[company].prices[day][i][0] for i in range(1, 6)] + [self.companies[company].prices[day][6]]
                        snap.add_node_feature(self.com2id[company], feature, self.feature_scale)
                        if dates[date][sample_idx] is None:
                            dates[date][sample_idx] = snap
                        sample_idx += 1
                    # add the movement of the last hour, which is to be predicted without features
                    # -1 is not the last hour, but the close price of the day, which is stored manually
                    # prev_day_close = self.companies[company].prices[prev_date][-1]  # the previous day close
                    # today_open = self.companies[company].prices[date][1][0]  # the today open

                    #NOTE ---  添加最后一个特征（last day close hour feature） and label of sample and news，之前的是20天open的特征，是我们需要改的
                    # the actual snap that contains the news and the opening movement
                    if dates[date][20] is None:
                        # this is for snap of different companies
                        snap = GraphSnap(self.company_list, prev_date, self.config.node_num, 'close')
                    else:
                        # we use 6 instead of -1 to prevent exception where there is not enough hour units
                        snap = dates[date][20]
                    feature = [0] + [self.companies[company].prices[prev_date][i][-1] for i in range(1, 6)] + [self.companies[company].prices[prev_date][6]]
                    snap.add_node_feature(self.com2id[company], feature, self.feature_scale)
                    if dates[date][20] is None:
                        dates[date][20] = snap
                    # snap.add_node_movement(self.com2id[company], (today_open - prev_day_close) / prev_day_close)
                    #NOTE  --- this is label of each sample 
                    # from ipdb import set_trace; set_trace()
                    if dates[date][21] is None:
                        # this is for snap of different companies
                        snap = GraphSnap(self.company_list, date, self.config.node_num, 'open')
                    else:
                        snap = dates[date][21]
                    today_open_volume = self.companies[company].prices[date][-2][0]
                    today_total_volume = self.companies[company].prices[date][-1]
                    # from ipdb import set_trace; set_trace() 
                    snap.add_node_movement(self.com2id[company], math.log(today_open_volume))
                    if dates[date][21] is None:
                        dates[date][21] = snap
                    # from ipdb import set_trace; set_trace() 
                    #NOTE ---- add news for close hour
                    if date in self.companies[company].news: # if current day there exists news .
                        daily_news = self.companies[company].news[date]
                        for i in range(len(daily_news)):
                            hour = int(daily_news[i][-1])
                            # here put in a news_list instead of the news itself according to the add_node_text method
                            if hour >= 5:
                                # the news after 5 is actually the news from yesterday, moved to this day manually
                                # the news belongs to the opening hour
                                if dates[date] is not None:  # overnight news
                                    dates[date][21].add_node_text(self.com2id[company], [self.companies[company].news[date][i]])
                    # to deal with the weekend situation, where to put the news from the weekend to today
                    # if today is not weekend, limit_day_num should be equal to 0
                    for day_delta in range(1, limit_day_num + 1):
                        target_date = date - datetime.timedelta(days=day_delta)
                        if target_date in self.companies[company].news:
                            daily_news = self.companies[company].news[target_date]
                            for i in range(len(daily_news)):
                                if dates[date] is not None:
                                    dates[date][21].add_node_text(self.com2id[company],
                                                                           [self.companies[company].news[target_date][i]])

        data = [(date, dates[date]) for date in dates]
        print('date number', len(data))
        data.sort(key=lambda k: k[0])
        train_spans, dev_spans, test_spans = [], [], []
        for i in range(0, len(data)):
            date = data[i][0]
            dataset = judge_dataset(date, self.config)
            if dataset == 0:
                # contains the daily snaps
                train_spans.append(GraphSpan(data[i][1]))
            elif dataset == 1:
                dev_spans.append(GraphSpan(data[i][1]))
            elif dataset == 2:
                test_spans.append(GraphSpan(data[i][1]))
        # statistic_span(train_spans, test_spans, dev_spans)
        # from ipdb import set_trace; set_trace() 
        return train_spans, dev_spans, test_spans

    @staticmethod
    def padding(batch, max_len):
        result = []
        mask_batch = []
        for s in batch:
            l = copy.deepcopy(s)
            m = [1. for _ in range(len(l))]
            l = l[:max_len]
            m = m[:max_len]
            while len(l) < max_len:
                l.append(0)
                m.append(0.)
            result.append(l)
            mask_batch.append(m)
        return result, mask_batch

def statistic_span(train_spans, test_spans, dev_spans):
        train_pos, train_neg, train_neu, train_pos_news, train_neg_news, train_neu_news = 0, 0, 0, 0, 0, 0
        test_pos, test_neg, test_neu, test_pos_news, test_neg_news, test_neu_news = 0, 0, 0, 0, 0, 0
        valid_pos, valid_neg, valid_neu, valid_pos_news, valid_neg_news, valid_neu_news = 0, 0, 0, 0, 0, 0

        for span in train_spans:
            for movement in span.span_movement:
                if movement == 0: train_pos += 1
                elif movement == 1: train_neg += 1
                elif movement == 2: train_neu += 1 
                if span.has_news:
                    if movement == 0: train_pos_news += 1
                    elif movement == 1: train_neg_news += 1
                    elif movement == 2: train_neu_news += 1    

        for span in test_spans:
            for movement in span.span_movement:
                if movement == 0: test_pos += 1
                elif movement == 1: test_neg += 1
                elif movement == 2: test_neu += 1 
                if span.has_news:
                    if movement == 0: test_pos_news += 1
                    elif movement == 1: test_neg_news += 1
                    elif movement == 2: test_neu_news += 1    

        for span in dev_spans:
            for movement in span.span_movement:
                if movement == 0: valid_pos += 1
                elif movement == 1: valid_neg += 1
                elif movement == 2: valid_neu += 1 
                if span.has_news:
                    if movement == 0: valid_pos_news += 1
                    elif movement == 1: valid_neg_news += 1
                    elif movement == 2: valid_neu_news += 1      
        print("pos, neg, neu")
        print(train_pos, train_neg, train_neu)
        print(test_pos, test_neg, test_neu)
        print(valid_pos, valid_neg, valid_neu)
        from ipdb import set_trace; set_trace()        


def judge_dataset(date, config):
    train_start = datetime.datetime.strptime(config.train_start, "%Y-%m-%d")
    dev_start = datetime.datetime.strptime(config.dev_start, "%Y-%m-%d")
    test_start = datetime.datetime.strptime(config.test_start, "%Y-%m-%d")
    test_end = datetime.datetime.strptime(config.test_end, "%Y-%m-%d")
    if train_start <= date < dev_start:
        return 0
    elif dev_start <= date < test_start:
        return 1
    elif test_start <= date <= test_end:
        return 2
    return -1


def extract_ohlcv_from_hourly(open_price, high, low, close, volume):
    """
    This method is not only designed for extracting information from a specific time period. The parameters should all
    be lists within the specific period.
    :param open_price:
    :param high:
    :param low:
    :param close:
    :param volume:
    :return:
    """
    return open_price[0], max(high), min(low), close[-1], sum(volume)


def map_hourly_price(config):
    csv.field_size_limit(sys.maxsize)
    dir = '../topix500'
    company_names = os.listdir(dir)
    for fname in company_names:
        if 'hourly' not in fname:
            with open(os.path.join(dir, fname)) as csvfile:
                data = {}
                reader = csv.reader(csvfile)
                next(reader)  # first line is header
                for row in reader:
                    #date, time, open_price, high, low, close, volume = row
                    _, close, date, high, low, open_price, time, volume = row
                    date = datetime.datetime.strptime(date, "%Y-%m-%d")
                    time = datetime.datetime.strptime(time, "%H:%M")
                    if judge_dataset(date, config) < 0:
                        continue
                    if date not in data:
                        data[date] = {}
                    if time.hour not in data[date]:
                        data[date][time.hour] = []
                    data[date][time.hour].append(
                        [float(open_price), float(high), float(low), float(close), float(volume), time.minute])
                write = open(os.path.join(dir, fname.split('.')[0] + '-hourly.csv'), 'w')
                write.write('date,time,o,h,l,c,v\n')
                for date in data:
                    for hour in data[date]:
                        hour_data = sorted(data[date][hour], key=lambda k: k[-1])
                        open_price = [d[0] for d in hour_data]
                        high = [d[1] for d in hour_data]
                        low = [d[2] for d in hour_data]
                        close = [d[3] for d in hour_data]
                        volume = [d[4] for d in hour_data]
                        hour_result = extract_ohlcv_from_hourly(open_price, high, low, close, volume)
                        write.write(date.strftime('%Y-%m-%d') + ',' + str(hour) + ':00' + ',' + ','.join(
                            [str(d) for d in hour_result]) + '\n')
                write.close()


def read_price(fname, config):
    # read the trade history of one company
    csv.field_size_limit(sys.maxsize)
    data = {}
    with open(fname) as csvfile:
        reader = csv.reader(csvfile)
        #next(reader)  # first line is header
        for row in reader:
            date, time, open_price, high, low, close, volume = row
            #_, close, date, high, low, open_price, time, volume = row
            date = datetime.datetime.strptime(date, "%Y-%m-%d")
            time = datetime.datetime.strptime(time, "%H:%M")
            if judge_dataset(date, config) < 0:
                continue
            if date not in data:
                data[date] = []
            # movement = (close - open_price) / float(open_price)
            data[date].append([float(open_price), float(high), float(low), float(close), float(volume), int(time.hour)])
    result_data = {}
    # movements = []
    volume_movements = []
    for date in data:
        # 6 here to exclude 13:00 useless data
        daily_data = data[date]
        open_price = [d[0] for d in daily_data][:6]
        high_price = [d[1] for d in daily_data][:6]
        low_price = [d[2] for d in daily_data][:6]
        close_price = [d[3] for d in daily_data][:6]
        volume = [d[4] for d in daily_data][:6]
        total_volume = sum(volume)
        movement = [ (o - c) / float(o) for c, o in zip(open_price, close_price)]
        volume_movements.append(volume[0]/total_volume)
        # movements.append(movement[-1])
        day_close = data[date][-1][3]
        # 5 here to exclude 14:00 data from features
        result_data[date] = [movement[:6], open_price[:6], high_price[:6], low_price[:6], close_price[:6], volume[:6], total_volume]
    if len(result_data) == 0:
        return None
    highest_price = max([result_data[d][2] for d in result_data])
    lowest_price = min([result_data[d][3] for d in result_data])
    highest_vol = max([result_data[d][5] for d in result_data])
    lowest_vol = min([result_data[d][5] for d in result_data])
    # movement_std = np.std(movements)
    volume_movement_std = np.std(volume_movements)
    volume_movement_mean = np.mean(volume_movements)
    return result_data, volume_movement_std, highest_price, lowest_price, highest_vol, lowest_vol, volume_movement_mean


def extract_news_headline_vocab(companies):
    word_count = {}
    for company in companies:
        for news_date in companies[company]:
            news_list = companies[company][news_date]
            for news in news_list:
                headline = news[1]
                words = word_tokenize(headline.strip())
                for word in words:
                    if word not in word_count:
                        word_count[word] = 0
                    word_count[word] += 1
    word_count = [(word, word_count[word]) for word in word_count]
    word_count.sort(key=lambda k: k[1], reverse=True)
    write = open('headline_vocab.txt', 'w')
    for pair in word_count:
        write.write(pair[0] + ' ' + str(pair[1]) + '\n')
    write.close()


def read_company_list(list_file='group-code.csv'):
    core30 = set()
    large70 = set()
    lines = open(list_file).readlines()
    for i in range(1, len(lines)):
        code, group = lines[i].strip().split(',')
        if group == 'CORE30':
            core30.add(code)
        elif group == 'LARGE70':
            large70.add(code)
    return core30, large70


def read_data(config):
    print('reading date')
    companies = {}
    company_names = os.listdir(config.price_dir)
    news = json.load(open(config.news_json))
    prices = {}
    high_price = 0
    low_price = 0
    high_vol = 0
    low_vol = 0
    # from ipdb import set_trace; set_trace() 
    for company_file in company_names:
        # print('price')
        company_name = (company_file.split('.')[0]).split('-')[0]
        # print(company_name, company_name in company_set)
        if company_name not in news:
            continue
        prices[company_name] = read_price(os.path.join(config.price_dir, company_file), config)
        if prices[company_name] is None or company_name not in news:
            continue
        if company_name not in companies:
            companies[company_name] = Company(company_name)
        companies[company_name].set_prices(prices[company_name][0])
        companies[company_name].set_movement_std(prices[company_name][1])
        companies[company_name].set_movement_mean(prices[company_name][6])  #NOTE
        high_price = max(high_price, max(prices[company_name][2]))
        low_price = min(low_price, min(prices[company_name][3]))
        high_vol = max(high_vol, max(prices[company_name][4]))
        low_vol = min(low_vol, min(prices[company_name][5]))
    for company in news:
        # print('news')
        # print(company, company in company_set)
        company_name = company
        if company_name not in companies:
            continue
            # companies[company_name] = Company(company_name)
        companies[company_name].set_news(news[company])
    return companies, prices, news, high_price, low_price, high_vol, low_vol


def write_stocknet_format(companies):
    for company in companies:
        write = open(company + '.txt', 'w')


if __name__ == '__main__':
    # companies = json.load(open('company_news.json'))
    # extract_news_headline_vocab(companies)
    config = utils.read_config('../config.yaml')
    map_hourly_price(config)

