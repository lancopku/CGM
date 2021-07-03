import csv
import json
import os
import copy
import numpy as np
import utils
import datetime
import pickle
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
        self.movement_std = 0

    def set_prices(self, price):
        self.prices = price

    def set_news(self, news):
        self.news = {datetime.datetime.strptime(date, '%Y-%m-%d'): news[date] for date in news}

    def set_movement_std(self, std):
        self.movement_std = std

    def judge_label(self, movement):
        # using this setting because of the weight in loss function
        # print(movement, self.std)
        if movement < -0.5 * self.movement_std:
            return 0
        elif movement > 0.5 * self.movement_std:
            return 1
        else:
            return 2


class GraphSnap:
    def __init__(self, companies, date, node_num, is_hourly=False):
        self.companies = companies
        self.nodes = [i for i in range(node_num)]
        self.node_headline = [[] for _ in range(node_num)]
        # self.node_body_bert = [np.zeros(768) for _ in range(node_num)]
        if is_hourly:
            self.node_features = [[0 for _ in range(7)] for _ in range(node_num)]
        else:
            # here should use 7 (original digit feature number) instead of feature_size, because the features will be converted later
            self.node_features = [[[0 for _ in range(5)] for _ in range(7)] for _ in range(node_num)]
        self.date = date
        self.movement = [2 for _ in range(node_num)]

    def add_node_text(self, node_id, news_list):
        for news in news_list:
            # id, headline, body, topic, rics, time
            n = News(news[0], news[1], news[2], news[3], news[4], news[5])
            self.node_headline[node_id].append(n.headline)
            # self.node_body_bert[node_id] += n.get_body_fixed_vec()

    def add_node_movement(self, node_id, movement):
        self.movement[node_id] = self.companies[node_id].judge_label(movement)

    def add_node_feature(self, node_id, feature, feature_scale, is_hourly=False):
        def scale(p, high, low):
            return (p - low) / float(high - low)

        if is_hourly:
            # feature[0] is the digit number of movement
            feature[1] = scale(feature[1], feature_scale[0], feature_scale[1])  # open
            feature[2] = scale(feature[2], feature_scale[0], feature_scale[1])  # high
            feature[3] = scale(feature[3], feature_scale[0], feature_scale[1])  # low
            feature[4] = scale(feature[4], feature_scale[0], feature_scale[1])  # close
            feature[5] = scale(feature[5], feature_scale[2], feature_scale[3])  # volume
            feature.append(self.companies[node_id].judge_label(feature[0]))  # label of movement
        else:
            feature[1] = [scale(d, feature_scale[0], feature_scale[1]) for d in feature[1]]  # open
            feature[2] = [scale(d, feature_scale[0], feature_scale[1]) for d in feature[2]]  # high
            feature[3] = [scale(d, feature_scale[0], feature_scale[1]) for d in feature[3]]  # low
            feature[4] = [scale(d, feature_scale[0], feature_scale[1]) for d in feature[4]]  # close
            feature[5] = [scale(d, feature_scale[2], feature_scale[3]) for d in feature[5]]  # volume
            feature.append(
                [self.companies[node_id].judge_label(movement) for movement in feature[0]])  # label of movement
        assert len(feature) == 7, feature
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
    def __init__(self, span, is_hourly=False):
        # each date in the span consists of (date, Snap)
        self.data = span
        if is_hourly:
            # need to exclude the last hour
            self.size = len(self.data) - 1
            self.span_news_mask, self.span_time_news_mask = self.get_span_news_mask(
                [d.node_headline for d in self.data[:-1]])
            self.span_text = [self.get_span_text(d.node_headline) for d in self.data[:-1]]
            # self.bert_vec = [d.node_body_bert for d in self.data[:-1]]
            self.span_nodes = [d.nodes for d in self.data[:-1]]
            # last_movement, open_price, high, low, close, volume, close-open, high-low
            self.span_features = self.convert_feature(
                [d.node_features for d in self.data[:-1]], is_hourly)
            self.span_movement = [node_movement for node_movement in self.data[-1].movement]
            self.span_movement_mask = self.data[-1].get_movement_mask()
            self.hourly_movement = [snap.movement for snap in self.data[1:]]
            self.hourly_movement_mask = [snap.get_movement_mask() for snap in self.data[1:]]
        else:
            self.size = len(self.data)
            self.span_text = [self.get_span_text(d[1].node_headline) for d in self.data]
            # self.bert_vec = [d[1].node_body_bert for d in self.data]
            self.span_nodes = [d[1].nodes for d in self.data]
            # movement, open_price, high, low, close, volume, close-open, high-low
            self.span_features = self.convert_feature(
                [d[1].node_features for d in self.data], is_hourly)  # excluding movement for now
            self.span_movement = [node_movement for node_movement in self.data[-1][1].movement]
            self.span_movement_mask = self.data[-1][1].get_movement_mask()
        self.movement_num = sum(self.span_movement_mask)
        # self.valid_node = [d[1].valid_node for d in self.data]

    @staticmethod
    def get_span_news_mask(node_headline_list):
        # node_headline_list: time, node, news_list
        node_mask = []  # mask for each node
        for node in range(len(node_headline_list[0])):
            # along the node
            time_mask = []
            for time in range(len(node_headline_list)):
                # along the time of each node
                time_mask.append(any([len(text) > 1 for text in node_headline_list[time][node]]))
            assert len(time_mask) == 5, len(time_mask)
            node_mask.append(any(time_mask))
        assert len(node_mask) == 101, len(node_mask)
        time_node_mask = []
        for time in range(len(node_headline_list)):
            mask = []
            for node in range(len(node_headline_list[time])):
                mask.append(any([len(text) > 1 for text in node_headline_list[time][node]]))
            time_node_mask.append(mask)
        return node_mask, time_node_mask

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
        self.span_text:     (span, node, seq)
        :param word_vocab:
        :return:
        '''
        mlen = max(
            [max([len(node_text) for node_text in snap_text]) for snap_text in self.span_text])
        if max_len is not None:
            max_len = min(max_len, mlen)
        else:
            max_len = mlen
        result = []
        span_mask = []
        for snap_text in self.span_text:
            snap_result = []
            snap_mask = []
            for node_text in snap_text:
                words = [word_vocab.word2id(word) for word in node_text][:max_len]
                mask = [1 for _ in words]
                snap_result.append(words + [word_vocab.PAD_token] * (max_len - len(words)))
                snap_mask.append(mask + [0] * (max_len - len(mask)))
                assert len(snap_result) == len(snap_mask)
            result.append(snap_result)
            span_mask.append(snap_mask)
        return result, span_mask

    # TODO
    @staticmethod
    def convert_feature(span_features, is_hourly=False):
        '''
        features: movement, open, high, low, close, volume
        :param span_features:  (span, node, features)
        :return:
        '''
        features = []
        if is_hourly:
            assert len(span_features) == 5, len(span_features)  # 5 hours
        else:
            assert len(span_features) == 6, len(span_features)  # 6 days
        for i in range(len(span_features)):
            node_features = []
            assert len(span_features[i]) == 101, len(span_features)  # 101 companies
            for node_id in range(len(span_features[i])):
                if is_hourly:
                    assert len(span_features[i][node_id]) == 7, len(span_features[i][node_id])  # 7 source features
                    # 0 is the digit number of last movement,
                    day_features = [span_features[i][node_id][0]] + [span_features[i][node_id][6]] + \
                                   [span_features[i][node_id][1]] + [span_features[i][node_id][2]] + \
                                   [span_features[i][node_id][3]] + [span_features[i][node_id][4]] + \
                                   [span_features[i][node_id][5]] + \
                                   [span_features[i][node_id][4] - span_features[i][node_id][1]] + \
                                   [span_features[i][node_id][2] - span_features[i][node_id][3]]
                else:
                    day_features = []
                    assert len(span_features[i][node_id]) == 7, len(span_features[i][node_id])  # 7 source features
                    assert len(span_features[i][node_id][1]) == 5, len(span_features[i][node_id][1])  # 5 hours
                    for hour in range(
                            len(span_features[i][node_id][1])):  # the trade info of last hour has already been excluded
                        day_features.append(
                            [span_features[i][node_id][0][hour]] + [span_features[i][node_id][6]] +  # movement
                            [span_features[i][node_id][1][hour]] + [span_features[i][node_id][2][hour]] +
                            [span_features[i][node_id][3][hour]] + [span_features[i][node_id][4][hour]] +
                            [span_features[i][node_id][5][hour]] +
                            [span_features[i][node_id][4][hour] - span_features[i][node_id][1][hour]] +
                            [span_features[i][node_id][2][hour] - span_features[i][node_id][3][hour]])
                node_features.append(day_features)
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

    def judge_dataset(self, date):
        train_start = datetime.datetime.strptime(self.config.train_start, "%Y-%m-%d")
        dev_start = datetime.datetime.strptime(self.config.dev_start, "%Y-%m-%d")
        test_start = datetime.datetime.strptime(self.config.test_start, "%Y-%m-%d")
        test_end = datetime.datetime.strptime(self.config.test_end, "%Y-%m-%d")
        if train_start <= date < dev_start:
            return 0
        elif dev_start <= date < test_start:
            return 1
        elif test_start <= date <= test_end:
            return 2
        return -1

    def set_mapping(self):
        for company in self.companies:
            self.com2id[company] = len(self.id2com)
            self.id2com.append(company)
        for id in range(len(self.id2com)):
            self.company_list.append(self.companies[self.id2com[id]])

    def load_adjacency(self):
        adj = [[0 for _ in range(self.node_num)] for _ in range(self.node_num)]
        adj_dict = json.load(open('./graph/corr_table.json'))
        for c1 in adj_dict:
            for c2 in adj_dict[c1]:
                score = abs(float(adj_dict[c1][c2]))
                if score >= self.config.edge_threshold:
                    adj[self.com2id[c1]][self.com2id[c2]] = score
        return adj

    def extract_data_span(self, is_hourly):
        dates = {}  # dict of GraphSnaps

        if is_hourly:
            for company in self.companies:
                for date in self.companies[company].prices:
                    if date not in dates:
                        dates[date] = [None for _ in range(6)]
                    for hour in range(5):
                        if dates[date][hour] is None:
                            snap = GraphSnap(self.company_list, date, self.config.node_num, is_hourly)
                        else:
                            snap = dates[date][hour]
                        # index 0 is movement
                        snap.add_node_movement(self.com2id[company], self.companies[company].prices[date][0][hour])
                        prev_movement = 0 if hour == 0 else self.companies[company].prices[date][0][hour - 1]
                        # 1 ~ 6 are the indices of features excluding movement
                        feature = [prev_movement] + [self.companies[company].prices[date][i][hour] for i in range(1, 6)]
                        snap.add_node_feature(self.com2id[company], feature, self.feature_scale, is_hourly)
                        if dates[date][hour] is None:
                            dates[date][hour] = snap
                    # add the movement of the last hour, which is to be predicted without features
                    if dates[date][5] is None:
                        snap = GraphSnap(self.company_list, date, self.config.node_num, is_hourly)
                    else:
                        snap = dates[date][5]
                    # index 0 is movement, 5 is the last hour
                    # we use 5 instead of -1 to prevent exception where there is not enough hour units
                    snap.add_node_movement(self.com2id[company], self.companies[company].prices[date][0][5])
                    if dates[date][5] is None:
                        dates[date][5] = snap
                for date in self.companies[company].news:
                    if date in dates:
                        daily_news = self.companies[company].news[date]
                        for i in range(len(daily_news)):
                            hour = int(daily_news[i][-1])
                            # here put in a news_list instead of the news itself according to the add_node_text method
                            if hour >= 5:
                                # the news after 5 is actually the news from yesterday, moved to this day manually
                                dates[date][0].add_node_text(self.com2id[company],
                                                             [self.companies[company].news[date][i]])
                            else:
                                dates[date][hour].add_node_text(self.com2id[company],
                                                                [self.companies[company].news[date][i]])
            data = [(date, dates[date]) for date in dates]
            print('date number', len(data))
            data.sort(key=lambda k: k[0])
            train_spans, dev_spans, test_spans = [], [], []
            for i in range(0, len(data)):
                date = data[i][0]
                dataset = self.judge_dataset(date)
                if dataset == 0:
                    # contains the daily snaps
                    train_spans.append(GraphSpan(data[i][1], is_hourly))
                elif dataset == 1:
                    dev_spans.append(GraphSpan(data[i][1], is_hourly))
                elif dataset == 2:
                    test_spans.append(GraphSpan(data[i][1], is_hourly))
        else:
            for company in self.companies:
                for date in self.companies[company].prices:
                    if date not in dates:
                        # snap here is daily
                        dates[date] = GraphSnap(self.company_list, date, self.config.node_num, is_hourly)
                    # 0 is the movement index, -1 is the last hour of the day
                    dates[date].add_node_movement(self.com2id[company], self.companies[company].prices[date][0][-1])
                    dates[date].add_node_feature(self.com2id[company], self.companies[company].prices[date],
                                                 self.feature_scale)  # including movement
                for date in self.companies[company].news:
                    if date not in dates:
                        # no trading record in this day, which means that day is on weekend, choose the previous day instead
                        if date - datetime.timedelta(days=1) in dates:
                            dates[date - datetime.timedelta(days=1)].add_node_text(self.com2id[company],
                                                                                   self.companies[company].news[date])
                        elif date - datetime.timedelta(days=2) in dates:
                            dates[date - datetime.timedelta(days=2)].add_node_text(self.com2id[company],
                                                                                   self.companies[company].news[date])
                    else:
                        dates[date].add_node_text(self.com2id[company], self.companies[company].news[date])
            data = [(date, dates[date]) for date in dates]
            data.sort(key=lambda k: k[0])
            train_spans, dev_spans, test_spans = [], [], []
            for i in range(self.config.time_span, len(data)):
                # time_span is the history length we want to consider
                left = i - self.config.time_span if i > self.config.time_span else 0
                date = data[i][0]
                dataset = self.judge_dataset(date)
                if dataset == 0:
                    train_spans.append(GraphSpan(data[left: i + 1]))
                elif dataset == 1:
                    dev_spans.append(GraphSpan(data[left: i + 1]))
                elif dataset == 2:
                    test_spans.append(GraphSpan(data[left: i + 1]))
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


def interpolate(day1, day2):
    return [(num1 + num2) / 2 for num1, num2 in zip(day1, day2)]


def read_price(fname):
    # read the trade history of one company
    csv.field_size_limit(sys.maxsize)
    data = {}
    with open(fname) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # first line is header
        for row in reader:
            date, time, open_price, high, low, close, volume = row
            date = datetime.datetime.strptime(date, "%Y-%m-%d")
            time = datetime.datetime.strptime(time, "%H:%M")
            if date not in data:
                data[date] = []
            # movement = (close - open_price) / float(open_price)
            data[date].append([float(open_price), float(high), float(low), float(close), float(volume), int(time.hour)])
    result_data = {}
    desserted = []
    movements = []
    for date in data:
        # 6 here to exclude 13:00 useless data
        daily_data = []
        # i: 0~5, hour 9~14
        hour = 9
        for i in range(min(len(data[date]), 6)):
            if data[date][i][-1] != hour:
                if hour == 9:
                    daily_data.append(data[date][i])
                elif hour == 14:
                    daily_data.append(data[date][i])
                else:
                    daily_data.append(interpolate(data[date][i - 1], data[date][i]))
                hour += 1
            if data[date][i][-1] == 15:
                break
            daily_data.append(data[date][i])
            hour += 1
        if hour == 14:
            daily_data.append(daily_data[-1])
        if len(daily_data) < 6:
            # the desserted data will be padded with all 0, which will be masked during prediction
            desserted.append(data[date])
            continue
        assert len(daily_data) == 6, (data[date], daily_data)
        open_price = [d[0] for d in daily_data][:6]
        high_price = [d[1] for d in daily_data][:6]
        low_price = [d[2] for d in daily_data][:6]
        close_price = [d[3] for d in daily_data][:6]
        volume = [d[4] for d in daily_data][:6]
        movement = [(c - o) / float(o) for c, o in zip(open_price, close_price)]
        movements.append(movement[-1])
        day_close = data[date][-1][3]
        # 5 here to exclude 14:00 data from features
        result_data[date] = [movement[:6], open_price[:5], high_price[:5], low_price[:5], close_price[:5], volume[:5]]
    print('desserted number', len(desserted))
    highest_price = max([result_data[d][2] for d in result_data])
    lowest_price = min([result_data[d][3] for d in result_data])
    highest_vol = max([result_data[d][-1] for d in result_data])
    lowest_vol = min([result_data[d][-1] for d in result_data])
    movement_std = np.std(movements)
    return result_data, movement_std, highest_price, lowest_price, highest_vol, lowest_vol


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


def read_data(company_set, price_dir, news_in_json):
    print('reading date')
    companies = {}
    company_names = os.listdir(price_dir)
    prices = {}
    high_price = 0
    low_price = 0
    high_vol = 0
    low_vol = 0
    for company_file in company_names:
        # print('price')
        company_name = company_file.split('-')[0]
        # print(company_name, company_name in company_set)
        if company_name not in company_set:
            continue
        prices[company_name] = read_price(os.path.join(price_dir, company_file))
        if company_name not in companies:
            companies[company_name] = Company(company_name)
        companies[company_name].set_prices(prices[company_name][0])
        companies[company_name].set_movement_std(prices[company_name][1])
        high_price = max(high_price, max(prices[company_name][2]))
        low_price = min(low_price, min(prices[company_name][3]))
        high_vol = max(high_vol, max(prices[company_name][4]))
        low_vol = min(low_vol, min(prices[company_name][5]))
    news = json.load(open(news_in_json))
    for company in news:
        # print('news')
        # print(company, company in company_set)
        company_name = company
        if company_name not in company_set:
            continue
        if company_name not in companies:
            companies[company_name] = Company(company_name)
        companies[company_name].set_news(news[company])
    return companies, prices, news, high_price, low_price, high_vol, low_vol


if __name__ == '__main__':
    companies = json.load(open('company_news.json'))
    extract_news_headline_vocab(companies)
