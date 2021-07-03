from __future__ import division
from __future__ import print_function

import time
import argparse
import torch
import torch.nn as nn
from optims import Optim
import utils
import lr_scheduler as L
from models import *
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import sys
import dcca 
import os
import random
from Data import *
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
current_dir = os.getcwd()
sys.path.insert(0, parent_dir)


def parse_args():
    parser = argparse.ArgumentParser(description='train.py')
    parser.add_argument('-gpus', default=[1], type=int,
                        help="Use CUDA on the listed devices.")
    parser.add_argument('-restore', type=str, default=None,
                        help="restore checkpoint")
    parser.add_argument('-seed', type=int, default=1234,
                        help="Random seed")
    parser.add_argument('-use_medium', default=False, action='store_true',
                        help="whether to use the third class medium during training")
    parser.add_argument('-notrain', default=False, action='store_true',
                        help="train or not")
    parser.add_argument('-multi_turn', default=False, action='store_true',
                        help="train multiple times and test")
    parser.add_argument('-log', default='', type=str,
                        help="log directory")
    parser.add_argument('-verbose', default=False, action='store_true',
                        help="verbose")
    parser.add_argument('-debug', default=False, action="store_true",
                        help='whether to use debug mode')
    parser.add_argument('-config', default='', type=str,
                        help="config file")

    opt = parser.parse_args()
    # 用config.data来得到config中的data选项
    config = utils.read_config(opt.config)
    return opt, config


# set opt and config as global variables
args, config = parse_args()
random.seed(args.seed)
np.random.seed(args.seed)


# Training settings

def set_up_logging():
    # log为记录文件
    # config.log是记录的文件夹, 最后一定是/
    # opt.log是此次运行时记录的文件夹的名字
    if not os.path.exists(config.log):
        os.mkdir(config.log)
    if args.log == '':
        log_path = config.log + utils.format_time(time.localtime()) + '/'
    else:
        log_path = config.log + args.log + '/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    logging = utils.logging(log_path + 'log.txt')  # 往这个文件里写记录
    logging_csv = utils.logging_csv(log_path + 'record.csv')  # 往这个文件里写记录
    logging_test = utils.logging_test(log_path + 'test_output.csv')
    for k, v in config.items():
        logging("%s:\t%s\n" % (str(k), str(v)))
    logging("\n")
    return logging, logging_csv, logging_test, log_path

logging, logging_csv, logging_test, log_path = set_up_logging()
use_cuda = torch.cuda.is_available()


def train(model, dataloader, scheduler, optim, updates):
    scores = []
    score = 0.
    best_epoch = 0
    min_mse = 1e8
    adjs = dataloader.adjs
    print('training')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cca_loss_func = dcca.cca_loss(config.hidden_size, True, device)
    for epoch in range(1, config.epoch + 1):
        total_right = 0
        total_num = 0
        total_cca_loss = 0.
        total_loss = 0.
        start_time = time.time()

        if config.schedule:
            scheduler.step()
            print("Decaying learning rate to %g" % scheduler.get_lr()[0])
        elif config.learning_rate_decay < 1:
            optim.updateLearningRate(score, epoch)
        model.train()

        train_data = dataloader.train
        random.shuffle(train_data)
        # from ipdb import set_trace; set_trace() 
        for span in tqdm(train_data, disable=not args.verbose):
            model.zero_grad()
            span_nodes, node_text, text_mask, node_features, last_movement, movement_mask, news_mask, movement_num = span
            if movement_num == 0:
                continue
            if use_cuda:
                node_text, node_features = node_text.cuda(), node_features.cuda()
                text_mask = text_mask.cuda()
                adjs = [adjs[i].cuda() for i in range(len(adjs))]
                span_nodes = span_nodes.cuda()
                last_movement = last_movement.cuda()
                movement_mask = movement_mask.cuda()
                news_mask = news_mask.cuda()
            # print('last movement', last_movement)
            if args.use_medium:
                weight = torch.Tensor([1, 1, 1])
                total_mask = news_mask
                # total_mask = movement_mask
            else:
                weight = torch.Tensor([1, 1, 0])
                total_mask = movement_mask * news_mask
            total_num += total_mask.sum().item()
            # from ipdb import set_trace; set_trace()
            # if total_num == 0: continue
            last_output, cca_volume, cca_price = model(span_nodes, node_text, text_mask, node_features, adjs)
            # right_num = torch.sum((last_output.max(-1)[1] == last_movement).float() * total_mask ).item()
            # total_right += right_num

            if use_cuda:
                weight = weight.cuda()
            # last_loss = F.cross_entropy(last_output, last_movement, weight=weight, reduction='none') * total_mask
            # from ipdb import set_trace; set_trace() 
            last_loss = F.mse_loss(last_output.squeeze(-1), last_movement, reduction='none') * total_mask
            cca_loss = cca_loss_func(cca_volume, cca_price).sum()
            loss = last_loss.sum() + cca_loss
            # from ipdb import set_trace; set_trace() 
            loss.backward()
            total_loss += loss.data.item()
            total_cca_loss += cca_loss.data.item()

            optim.step()
            updates += 1  # 进行了一次更新

        print('train total movement number', total_num)
        # logging中记录的是每次更新时的epoch，time，updates，correct等基本信息.
        # 还有score分数的信息
        logging("time: %6.3f, epoch: %3d, updates: %8d, train loss: %6.3f, train cca_loss:%6.3f\n"
                % (time.time() - start_time, epoch, updates, total_loss, total_cca_loss))
        logging("train_loss: %6.3f train_cca_loss: %6.3f \n"%(total_loss, total_cca_loss))
        # logging("train_acc: %.3f\n"%(total_right / float(total_num)))
        score = eval(model, dataloader, epoch, updates, do_test=False)
        scores.append(score)
        if score <= min_mse:
            save_model(log_path + 'best_model_checkpoint.pt', model, optim, updates)
            min_mse = score
            best_epoch = epoch
        model.train()

    model = load_model(log_path + 'best_model_checkpoint.pt', model)
    os.mknod(log_path + 'best_epoch_' +str(best_epoch))
    test_mse = eval(model, dataloader, -1, -1, True)
    return min_mse, test_mse


def eval(model, dataloader, epoch, updates, do_test=False):
    model.eval()
    y_pred, y_true = [], []
    total_num = 0
    if do_test:
        data = dataloader.test
    else:
        data = dataloader.dev
    adjs = dataloader.adjs
    for span in tqdm(data, disable=not args.verbose):
        model.zero_grad()
        span_nodes, node_text, text_mask, node_features, last_movement, movement_mask, news_mask, movement_num = span
        if movement_num == 0:
            continue
        if use_cuda:
            node_text, node_features = node_text.cuda(), node_features.cuda()
            text_mask = text_mask.cuda()
            adjs = [adjs[i].cuda() for i in range(len(adjs))]
            span_nodes = span_nodes.cuda()
            last_movement = last_movement.cuda()
            movement_mask = movement_mask.cuda()
            news_mask = news_mask.cuda()
        # print('last movement', last_movement)
        if args.use_medium:
            weight = torch.Tensor([1, 1, 1])
            total_mask = news_mask
            # total_mask = movement_mask
        else:
            weight = torch.Tensor([1, 1, 0])
            total_mask = movement_mask * news_mask
        last_output, _, _ = model(span_nodes, node_text, text_mask, node_features, adjs)
        for test_output_item, test_label_item, m in zip(last_output.cpu().detach().numpy().tolist(), last_movement.cpu().detach().numpy().tolist(), total_mask.cpu().detach().numpy().tolist()):
            if m :
                y_pred.append(test_output_item)
                y_true.append(test_label_item)
                if do_test: logging_test([test_output_item, test_label_item])
    
        total_num += total_mask.sum().item()
        # total_right += right_num
    # acc = total_right / float(total_num)
    # from ipdb import set_trace; set_trace() 
    rmse = mean_squared_error(y_true, y_pred)
    logging_csv([epoch, updates, rmse])
    print('eval total movement number', total_num)
    print('evaluating rmse %.6f' % rmse)
    return rmse


def save_model(path, model, optim, updates):
    '''保存的模型是一个字典的形式, 有model, config, optim, updates.'''

    # 如果使用并行的话使用的是model.module.state_dict()
    model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()
    checkpoints = {
        'model': model_state_dict,
        'config': config,
        'optim': optim,
        'updates': updates}
    torch.save(checkpoints, path)


def load_model(path, model):
    checkpoints = torch.load(path)
    model.load_state_dict(checkpoints['model'])
    return model


def main(vocab, dataloader):
    # 设定种子
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    # checkpoint
    if args.restore:  # 存储已有模型的路径
        print('loading checkpoint...\n')
        checkpoints = torch.load(os.path.join(log_path, args.restore))

    torch.backends.cudnn.benchmark = True

    # model
    print('building model...\n')
    # configure the model
    # Model and optimizer
    # model = GLSTM(config, vocab)
    model = CGM(config, vocab)
    if args.restore:
        model.load_state_dict(checkpoints['model'])
    if use_cuda:
        model.cuda()
    if len(args.gpus) > 1:  # 并行
        model = nn.DataParallel(model, device_ids=args.gpus, dim=1)
    logging(repr(model) + "\n\n")  # 记录这个文件的框架

    # total number of parameters
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]

    logging('total number of parameters: %d\n\n' % param_count)

    # updates是已经进行了几个epoch, 防止中间出现程序中断的情况.
    if args.restore:
        updates = checkpoints['updates']
        ori_updates = updates
    else:
        updates = 0

    # optimizer
    if args.restore:
        optim = checkpoints['optim']
    else:
        optim = Optim(config.optim, config.learning_rate, config.max_grad_norm, lr_decay=config.learning_rate_decay,
                      start_decay_at=config.start_decay_at)

    optim.set_parameters(model.parameters())
    if config.schedule:
        scheduler = L.CosineAnnealingLR(optim.optimizer, T_max=config.epoch)
    else:
        scheduler = None

    if not args.notrain:
        max_acc, test_acc = train(model, dataloader, scheduler, optim, updates)
        logging("Best accuracy: %.2f, test accuracy: %.2f\n" % (max_acc * 100, test_acc * 100))
        return test_acc
    else:
        assert args.restore is not None
        eval(model, vocab, dataloader, 0, updates, do_test=True)


if __name__ == '__main__':
    vocab = Vocab(config.vocab_file, config.emb_file, emb_size=config.emb_size,
                  vocab_size=config.vocab_size, use_pre_emb=config.use_pre_emb)
    # Load data
    start_time = time.time()
    print('loading data...\n')
    dataloader = DataLoader(config, vocab, debug=args.debug)
    print("DATA loaded!")
    # data
    print('loading time cost: %.3f' % (time.time() - start_time))
    main(vocab, dataloader)
