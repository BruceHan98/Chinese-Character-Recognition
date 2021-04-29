import os
import time
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torchnet import meter

from dataset import DataSet
from network import Network
from config import config
from utils.log import logger
from dataset import data_loader


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def train(config):
    print(config)
    # set random seed
    setup_seed(20)

    train_loader = data_loader(config, mode='train')
    val_loader = data_loader(config, mode='val')

    model = Network(config.class_num)

    if config.checkpoint:
        logger.info("Loading pretrained model: %s" % config.checkpoint)
        model.load_state_dict(torch.load(config.checkpoint))

    device = torch.device('cuda:0' if config.use_gpu else 'cpu')
    criterion = nn.CrossEntropyLoss().to(device)
    optim = torch.optim.Adam(params=model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    model.to(device)

    total_iter = int(DataSet(config, mode='train').__len__() / config.batch_size)

    logger.info("Start training...")
    for epoch in range(config.epoch_num):
        model.train()
        train_loss = 0
        num_correct = 0
        for step, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optim.zero_grad()
            outs = model(inputs)
            loss = criterion(outs, targets)
            loss.backward()
            optim.step()

            train_loss += loss.item()
            pred = outs.argmax(dim=1)
            num_correct += torch.eq(pred, targets).sum().float().item()
            if (step + 1) % config.print_freq == 0:
                train_loss = train_loss / config.print_freq
                train_acc = num_correct / (config.batch_size * config.print_freq)
                logger.info('epoch: [%d/%d], iter: %4d/%4d, lr: %.6f, loss: %.5f, acc: %.5f' %
                            (epoch + 1, config.epoch_num, step + 1, total_iter, optim.param_groups[0]['lr'], train_loss, train_acc))
                train_loss = 0
                num_correct = 0

        if (epoch + 1) % config.lr_decay_freq == 0:
            if optim.param_groups[0]['lr'] > 0.000001:
                lr = config.lr * (0.5 ** int((epoch + 1) / config.lr_decay_freq))
                for param_group in optim.param_groups:
                    param_group['lr'] = lr

        model_save_name = time.strftime(f'epoch-{epoch}-%Y-%m-%d-%H-%M.pth')
        model_save_path = os.path.join(config.model_dir, model_save_name)
        torch.save(model.state_dict(), model_save_path)
        logger.info("Model saved in: %s" % model_save_path)

        # validation
        if (epoch + 1) >= config.val_start and (epoch + 1) % config.val_freq == 0 or (epoch + 1) == config.val_start:
            logger.info("Start Validation...")
            _, acc = val(model, val_loader)
            logger.info("Validation accuracy: %.3f" % acc)

        # test
        if (epoch + 1) >= config.test_start and (epoch + 1) % config.test_freq == 0 or (epoch + 1) == config.test_start:
            test(config, model_save_path)


def val(model, loader):
    confusion_matrix = meter.ConfusionMeter(config.class_num)
    model.eval()
    with torch.no_grad():
        for step, (data, target) in tqdm(enumerate(loader)):
            device = torch.device('cuda:0' if config.use_gpu else 'cpu')
            val_input = data.to(device)
            out = model(val_input)
            confusion_matrix.add(out.detach(), target.detach())

    cm_value = confusion_matrix.value()
    correct_sum = 0
    for i in range(config.class_num):
        correct_sum += cm_value[i][i]
    accuracy = 100. * correct_sum / (cm_value.sum())
    return confusion_matrix, accuracy


def test(config, model_path=''):
    test_loader = data_loader(config, mode='test')
    print(config)
    logger.info("Start testing...")
    model = Network(config.class_num).eval()

    if model_path:
        model.load_state_dict(torch.load(model_path))
    elif config.checkpoint:
        logger.info("Loading pretrained model: %s" % config.checkpoint)
        model.load_state_dict(torch.load(config.checkpoint))
    else:
        logger.error("No model file to load")

    device = torch.device('cuda:0' if config.use_gpu else 'cpu')
    model.to(device)

    results = list()

    total_iter = int(DataSet(config, mode='test').__len__() / config.batch_size)

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outs = model(inputs)
            pred = torch.max(outs, 1)[1]

            acc = (pred == targets).float().sum() / len(targets)
            results += ((pred == targets).float().to('cpu').numpy().tolist())

            logger.info('iter: %d/%d | acc: %.5f' % (step + 1, total_iter, acc))

        results = np.array(results)
        logger.info('Top 1 acc: %.5f' % (np.sum(results) / len(results)))


if __name__ == '__main__':
    if config.mode == 'train':
        train(config)
    elif config.mode == 'test':
        test(config)
