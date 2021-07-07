import os
import torch

from colbert.utils.runs import Run
from colbert.utils.utils import print_message, save_checkpoint
from colbert.parameters import SAVED_CHECKPOINTS


def print_progress(scores):
    positive_avg, negative_avg = round(scores[:, 0].mean().item(), 2), round(scores[:, 1].mean().item(), 2)
    print("#>>>   ", positive_avg, negative_avg, '\t\t|\t\t', positive_avg - negative_avg)


def manage_checkpoints(args, colbert, optimizer, epoch_idx):
    arguments = args.input_arguments.__dict__

    path = os.path.join(Run.path, 'checkpoints')

    if not os.path.exists(path):
        os.mkdir(path)

    name = os.path.join(path, "colbert-{}.dnn".format(epoch_idx))
    save_checkpoint(name, epoch_idx, 0, colbert, optimizer, arguments)

    # if batch_idx in SAVED_CHECKPOINTS:
    #     name = os.path.join(path, "colbert-{}.dnn".format(batch_idx))
    #     save_checkpoint(name, 0, batch_idx, colbert, optimizer, arguments)
