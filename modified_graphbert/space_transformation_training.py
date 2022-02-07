import logging
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from helpers.create_dataset import load_src_embeddings, load_tgt_embeddings
from helpers.logging_helpers import Writer, ExitOnExceptionHandler
from helpers.parse_args import load_args_transformation
from space_transformation.dataset import EmbDataset
from space_transformation.projection import ProjectionLayer


# ----------------------------------------------------------------
# create datasets
# ----------------------------------------------------------------

def create_tgt_embeddings_matrix(shape, embeddings, synsets):
    tgt_matrix = np.zeros(shape)
    for i, synset in enumerate(synsets):
        tgt_matrix[i, :] = embeddings[synset]
    return tgt_matrix


def train_dev_split(matrix, synsets_ordered, tgt_embeddings):
    train_src_matrix, dev_src_matrix, train_synsets, dev_synsets = train_test_split(matrix, synsets_ordered,
                                                                                    test_size=100 / matrix.shape[0])

    train_tgt_matrix = create_tgt_embeddings_matrix(train_src_matrix.shape, tgt_embeddings, train_synsets)
    dev_tgt_matrix = create_tgt_embeddings_matrix(dev_src_matrix.shape, tgt_embeddings, dev_synsets)

    return train_src_matrix, dev_src_matrix, train_tgt_matrix, dev_tgt_matrix


# ----------------------------------------------------------------
# create dataloaders
# ----------------------------------------------------------------

def load_data(node_path, src_embeddings_path, batch_size):
    tgt_embeddings = load_tgt_embeddings(node_path)
    src_matrix, src_synsets_ordered = load_src_embeddings(src_embeddings_path)

    train_src_matrix, dev_src_matrix, train_tgt_matrix, dev_tgt_matrix = train_dev_split(src_matrix,
                                                                                         src_synsets_ordered,
                                                                                         tgt_embeddings)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    train_dataset = EmbDataset(train_src_matrix, train_tgt_matrix)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    dev_dataset = EmbDataset(dev_src_matrix, dev_tgt_matrix)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size)

    return train_loader, dev_loader, train_src_matrix.shape[-1], train_tgt_matrix.shape[-1]


# ----------------------------------------------------------------
# create dataloaders
# ----------------------------------------------------------------

def main():
    args = load_args_transformation()
    # Load data
    logging.info('Loading datasets')
    train_loader, dev_loader, input_shape, output_shape = load_data(args.node_path, args.src_embeddings_path,
                                                                    args.batch_size)
    logging.info('Initializing model')
    model = ProjectionLayer(emb_size=input_shape, hidden_sizes=args.hidden_states, target_size=output_shape,
                            device=args.device).to(args.device)
    loss_function = nn.CosineEmbeddingLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
    logging.info('Training process is started')
    model.train_and_evaluate(train_loader, dev_loader, loss_function, optimizer, scheduler, args.epochs)
    logging.info('Saving trained model')
    model.save(args.out_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.StreamHandler(),
                                  ExitOnExceptionHandler(
                                      filename=f'graph-bert_training_{datetime.now().strftime("%Y_%m_%d_%H-%M-%S")}.log'
                                  )])
    sys.stdout = Writer(logging.info)
    sys.stderr = Writer(logging.error)
    main()
