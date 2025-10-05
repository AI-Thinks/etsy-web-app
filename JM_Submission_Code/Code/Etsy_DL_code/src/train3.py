import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data2 import get_data
from model2 import Model
from utils import TrainingProcessTracker, create_argument_parser, MODEL_PATH

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_model(net, criterion, data_loader, tracker):
    print('Staring the evaluation ...')
    net.eval()
    with torch.no_grad():
        for images, texts, structured, price in tqdm(data_loader, desc='Validation'):
            images = images.to(device).float()
            texts = torch.stack(texts).T.to(device).float()
            structured = torch.stack(structured).T.to(device).float()
            price = price.to(device).float()

            outputs = net(images, texts, structured)
            loss = criterion(outputs.squeeze(), price.squeeze())

            tracker.log_val_step(loss, images, outputs, price)


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def run_train(args, sweep_mode=False):
    print('Using device:', device)
    train_loader, val_loader = get_data(args)

    print('train data size:', len(train_loader.dataset))
    print('val   data size:', len(val_loader.dataset))
    print(f"\nRunning training phase for: {args.extra_features}\n")

    model = Model(
        args.encoder_model,
        'texts' in args.extra_features,
        'structured' in args.extra_features,
        'visual' in args.extra_features
    )
    model = model.to(device)
    # print(model)
    # exit()

    print('model params (trainable, total):', count_parameters(model))

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    with TrainingProcessTracker(model, args, sweep_mode) as tracker:
        for epoch in range(args.num_epochs):  # loop over the dataset multiple times
            model.train()

            tracker.start_epoch(epoch)

            for i, (images, texts, structured, price) in enumerate(tqdm(train_loader, desc='Training')):
                images = torch.stack(texts).T.to(device).float()
                texts = torch.stack(texts).T.to(device).float()
                structured = torch.stack(structured).T.to(device).float()
                price = price.to(device).float()

                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = model(images, texts, structured)
                loss = criterion(outputs.squeeze(), price.squeeze())

                loss.backward()
                optimizer.step()

                tracker.log_train_step(loss, images, outputs, price)

            evaluate_model(model, criterion, val_loader, tracker)

            epochs_since_best_result = tracker.end_epoch(epoch)
            print()
            if epochs_since_best_result >= args.max_epochs_after_best_result:
                print(f'Stopping early at {epoch=}')
                break

    import pathlib
    save_dir = pathlib.Path(MODEL_PATH).parent  # Directory where model is saved
    model.to_file(save_dir)

    print('Finished Training')


def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    run_train(args)


if __name__ == '__main__':
    main()