import argparse
import hashlib
import json
import math
import os
import pathlib
import pickle
import platform
import time
import uuid
from functools import cache

import numpy as np
import requests
import wandb
from sklearn.metrics import r2_score, mean_squared_error

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
DATA_DIR = pathlib.Path(os.getenv('DATA_DIR', PROJECT_ROOT / 'data'))
PROJECT_ID = 'ETSY-PRICE-PREDICTION'
WANDB_ENTITY = os.getenv('WANDB_ENTITY', 'doghonadze-nk')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# EMBEDDINGS_FILE_NAME = 'embeddings_dict.json.gz'
EMBEDDINGS_FILE_NAME = '/Users/nevinselby/Documents/UWMadison/DataAnalystIntern/Project 2/nevin/joint_embedding_dict.json.gz'
# EMBEDDINGS_FILE_NAME = '/Users/nevinselby/Documents/UWMadison/DataAnalystIntern/Project 2/nevin/agg_joint_embedding_dict.json.gz'
MODEL_PATH = '/Users/nevinselby/Documents/UWMadison/DataAnalystIntern/Project 2/model.json'
# MODEL_PATH = '/Users/nevinselby/Documents/UWMadison/DataAnalystIntern/Project 2/nevin/agg_joint_model.pth'
OUT_FOLDER = "/Users/nevinselby/Documents/UWMadison/DataAnalystIntern/Project 2/out"

STRUCTURED_FEATURES = {
    'width_inches': ('Painting Width In Inches ', float),
    'height_inches': ('Painting Height in Inches', float),
    'num_images': ('Number of Images In the Listing', int),
    'is_rare_find': ('Is Rare Find', bool),
    'is_only_one_available': ('Is Only One Available', bool),
    'is_handmade': ('Is Handmade', bool),
    'number_of_reviews': ('Number of seller reviews', float),
    'sales': ('Number of Seller Sales', float),
    'admirers': ('Number of Seller Admirers', float),
    'materials_canvas': ('Canvas', bool),
    'materials_oil': ('Oil', bool),
    'materials_acrylic': ('Acrylic', bool),
    'framed': ('Framed', bool),
}


def load_json(file_path):
    with open(file_path) as inp:
        return json.load(inp)


def write_json(data, file_path):
    with open(file_path, 'w') as out:
        json.dump(data, out, indent=4, ensure_ascii=False)
    return file_path


def load_pickle(file_path):
    with open(file_path, 'rb') as inp:
        return pickle.load(inp)


def write_pickle(data, file_path):
    with open(file_path, 'wb') as out:
        pickle.dump(data, out)


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


class AverageMeter:
    def __init__(self, name):
        self.name = name
        self.sum = 0
        self.count = 0

    @property
    def avg(self):
        return self.sum / self.count

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    def __str__(self):
        return f'{self.name}: {self.avg:.4f}'

    def reset(self):
        self.sum = 0
        self.count = 0


class ModelCheckpointTracker:
    def __init__(self, model):
        self._model = model
        self._files = []

    def save_model(self):
        self._files = self._model.to_file(pathlib.Path(wandb.run.dir))

    def save_to_wandb(self):
        for file in self._files:
            wandb.save(file.name, policy='now')
            print(f'Saved the file {file.name}')


def copy_file(src_path, dest_path):
    with src_path.open('rb') as inp:
        dest_path.parent.mkdir(exist_ok=True, parents=True)
        with dest_path.open('wb') as out:
            out.write(inp.read())


class PredictionsTracker:
    def __init__(self, name):
        self.name = name

        self.all_outputs = []
        self.all_targets = []
        self.all_items = []

    def step(self, inputs, outputs, targets):
        # self.all_items.extend(inputs)
        self.all_targets.extend(targets.tolist())
        self.all_outputs.extend(outputs.tolist())

    def reset(self):
        self.all_outputs = []
        self.all_targets = []
        self.all_items = []

    def r2_score(self):
        return r2_score(np.array(self.all_targets), np.array(self.all_outputs))

    def mse_score(self):
        return mean_squared_error(np.array(self.all_targets), np.array(self.all_outputs))


class TrainingProcessTracker:
    def __init__(self, model, args, sweep_mode):
        self._args = args
        self._model = model
        self._sweep_mode = sweep_mode

        self._cur_epoch = 0

        self._train_predictions = PredictionsTracker('training predictions')
        self._train_loss_meter = AverageMeter('avg_train_loss')

        self._val_predictions = PredictionsTracker('val predictions')
        self._val_loss_meter = AverageMeter('avg_val_loss')

        self._epoch_start_time = time.perf_counter()

        self._best_score_epoch = 0
        self._summary_log = {}

        self._model_tracker = ModelCheckpointTracker(model)

        self._best_train_loss = math.inf
        self._best_train_r2 = -math.inf

        self._best_val_loss = math.inf
        self._best_val_r2 = -math.inf

    def __enter__(self):
        if not self._sweep_mode:
            wandb.init(
                project=PROJECT_ID, name=f'{uuid.uuid4()}_{platform.node()}',
                save_code=False, tags=[], config=self._args
            )

        wandb.watch(self._model, log='all', log_graph=True)
        # wandb.run.log_code(pathlib.Path(__file__).parent).wait()

        # save all the source code
        wandb_dir = pathlib.Path(wandb.run.dir)
        for file in (PROJECT_ROOT / 'src').iterdir():
            if file.is_file() and file.name.endswith('py'):
                copy_file(file, wandb_dir / 'code' / 'src' / file.name)
        wandb.save('src', policy='now')

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # log summary
        wandb.run.summary.update(self._summary_log)
        if not self._sweep_mode:
            wandb.finish()

    def log_train_step(self, train_loss, inputs, outputs, targets):
        self._train_loss_meter.update(train_loss.item(), len(outputs))
        self._train_predictions.step(inputs, outputs, targets)
        wandb.log({'train_loss': train_loss.item()})

    def log_val_step(self, val_loss, inputs, outputs, targets):
        self._val_loss_meter.update(val_loss.item(), len(outputs))
        self._val_predictions.step(inputs, outputs, targets)
        # wandb.log({'val_loss': val_loss.item()})

    def start_epoch(self, epoch):
        self._epoch_start_time = time.perf_counter()
        self._cur_epoch = epoch

    @property
    def avg_train_loss(self):
        return self._train_loss_meter.avg

    @property
    def avg_val_loss(self):
        return self._val_loss_meter.avg

    def end_epoch(self, epoch):
        self._cur_epoch = epoch
        print(f'Epoch {epoch} ended ==========')

        epoch_time = time.perf_counter() - self._epoch_start_time
        print(f'Epoch time {epoch_time :.2f} sec')

        val_r2 = self._val_predictions.r2_score()
        train_r2 = self._train_predictions.r2_score()

        mean_train_loss = self._train_loss_meter.avg
        mean_val_loss = self._val_loss_meter.avg

        self._best_train_loss = min(self._best_train_loss, mean_train_loss)
        self._best_train_r2 = max(self._best_train_r2, train_r2)

        self._best_val_loss = min(self._best_val_loss, mean_val_loss)
        self._best_val_r2 = max(self._best_val_r2, val_r2)

        best_train_loss = self._best_train_loss
        best_train_r2 = self._best_train_r2

        best_val_loss = self._best_val_loss
        best_val_r2 = self._best_val_r2

        log_step = {
            'epoch_train_loss': mean_train_loss,
            'best_train_loss': best_train_loss,
            'epoch_train_r2': train_r2,
            'best_train_r2': best_train_r2,

            'epoch_val_loss': mean_val_loss,
            'best_val_loss': best_val_loss,
            'epoch_val_r2': val_r2,
            'best_val_r2': best_val_r2,
        }
        wandb.log(log_step)

        print(
            self._train_loss_meter,  f'{best_train_loss=:.4f}', f'{train_r2=:.4f}', f'{best_train_r2=:.4f}',
            self._val_loss_meter, f'{best_val_loss=:.4f}', f'{val_r2=:.4f}', f'{best_val_r2=:.4f}'
        )

        if mean_val_loss <= best_val_loss:
            self._best_score_epoch = epoch
            self._model_tracker.save_model()
            self._model_tracker.save_to_wandb()

        self._train_loss_meter.reset()
        self._train_predictions.reset()
        self._val_loss_meter.reset()
        self._val_predictions.reset()

        return epoch - self._best_score_epoch


def download_files(run, run_id):
    temp_project_dir = PROJECT_ROOT / 'runs' / run_id
    temp_project_dir.mkdir(parents=True, exist_ok=True)
    for file in run.files():
        if file.name.endswith('json') or file.name.endswith('pt'):
            out_path = temp_project_dir / file.name
            if out_path.exists():
                print('Skipped:', out_path)
            else:
                print('Downloading:', out_path)
                file.download(temp_project_dir, replace=True)
    return temp_project_dir


@cache
def get_text_embedding(text):
    resp = requests.post(
        'https://api.openai.com/v1/embeddings',
        headers={'Authorization': f'Bearer {OPENAI_API_KEY}'},
        json={'input': text, 'model': 'text-embedding-ada-002'}
    )
    resp.raise_for_status()
    return resp.json()['data'][0]['embedding']


@cache
def get_hash(text, method='sha256'):
    hasher = getattr(hashlib, method)()
    hasher.update(text.encode('utf-8'))
    return hasher.hexdigest()


def create_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../out/', help='directory with data')
    parser.add_argument('--debug', type=bool, default=False, help='include additional debugging ifo')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--train_portion', type=float, default=0.8)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--unfreeze_epoch', type=int, default=2)
    parser.add_argument('--encoder_model', default='resnet18', choices=[
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
    ])
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument('--extra_features', nargs='+', choices=['texts', 'structured', 'visual'], default=['visual'])
    parser.add_argument('--max_epochs_after_best_result', type=int, default=20)
    return parser
