import gzip
import json
import pathlib
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from utils import get_hash, EMBEDDINGS_FILE_NAME, create_argument_parser

STRUCTURED_FEATURES = {
    'width_inches': float,
    'height_inches': float,
    'num_images': int,
    'is_rare_find': bool,
    'is_only_one_available': bool,
    'is_handmade': bool,
    'number_of_reviews': float,
    'sales': float,
    'admirers': float,
    'materials_canvas': bool,
    'materials_oil': bool,
    'materials_acrylic': bool,
    'framed': bool,
}


class EmbeddingsLoader:
    def __init__(self, embeddings_file_path):
        print('Loading embeddings')
        with gzip.open(embeddings_file_path, 'rt', encoding='utf-8') as f:
            self.embeddings = json.load(f)
        print('Loaded embeddings')

    def __call__(self, text):
        return self.embeddings[get_hash(text)]


class MyDataSet(Dataset):
    def __init__(self, data, embeddings_loader, img_transform=None):
        super().__init__()
        self.embeddings_loader = embeddings_loader
        self.img_transform = img_transform
        self.data = []
        for items in tqdm(data.values(), desc='Processing and Loading the dataset'):
            new_items = []
            for item in items:
                # full_embedding = self.embeddings_loader(f"{item['listing_id']}")
                # text_embedding = full_embedding[512:]
                # print("\nType of text_embed:", type(text_embedding))
                # img_embedding = torch.tensor(full_embedding[:512])
                # print("\nType of img_embed:", type(img_embedding))
                
                
                text_embedding = self.embeddings_loader(f"{item['listing_id']}")
                
                structured = [
                    item['structured_features'][key] / 100 for key in sorted(item['structured_features'].keys())
                ]

                img_path = item['image_path']
                with open(img_path, 'rb') as inp:
                    image = Image.open(inp)
                    if self.img_transform:
                        image = self.img_transform(image)

                # print(type(text_embedding))
                # print(type(image))
                # exit()

                log_price_usd = item['log_price_usd']
                new_items.append((
                    # img_embedding, text_embedding, structured, log_price_usd
                    image, text_embedding, structured, log_price_usd
                ))
            self.data.append(new_items)
            # print("\nData from MyDataSet:\n", len(new_items))
            # print(new_items[0][0].shape, len(new_items[0][1]))
            # print(new_items[1][0].shape, len(new_items[1][1]))
            # print(new_items[2][0].shape, len(new_items[2][1]))
            # exit()

    def __getitem__(self, index):
        return random.choice(self.data[index])

    def __len__(self):
        return len(self.data)


class ScaledDataSet(Dataset):
    def __init__(self, dataset, scaler=None):
        self.dataset = dataset
        self.scaler = scaler

        if self.scaler is None:
            print('Using the default scaler values')
            self.scaler = StandardScaler()
            self.scaler.mean_ = np.array([5.44])
            self.scaler.var_ = np.array([0.86])
            self.scaler.scale_ = np.sqrt(self.scaler.var_)

            # print('Fitting the scaler to the training data')
            # self.scaler = StandardScaler()
            #
            # prices = []
            # for *_, price_usd in self.dataset:
            #     prices.append([price_usd])
            # self.scaler.fit(np.array(prices))
            # print(f'{self.scaler.mean_=}, {self.scaler.var_=}')

    def __getitem__(self, index):
        *outputs, price_usd = self.dataset[index]
        price_usd = self.scaler.transform(np.array([[price_usd]])).item()
        return *outputs, price_usd

    def __len__(self):
        return len(self.dataset)


def extract_extra_features(row):
    # materials = eval(row.materials)
    return {
        'width_inches': np.log(row.width_inches),
        'height_inches': np.log(row.height_inches),
        'num_images': np.log(row.num_images),
        'is_rare_find': row.is_rare_find,
        'is_only_one_available': row.is_only_one_available,
        'is_handmade': row.is_handmade,
        'number_of_reviews': np.log(row.seller_number_of_reviews + 1e-6),
        'sales': np.log(row.sales + 1e-6),
        'admirers': np.log(row.admirers + 1e-6),
        'materials_canvas': row.materials_canvas,
        'materials_oil': row.materials_oil,
        'materials_acrylic': row.materials_acrylic,
        'framed': row.framed
    }


def read_data(out_dir):
    data = defaultdict(list)

    all_dataframes = []
    for seller_dir in out_dir.iterdir():
        if seller_dir.is_file():
            continue

        images_dir = seller_dir / 'images_small'

        data_df_old = pd.read_csv(
            seller_dir.parent / 'etsy_final.csv.zip',
            usecols=['listing_id', 'images', 'painting_title', 'painting_description']
        )
        data_df_new = pd.read_csv(
            seller_dir.parent / 'EtsyCleanMainData.csv', usecols=[
                'seller_name',
                'seller_url',
                'seller_rating',
                'seller_location',
                'seller_number_of_reviews',
                'listing_id',
                'listing_url',
                'main_image_url',
                'num_images',
                # 'painting_title',
                'num_words_in_title',
                # 'painting_description',
                'painting_description_num_words',
                'is_rare_find',
                'is_only_one_available',
                # 'is_handmade',
                # 'highlights',
                # 'materials',
                # 'returns_accepted',
                # 'return_window',
                'sales',
                'admirers',
                # 'width_inches',
                # 'height_inches',
                # 'price_usd',
                'painting_valid',
                'location_clean',
                'material',
                # 'canvasMaterialOld', 'mixedmediaMaterialOld', 'oilMaterialOld', 'acrylicMaterialOld',
                'canvasMaterial_New',
                'mixedmediaMaterial_New',
                'oilMaterial_New',
                'acrylicMaterial_New',
                'isHandmade_New',
                'isFramed_New',
                'width_clean',
                'height_clean',
                'price_clean'
            ]
        ).rename(columns={
            'price_clean': 'price_usd',
            'height_clean': 'height_inches',
            'width_clean': 'width_inches',
            'isFramed_New': 'framed',
            'isHandmade_New': 'is_handmade',
            'acrylicMaterial_New': 'materials_acrylic',
            'oilMaterial_New': 'materials_oil',
            'mixedmediaMaterial_New': 'mixed_material',
            'canvasMaterial_New': 'materials_canvas',
        })

        data_df = pd.merge(data_df_old, data_df_new, on='listing_id', how='inner').reset_index(drop=True)

        data_df = data_df[data_df.seller_name == seller_dir.name]
        
        data_df = data_df.dropna(subset=['seller_name', 'listing_id', 'images', 'width_inches', 'height_inches', 'price_usd'])

        data_df = data_df[(data_df.width_inches > 0) & (data_df.height_inches > 0) & (data_df.price_usd > 0)]
        all_dataframes.append(data_df)

        for index, row in data_df.iterrows():
            for image_url in eval(row.images):
                image_id = image_url.split('/')[-1]
                data[str(row.listing_id)].append({
                    'seller_name': row.seller_name,
                    'listing_url': row.listing_url,
                    'painting_title': row.painting_title,
                    'painting_description': row.painting_description,
                    'seller_name': seller_dir.name,
                    'image_url': image_url,
                    'image_path': images_dir / image_id,
                    'listing_id': str(row.listing_id),
                    'log_price_usd': np.log(row.price_usd),
                    'structured_features': extract_extra_features(row),
                })

    with open(out_dir / 'train_test_split.json') as f:
        split = json.load(f)
        train_listing_ids = split['train_listing_ids']
        test_listing_ids = split['test_listing_ids']
        train_data = {k: v for k, v in data.items() if k in train_listing_ids}
        test_data = {k: v for k, v in data.items() if k in test_listing_ids}

    print('Train test size:', len(train_data), len(test_data))
    return train_data, test_data


def get_transforms():
    resize_shape = (224, 224)
    train_transform = transforms.Compose([
        transforms.Resize(resize_shape),

        # transforms.ColorJitter(),
        # transforms.RandomAffine(5),
        # transforms.RandomGrayscale(p=0.1),

        transforms.ToTensor(),

        # transforms.RandomErasing(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(resize_shape),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def get_data(args):
    data_dir = pathlib.Path(args.data_dir)

    train_data, test_data = read_data(data_dir)
    # embeddings_loader = EmbeddingsLoader(data_dir / EMBEDDINGS_FILE_NAME)
    embeddings_loader = EmbeddingsLoader(EMBEDDINGS_FILE_NAME)

    # print(train_data, test_data)

    train_transform, val_transform = get_transforms()

    train_dataset = MyDataSet(train_data, embeddings_loader, train_transform)
    val_dataset = MyDataSet(test_data, embeddings_loader, val_transform)

    train_dataset = ScaledDataSet(train_dataset)
    val_dataset = ScaledDataSet(val_dataset, train_dataset.scaler)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers,
        # prefetch_factor=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=4*args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers,
        # prefetch_factor=4
    )
    return train_loader, val_loader


def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    train_loader, val_loader = get_data(args)
    for batch in tqdm(train_loader):
        continue


if __name__ == '__main__':
    main()
