import pathlib
import random

import torch
import torch.nn as nn
import torchvision.models

from data2 import STRUCTURED_FEATURES
from utils import write_json, load_json


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class Model(nn.Module):
    def __init__(self, encoder_model_name='resnet18', texts=False, structured=False, visual=False, sense_dropout=0.5):
        super(Model, self).__init__()

        self.encoder_model_name = encoder_model_name
        self.texts = texts
        self.structured = structured
        self.visual = visual
        self.sense_dropout = sense_dropout

        fc_dim = 0

        if self.visual:
            encoder = getattr(torchvision.models, encoder_model_name)(weights='IMAGENET1K_V1')

            fc_dim = encoder.fc.in_features
            encoder = nn.Sequential(*list(encoder.children()))[:-4]
            for p in encoder.parameters():
                p.requires_grad_(False)
                fc_dim = p.shape[0]

            self.image_encoder = nn.Sequential(
                encoder,
                nn.AdaptiveMaxPool2d(1),

                Flatten()
            )

        if self.texts:
            self.fc_downsample = nn.Sequential(nn.Linear(1024, 256))
            fc_dim += self.fc_downsample[-1].out_features

        if self.structured:
            fc_dim += len(STRUCTURED_FEATURES)  # number of features in structured data

        self.fc = nn.Sequential(
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(),

            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(),

            nn.Linear(fc_dim, 1),
        )

    def forward(self, images, texts, structured):
        
        final_embeddings = []

        if self.visual:
            final_embeddings.append(self.image_encoder(images))

        if self.texts:
            text_embeddings = self.fc_downsample(texts)
            if self.training and self.visual and random.random() < self.sense_dropout:
                text_embeddings = torch.zeros_like(text_embeddings)
            final_embeddings.append(text_embeddings)

        if self.structured:
            if self.training and self.visual and random.random() < self.sense_dropout:
                structured = torch.zeros_like(structured)
            final_embeddings.append(structured)

        final_emb = torch.cat(final_embeddings, dim=1)
        return self.fc(final_emb)

    def unfreeze_encoder(self):
        if self.visual:
            for p in self.image_encoder.parameters():
                p.requires_grad_(True)

    def to_file(self, save_dir: pathlib.Path):
        pt_path = save_dir / f'model_params_{self.visual}_{self.texts}_{self.structured}.pt'
        torch.save(self.state_dict(), pt_path)
        return [
            pt_path,
            write_json(
                {
                    'visual': self.visual,
                    'texts': self.texts,
                    'structured': self.structured,
                    'encoder_model_name': self.encoder_model_name,
                    'state_dict': pt_path.name
                },
                save_dir / f'model_{self.visual}_{self.texts}_{self.structured}.json',
            ),
        ]

    @staticmethod
    def from_file(file_path):
        file_path = pathlib.Path(file_path)  # Ensure it's a Path object
        data = load_json(file_path)
        model = Model(
            encoder_model_name=data['encoder_model_name'],
            visual=data['visual'],
            texts=data['texts'],
            structured=data['structured']
        )
        state_dict_path = file_path.with_name(data['state_dict'])
        model.load_state_dict(torch.load(state_dict_path, map_location='cpu'))
        return model.eval()
