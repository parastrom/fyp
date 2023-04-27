import sys
import os
import traceback
import logging
import json
import attr
import torch

from text2sql.models import encoder
from text2sql.models.sql_decoder import decoder


class EncDecModel(torch.nn.Module):
    """Dygraph version of BoomUp Model"""

    def __init__(self, config, label_encoder, model_version='v2'):
        super(EncDecModel, self).__init__()

        self._config = config
        self._model_version = model_version

        assert model_version in ('v2', ), "model_version only support v2"
        self.encoder = encoder.Text2SQLEncoder(config)
        self.decoder = decoder.Text2SQLDecoder(
            label_encoder,
            dropout=0.2,
            desc_attn='mha',
            use_align_mat=True,
            use_align_loss=True)

    def forward(self, inputs, labels=None, db=None, is_train=True):
        if is_train:
            assert labels is not None, "labels should not be None while training"
            return self._train(inputs, labels)
        else:
            assert db is not None, "db should not be None while inferencing"
            return self._inference(inputs, db)

    def _train(self, inputs, labels):
        # timer = text2sql.utils.Timer()
        enc_results = self.encoder(inputs)
        # logging.info(f'Encode Time: {timer.interval():.2f}')
        lst_loss = []
        for orig_inputs, label_info, enc_result in zip(inputs['orig_inputs'],
                                                       labels, enc_results):
            loss = self.decoder.compute_loss(orig_inputs, label_info,
                                             enc_result)
            lst_loss.append(loss)
        # logging.info(f'Decode Time: {timer.interval():.2f}')

        return torch.mean(torch.stack(lst_loss, dim=0), dim=0)

    def _inference(self, inputs, db):
        enc_state = self.encoder(inputs)
        if self._model_version == 'v1':
            return self.decoder.inference(enc_state[0], db)
        elif self._model_version == 'v2':
            return self.decoder.inference(enc_state[0], db,
                                          inputs['orig_inputs'][0].values)


if __name__ == "__main__":
    """run some simple test cases"""
    pass
