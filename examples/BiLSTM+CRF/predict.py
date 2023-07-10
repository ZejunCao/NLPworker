#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Cao Zejun
# @Time     : 2023/7/10 14:38
# @File     : predict.py
# @ Software: PyCharm
# @System   : Windows
# @desc     : 预测脚本
import json

import torch

from config import Config
from model import BiLSTM_CRF
from data_processor import Loader
from theshy.base.predict import PredictBase
from theshy.metrics.ner import chunks_extract


class Predictor(PredictBase):
    def pred_one_batch(self, batch_data):
        with torch.no_grad():
            text = batch_data['text'].to(self.device)
            label = batch_data['label'].to(self.device)
            seq_len = batch_data['seq_len'].to(self.device)

            batch_tag = self.model(text, label, seq_len)

            self.all_inputs.extend([t for t in batch_data['original_text']])
            self.all_preds.extend([[self.config.label_map_inv[t] for t in l] for l in batch_tag])

    def format_output(self, inputs, preds):
        format_results = []
        for i in range(len(preds)):
            result = chunks_extract(preds[i])
            test_format = {"id": i, "text": ''.join(inputs[i]), "label": {}}
            for r in result:
                text = ''.join(inputs[i][r['st_idx']: r['end_idx']])
                if test_format["label"].get(r["label"]):
                    if test_format["label"][r["label"]].get(text):
                        test_format["label"][r["label"]][text].append([r['st_idx'], r['end_idx']])
                    else:
                        test_format["label"][r["label"]][text] = [[r['st_idx'], r['end_idx']]]
                else:
                    test_format["label"][r["label"]] = {text: [[r['st_idx'], r['end_idx']]]}
            format_results.append(test_format)

        with open('data/cluener_public/test_submit.json', 'w', encoding='utf-8') as fp:
            for line in format_results:
                fp.write(json.dumps(line, ensure_ascii=False) + '\n')

        return format_results


if __name__ == '__main__':
    device = "cpu" if torch.cuda.is_available() else "cpu"
    config = Config()

    config.state = 'pred'
    test_dataloader = Loader(config)

    # torch.manual_seed(1234)
    model = BiLSTM_CRF(config, device).to(device)

    predictor = Predictor(model)
    predictor.pred(test_dataloader)
