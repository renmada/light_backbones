import paddle
import torch
import os
from collections import OrderedDict
import sys


def convert(file_path):
    path, name = os.path.split(file_path)
    out = OrderedDict()
    for k, v in torch.load(file_path, map_location='cpu').items():
        if 'num_batches_tracked' in k:
            continue
        k = k.replace('running_mean', '_mean').replace('running_var', '_variance')
        if 'weight' in k:
            if v.ndim == 2 and 'embedding' not in k:
                v = v.t()
            else:
                print(k, v.shape)

        out[k] = v.numpy().astype('float32')
    paddle.save(out, os.path.join(path, file_path.replace('.pt', '.pdparams')))


if __name__ == '__main__':
    # model_path = sys.argv[1]
    model_path = 'mobilevit_s.pt'
    convert(model_path)
