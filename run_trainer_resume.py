# -*- coding: utf-8 -*-
import sys

sys.dont_write_bytecode = True

import os

from core.config import Config
from core import Trainer

PATH = "./results/SKDModel-CUB_200_2011_FewShot-resnet12-5-1-Dec-07-2021-11-20-04"

if __name__ == "__main__":
    config = Config(os.path.join(PATH, "config.yaml"), is_resume=True).get_config_dict()
    trainer = Trainer(config)
    trainer.train_loop()
