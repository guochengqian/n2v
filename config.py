import os
import sys
import argparse
import shutil
import random
import numpy as np
import logging
import time
import uuid
import pathlib
import glob


class Config:
    def __init__(self):
        # ===> argparse
        parser = argparse.ArgumentParser(description='GraphUnet for self supervised denoising')
        # ----------------- Log related
        parser.add_argument('--exp_name', type=str, default='N2V', help='Name of the experiment')
        parser.add_argument('--root_dir', type=str, default='log',
                            help='the dir of experiment results, ckpt and logs')

        # ----------------- Training related
        parser.add_argument('--phase', type=str, default='train', metavar='N',
                            choices=['train', 'test'])
        parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu?')
        parser.add_argument('--batch_size', type=int, default=128, metavar='batch_size', help='Size of batch)')
        parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of episode to train ')
        parser.add_argument('--lr', type=float, default=4e-4, metavar='LR', help='learning rate (default: 0.001)')

        # ----------------- Testing related
        parser.add_argument('--pretrained_model', type=str, default='', metavar='N',
                            help='Pretrained model path')

        # ----------------- Model related
        parser.add_argument('--patch_size', default=64, type=int)
        parser.add_argument('--perc_pix', default=0.198, type=float)
        parser.add_argument('--depth', default=2, type=int)
        parser.add_argument('--k', default=9, type=int)
        parser.add_argument('--region_size', default=7, type=int)
        parser.add_argument('--use_att', action='store_true')

        args = parser.parse_args()
        self.args = args

        # ===> generate log dir
        if self.args.phase == 'train':
            self._generate_exp_directory()
            # loss
            self.args.epoch = -1
            self.args.step = -1

        else:
            self.args.exp_dir = os.path.dirname(self.args.pretrained_model)
            self.args.res_dir = os.path.join(self.args.exp_dir, "result")
            pathlib.Path(self.args.res_dir).mkdir(parents=True, exist_ok=True)

        self._configure_logger()
        self._print_args()

    def _get_args(self):
        return self.args

    def _generate_exp_directory(self):
        """
        Helper function to create checkpoint folder. We save
        model checkpoints using the provided model directory
        but we add a sub-folder for each separate experiment:
        """
        timestamp = time.strftime('%Y%m%d-%H%M%S')

        self.args.jobname = '{}-p{}-B{}-pix{}-lr{}'.format(
            self.args.exp_name, self.args.patch_size, self.args.batch_size,
            self.args.perc_pix, self.args.lr)
        experiment_string = '_'.join([self.args.jobname, timestamp, str(uuid.uuid4())])
        self.args.exp_dir = os.path.join(self.args.root_dir, experiment_string)
        self.args.ckpt_dir = os.path.join(self.args.exp_dir, "checkpoint")
        self.args.res_dir = os.path.join(self.args.exp_dir, "result")
        pathlib.Path(self.args.exp_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.args.ckpt_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.args.res_dir).mkdir(parents=True, exist_ok=True)

    def _configure_logger(self):
        """
        Configure logger on given level. Logging will occur on standard
        output and in a log file saved in model_dir.
        """
        self.args.loglevel = "info"
        numeric_level = getattr(logging, self.args.loglevel.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: {}'.format(self.args.loglevelloglevel))

            # configure logger to display and save log data
        # log_format = logging.Formatter('%(asctime)s [%(levelname)-5.5s] [%(filename)s:%(lineno)04d] %(message)s')
        log_format = logging.Formatter('%(asctime)s %(message)s')
        logger = logging.getLogger()
        logger.setLevel(numeric_level)

        file_handler = logging.FileHandler(os.path.join(self.args.exp_dir,
                                                        '{}.log'.format(os.path.basename(self.args.exp_dir))))
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

        file_handler = logging.StreamHandler(sys.stdout)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
        logging.root = logger
        logging.info("saving log, checkpoint and back up code in folder: {}".format(self.args.exp_dir))

    def _print_args(self):
        logging.info("==========       args      =============")
        for arg, content in self.args.__dict__.items():
            logging.info("{}:{}".format(arg, content))
        logging.info("==========     args END    =============")
        logging.info("\n")
        logging.info('===> Phase is {}.'.format(self.args.phase))
