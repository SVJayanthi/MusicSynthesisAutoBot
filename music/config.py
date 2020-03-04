# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 23:38:34 2020

@author: srava
"""

from fastai.text.models.transformer import tfmerXL_lm_config, Activation
# from .vocab import MusicVocab

def default_config():
    config = tfmerXL_lm_config.copy()
    config['act'] = Activation.GeLU

    config['mem_len'] = 512
    config['d_model'] = 512
    config['d_inner'] = 2048
    config['n_layers'] = 16
    
    config['n_heads'] = 8
    config['d_head'] = 64

    return config

def music_config():
    config = default_config()
    config['encode_position'] = True
    return config