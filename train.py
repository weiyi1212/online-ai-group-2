import os
import sys
import random
import numpy as np
import pandas as pd
from keras.layers import Embedding, BatchNormalization, Input, Dense, LSTM, concatenate
from keras.optimizers import adam_v2
import keras.callbacks as kc
import keras.backend as K
from keras.utils import Sequence
from keras.models import Model



def prepare_data(traindf):
    for_feats = traindf.loc[
        lambda x: x.session_position <= (x.session_length/2)
    ].copy()

    for_feats.skip_1 = for_feats.skip_1.astype('int64')
    for_feats.skip_2 = for_feats.skip_2.astype('int64')
    for_feats.skip_3 = for_feats.skip_3.astype('int64')
    for_feats.not_skipped = for_feats.not_skipped.astype('int64')
    for_feats.hist_user_behavior_is_shuffle = for_feats.hist_user_behavior_is_shuffle.astype('int64')
    for_feats.premium = for_feats.premium.astype('int64')

    for_feats.date = pd.to_datetime(for_feats.date)
    for_feats['wkdy'] = for_feats.date.dt.dayofweek
    for_feats['day'] = for_feats.date.dt.day
    for_feats['month'] = for_feats.date.dt.month
    for_feats['year'] = for_feats.date.dt.year
    for_feats.drop(columns=['date'], inplace=True)

    for_feats.drop(columns=['track_id_clean'], inplace=True)

    where_to_replace = for_feats.hist_user_behavior_reason_start.isin([
        'endplay', 'popup', 'uriopen', 'clickside'
    ]).copy()
    for_feats.loc[where_to_replace, 'hist_user_behavior_reason_start'] = 'merged'

    where_to_replace = for_feats.hist_user_behavior_reason_end.isin([
        'clickrow', 'appload', 'popup', 'uriopen', 'clickside', 'logout'
    ]).copy()
    for_feats.loc[where_to_replace, 'hist_user_behavior_reason_end'] = 'merged'

    for_feats.sort_values(['session_id', 'session_position'], inplace=True)

    traindf = traindf.loc[
        lambda x: x.session_position > (x.session_length/2)
    ].sort_values(['session_id', 'session_position'])

    traindf = traindf.loc[:, [
        'session_id', 'session_position', 'session_length',
        'track_id_clean', 'track_slno', 'skip_2'
    ]].copy()

    traindf.sort_values(['session_id', 'session_position'], inplace=True)

    return (traindf.reset_index(drop=True),
            for_feats.reset_index(drop=True))

cols_to_select = [
    'context_switch',
    'context_type',
    'day',
    'hist_user_behavior_is_shuffle',
    'hist_user_behavior_n_seekback',
    'hist_user_behavior_n_seekfwd',
    'hist_user_behavior_reason_end',
    'hist_user_behavior_reason_start',
    'hour_of_day',
    'long_pause_before_play',
    'month',
    'no_pause_before_play',
    'not_skipped',
    'premium',
    'session_position',
    'short_pause_before_play',
    'skip_1',
    'skip_2',
    'skip_3',
    'wkdy']




session_embed = Embedding(
    input_dim=None,
    output_dim=None,
    weights=None,
    trainable=False,
    mask_zero=False,
    name='session_embed')

track_embed = Embedding(
    input_dim=None,
    output_dim=None,
    weights=None,
    trainable=False,
    mask_zero=False,
    name='track_embed')
session_bn = BatchNormalization(name='bn1')
session_transformer = Dense(64, activation='relu', name='session_transformer')

session_input = Input(shape=(None,), dtype='int64', name='session_input')
x1 = session_embed(session_input)
x1 = session_bn(x1)
x1 = session_transformer(x1)

track_bn = BatchNormalization(name='track_bn')
track_transformer = Dense(64, activation='relu', name='track_transformer')

prehist_tracks_input = Input(shape=(None,), dtype='int64', name='prehist_tracks_input')
x2 = track_embed(prehist_tracks_input)
x2 = track_bn(x2)
x2 = track_transformer(x2)

topred_tracks_input = Input(shape=(None,), dtype='int64', name='topred_tracks_input')
x3 = track_embed(topred_tracks_input)
x3 = track_bn(x3)
x3 = track_transformer(x3)

x = concatenate([x1, x2], axis=-1)

model = Model(inputs=None,
              outputs=None)


