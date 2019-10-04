#  Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  A copy of the License is located at
#  
#      http://www.apache.org/licenses/LICENSE-2.0
#  
#  or in the "license" file accompanying this file. This file is distributed 
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either 
#  express or implied. See the License for the specific language governing 
#  permissions and limitations under the License.
from __future__ import print_function

import argparse
import logging
import os
import pickle as pkl

from sagemaker_xgboost_container.checkpointing import save_checkpoint, load_checkpoint

import xgboost as xgb


CHECKPOINTS_DIR = '/opt/ml/checkpoints'
def add_checkpointing(callbacks):
    if os.path.exists(CHECKPOINTS_DIR):
        callbacks.append(save_checkpoint(CHECKPOINTS_DIR))
    # If there are no previous checkpoints in CHECKPOINTS_DIR, load_checkpoint() returns (None, 0).
    # If there are previous checkpoints in CHECKPOINTS_DIR because the instance was interrupted after
    # iteration M, for example, load_checkpoint() will return ('/path/to/xgboost-checkpoint', M).
    # If we initially wanted to train for total N rounds, we now train for N - M rounds after resuming,
    previous_checkpoint, start_iteration = load_checkpoint(CHECKPOINTS_DIR)
    return previous_checkpoint, start_iteration


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here
    parser.add_argument('--num_round', type=int)
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--eta', type=float, default=0.2)
    parser.add_argument('--objective', type=str, default='reg:squarederror')
    
    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    
    args = parser.parse_args()
    
    dtrain = xgb.DMatrix(args.train)
    dval = xgb.DMatrix(args.validation)

    watchlist = [(dtrain, 'train'), (dval, 'validation')] if dval is not None else [(dtrain, 'train')]

    train_hp = {
        'max_depth': args.max_depth,
        'eta': args.eta,
        'gamma': args.gamma,
        'min_child_weight': args.min_child_weight,
        'subsample': args.subsample,
        'silent': args.silent,
        'objective': args.objective}
    
    callbacks = []
    prev_checkpoint, start_iteration = add_checkpointing(callbacks)
    else: 
        prev_checkpoint, start_iteration = None, 0
    bst = xgb.train(
        params=train_hp,
        dtrain=dtrain,
        evals=watchlist,
        num_boost_round=args.num_round - start_iteration,
        xgb_model=prev_checkpoint,
        callbacks=callbacks
    )
    
    model_location = args.model_dir + '/xgboost-model'
    pkl.dump(bst, open(model_location, 'wb'))
    logging.info("Stored trained model at {}".format(model_location))
    

def model_fn(model_dir):
    """ Deserialized and return fitted model.
    Note that this should have the same name as the serialized model in the main method
    """
    model_file = 'xgboost-model'
    logging.info(model_file)
    booster = pkl.load(open(os.path.join(model_dir, model_file), 'rb'))
    return booster