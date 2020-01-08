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
import csv
import json
import logging
import numpy as np
import os
import pickle as pkl

from sagemaker_containers import entry_point
import xgboost as xgb

from sagemaker_xgboost_container import distributed
from sagemaker_xgboost_container.data_utils import get_dmatrix

def _xgb_train(params,
               dtrain,
               evals,
               num_boost_round,
               model_dir,
               is_master,
               feature_names=None,
               shap_ref_data=None):
    """Run xgb.train on arguments and compute SHAP values

    This is our internal execution function that reflects xgb.train.

    :param params: Argument dictionary forwarded to run xgb.train().
    :param dtrain: Training Dmatrix dataset
    :param evals: List of evaluations to record during training
    :param is_master: True if current node is master host in distributed
                       training, or is running single node training job.
                       Note that rabit_run will include this argument.
    """
    booster = xgb.train(params=params,
                        dtrain=dtrain,
                        evals=evals,
                        num_boost_round=num_boost_round)
    if is_master:
        if shap_ref_data is not None:
            booster.feature_names = feature_names
            shap_values = get_shap_values(booster, shap_ref_data, feature_names)
            pkl.dump(shap_values, open(os.path.join(model_dir, 'shap_values'), 'wb'))
            pkl.dump(shap_ref_data, open(os.path.join(model_dir, 'reference_data'), 'wb'))

        model_location = model_dir + '/xgboost_model'
        pkl.dump(booster, open(os.path.join(model_dir, 'xgboost_model'), 'wb'))
        logging.info("Stored trained model at {}".format(model_location))


def get_shap_values(booster, data, feature_names):
    if not isinstance(data, xgb.DMatrix):
        data = xgb.DMatrix(data, feature_names=feature_names)
    return booster.predict(data, pred_contribs=True)


def build_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_depth', type=int,)
    parser.add_argument('--eta', type=float)
    parser.add_argument('--subsample', type=float, default=1.0)
    parser.add_argument('--verbosity', type=int, default=1)
    parser.add_argument('--objective', type=str, default='reg:squarederror')
    parser.add_argument('--num_round', type=int)

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output_data_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--feature_names', type=str, default=os.environ.get('SM_CHANNEL_FEATURE_NAMES'))

    parser.add_argument('--sm_hosts', type=str, default=os.environ['SM_HOSTS'])
    parser.add_argument('--sm_current_host', type=str, default=os.environ['SM_CURRENT_HOST'])
    return parser


if __name__ == '__main__':
    parser = build_argument_parser()
    args, _ = parser.parse_known_args()

    # Get SageMaker host information from runtime environment variables
    sm_hosts = json.loads(os.environ['SM_HOSTS'])
    sm_current_host = args.sm_current_host

    dtrain = get_dmatrix(args.train, 'csv')
    dtest = get_dmatrix(args.test, 'csv')

    # test dataset is also used for SHAP computation
    # args.test can be a channel or directory. Pick the first file in the directory
    if os.path.isfile(args.test):
        test_csv_file = args.test
    else:
        test_csv_file = [f for f in os.listdir(args.test)
                               if os.path.isfile(os.path.join(args.test, f))][0]
    MAX_REF_DATA_SIZE = 10000
    shap_ref_data = np.loadtxt(os.path.join(args.test, test_csv_file),
                               delimiter=',',
                               max_rows=MAX_REF_DATA_SIZE)
    # XGBoost datasets are assumed to have their first column as the 'output/label' column
    # which is not needed for SHAP
    shap_ref_data = np.delete(shap_ref_data, 0, axis=1)

    # convert feature names from comma-separated file row to a list
    if args.feature_names:
        with open(os.path.join(args.feature_names, 'feature_names.csv')) as f:
            reader = csv.reader(f)
            feature_names = next(reader)
    else:
        # if feature names are not available or invalid then create a dummy list
        feature_names = ['f{0:0>2d}'.format(i) for i in range(shap_ref_data.shape[1])]


    watchlist = [(dtrain, 'train'), (dtest, 'test')]
    train_hp = {
        'max_depth': args.max_depth,
        'eta': args.eta,
        'subsample': args.subsample,
        'verbosity': args.verbosity,
        'objective': args.objective}

    xgb_train_args = dict(
        params=train_hp,
        dtrain=dtrain,
        shap_ref_data=shap_ref_data,
        feature_names=feature_names,
        evals=watchlist,
        num_boost_round=args.num_round,
        model_dir=args.model_dir
    )

    if len(sm_hosts) > 1:
        # Wait until all hosts are able to find each other
        entry_point._wait_hostname_resolution()

        # Execute training function after initializing rabit.
        distributed.rabit_run(
            exec_fun=_xgb_train,
            args=xgb_train_args,
            include_in_training=(dtrain is not None),
            hosts=sm_hosts,
            current_host=sm_current_host,
            update_rabit_args=True
        )
    else:
        # If single node training, call training method directly.
        xgb_train_args['is_master'] = True
        _xgb_train(**xgb_train_args)


def model_fn(model_dir):
    """Deserialized and return fitted model.

    Note that this should have the same name as the serialized model in the
       _xgb_train method
    """
    model_file = 'xgboost_model'
    logging.info(model_file)
    booster = pkl.load(open(os.path.join(model_dir, model_file), 'rb'))
    return booster
