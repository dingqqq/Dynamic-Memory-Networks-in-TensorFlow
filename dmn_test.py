from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--babi_task_id", help="specify babi task 1-20 (default=1)")
parser.add_argument("-t", "--dmn_type", help="specify type of dmn (default=original)")
args = parser.parse_args()

print("args.dmn_type: {}".format(args.dmn_type))
dmn_type = args.dmn_type if args.dmn_type is not None else "plus"

if dmn_type == "original":
    print("Using dmn_original")
    from dmn_original import Config
    config = Config()
elif dmn_type == "plus":
    print("Using dmn_plus")
    from dmn_plus import Config
    config = Config()
else:
    raise NotImplementedError(dmn_type + ' DMN type is not currently implemented')

print("args.babi_task_id: {}".format(args.babi_task_id))
if args.babi_task_id is not None:
    config.babi_id = args.babi_task_id

config.strong_supervision = False

config.train_mode = False

print('Testing DMN ' + dmn_type + ' on babi task', config.babi_id)

# Create model
with tf.variable_scope('DMN') as scope:
    if dmn_type == "original":
        print("Using dmn_original")
        from dmn_original import DMN
        model = DMN(config)
    elif dmn_type == "plus":
        print("Using dmn_plus")
        from dmn_plus import DMN_PLUS
        model = DMN_PLUS(config)


print('==> Initializing variables')
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as session:
    session.run(init)

    print('==> Restoring weights')
    saver.restore(session, 'weights/task' + str(model.config.babi_id) + '.weights')

    print('==> Running DMN')
    test_loss, test_accuracy = model.run_epoch(session, model.test)

    print('')
    print('Test accuracy: {}'.format(test_accuracy))
