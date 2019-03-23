from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from toxicity_classifier.classifier import ToxicityClassifier


def restore():
    sess = tf.Session()
    tox_model = ToxicityClassifier(session=sess)
    return tox_model


def main():
    restore()


if __name__ == '__main__':
    main()
