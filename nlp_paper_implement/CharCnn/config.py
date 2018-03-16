import tensorflow as tf

tf.app.flags.DEFINE_string("train_dir", "./dataset/TREC_QA/trec_train.csv", "trec-qa train dataset dir")
tf.app.flags.DEFINE_string("test_dir", "./dataset/TREC_QA/trec_test.csv", "trec-qa test dataset dir")
tf.app.flags.DEFINE_string("model_save", "./model/", "checkpoint dir")
tf.app.flags.DEFINE_string("checkpoint", "./model/model.ckpt", 'checkpoint')

FLAGS = tf.app.flags.FLAGS