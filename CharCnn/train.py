from model import CharCnn
from config import FLAGS
import preprocessing


if __name__ == '__main__':

    x_train, y_train, x_test, y_test = preprocessing.load_data(FLAGS.train_dir, FLAGS.test_dir)

    x_train_idx = preprocessing.convert_str2idx(x_train)
    x_test_idx = preprocessing.convert_str2idx(x_test)

    y_train = preprocessing.one_hot(y_train)
    y_test = preprocessing.one_hot(y_test)

    char_cnn = CharCnn(sequence_length=300, num_char=70, batch_size=128,
                       iteration=50, init_lr=0.001, n_class=6, embedding_size=100, num_filter=128,
                       filter_size=(7, 7, 3, 3, 3, 3), hidden_unit=1024, step_size=2000, decay=0.9)
    char_cnn.train(x_train_idx, y_train)
    char_cnn.test(x_test_idx, y_test)