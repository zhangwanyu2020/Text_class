import tensorflow.keras.backend as K
from pprint import pprint

def micro_f1(y_true, y_pred):
    """F1 metric.
    Computes the micro_f1 and macro_f1, metrics for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=0)
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=0)
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=0)

    """Micro_F1 metric.
    """
    precision = K.sum(true_positives) / (K.sum(predicted_positives)+K.epsilon())
    recall = K.sum(true_positives) / (K.sum(possible_positives)+K.epsilon())
    micro_f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return micro_f1


def macro_f1(y_true, y_pred):
    """F1 metric.
    Computes the micro_f1 and macro_f1, metrics for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=0)
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=0)
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=0)

    """Macro_F1 metric.
    """
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    macro_f1 = K.mean(2 * precision * recall / (precision + recall + K.epsilon()))
    return macro_f1


def train(train_x, train_y, vocab_size, sequence_length, save_path, embedding_matrix, args):
    model = text_cnn(sequence_length,
                    vocab_size,
                    args.embedding_size,
                    args.num_classes,
                    args.num_filters,
                    args.filter_sizes,
                    args.regularizers_lambda,
                    args.dropout_rate,
                    embedding_matrix=embedding_matrix)
    model.summary()
    model.compile(tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                  loss='binary_crossentropy',
                  metrics=[micro_f1,macro_f1])
    history = model.fit(x=train_x,
                        y=train_y,
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        steps_per_epoch=18060//args.batch_size,
                        validation_split=args.fraction_validation,
                        shuffle=True)
    tf.keras.models.save_model(model,save_path)
    pprint(history.history)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='This is the TextCNN train project.')
    parser.add_argument('-t', '--test_sample_percentage', default=0.1, type=float, help='The fraction of test data.(default=0.1)')
    parser.add_argument('-p', '--padding_size', default=256, type=int, help='Padding size of sentences.(default=128)')
    parser.add_argument('-e', '--embedding_size', default=512, type=int, help='Word embedding size.(default=512)')
    parser.add_argument('-f', '--filter_sizes', default="2,4,6", help='Convolution kernel sizes.(default=3,4,5)')
    parser.add_argument('-n', '--num_filters', default=32, type=int, help='Number of each convolution kernel.(default=128)')
    parser.add_argument('-d', '--dropout_rate', default=0.1, type=float, help='Dropout rate in softmax layer.(default=0.5)')
    parser.add_argument('-c', '--num_classes', default=95, type=int, help='Number of target classes.(default=18)')
    parser.add_argument('-l', '--regularizers_lambda', default=0.01, type=float, help='L2 regulation parameter.(default=0.01)')
    parser.add_argument('-b', '--batch_size', default=256, type=int, help='Mini-Batch size.(default=64)')
    parser.add_argument('-lr', '--learning_rate', default=0.01, type=float, help='Learning rate.(default=0.005)')
    parser.add_argument('-vocab_size', default=50000, type=int, help='Limit vocab size.(default=50000)')
    parser.add_argument('--epochs', default=8, type=int, help='Number of epochs.(default=10)')
    parser.add_argument('--fraction_validation', default=0.01, type=float, help='The fraction of validation.(default=0.05)')
    parser.add_argument('--results_dir', default='./results/', type=str, help='The results dir including log, model, vocabulary and some images.(default=./results/)')
    args = parser.parse_args(args=[])
    print('Parameters:', args, '\n')

    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    train_x, train_y = get_data('./train_x.npy','./train_y.npy')
    embedding_matrix = build_w2v('corpus.txt','vocab.txt', w2v_model_path='w2v.model', min_count=0)
    train(train_x, train_y, args.vocab_size, args.padding_size, os.path.join(args.results_dir, 'TextCNN.h5'), embedding_matrix, args)
