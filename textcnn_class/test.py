#test
def f1_np(y_true, y_pred):
    """F1 metric.
    Computes the micro_f1 and macro_f1, metrics for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)), axis=0)
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)), axis=0)
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)), axis=0)

    """Macro_F1 metric.
    """
    precision = true_positives / (predicted_positives + 1e-8)
    recall = true_positives / (possible_positives + 1e-8)
    macro_f1 = np.mean(2 * precision * recall / (precision + recall + 1e-8))

    """Micro_F1 metric.
    """
    precision = np.sum(true_positives) / np.sum(predicted_positives)
    recall = np.sum(true_positives) / np.sum(possible_positives)
    micro_f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return micro_f1, macro_f1


def test(model, x_test, y_test):
    y_pred = model.predict(x=x_test, batch_size=1, verbose=1)
    # print(y_pred)
    metrics = [f1_np]
    result = {}
    for func in metrics:
        result[func.__name__] = func(y_test, y_pred)
    pprint(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This is the TextCNN test project.')
    parser.add_argument('--results_dir', default='./results/', type=str,
                        help='The results dir including log, model, vocabulary and some images.')
    args = parser.parse_args(args=[])
    print('Parameters:', args)

    x_test, y_test = get_data('./test_x.npy','./test_y.npy')
    print("Loading model...")
    model = load_model(os.path.join(args.results_dir, 'TextCNN.h5'),custom_objects={"micro_f1": micro_f1, "macro_f1": macro_f1})
    test(model, x_test, y_test)