import matplotlib.pyplot as plt
import time
import argparse
import numpy as np
import tensorflow
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Flatten
from keras.callbacks import EarlyStopping
import sklearn
from sklearn import metrics
import os
import heapq
import pickle
from numpy.random import seed
from keras_self_attention import SeqSelfAttention

# seed(0)
# tensorflow.random.set_seed(0)


def difference(x):
    x_diff = np.zeros(x.shape)
    for i in range(x.shape[1] - 1, 0, -1):
        x_diff[:, i] = x[:, i] - x[:, i - 1]
    x_diff[:, 0] = 0
    return x_diff


def get_benefit(x_mat, y_arr, labels, misclassification_cost, multivariate=False):
    if multivariate:
        T = max([d.shape[0] for d in x_mat])
        N = len(x_mat)
    else:
        T = x_mat.shape[1]  # length of each time series
        N = x_mat.shape[0]  # number of time series

    benefit = {l: np.zeros((N, T)) for l in labels}
    for i, y in enumerate(y_arr):
        for l in labels:
            benefit[l][i, :] = (T - 1 - np.arange(T)) - misclassification_cost[y, l]

    return benefit


def split_sequences(sequences, shingle_size):
    x, y = [], []
    for sequence in sequences:
        for i in range(len(sequence) - shingle_size + 1):
            end_ix = i + shingle_size
            if end_ix > len(sequence):
                break
            seq_x = sequence[i:end_ix, :-1]
            seq_y = sequence[end_ix - 1, -1]
            x.append(seq_x)
            y.append(seq_y)
    return np.array(x), np.array(y)


def fit_lstm(train_x, benefit_l, num_hidden_units=50, shingle_size=10, num_epochs=30, num_features=1, attention=False):
    model = Sequential()
    if attention:
        model.add(LSTM(num_hidden_units, activation='relu', input_shape=(shingle_size, num_features),
                       return_sequences=True))  #
        model.add(SeqSelfAttention(attention_activation='sigmoid', history_only=True))
        model.add(Flatten())
    else:
        model.add(LSTM(num_hidden_units, activation='relu', input_shape=(shingle_size, num_features)))

    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    if num_features == 1:
        N = train_x.shape[0]
        data = [np.vstack((train_x[i, :], benefit_l[i, :])).T for i in range(N)]
        train_x, train_y = split_sequences(data, shingle_size)  # vectorize=False
        print(train_x.shape, train_y.shape)
    else:
        train_x, train_y = split_multivariate_sequences(train_x, benefit_l, shingle_size=shingle_size)
        print(train_x.shape, train_y.shape)

    es = EarlyStopping(monitor='val_loss', verbose=1, patience=20)
    model.fit(train_x, train_y, epochs=num_epochs, verbose=1, validation_split=0.1, callbacks=[es])

    predictions = model.predict(train_x)[:, 0]
    rmse = np.sqrt(sklearn.metrics.mean_squared_error(train_y, predictions))
    print('RMSE: %.3f' % rmse)
    return model


def predict_lstm(model, test_x, shingle_size=10):
    N = test_x.shape[0]
    data = [np.vstack((test_x[i, :], np.ones(test_x.shape[1]))).T for i in range(N)]
    test_x, _ = split_sequences(data, shingle_size) # vectorize=False
    return model.predict(test_x)[:, 0]


def split_multivariate_sequences(data_x, benefit=None, shingle_size=10):
    x, y = [], []
    for i in range(len(data_x)):
        for j in range(data_x[i].shape[0] - shingle_size + 1):
            x.append(data_x[i][j:j + shingle_size, :])
            if benefit is not None:  # test time
                y.append(benefit[i, j + shingle_size - 1])

    x = np.array(x)
    if benefit is None:
        y = None
    else:
        y = np.array(y)

    return x, y


def predict_lstm_multivariate(model, test_x, shingle_size=10):
    N = len(test_x)
    lengths = [d.shape[0] for d in test_x] # assuming test_x dim is N X T X dim
    # lengths = np.cumsum(lengths) # split indices based on lengths
    test_x, _ = split_multivariate_sequences(test_x, shingle_size=shingle_size) # vectorize=False

    predictions = model.predict(test_x)[:, 0]


    reshaped_predictions = []
    start_idx = 0
    for series_len in lengths:
        end_idx = series_len
        reshaped_predictions.append(predictions[start_idx:start_idx + end_idx])
        start_idx = series_len

    # predictions = np.split(predictions, lengths)
    # reshaped_predictions = [d.reshape(1, -1) for d in predictions]
    assert N == len(reshaped_predictions)
    return reshaped_predictions # model.predict(test_x)[:, 0]


def load_ucr_data(name, path='data/ucr/'):
    train_path = os.path.join(path, name, name + '_TRAIN.tsv')
    test_path = os.path.join(path, name, name + '_TEST.tsv')
    train = np.loadtxt(train_path)
    train_y = train[:, 0].astype(int)
    train_x = difference(train[:, 1:])

    test = np.loadtxt(test_path)
    test_y = test[:, 0].astype(int)
    test_x = difference(test[:, 1:])  # successive differences

    return train_x, train_y, test_x, test_y


def trainer(args):
    train_x, train_y, test_x, test_y = load_ucr_data(args.dataset_name, args.data_path)

    labels = np.unique(train_y)

    # get K from args
    if args.K is None:
        miss_cost = int(train_x.shape[1] * args.K_n_ratio)
    else:
        miss_cost = args.K

    misclassification_cost = {(l1, l2): 0 if l1 == l2 else miss_cost  # cost of l1 being classified as l2
                              for l1 in labels for l2 in labels}

    benefit = get_benefit(train_x, train_y, labels, misclassification_cost)

    print("Number of test samples", len(test_y))
    print("Length of each test sample", len(test_x[0]))

    # check if training is required or not
    if args.load_existing_model:
        models = {}
        # scaler_transforms = {}
        for l in labels:
            models[l] = load_model(os.path.join(args.model_save_path, 'lbl' + str(l) + '_K' + str(miss_cost) +'.h5'))
    else:
        models = {}
        start = time.time()
        for l in labels:
            print('label', l)
            models[l] = fit_lstm(train_x, benefit[l], num_epochs=args.epochs, shingle_size=args.shingle_size,
                                 attention=args.attention)
        end = time.time()
        print("Total traintime:", end - start)

        # saving models
        if args.model_save_path:
            try:
                os.makedirs(args.model_save_path)
            except FileExistsError:
                # directory already exists
                pass
            print("Saving model")
            for l, model in models.items():
                model.save(os.path.join(args.model_save_path, 'lbl' + str(l) + '_K' + str(miss_cost) +'.h5'))

    # evaluation
    test_y_pred = {l: predict_lstm(model, test_x, shingle_size=args.shingle_size).reshape(test_x.shape[0], -1)
                   for l, model in models.items()}

    counts = {'correct': np.zeros(test_x.shape[1]),
              'wrong': np.zeros(test_x.shape[1])}
    prediction_times = []
    prediction_labels = []
    no_action_from_model = 0
    start = time.time()

    # minimum gap in benefit before outputting a label
    min_gap = 0
    if args.min_decision_gap:
        b_at_0 = [benefit[l][0][0] for l in labels]
        top_2_b = heapq.nlargest(2, b_at_0)
        min_gap = (top_2_b[0]-top_2_b[1])*args.min_decision_gap


    #  test evaluation
    for i in range(test_x.shape[0]):
        pred_label, pred_time = np.random.choice(labels), test_x.shape[1]
        for t in range(test_y_pred[labels[0]].shape[1]):
            b = np.array([test_y_pred[l][i, t] for l in labels])
            top_2_pred_b = heapq.nlargest(2, b)
            if top_2_pred_b[0] - top_2_pred_b[1] >= min_gap:
                if np.max(b) > 0 and np.sum(b == np.max(b)) == 1:
                    pred_label, pred_time = labels[np.argmax(b)], t
                    break

        if pred_label == test_y[i]:
            try:
                counts['correct'][pred_time + args.shingle_size] += 1
                prediction_labels.append(pred_label)
            except IndexError:
                no_action_from_model += 1
                pred_label = labels[np.argmax(b)]
                prediction_labels.append(pred_label) # adding just based on the last time tick comparison
                continue
        else:
            try:
                counts['wrong'][pred_time + args.shingle_size] += 1
                prediction_labels.append(pred_label)
            except IndexError:
                no_action_from_model += 1
                pred_label = labels[np.argmax(b)]
                prediction_labels.append(pred_label) # adding just based on the last time tick comparison
                continue
        prediction_times.append(pred_time + args.shingle_size)
        # prediction_labels.append(pred_label)

    end = time.time()
    print("Total test time:", end - start)
    print("Average test time:", (end - start) / len(test_x))

    avg_earliness = np.average(prediction_times) / len(test_x[0])
    print("Earliness:", avg_earliness)
    print("No action:", no_action_from_model)

    ####################
    avg_earliness = np.average(prediction_times) / len(test_x[0])
    accuracy = metrics.accuracy_score(test_y, prediction_labels) #(np.sum(counts['correct']) / (test_x.shape[0] - no_action_from_model))
    acc_total = (np.sum(counts['correct']) / test_x.shape[0])
    print("Accuracy (all):", accuracy, "Accuracy (remove no action)", acc_total)
    print("Earliness:", avg_earliness)
    print("No action from model:", no_action_from_model)

    print("True:", len(test_y), ", Pred:", len(prediction_labels))
    print("Precision:", metrics.precision_score(test_y, prediction_labels))
    print("Recall:", metrics.recall_score(test_y, prediction_labels))
    print("F1:", metrics.f1_score(test_y, prediction_labels))

    # write report to a file for later use
    if args.report_results_file is not None:
        print("Writing results to file")
        with open(os.path.join(args.model_save_path, args.report_results_file), "a+") as f:
            f.write('\t'.join(['K', 'K_n_ratio', 'accuracy', 'acc_total', 'earliness', 'no_decision', 'min_gap']))
            f.write('\n')
            f.write('\t'.join([str(miss_cost), str(args.K_n_ratio), str(accuracy), str(acc_total), str(avg_earliness),
                               str(no_action_from_model), str(args.min_decision_gap)]))
            f.write('\n')


def load_mimic_data(path='data/mimic'):
    trx_path = os.path.join(path, 'train_x_subsampled.pkl')
    tex_path = os.path.join(path, 'test_x.pkl')
    try_path = os.path.join(path, 'train_y_subsampled.pkl')
    tey_path = os.path.join(path, 'test_y.pkl')

    with open(trx_path, 'rb') as f:
        train_x = pickle.load(f)

    with open(try_path, 'rb') as f:
        train_y = pickle.load(f)

    with open(tex_path, 'rb') as f:
        test_x = pickle.load(f)

    with open(tey_path, 'rb') as f:
        test_y = pickle.load(f)

    return train_x, train_y, test_x, test_y


def load_eeg_data(path='data/raw_data.pkl'):
    with open(path, 'rb') as f:
        X, test_X, y, test_y, train_pid, test_pid = pickle.load(f)

    # keep only the one with length at least 24 and restrict only to 96 time steps
    X_train = []
    y_train_mult = []
    for i in range(len(X)):
        if len(X[i]) >= 24:
            val = np.nan_to_num(X[i][:96])
            val[val > 1000] = 1000.
            val[val < -1000] = -1000.
            X_train.append(val)
            y_train_mult.append(y[i])


    X_test = []
    y_test_mult = []
    for i in range(len(test_X)):
        if len(test_X[i]) >= 24:
            val = np.nan_to_num(test_X[i][:96])
            val[val > 1000] = 1000.
            val[val < -1000] = -1000.
            X_test.append(val)
            y_test_mult.append(test_y[i])

    # converts targets to binary targets -- from 0, 1, 2, 3, 4 ---> 0, 1 with 1 being zombie
    y_train = []
    y_test = []
    for lbl in y_train_mult:
        if lbl == 0:
            y_train.append(0)
        else:
            y_train.append(1)

    for lbl in y_test_mult:
        if lbl == 0:
            y_test.append(0)
        else:
            y_test.append(1)

    return X_train, y_train, X_test, y_test


def trainer_multivariate(args):
    """
    In this method we will trin just one model for label 1
    :param args:
    :return:
    """
    # dataset will be list of numpy array of varying lengths
    if args.dataset_name == 'mimic':
        train_x, train_y, test_x, test_y = load_mimic_data(args.data_path)
    else:
        train_x, train_y, test_x, test_y = load_eeg_data(args.data_path)

    print("Trainin and test counts:", np.unique(train_y, return_counts=True), np.unique(test_y, return_counts=True))

    # find unique lables
    labels = np.unique(train_y)
    num_features = train_x[0].shape[-1]

    # max length of a series
    max_length = max([d.shape[0] for d in train_x])

    # get K from args
    if args.K is None:
        miss_cost = int(max_length * args.K_n_ratio)
    else:
        miss_cost = args.K

    misclassification_cost = {(0, 1): miss_cost,  # cost of l1 being classified as l2
                              (1, 1): 0}

    # get benefit only for the label 1
    benefit = get_benefit(train_x, train_y, labels[1:], misclassification_cost, multivariate=True)

    print("Number of test samples", len(test_y))
    print("Length of each test sample", len(test_x[0]))

    # check if training is required or not
    if args.load_existing_model:
        models = {}
        # scaler_transforms = {}
        for l in labels[1:]:
            models[l] = load_model(os.path.join(args.model_save_path, 'lbl' + str(l) + '_K' + str(miss_cost) +'.h5'))
    else:
        models = {}
        start = time.time()
        for l in labels[1:]:
            print('label', l)
            models[l] = fit_lstm(train_x, benefit[l], num_epochs=args.epochs, shingle_size=args.shingle_size,
                                 num_features=num_features)
        end = time.time()
        print("Total traintime:", end - start)

        # saving models
        if args.model_save_path:
            try:
                os.makedirs(args.model_save_path)
            except FileExistsError:
                # directory already exists
                pass
            print("Saving model")
            for l, model in models.items():
                model.save(os.path.join(args.model_save_path, 'lbl' + str(l) + '_K' + str(miss_cost) +'.h5'))


    # evaluation --- need to change so that variable size examples are predicted one by one
    # it wil contain list of predictions
    test_y_pred = {l: predict_lstm_multivariate(model, test_x, shingle_size=args.shingle_size)
                   for l, model in models.items()}

    counts = {'correct': np.zeros(max_length),
              'wrong': np.zeros(max_length)}

    always_prediction_times = np.ones(len(test_x))*max_length
    prediction_times = []
    prediction_labels = []
    earliness_fraction = []
    no_action_from_model = 0
    start = time.time()


    #  test evaluation
    for i in range(len(test_x)):
        pred_label, pred_time = np.random.choice(labels), max_length
        for t in range(len(test_y_pred[labels[1]][i])):
            b = np.array([test_y_pred[l][i][t] for l in labels[1:]])
            if np.max(b) > 0 and np.sum(b == np.max(b)) == 1:
                pred_label, pred_time = labels[np.argmax(b)], t
                break

        if pred_label == test_y[i]:
            try:
                counts['correct'][pred_time + args.shingle_size] += 1
                prediction_labels.append(pred_label)
            except IndexError:
                no_action_from_model += 1
                pred_label = labels[np.argmax(b)]
                prediction_labels.append(pred_label) # adding just based on the last time tick comparison
                continue
        else:
            try:
                counts['wrong'][pred_time + args.shingle_size] += 1
                prediction_labels.append(pred_label)
            except IndexError:
                no_action_from_model += 1
                pred_label = labels[np.argmax(b)]
                prediction_labels.append(pred_label) # adding just based on the last time tick comparison
                continue
        earliness_fraction.append((pred_time + args.shingle_size)/len(test_y_pred[labels[1]][i]))
        prediction_times.append(pred_time + args.shingle_size)
        always_prediction_times[i] = pred_time + args.shingle_size
        # prediction_labels.append(pred_label)

    end = time.time()
    print("Total test time:", end - start)
    print("Average test time:", (end - start) / len(test_x))

    avg_earliness = np.average(earliness_fraction)  # np.average(prediction_times) / len(test_x[0])
    print("Earliness:", avg_earliness)
    print("No action:", no_action_from_model)

    accuracy = metrics.accuracy_score(test_y, prediction_labels) #(np.sum(counts['correct']) / (test_x.shape[0] - no_action_from_model))
    acc_total = (np.sum(counts['correct']) / len(test_x))
    print("Accuracy (all):", accuracy, "Accuracy (remove no action)", acc_total)
    print("Earliness:", avg_earliness)
    print("No action from model:", no_action_from_model)

    print("True:", len(test_y), ", Pred:", len(prediction_labels))
    print("Precision:", metrics.precision_score(test_y, prediction_labels))
    print("Recall:", metrics.recall_score(test_y, prediction_labels))
    print("F1:", metrics.f1_score(test_y, prediction_labels))

    series_lens = [d.shape[0] for d in test_x]
    benefit_val = calculate_benefit(series_lens, always_prediction_times, prediction_labels, test_y, K=300, d_label=1)
    print("Incurred benefit:", benefit_val)

    # write report to a file for later use
    if args.report_results_file is not None:
        print("Writing results to file")
        with open(os.path.join(args.model_save_path, args.report_results_file), "a+") as f:
            f.write('\t'.join(['K', 'K_n_ratio', 'accuracy', 'acc_total', 'earliness', 'no_decision', 'min_gap']))
            f.write('\n')
            f.write('\t'.join([str(miss_cost), str(args.K_n_ratio), str(accuracy), str(acc_total), str(avg_earliness),
                               str(no_action_from_model), str(args.min_decision_gap)]))
            f.write('\n')


# find benefit of this model
def calculate_benefit(series_lens, pred_tau, pred_class, labels_test, K=300,d_label=None):
    # d_label is the death label. Then we need to claculate the cost wrt to this
    total_benefit = 0
    if d_label:
        for i in range(len(labels_test)):
            pred_label, pred_time = pred_class[i], pred_tau[i]
            if labels_test[i] == d_label and pred_label == labels_test[i]:  # pred death, actual death
                total_benefit += series_lens[i] - pred_time
            elif labels_test[i] != d_label and pred_label == d_label: # actual survive, predited death
                total_benefit += series_lens[i] - pred_time - K
            else:
                total_benefit += 0

    else:
        for i in range(len(labels_test)):
            pred_label, pred_time = pred_class[i], pred_tau[i]

            if pred_label == labels_test[i]:  # pred death, actual death
                total_benefit += series_lens[i] - pred_time
            else:
                total_benefit += series_lens[i] - pred_time - K

    return total_benefit


def benefit_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='ECG200', help='UCR Dataset') # None
    parser.add_argument('--model_save_path', type=str, default="../kerasmodels/",
                        help="Path to save trained model")  # None
    parser.add_argument('--report_results_file', type=str, default=None, help="File to save experiment details "
                                                                              "in models folder")  # None
    parser.add_argument('--data_path', type=str, default='../datasets/ucr/', help='Data path')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--dropout', type=float, default=0., help='LSTM Dropout')
    parser.add_argument('--attention', type=int, default=0, help='Attention')
    parser.add_argument('--num_layers', type=int, default=1, help='LSTM Layers')
    parser.add_argument('--hidden_size', type=int, default=16, help='Hidden Dim of LSTM')
    parser.add_argument('--num_features', type=int, default=1, help='Input feature size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--min_t', type=int, default=5, help='Minimum length for evaluation')
    parser.add_argument('--max_t', type=int, default=None, help='Maximum length for evaluation')
    parser.add_argument('--K', type=float, default=None, help='Misclassification Cost') # K is None or K_n_ratio is None
    parser.add_argument('--K_n_ratio', type=float, default=None, help='Misclassification Cost as a ratio')
    parser.add_argument('--shingle_size', type=int, default=10, help='Shingle size to use')
    parser.add_argument('--load_existing_model', type=str, default=False, help="Loads existing model to test") # False
    parser.add_argument('--min_decision_gap', type=float, default=0.4, help='Decision threshold in multi-class')

    args, _ = parser.parse_known_args()

    return args


def main():
    args = benefit_parse_args()

    args.dataset_name = 'ECG200'
    args.data_path = '../datasets/'

    args.report_results_file = None

    args.epochs = 10
    args.shingle_size = 10
    args.load_existing_model = 'False'
    args.min_decision_gap = 0.6

    args.model_save_path = None
    args.K = 10
    args.load_existing_model = False

    trainer(args)
    # trainer_multivariate(args)


if __name__ == '__main__':
    main()
