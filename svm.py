import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, accuracy_score, classification_report, confusion_matrix
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.svm import SVC
import six
import matplotlib as mpl
import os

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True, help='file path of input csv')
parser.add_argument('-o', '--output_dir', required=True, help='destination directory')
parser.add_argument('-t', '--test_size', default=0.5, type=float, help='ratio of test sample')
parser.add_argument('-r', '--reduce', default=1, type=int)
parser.add_argument('-s', '--seed', default=42, type=int, help='random seed')
args = parser.parse_args()
name = os.path.splitext(os.path.basename(args.input))[0]

def main():
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    seed = args.seed

    print('Loading csv...')
    df = pd.read_csv(args.input)
    gmi_cols = df.columns[:np.where(df.columns=='FSC-H')[0][0]]
    print('used gmi')
    print(set([c.split('_')[0] for c in gmi_cols]))
    print()

    print('Enter comma-separated modality names to be used for training')
    print('e.g.: fsgmi,bsgmi,fsc,bsc')
    modality = input()
    modality = modality.split(',')
    print()
    print('All gate name')
    print(df.columns[np.where(df.columns=='BSC-A')[0][0]+1:].values)
    pos_name = input('Select gate name of positive cells: ')
    neg_name = input('Select gate name of negative cells: ')
    idx = df[pos_name] | df[neg_name]
    df = df.loc[idx, :]
    df_train, df_test = train_test_split(df, test_size=float(args.test_size), shuffle=True, 
                                         stratify=df[pos_name], random_state=seed)
    print()
    print('train')
    print(f'pos:{pos_name}  {df_train[pos_name].sum()} cells')
    print(f'neg:{neg_name}  {df_train[neg_name].sum()} cells')
    print()
    print('test')
    print(f'pos:{pos_name}  {df_test[pos_name].sum()} cells')
    print(f'neg:{neg_name}  {df_test[neg_name].sum()} cells')
    print()
    print('Enter the number of cells to be used')
    while True:
        n_tr_pos = int(input('train_pos: '))
        if n_tr_pos <= df_train[pos_name].sum():
            break
        else:
            print(f'Up to {df_train[pos_name].sum()}')
            print()
    print()

    while True:
        n_tr_neg = int(input('train_neg: '))
        if n_tr_neg <= df_train[neg_name].sum():
            break
        else:
            print(f'Up to {df_train[neg_name].sum()}')
            print()
    print()
        
    while True:
        n_ts_pos = int(input('test_pos: '))
        if n_ts_pos <= df_test[pos_name].sum():
            break
        else:
            print(f'Up to {df_test[pos_name].sum()}')
            print()
    print()
        
    while True:
        n_ts_neg = int(input('test_neg: '))
        if n_ts_neg <= df_test[neg_name].sum():
            break
        else:
            print(f'Up to {df_test[neg_name].sum()}')
            print()
    print()
    print('Learning...')

    df_train_pos = df_train.loc[df_train[pos_name], :].sample(n=n_tr_pos)
    df_train_neg = df_train.loc[df_train[neg_name], :].sample(n=n_tr_neg)
    df_test_pos = df_test.loc[df_test[pos_name], :].sample(n=n_ts_pos)
    df_test_neg = df_test.loc[df_test[neg_name], :].sample(n=n_ts_neg)

    df_train = pd.concat([df_train_pos, df_train_neg], axis=0)
    df_test = pd.concat([df_test_pos, df_test_neg], axis=0)
    y_train = np.array([1] * n_tr_pos + [0] * n_tr_neg)
    y_test = np.array([1] * n_ts_pos + [0] * n_ts_neg)

    X_train = []
    X_test = []
    fsc_cols = ['FSC-A', 'FSC-H', 'FSC-W']
    bsc_cols = ['BSC-A', 'BSC-H', 'BSC-W']
    for m in modality:
        if m == 'fsc':
            train = df_train.loc[:, fsc_cols].values
            test = df_test.loc[:, fsc_cols].values
        elif m == 'bsc':
            train = df_train.loc[:, bsc_cols].values
            test = df_test.loc[:, bsc_cols].values
        else:
            col_idx = [m in c for c in df_train.columns]
            train = df_train.loc[:, col_idx].values
            test = df_test.loc[:, col_idx].values
            if int(args.reduce) > 1:
                train = np.mean(np.reshape(train, (train.shape[0], -1, int(args.reduce))), axis=2)
                test = np.mean(np.reshape(test, (test.shape[0], -1, int(args.reduce))), axis=2)
        X_train.append(train)
        X_test.append(test)
    X_train = np.concatenate(X_train, axis=1)
    X_test = np.concatenate(X_test, axis=1)

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    def objective(trial):
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        fold_splits = skf.split(X_train, y_train)
        AUC = []
    
        C = trial.suggest_float('C', 1e-1, 1e6)
        gamma = trial.suggest_float('gamma', 1e-8, 1e0)
    
        for train_idx, val_idx in fold_splits:
            X_tr, X_val = X_train[train_idx, :], X_train[val_idx, :]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
            svc = SVC(kernel='rbf', C=C, gamma=gamma, random_state=seed)
            svc.fit(X_tr, y_tr)
            y_pred = svc.decision_function(X_val)
            AUC.append(roc_auc_score(y_val, y_pred))
        return np.array(AUC).mean()
    
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(objective, n_trials=10)
    C = study.best_trial.params['C']
    gamma = study.best_trial.params['gamma']
    svc = SVC(kernel='rbf', C=C, gamma=gamma, random_state=seed)
    svc.fit(X_train, y_train)
    y_pred_df = svc.decision_function(X_test)
    y_pred_class = svc.predict(X_test)
    print('Accuracy:  ', accuracy_score(y_pred_class, y_test))
    print('ROC-AUC:  ', roc_auc_score(y_test, y_pred_df))
    print()

    labels = [neg_name, pos_name]
    report = classification_report(y_test, y_pred_class, 
                                   target_names=labels, output_dict=True)
    header = ['class', 'precision', 'recall', 'f1-score']
    d = [labels]
    for col in header[1:]:
        d.append([np.round(report[label][col], 3) for label in labels])
    df_table = pd.DataFrame(np.array(d).T, columns=header)
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.axis('off')
    mpl_table = ax.table(cellText=df_table.values, bbox=[0, 0, 1, 1], colLabels=df_table.columns)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(14)
    row_colors = ["#f1f1f2", "w"]
    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor('w')
        if k[0] == 0 or k[1] < 0:
            cell.set_text_props(weight="bold", color="w")
            cell.set_facecolor("#40466e")
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
    fig.savefig(os.path.join(args.output_dir, name + '_table.png'), format="png", dpi=100)

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_df)
    fig = plt.figure(figsize=[5,5])
    plt.plot(recall*100,precision*100, c='k')
    plt.xlim(0,105)
    plt.ylim(0,105)
    plt.xticks(np.arange(0, 105, 20))
    plt.yticks(np.arange(0, 105, 20))
    plt.tick_params(labelsize=25)
    plt.xlabel('Recall [%]', fontsize=25)
    plt.ylabel('Precision [%]', fontsize=25)
    #plt.title('PR Curve')
    fig.savefig(os.path.join(args.output_dir, name + '_PR_curve.png'), format="png", dpi=100)

    mpl.rcParams["font.size"] = 15

    cm = confusion_matrix(y_test, y_pred_class)
    _cm_sum = cm.sum(axis=1)[:, np.newaxis]
    _cm_sum[_cm_sum == 0] = 1
    cm_ = cm.astype("float") / _cm_sum

    fig, ax = plt.subplots()
    im = ax.imshow(cm_, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        title='Confusion matrix',
        ylabel="True label",
        xlabel="Predicted label",
        )
    fmt = "d"
    thresh = (cm_.max() + cm_.min()) / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm_[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, name + '_confusion_matrix.png'), format="png", dpi=100)

    positive = y_pred_df[y_test == 1]
    negative = y_pred_df[y_test == 0]
    xmin = -1.5*max(abs(np.percentile(y_pred_df[y_test==1],99)), 
                    abs(np.percentile(y_pred_df[y_test==0],1)))
    xmax = 1.5*max(abs(np.percentile(y_pred_df[y_test==1],99)), 
                   abs(np.percentile(y_pred_df[y_test==0],1)))
    bins = np.linspace(xmin, xmax, 25)
    hist_pos = np.histogram(positive, bins)[0]
    hist_neg = np.histogram(negative, bins)[0]
    ymax = max(hist_pos.max(), hist_neg.max()) * 1.3

    fig = plt.figure(figsize=(6,6))
    plt.hist(positive, color='r', bins=bins, alpha=0.5, label=pos_name)
    plt.hist(negative, color='b', bins=bins, alpha=0.5, label=neg_name)
    plt.xlabel("SVM score [a.u.]", fontsize=25)
    plt.ylabel('Counts [cells]', fontsize=25)
    plt.ylim(0, ymax)
    plt.legend(fontsize=15)
    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, name + '_histgram.png'), format="png", dpi=100)

if __name__ == "__main__":
    main()
