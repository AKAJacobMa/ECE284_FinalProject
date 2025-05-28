import pickle
import os
import numpy as np
import pandas as pd
import json

from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

import optuna
from optuna.samplers import TPESampler


def read_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')

    part = 'chest'
    chest_dict = {
        f'{part}_ACC_x': data['signal'][part]['ACC'][:, 0],
        f'{part}_ACC_y': data['signal'][part]['ACC'][:, 1],
        f'{part}_ACC_z': data['signal'][part]['ACC'][:, 2],
        f'{part}_ECG': data['signal'][part]['ECG'][:, 0],
        f'{part}_EMG': data['signal'][part]['EMG'][:, 0],
        f'{part}_EDA': data['signal'][part]['EDA'][:, 0],
        f'{part}_Temp': data['signal'][part]['Temp'][:, 0],
        f'{part}_Resp': data['signal'][part]['Resp'][:, 0], 
        'label': data['label']
    }

    chest_df = pd.DataFrame(chest_dict)
    
    return chest_df


def process_dataframe(df, label_column, valid_labels):
    """
    Filter, normalize features, and reset index of a DataFrame.

    Parameters:
        df (pd.DataFrame): Input data
        label_column (str): Name of the label column
        valid_labels (list): Allowed label values

    Returns:
        pd.DataFrame: Processed DataFrame
    """
    # Filter rows with valid labels
    df = df[df[label_column].isin(valid_labels)].copy()

    # Separate features and label
    features = df.drop(columns=[label_column])
    labels = df[label_column]

    # Normalize features using StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)

    # Combine features and label
    df_processed = features_scaled_df.copy()
    df_processed[label_column] = labels.reset_index(drop=True)

    # Reset index
    df_processed.reset_index(drop=True, inplace=True)

    return df_processed




def print_callback(study, trial):
    print(f"Trial {trial.number}: Value={trial.value:.4f}, Params={trial.params}")

def objective_rf(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 300)
    max_depth = trial.suggest_int('max_depth', 2, 32)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )

    return cross_val_score(clf, trainX, trainY, cv=5).mean()



def compute_confusion_counts_dict(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))

    overalltp = 0
    overallfp = 0
    overallfn = 0
    overalltn = 0
    result = {}


    for label in labels:
        tp = np.sum((y_pred == label) & (y_true == label)).item()
        fp = np.sum((y_pred == label) & (y_true != label)).item()
        fn = np.sum((y_pred != label) & (y_true == label)).item()
        tn = np.sum((y_pred != label) & (y_true != label)).item()
        
        result[int(label)] = {
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn
        }

        overalltp += tp
        overallfp += fp
        overallfn += fn
        overalltn += tn

    result['overall'] = {
        'TP': overalltp,
        'FP': overallfp,
        'FN': overallfn,
        'TN': overalltn
    }

    return result


if __name__ == "__main__":

    model_name = 'rf'
    training_records_path = 'training_records'

    chest_df = read_pickle( os.path.join('WESAD', 'S2','S2.pkl' ) )
    chest_df = process_dataframe(chest_df, 'label', [1, 2, 3, 4])
    # print number of total data points
    print(f"Total data points: {len(chest_df)}")


    # count the percentage of each label
    label_counts = chest_df['label'].value_counts(normalize=True) * 100
    print(label_counts)


    chest_np = chest_df.to_numpy()


    # get the indices of 4 classes
    indices = {
        1: np.where(chest_np[:, -1] == 1)[0],
        2: np.where(chest_np[:, -1] == 2)[0],
        3: np.where(chest_np[:, -1] == 3)[0],
        4: np.where(chest_np[:, -1] == 4)[0]
    }

    print(f"# of data samples class 1: {len(indices[1])}")
    print(f"# of data samples class 2: {len(indices[2])}")
    print(f"# of data samples class 3: {len(indices[3])}")
    print(f"# of data samples class 4: {len(indices[4])}")

    num_samples_per_class = 5000
    np.random.seed(42)  # For reproducibility
    class1_indices = np.random.choice(indices[1], num_samples_per_class, replace=False)
    class2_indices = np.random.choice(indices[2], num_samples_per_class, replace=False)
    class3_indices = np.random.choice(indices[3], num_samples_per_class, replace=False)
    class4_indices = np.random.choice(indices[4], num_samples_per_class, replace=False)

    # Combine the selected indices
    selected_indices = np.concatenate([class1_indices, class2_indices, class3_indices, class4_indices])
    # create a new 2d array with the selected indices
    chest_np_selected = chest_np[selected_indices]
    # Shuffle the selected data
    np.random.shuffle(chest_np_selected)

    print(f"final 2d array shape: {chest_np_selected.shape}")

    X = chest_np_selected[:, :-1]  # Features
    y = chest_np_selected[:, -1]   # Labels

    numdata = len(y)

    trainX = X[:int(numdata * 0.8)]
    trainY = y[:int(numdata * 0.8)]
    testX = X[int(numdata * 0.8):]
    testY = y[int(numdata * 0.8):]
    
    sampler = TPESampler(seed=10) 
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_rf, n_trials=5, callbacks=[print_callback])


    best_params = study.best_trial.params
    print()
    print("best parameters:")
    for d in best_params:
        print(f"{d}: {best_params[d]}")
    print()
    with open( os.path.join(training_records_path, f'{model_name}_best_params.json'), 'w') as f:
        json.dump(best_params, f, indent=4)

    best_model = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        max_features=best_params['max_features'],
        random_state=42
    )
    
    best_model.fit(trainX, trainY)
    
    # Evaluate the model
    y_train_pred = best_model.predict(trainX)
    y_test_pred = best_model.predict(testX)
    clr_train = classification_report(trainY, y_train_pred, output_dict=True)
    clr_test = classification_report(testY, y_test_pred, output_dict=True)

    confusion_counts_train = compute_confusion_counts_dict(trainY, y_train_pred)
    confusion_counts_train['ypred'] = y_train_pred.tolist()
    confusion_counts_train['ytrue'] = trainY.tolist()
    confusion_counts_test = compute_confusion_counts_dict(testY, y_test_pred)
    confusion_counts_test['ypred'] = y_test_pred.tolist()
    confusion_counts_test['ytrue'] = testY.tolist()


    with open(os.path.join(training_records_path, f'{model_name}_confusion_counts_train.json'), 'w') as f:
        json.dump(confusion_counts_train, f, indent=4)
    with open(os.path.join(training_records_path, f'{model_name}_confusion_counts_test.json'), 'w') as f:
        json.dump(confusion_counts_test, f, indent=4)

    print(f'\n*****  Overall Train Accuracy: {clr_train["accuracy"]:.4f} -- Overall Test Accuracy: {clr_test["accuracy"]:.4f}  *****\n')