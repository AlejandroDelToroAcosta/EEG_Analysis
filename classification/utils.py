import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.utils import resample, shuffle

def train(files, n_epochs, path_data):
    results = []
    power_features = ['alpha_power_Pz', 'alpha_power_PO7', 'theta_power_PO8']
    ispc_features = ['ispc_PO7_PO8_alpha_signal', 'ispc_Pz_PO8_alpha_signal', 'ispc_PO7_Pz_theta_signal']
    for filename in files:
        df = pd.read_csv(path_data + filename)

        if len(df) < n_epochs:
            print(f"Sujeto {filename} saltado: Solo tiene {len(df)} épocas.")
            continue

        for feat in power_features:
            df[f'{feat}_dB'] = 10 * np.log10(df[f'{feat}_signal'] / (df[f'{feat}_baseline'] + 1e-6))

        selected_cols = [f'{f}_dB' for f in power_features] + ispc_features
        X = df[selected_cols].fillna(0)
        y = df['label']


        n_splits = min(5, y.value_counts().min())
        if n_splits < 2:
            print(f"Sujeto {filename} saltado: No hay suficientes muestras de ambas clases.")
            continue

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        subj_accuracies = []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            train_df = pd.concat([X_train, y_train], axis=1)
            df_maj = train_df[train_df.label == 0]
            df_min = train_df[train_df.label == 1]

            if 0 < len(df_min) < len(df_maj):
                df_min_up = resample(df_min, replace=True, n_samples=len(df_maj), random_state=42)
                train_df_bal = pd.concat([df_maj, df_min_up])
                X_train_final = train_df_bal.drop('label', axis=1)
                y_train_final = train_df_bal['label']
            else:
                X_train_final, y_train_final = X_train, y_train

            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_final)
            X_test_scaled = scaler.transform(X_test)

            model = SVC(kernel="rbf", C=1.0, class_weight="balanced")
            model.fit(X_train_scaled, y_train_final)

            preds = model.predict(X_test_scaled)
            subj_accuracies.append(accuracy_score(y_test, preds))

        mean_acc = np.mean(subj_accuracies)
        results.append({'subject': filename, 'accuracy': mean_acc})
        print(f"Sujeto {filename}: Accuracy Real = {mean_acc:.4f}")

    return results


def visualize(results_df):
    print("\n--- RESUMEN FINAL ---")
    print(f"Precisión Media: {results_df['accuracy'].mean():.4f}")

    plt.figure(figsize=(10, 5))
    sns.histplot(results_df['accuracy'], bins=10, kde=True, color='teal')
    plt.axvline(0.5, color='red', linestyle='--', label="Azar (0.50)")
    plt.title("Distribución de Accuracy Intrasujeto (Sin Leakage)")
    plt.legend()
    plt.show()

    sns.boxplot(x=results_df['accuracy'], color='lightblue')
    plt.title("Dispersión de la Precisión entre Sujetos")
    plt.show()


def permutation_test(files, n_epochs, path_data, n_permutations):
    permutation_results = []
    power_features = ['alpha_power_Pz', 'alpha_power_PO7', 'theta_power_PO8']
    ispc_features = ['ispc_PO7_PO8_alpha_signal', 'ispc_Pz_PO8_alpha_signal', 'ispc_PO7_Pz_theta_signal']

    for filename in files:
        df = pd.read_csv(path_data + filename)
        if len(df) < n_epochs:
            print(f"Sujeto {filename} saltado: Solo tiene {len(df)} épocas.")
            continue

        for feat in power_features:
            df[f'{feat}_dB'] = 10 * np.log10(df[f'{feat}_signal'] / (df[f'{feat}_baseline'] + 1e-6))

        selected_cols = [f'{f}_dB' for f in power_features] + ispc_features
        X_data = df[selected_cols].fillna(0)
        y = df['label']

        def evaluate_model(X_in, y_in, is_shuffled=False):
            if is_shuffled:
                y_in = shuffle(y_in).reset_index(drop=True)

            n_splits = min(5, y_in.value_counts().min())
            if n_splits < 2: return None

            skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
            accs = []

            for train_idx, test_idx in skf.split(X_in, y_in):
                X_train, X_test = X_in.iloc[train_idx], X_in.iloc[test_idx]
                y_train, y_test = y_in.iloc[train_idx], y_in.iloc[test_idx]

                train_df = pd.concat([X_train, y_train], axis=1)
                df_maj = train_df[train_df.label == 0]
                df_min = train_df[train_df.label == 1]

                if 0 < len(df_min) < len(df_maj):
                    df_min_up = resample(df_min, replace=True, n_samples=len(df_maj))
                    X_tr_final = pd.concat([df_maj.drop('label', axis=1), df_min_up.drop('label', axis=1)])
                    y_tr_final = pd.concat([df_maj['label'], df_min_up['label']])
                else:
                    X_tr_final, y_tr_final = X_train, y_train

                scaler = RobustScaler()
                X_tr_s = scaler.fit_transform(X_tr_final)
                X_te_s = scaler.transform(X_test)

                from sklearn.svm import SVC
                model = SVC(kernel='rbf', class_weight="balanced", C=1.0)
                model.fit(X_tr_s, y_tr_final)
                accs.append(accuracy_score(y_test, model.predict(X_te_s)))
            return np.mean(accs)


        real_acc = evaluate_model(X_data, y, is_shuffled=False)
        if real_acc is None: continue

        null_distribution = [evaluate_model(X_data, y, is_shuffled=True) for _ in range(n_permutations)]
        null_distribution = np.array([a for a in null_distribution if a is not None])

        p_value = (np.sum(null_distribution >= real_acc) + 1) / (len(null_distribution) + 1)

        permutation_results.append({
            'subject': filename,
            'real_accuracy': real_acc,
            'null_mean': np.mean(null_distribution),
            'p_value': p_value
        })

        print(f"Sujeto: {filename} | Real: {real_acc:.3f} | Azar: {np.mean(null_distribution):.3f} | p: {p_value:.3f}")

    return pd.DataFrame(permutation_results)

def visualize_permutation(perm_df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=perm_df, x='null_mean', y='real_accuracy', hue=(perm_df['p_value'] < 0.05))
    plt.plot([0.4, 0.9], [0.4, 0.9], 'r--', label="Línea de Identidad (Azar)")
    plt.title("Real vs Azar (Cada punto es un sujeto)")
    plt.xlabel("Precisión Media por Azar (Etiquetas barajadas)")
    plt.ylabel("Precisión Real del Modelo")
    plt.legend(title="Significativo (p < 0.05)")
    plt.show()

    print(f"\nSujetos significativos: {np.sum(perm_df['p_value'] < 0.05)} de {len(perm_df)}")

def train_unified_subjects(files, n_epochs, path_data):
    subjects_dict = {}
    for f in files:
        match = re.search(r'subject_(\d+)', f)
        if match:
            subj_id = match.group(1)
            if subj_id not in subjects_dict:
                subjects_dict[subj_id] = []
            subjects_dict[subj_id].append(f)

    print(f"Detectados {len(subjects_dict)} sujetos únicos.")

    power_features = ['alpha_power_Pz', 'alpha_power_PO7', 'theta_power_PO8']
    ispc_features = ['ispc_PO7_PO8_alpha_signal', 'ispc_Pz_PO8_alpha_signal', 'ispc_PO7_Pz_theta_signal']
    results = []

    for subj_id, file_list in subjects_dict.items():
        df_list = [pd.read_csv(os.path.join(path_data, f)) for f in file_list]
        df = pd.concat(df_list, ignore_index=True)

        if len(df) < n_epochs:
            print(f"Sujeto {subj_id} saltado: Solo tiene {len(df)} épocas totales.")
            continue

        for feat in power_features:
            df[f'{feat}_dB'] = 10 * np.log10(df[f'{feat}_signal'] / (df[f'{feat}_baseline'] + 1e-6))

        selected_cols = [f'{f}_dB' for f in power_features] + ispc_features
        X = df[selected_cols].fillna(0)
        y = df['label']

        n_splits = min(5, y.value_counts().min())
        if n_splits < 2:
            print(f"Sujeto {subj_id} saltado: No hay suficientes muestras de ambas clases.")
            continue

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        subj_accuracies = []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            train_df = pd.concat([X_train, y_train], axis=1)
            df_maj = train_df[train_df.label == 0]
            df_min = train_df[train_df.label == 1]

            if 0 < len(df_min) < len(df_maj):
                df_min_up = resample(df_min, replace=True, n_samples=len(df_maj), random_state=42)
                train_df_bal = pd.concat([df_maj, df_min_up])
                X_train_final = train_df_bal.drop('label', axis=1)
                y_train_final = train_df_bal['label']
            else:
                X_train_final, y_train_final = X_train, y_train

            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_final)
            X_test_scaled = scaler.transform(X_test)

            model = SVC(kernel="rbf", C=1.0, class_weight="balanced")
            model.fit(X_train_scaled, y_train_final)

            preds = model.predict(X_test_scaled)
            subj_accuracies.append(accuracy_score(y_test, preds))

        mean_acc = np.mean(subj_accuracies)
        results.append({'subject': subj_id, 'accuracy': mean_acc})
        print(f"Sujeto {subj_id}: Accuracy Real = {mean_acc:.4f}")

    return results



def permutation_test_unified(files, n_epochs_min, path_data, n_permutations):
    subjects_dict = {}
    for f in files:
        match = re.search(r'subject_(\d+)', f)
        if match:
            subj_id = match.group(1)
            if subj_id not in subjects_dict:
                subjects_dict[subj_id] = []
            subjects_dict[subj_id].append(f)


    power_features = ['alpha_power_Pz', 'alpha_power_PO7', 'theta_power_PO8']
    ispc_features = ['ispc_PO7_PO8_alpha_signal', 'ispc_Pz_PO8_alpha_signal', 'ispc_PO7_Pz_theta_signal']
    permutation_results = []

    for subj_id, file_list in subjects_dict.items():
        df_list = [pd.read_csv(os.path.join(path_data, f)) for f in file_list]
        df = pd.concat(df_list, ignore_index=True)

        if len(df) < n_epochs_min:
            print(f"Sujeto {subj_id} saltado: Solo tiene {len(df)} épocas totales.")
            continue

        # Feature Engineering
        for feat in power_features:
            df[f'{feat}_dB'] = 10 * np.log10(df[f'{feat}_signal'] / (df[f'{feat}_baseline'] + 1e-6))

        selected_cols = [f'{f}_dB' for f in power_features] + ispc_features
        X_data = df[selected_cols].fillna(0)
        y_data = df['label']

        def evaluate_model(X_in, y_in, is_shuffled=False):
            if is_shuffled:
                y_in = shuffle(y_in).reset_index(drop=True)

            n_splits = min(5, y_in.value_counts().min())
            if n_splits < 2: return None

            skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
            accs = []

            for train_idx, test_idx in skf.split(X_in, y_in):
                X_train, X_test = X_in.iloc[train_idx], X_in.iloc[test_idx]
                y_train, y_test = y_in.iloc[train_idx], y_in.iloc[test_idx]

                train_df = pd.concat([X_train, y_train], axis=1)
                df_maj = train_df[train_df.label == 0]
                df_min = train_df[train_df.label == 1]

                if 0 < len(df_min) < len(df_maj):
                    df_min_up = resample(df_min, replace=True, n_samples=len(df_maj))
                    X_tr_f = pd.concat([df_maj.drop('label', axis=1), df_min_up.drop('label', axis=1)])
                    y_tr_f = pd.concat([df_maj['label'], df_min_up['label']])
                else:
                    X_tr_f, y_tr_f = X_train, y_train

                scaler = RobustScaler()
                X_tr_s = scaler.fit_transform(X_tr_f)
                X_te_s = scaler.transform(X_test)

                model = SVC(kernel='rbf', C=1.0)
                model.fit(X_tr_s, y_tr_f)
                accs.append(accuracy_score(y_test, model.predict(X_te_s)))
            return np.mean(accs)

        real_acc = evaluate_model(X_data, y_data, is_shuffled=False)
        if real_acc is None: continue

        null_dist = [evaluate_model(X_data, y_data, is_shuffled=True) for _ in range(n_permutations)]
        null_dist = np.array([a for a in null_dist if a is not None])

        p_value = (np.sum(null_dist >= real_acc) + 1) / (len(null_dist) + 1)

        permutation_results.append({
            'subject': subj_id,
            'n_epochs': len(df),
            'real_accuracy': real_acc,
            'null_mean': np.mean(null_dist),
            'p_value': p_value
        })

        print(
            f"ID: {subj_id} | Épocas: {len(df)} | Real: {real_acc:.3f} | Azar: {np.mean(null_dist):.3f} | p: {p_value:.3f}")

    return pd.DataFrame(permutation_results)
