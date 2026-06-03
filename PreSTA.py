import argparse
import os

import numpy as np
from tqdm import tqdm

from base_tools import dump_pkl, load_pkl, makedirs


def compute_plcc_stats(features, gts):
    mu_X = np.mean(features, axis=0)
    mu_Y = np.mean(gts)

    sigma_X = np.std(features, axis=0)
    sigma_Y = np.std(gts)

    cov_XY = np.mean((features - mu_X) * (gts.reshape(-1, 1) - mu_Y), axis=0)

    plcc = cov_XY / (sigma_X * sigma_Y)

    return mu_X, sigma_X, cov_XY, mu_Y, sigma_Y, plcc


def update_plcc_stats(n, mu_X, mu_Y, sigma_X, sigma_Y, cov_XY, x_k, y_k):
    if n <= 1:
        raise ValueError("Cannot remove the last sample")

    mu_X_new = (n * mu_X - x_k) / (n - 1)
    mu_Y_new = (n * mu_Y - y_k) / (n - 1)

    var_X_new = (n * sigma_X ** 2 - (x_k - mu_X) ** 2) / (n - 1)
    var_Y_new = (n * sigma_Y ** 2 - (y_k - mu_Y) ** 2) / (n - 1)

    sigma_X_new = np.sqrt(var_X_new)
    sigma_Y_new = np.sqrt(var_Y_new)

    cov_XY_new = (n * cov_XY - (x_k - mu_X) * (y_k - mu_Y)) / (n - 1)

    plcc_new = cov_XY_new / (sigma_X_new * sigma_Y_new)

    return mu_X_new, mu_Y_new, sigma_X_new, sigma_Y_new, cov_XY_new, plcc_new


def compute_distribution_distance(source_plcc, target_plcc):
    return np.sqrt(np.sum((source_plcc - target_plcc) ** 2))


def batch_update_plcc_stats(n, mu_X, mu_Y, sigma_X, sigma_Y, cov_XY,
                            candidate_features, candidate_gts):
    if n <= 1:
        raise ValueError("Cannot remove the last sample")

    n_candidates, n_features = candidate_features.shape

    mu_X_expanded = np.broadcast_to(mu_X, (n_candidates, n_features))
    mu_Y_expanded = np.broadcast_to(mu_Y, (n_candidates,))
    sigma_X_expanded = np.broadcast_to(sigma_X, (n_candidates, n_features))
    sigma_Y_expanded = np.broadcast_to(sigma_Y, (n_candidates,))
    cov_XY_expanded = np.broadcast_to(cov_XY, (n_candidates, n_features))

    mu_X_new = (n * mu_X_expanded - candidate_features) / (n - 1)
    mu_Y_new = (n * mu_Y_expanded - candidate_gts.reshape(-1, 1)) / (n - 1)
    mu_Y_new = mu_Y_new.squeeze()

    var_X_new = (n * sigma_X_expanded ** 2 -
                 (candidate_features - mu_X_expanded) ** 2) / (n - 1)
    var_Y_new = (n * sigma_Y_expanded ** 2 -
                 (candidate_gts - mu_Y_expanded) ** 2) / (n - 1)

    var_X_new = np.maximum(var_X_new, 1e-12)
    var_Y_new = np.maximum(var_Y_new, 1e-12)

    sigma_X_new = np.sqrt(var_X_new)
    sigma_Y_new = np.sqrt(var_Y_new)

    cov_XY_new = (n * cov_XY_expanded -
                  (candidate_features - mu_X_expanded) *
                  (candidate_gts.reshape(-1, 1) - mu_Y_expanded.reshape(-1, 1))) / (n - 1)

    plcc_new = cov_XY_new / (sigma_X_new * sigma_Y_new.reshape(-1, 1))

    return mu_X_new, mu_Y_new, sigma_X_new, sigma_Y_new, cov_XY_new, plcc_new


def batch_compute_distribution_distance(plcc_candidates, target_plcc):
    target_expanded = np.broadcast_to(target_plcc, plcc_candidates.shape)
    distances = np.sqrt(np.sum((plcc_candidates - target_expanded) ** 2, axis=1))

    return distances


def compute_sou_tar_plcc(source_plcc, target_plcc):
    value = np.corrcoef(np.nan_to_num(source_plcc), np.nan_to_num(target_plcc))[0, 1]
    return float(np.nan_to_num(value))


def select_source_samples(SouData, target_plcc,
                          select_ratio=0.2,
                          select_samples=1,
                          early_stop_threshold=1e-6,
                          plcc_stop_threshold=0.9,
                          batch_size=None):
    source_features = np.array(SouData['features'])
    source_gts = np.array(SouData['gts'])

    n_samples = len(source_gts)
    mu_X, sigma_X, cov_XY, mu_Y, sigma_Y, source_plcc = compute_plcc_stats(source_features, source_gts)
    n_select = max(int(n_samples * select_ratio), select_samples)

    current_indices = list(range(n_samples))
    distance_history = [(
        None,
        None,
        compute_distribution_distance(source_plcc, target_plcc),
        compute_sou_tar_plcc(source_plcc, target_plcc)
    )]

    if batch_size is None:
        batch_size = min(len(current_indices), 1000)

    if distance_history[-1][3] > plcc_stop_threshold:
        print(f"Early stopping at {len(current_indices)} samples: SouTarPLCC {distance_history[-1][3]:.4f}")
        selected_data = {
            'names': [SouData['names'][i] for i in current_indices],
            'features': [SouData['features'][i] for i in current_indices],
            'gts': [SouData['gts'][i] for i in current_indices]
        }
        return selected_data, distance_history

    pbar = tqdm(total=len(current_indices) - n_select, desc="Removing samples")

    while len(current_indices) > n_select:
        current_distance = distance_history[-1][2]
        current_plcc = distance_history[-1][3]

        if current_plcc > plcc_stop_threshold:
            print(f"Early stopping at {len(current_indices)} samples: SouTarPLCC {current_plcc:.4f}")
            break

        n_remaining = len(current_indices)
        current_features = source_features[current_indices]
        current_gts_array = source_gts[current_indices]
        n_batches = (n_remaining + batch_size - 1) // batch_size

        best_distance = current_distance
        best_remove_local_idx = -1
        best_remove_idx = -1
        found_better = False

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_remaining)

            batch_features = current_features[start_idx:end_idx]
            batch_gts = current_gts_array[start_idx:end_idx]

            try:
                _, _, _, _, _, plcc_candidates = batch_update_plcc_stats(
                    n_remaining, mu_X, mu_Y, sigma_X, sigma_Y, cov_XY,
                    batch_features, batch_gts
                )

                distances = batch_compute_distribution_distance(plcc_candidates, target_plcc)
                batch_min_idx = np.argmin(distances)
                batch_min_distance = distances[batch_min_idx]

                if batch_min_distance < best_distance - early_stop_threshold:
                    best_distance = batch_min_distance
                    best_remove_local_idx = start_idx + batch_min_idx
                    best_remove_idx = current_indices[best_remove_local_idx]
                    found_better = True

            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue

        if not found_better:
            print(f"Early stopping at {len(current_indices)} samples: no further improvement")
            break

        if best_remove_idx != -1:
            x_k = source_features[best_remove_idx]
            y_k = source_gts[best_remove_idx]

            mu_X, mu_Y, sigma_X, sigma_Y, cov_XY, source_plcc = update_plcc_stats(
                len(current_indices), mu_X, mu_Y, sigma_X, sigma_Y, cov_XY, x_k, y_k
            )

            current_indices.pop(best_remove_local_idx)
            SouTarPLCC = compute_sou_tar_plcc(source_plcc, target_plcc)
            distance_history.append((best_remove_idx, best_remove_local_idx, best_distance, SouTarPLCC))
            pbar.update(1)

            if SouTarPLCC > plcc_stop_threshold:
                print(f"Early stopping at {len(current_indices)} samples: SouTarPLCC {SouTarPLCC:.4f}")
                break

    pbar.close()

    selected_data = {
        'names': [SouData['names'][i] for i in current_indices],
        'features': [SouData['features'][i] for i in current_indices],
        'gts': [SouData['gts'][i] for i in current_indices]
    }

    return selected_data, distance_history


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-db', type=str, default='kadid-10k')
    parser.add_argument('--target-db', type=str, nargs='+', required=True)
    parser.add_argument('--feature-dir', type=str, default='fs')
    parser.add_argument('--backbone', type=str, default='swin_base_patch4_window7_224')
    parser.add_argument('--output-dir', type=str, default='SpecSelBatch_L2_PLCCStop')
    parser.add_argument('--select-ratio', type=float, default=0.2)
    parser.add_argument('--select-samples', type=int, default=1)
    parser.add_argument('--early-stop-threshold', type=float, default=1e-6)
    parser.add_argument('--plcc-stop-threshold', type=float, default=0.9)
    parser.add_argument('--batch-size', type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    feature_name = '{}_hierarchical_features_ImageNet_{}.pkl'
    SouData = load_pkl(os.path.join(args.feature_dir, feature_name.format(args.source_db, args.backbone)))

    for TarDb in tqdm(args.target_db):
        print(args.source_db, '->', TarDb)

        TarData = load_pkl(os.path.join(args.feature_dir, feature_name.format(TarDb, args.backbone)))
        target_features = np.array(TarData['features'])
        target_gts = np.array(TarData['gts'])
        _, _, _, _, _, TarCor = compute_plcc_stats(target_features, target_gts)

        selected_data, distance_history = select_source_samples(
            SouData,
            TarCor,
            select_ratio=args.select_ratio,
            select_samples=args.select_samples,
            early_stop_threshold=args.early_stop_threshold,
            plcc_stop_threshold=args.plcc_stop_threshold,
            batch_size=args.batch_size
        )

        makedirs(args.output_dir)
        dump_pkl('{}/{}_for_{}_CorSel.pkl'.format(args.output_dir, args.source_db, TarDb), selected_data)
        dump_pkl('{}/{}_for_{}_CorSel_history.pkl'.format(args.output_dir, args.source_db, TarDb), distance_history)
