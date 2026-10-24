from pathlib import Path
import glob
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================================
# CONFIG: EDIT INPUT PATHS HERE
# ==========================================================
TRAIN_JSON_FILES = [
    # r"D:/FEMNIST/data/train/all_data_0.json",
    # r"D:/FEMNIST/data/train/all_data_1.json",
    # r"D:/FEMNIST/data/train/*.json",  # wildcard also works
]
# If the list is empty, all JSONs in this directory are loaded.
TRAIN_JSON_DIR = Path('./FEMNIST/train')

# Test shards optional. No test sample is mixed into train/online data.
TEST_JSON_FILES = []
TEST_JSON_DIR = Path('./FEMNIST/test')

NUM_CLIENTS = 8
SEED = 42
INITIAL_FRACTION = 0.40
NUM_ONLINE_ROUNDS = 6
MIN_SAMPLES = 100

# 'writer' preserves natural per-writer Non-IID partitions.
# 'dirichlet' ignores writer IDs and partitions per class like CIFAR-10.
PARTITION_MODE = 'writer'
ALPHA = 0.5  # used only when PARTITION_MODE == 'dirichlet'

OUTPUT_DIR = Path(f'./FEMNIST_Online_DFL_{PARTITION_MODE}_nodes{NUM_CLIENTS}')
SAVE_ALL_DATA = True  # Also save redundant complete node X/y files.

# ==========================================================
# INPUT DISCOVERY / JSON LOADING
# ==========================================================
def discover_jsons(path_list, default_dir, required=True):
    patterns = path_list if path_list else [str(default_dir / '**' / '*.json')]
    found = []
    for pattern in patterns:
        matches = glob.glob(str(pattern), recursive=True)
        if not matches and Path(pattern).is_file():
            matches = [str(Path(pattern))]
        found.extend(Path(p).resolve() for p in matches if Path(p).is_file())
    files = sorted(set(found))
    if required and not files:
        raise FileNotFoundError(
            'No FEMNIST JSON files found. Update TRAIN_JSON_FILES or TRAIN_JSON_DIR.'
        )
    return files


def to_images(x, source, user):
    """Accept flat 784-pixel or 28x28 images; return uint8 N x 28 x 28."""
    arr = np.asarray(x)
    if arr.size == 0:
        return np.empty((0, 28, 28), dtype=np.uint8)
    if arr.ndim == 2 and arr.shape[1] == 784:
        arr = arr.reshape(-1, 28, 28)
    elif arr.ndim != 3 or arr.shape[1:] != (28, 28):
        raise ValueError(f'Unsupported image shape {arr.shape} for {user} in {source}')
    if not np.all(np.isfinite(arr)):
        raise ValueError(f'Non-finite image values for {user} in {source}')
    # LEAF FEMNIST stores greyscale pixels in [0,1]. Some variants use [0,255].
    if arr.min() >= 0 and arr.max() <= 1.0:
        arr = arr * 255.0
    elif arr.min() < 0 or arr.max() > 255:
        raise ValueError(f'Pixels outside [0,1] or [0,255] for {user} in {source}')
    return np.rint(arr).clip(0, 255).astype(np.uint8)


def read_leaf_jsons(paths):
    images, targets, writers = [], [], []
    for path in paths:
        with path.open('r', encoding='utf-8') as fh:
            obj = json.load(fh)
        if 'user_data' not in obj or not isinstance(obj['user_data'], dict):
            raise ValueError(f'Expected LEAF user_data mapping in: {path}')
        users = obj.get('users', list(obj['user_data'].keys()))
        count = 0
        for user in users:
            samples = obj['user_data'][user]
            x = to_images(samples['x'], path.name, user)
            y = np.asarray(samples['y'], dtype=np.int64)
            if len(x) != len(y):
                raise ValueError(f'Length mismatch for writer {user} in {path}')
            if len(y) == 0:
                continue
            if np.any(y < 0):
                raise ValueError(f'Negative labels for writer {user} in {path}')
            images.append(x)
            targets.append(y)
            writers.extend([user] * len(y))
            count += len(y)
        print(f'Loaded {path.name}: {count:,} samples')
    if not images:
        raise ValueError('JSON files contain no usable FEMNIST samples.')
    return (np.concatenate(images), np.concatenate(targets),
            np.asarray(writers, dtype=str))

# ==========================================================
# PARTITION INTO DFL NODES
# ==========================================================
def writer_partition(writers, num_clients, seed):
    """Keep every writer on one node; greedy balance by sample count."""
    rng = np.random.default_rng(seed)
    unique, inverse, counts = np.unique(writers, return_inverse=True, return_counts=True)
    if len(unique) < num_clients:
        raise ValueError(f'Only {len(unique)} writers for {num_clients} nodes.')
    order = rng.permutation(len(unique))
    # Process largest writers first; randomize ties using the shuffled order.
    order = order[np.argsort(-counts[order], kind='stable')]
    writer_to_node = {}
    node_sizes = np.zeros(num_clients, dtype=np.int64)
    for writer_idx in order:
        smallest = np.flatnonzero(node_sizes == node_sizes.min())
        node = int(rng.choice(smallest))
        writer_to_node[str(unique[writer_idx])] = node
        node_sizes[node] += counts[writer_idx]
    client_indices = [[] for _ in range(num_clients)]
    for idx, writer in enumerate(writers):
        client_indices[writer_to_node[str(writer)]].append(idx)
    return [np.asarray(v, dtype=np.int64) for v in client_indices], writer_to_node


def dirichlet_partition(y, num_clients, alpha, min_samples, seed):
    if alpha <= 0:
        raise ValueError('ALPHA must be positive.')
    rng = np.random.default_rng(seed)
    for _ in range(10000):
        parts = [[] for _ in range(num_clients)]
        for label in np.unique(y):
            ids = np.flatnonzero(y == label)
            rng.shuffle(ids)
            proportions = rng.dirichlet(np.full(num_clients, alpha))
            cuts = (np.cumsum(proportions[:-1]) * len(ids)).astype(int)
            for n, p in enumerate(np.split(ids, cuts)):
                parts[n].extend(p.tolist())
        if min(map(len, parts)) >= min_samples:
            return [np.asarray(p, dtype=np.int64) for p in parts], {}
    raise RuntimeError('Could not meet MIN_SAMPLES with Dirichlet partition.')


def split_online(indices, rng):
    indices = np.asarray(indices, dtype=np.int64).copy()
    rng.shuffle(indices)
    n_initial = int(len(indices) * INITIAL_FRACTION)
    return indices[:n_initial], np.array_split(indices[n_initial:], NUM_ONLINE_ROUNDS)


def save_pair(folder, stem, x, y, indices):
    np.save(folder / f'{stem}_X.npy', x[indices])
    np.save(folder / f'{stem}_y.npy', y[indices])

# ==========================================================
# VISUALIZATIONS
# ==========================================================
def draw_heatmap(distribution, path):
    fig, ax = plt.subplots(figsize=(15, max(4, NUM_CLIENTS * 0.55)))
    img = ax.imshow(distribution, cmap='YlGnBu', aspect='auto')
    fig.colorbar(img, ax=ax, label='Number of samples')
    ax.set_xticks(range(distribution.shape[1]))
    ax.set_xticklabels(range(distribution.shape[1]), fontsize=7)
    ax.set_yticks(range(NUM_CLIENTS))
    ax.set_yticklabels([f'Node {i+1}' for i in range(NUM_CLIENTS)])
    ax.set(xlabel='FEMNIST label ID', ylabel='DFL node',
           title=f'FEMNIST {PARTITION_MODE.capitalize()} Non-IID Class Distribution')
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def draw_arrivals(summary, path):
    ax = summary.drop(columns='Total').T.plot(kind='bar', figsize=(12, 5))
    ax.set(xlabel='Training stage', ylabel='New samples', title='FEMNIST Online Data Arrival')
    ax.legend(title='DFL node', bbox_to_anchor=(1.02, 1))
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================================
# MAIN
# ==========================================================
def main():
    if not 0 < INITIAL_FRACTION < 1 or NUM_ONLINE_ROUNDS < 1 or NUM_CLIENTS < 1:
        raise ValueError('Check INITIAL_FRACTION, NUM_ONLINE_ROUNDS, NUM_CLIENTS.')
    train_paths = discover_jsons(TRAIN_JSON_FILES, TRAIN_JSON_DIR)
    test_paths = discover_jsons(TEST_JSON_FILES, TEST_JSON_DIR, required=False) if (
        TEST_JSON_FILES or TEST_JSON_DIR.is_dir()) else []
    # Protect against accidental overlap if train/test paths are configured alike.
    overlap = set(train_paths).intersection(test_paths)
    if overlap:
        raise ValueError(f'Train and test input paths overlap: {overlap}')

    print(f'Loading {len(train_paths)} training JSON file(s)...')
    x, y, writers = read_leaf_jsons(train_paths)
    if len(x) < NUM_CLIENTS * MIN_SAMPLES:
        raise ValueError('Not enough training samples for the requested minimum per node.')
    print(f'FEMNIST training: {len(y):,} images, {len(np.unique(writers)):,} writers, '
          f'{len(np.unique(y))} observed labels')

    if PARTITION_MODE == 'writer':
        parts, assignment = writer_partition(writers, NUM_CLIENTS, SEED)
    elif PARTITION_MODE == 'dirichlet':
        parts, assignment = dirichlet_partition(y, NUM_CLIENTS, ALPHA, MIN_SAMPLES, SEED)
    else:
        raise ValueError("PARTITION_MODE must be 'writer' or 'dirichlet'.")
    if min(map(len, parts)) < MIN_SAMPLES:
        raise ValueError('A node has fewer than MIN_SAMPLES; adjust parameters.')
    combined = np.concatenate(parts)
    assert len(combined) == len(y) and len(np.unique(combined)) == len(y)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + 1000)
    # FEMNIST typically has 62 labels (0..61), but handle actual IDs without guessing.
    num_classes = int(y.max()) + 1
    distribution = np.zeros((NUM_CLIENTS, num_classes), dtype=np.int64)
    records = []
    for node_id, indices in enumerate(parts, start=1):
        folder = OUTPUT_DIR / f'Node_{node_id}'
        folder.mkdir(exist_ok=True)
        initial, arrivals = split_online(indices, rng)
        reunited = np.concatenate([initial] + arrivals)
        assert len(reunited) == len(indices) and set(reunited) == set(indices)
        save_pair(folder, 'initial', x, y, initial)
        records.append({'Node': f'Node_{node_id}', 'Stage': 'Initial',
                        'Round': 0, 'Samples': len(initial)})
        for round_id, batch in enumerate(arrivals, 1):
            save_pair(folder, f'online_round_{round_id}', x, y, batch)
            records.append({'Node': f'Node_{node_id}', 'Stage': 'Online',
                            'Round': round_id, 'Samples': len(batch)})
        if SAVE_ALL_DATA:
            save_pair(folder, 'all', x, y, indices)
        distribution[node_id-1] = np.bincount(y[indices], minlength=num_classes)
        print(f'Node {node_id}: total={len(indices):,}, initial={len(initial):,}, '
              f'online={sum(map(len, arrivals)):,}')

    table = pd.DataFrame(distribution, columns=[f'Class_{c}' for c in range(num_classes)],
                         index=[f'Node_{i+1}' for i in range(NUM_CLIENTS)])
    table['Total'] = table.sum(axis=1)
    table.to_csv(OUTPUT_DIR / 'class_distribution.csv')
    arrivals_table = pd.DataFrame(records)
    arrivals_table.to_csv(OUTPUT_DIR / 'online_data_distribution.csv', index=False)
    summary = arrivals_table.pivot(index='Node', columns='Round', values='Samples').fillna(0).astype(int)
    summary.columns = ['Initial' if r == 0 else f'Round_{r}' for r in summary.columns]
    summary['Total'] = summary.sum(axis=1)
    summary.to_csv(OUTPUT_DIR / 'online_data_summary.csv')
    draw_heatmap(distribution, OUTPUT_DIR / 'class_heatmap.png')
    draw_arrivals(summary, OUTPUT_DIR / 'online_arrival.png')

    if assignment:
        with (OUTPUT_DIR / 'writer_node_mapping.json').open('w', encoding='utf-8') as f:
            json.dump({k: f'Node_{v+1}' for k,v in assignment.items()}, f, indent=2)
    test_count = 0
    if test_paths:
        tx, ty, tw = read_leaf_jsons(test_paths)
        np.save(OUTPUT_DIR / 'test_X.npy', tx)
        np.save(OUTPUT_DIR / 'test_y.npy', ty)
        test_count = len(ty)
        print(f'Separate test set saved: {test_count:,} samples')

    config = {
        'dataset': 'FEMNIST', 'train_json_files': list(map(str, train_paths)),
        'test_json_files': list(map(str, test_paths)),
        'partition_mode': PARTITION_MODE,
        'alpha': ALPHA if PARTITION_MODE == 'dirichlet' else None,
        'num_clients': NUM_CLIENTS, 'seed': SEED, 'min_samples': MIN_SAMPLES,
        'initial_fraction': INITIAL_FRACTION, 'num_online_rounds': NUM_ONLINE_ROUNDS,
        'training_samples': len(y), 'test_samples': test_count,
        'observed_labels': sorted(map(int, np.unique(y))),
        'image_storage': 'uint8, (N,28,28)', 'label_storage': 'int64, (N,)',
        'online_split': 'Random order per node (simulated arrival, not time-stamped)',
        'writer_note': 'Writer-based partitioning preserves user membership across nodes; '
                       'a writer\'s samples are randomly split into online stages.'
    }
    with (OUTPUT_DIR / 'config.json').open('w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    print('\n', summary)
    print(f'\nFinished: {OUTPUT_DIR.resolve()}')

if __name__ == '__main__':
    main()