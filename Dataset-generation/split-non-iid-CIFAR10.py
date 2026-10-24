from pathlib import Path
import json
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torchvision.datasets import CIFAR10

# ==========================================================
# CONFIGURATION
# ==========================================================
NUM_CLIENTS = 8
ALPHA = 0.5
SEED = 42
MIN_SAMPLES = 100
INITIAL_FRACTION = 0.40
NUM_ONLINE_ROUNDS = 6
DATA_DIR = Path('./Data')
OUTPUT_DIR = Path(f'./CIFAR10_Online_DFL_alpha{ALPHA}_nodes{NUM_CLIENTS}')


def dirichlet_partition(labels, num_clients, alpha, min_samples, seed):
    """Partition all training indices once without duplication or omission."""
    if num_clients < 1 or alpha <= 0:
        raise ValueError('NUM_CLIENTS must be positive and ALPHA > 0.')
    if min_samples * num_clients > len(labels):
        raise ValueError('MIN_SAMPLES is infeasible for the dataset size.')

    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    for _ in range(10000):
        client_indices = [[] for _ in range(num_clients)]
        for c in classes:
            class_indices = np.flatnonzero(labels == c)
            rng.shuffle(class_indices)
            proportions = rng.dirichlet(np.full(num_clients, alpha))
            boundaries = (np.cumsum(proportions[:-1]) * len(class_indices)).astype(int)
            for client_id, batch in enumerate(np.split(class_indices, boundaries)):
                client_indices[client_id].extend(batch.tolist())

        if all(len(idx) >= min_samples for idx in client_indices):
            out = []
            for idx in client_indices:
                shuffled = np.asarray(idx, dtype=np.int64)
                rng.shuffle(shuffled)
                out.append(shuffled)
            return out
    raise RuntimeError('Could not satisfy MIN_SAMPLES after 10,000 attempts.')


def split_online_data(indices, initial_fraction, num_rounds, rng):
    if not 0 < initial_fraction < 1 or num_rounds < 1:
        raise ValueError('INITIAL_FRACTION must be in (0,1), NUM_ONLINE_ROUNDS >= 1.')
    indices = np.asarray(indices, dtype=np.int64).copy()
    rng.shuffle(indices)
    initial_size = int(len(indices) * initial_fraction)
    return indices[:initial_size], np.array_split(indices[initial_size:], num_rounds)


def save_pair(folder, stem, images, labels, indices):
    np.save(folder / f'{stem}_X.npy', images[indices])
    np.save(folder / f'{stem}_y.npy', labels[indices])


def draw_heatmap(distribution, class_names, destination):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    image = ax.imshow(distribution, cmap='YlGnBu', aspect='auto')
    fig.colorbar(image, ax=ax, label='Number of samples')
    ax.set_xticks(range(len(class_names)), labels=class_names, rotation=35, ha='right')
    ax.set_yticks(range(len(distribution)), labels=[f'Node {i+1}' for i in range(len(distribution))])
    threshold = distribution.max() * 0.55
    for i in range(distribution.shape[0]):
        for j in range(distribution.shape[1]):
            value = int(distribution[i, j])
            ax.text(j, i, str(value), ha='center', va='center', fontsize=8,
                    color='white' if value > threshold else 'black')
    ax.set(title=f'CIFAR-10 Non-IID Distribution (alpha={ALPHA})',
           xlabel='Classes', ylabel='Nodes')
    fig.tight_layout()
    fig.savefig(destination, dpi=300, bbox_inches='tight')
    plt.close(fig)


def draw_online_arrivals(summary, destination):
    ax = summary.drop(columns='Total').T.plot(kind='bar', figsize=(12, 5), width=0.8)
    ax.set(title='Online Data Arrival Across Nodes', xlabel='Training stage',
           ylabel='Number of new samples')
    ax.legend(title='Nodes', bbox_to_anchor=(1.02, 1))
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(destination, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset = CIFAR10(root=str(DATA_DIR), train=True, download=True)
    test_dataset = CIFAR10(root=str(DATA_DIR), train=False, download=True)
    images = train_dataset.data  # uint8, (50000, 32, 32, 3)
    labels = np.asarray(train_dataset.targets, dtype=np.int64)
    class_names = train_dataset.classes

    print(f'CIFAR-10: {len(train_dataset)} training, {len(test_dataset)} testing samples')
    client_indices = dirichlet_partition(labels, NUM_CLIENTS, ALPHA, MIN_SAMPLES, SEED)
    all_indices = np.concatenate(client_indices)
    assert len(all_indices) == len(labels) and len(np.unique(all_indices)) == len(labels)

    online_rng = np.random.default_rng(SEED + 1000)
    distribution = np.zeros((NUM_CLIENTS, len(class_names)), dtype=int)
    records = []

    for client_id, indices in enumerate(client_indices, start=1):
        node_name = f'Node_{client_id}'
        node_dir = OUTPUT_DIR / node_name
        node_dir.mkdir(parents=True, exist_ok=True)
        initial_indices, online_batches = split_online_data(
            indices, INITIAL_FRACTION, NUM_ONLINE_ROUNDS, online_rng
        )
        combined = np.concatenate([initial_indices] + online_batches)
        assert len(combined) == len(indices)
        assert set(combined.tolist()) == set(indices.tolist())

        save_pair(node_dir, 'initial', images, labels, initial_indices)
        records.append({'Node': node_name, 'Stage': 'Initial', 'Round': 0,
                        'Samples': len(initial_indices)})

        for round_id, batch in enumerate(online_batches, start=1):
            save_pair(node_dir, f'online_round_{round_id}', images, labels, batch)
            records.append({'Node': node_name, 'Stage': 'Online', 'Round': round_id,
                            'Samples': len(batch)})

        save_pair(node_dir, 'all', images, labels, indices)
        distribution[client_id - 1] = np.bincount(labels[indices], minlength=len(class_names))
        print(f'{node_name}: total={len(indices)}, initial={len(initial_indices)}, '
              f'online={sum(map(len, online_batches))}')

    class_table = pd.DataFrame(distribution, columns=class_names,
                               index=[f'Node_{i}' for i in range(1, NUM_CLIENTS+1)])
    class_table['Total'] = class_table.sum(axis=1)
    class_table.to_csv(OUTPUT_DIR / 'class_distribution.csv')

    online_table = pd.DataFrame(records)
    online_table.to_csv(OUTPUT_DIR / 'online_data_distribution.csv', index=False)
    summary = online_table.pivot(index='Node', columns='Round', values='Samples').fillna(0).astype(int)
    summary.columns = ['Initial' if round_id == 0 else f'Round_{round_id}'
                       for round_id in summary.columns]
    summary['Total'] = summary.sum(axis=1)
    summary.to_csv(OUTPUT_DIR / 'online_data_summary.csv')

    draw_heatmap(distribution, class_names, OUTPUT_DIR / 'class_heatmap.png')
    draw_online_arrivals(summary, OUTPUT_DIR / 'online_arrival.png')

    config = {
        'dataset': 'CIFAR-10', 'num_clients': NUM_CLIENTS, 'alpha': ALPHA,
        'seed': SEED, 'min_samples': MIN_SAMPLES,
        'initial_fraction': INITIAL_FRACTION, 'num_online_rounds': NUM_ONLINE_ROUNDS,
        'total_training_samples': len(train_dataset),
        'total_test_samples': len(test_dataset),
        'partition': 'Dirichlet Non-IID',
        'online_split': 'Random incremental batches within each client',
        'image_storage': 'uint8, NHWC', 'labels': 'int64 class IDs',
        'note': 'Test images are kept separate; they are not partitioned into node training data.'
    }
    with (OUTPUT_DIR / 'config.json').open('w', encoding='utf-8') as handle:
        json.dump(config, handle, indent=4)

    print('\nOnline data summary:\n', summary)
    print(f'\nCompleted. Files saved in: {OUTPUT_DIR.resolve()}')


if __name__ == '__main__':
    main()