from time import perf_counter
from datasets import CachedDataset, MyDataset
from torch.utils.data import DataLoader
from torchvision import transforms as tf


def run_performance_test(dataset: MyDataset, batch_size: int, epoch_num: int):
    loader = DataLoader(dataset, batch_size, num_workers=0, shuffle=True)

    print(f'Running for {dataset.name()}:')
    if isinstance(dataset, CachedDataset):
        tic = perf_counter()
        # do first epoch to load the data
        for batch in loader:
            pass
        print(f'First epoch with caching the data took {perf_counter() - tic} seconds')

        dataset.use_cache = True
        # loader.num_workers = num_workers
        epoch_num -= 1

    ep1 = perf_counter()
    for i in range(epoch_num):
        for batch in loader:
            pass

    print(f'Mean epoch time is {(perf_counter() - ep1) / epoch_num} seconds, '
          f'dataset contains {len(dataset)} images, '
          f'with batch size of {batch_size}\n')


if __name__ == '__main__':
    transform = tf.Compose([
        tf.Resize(512),  # rescale
        tf.RandomResizedCrop(256),
        tf.ToTensor(),  # convert to [0, 1] range
    ])

    slow_dataset = MyDataset('../data/wikiart', transform=transform, img_limit=1000)
    fast_dataset = CachedDataset('../data/wikiart', transform=transform, img_limit=1000, max_cache_size=1000)

    run_performance_test(slow_dataset, batch_size=8, epoch_num=2)  # 25s per epoch for 1k imgs
    run_performance_test(fast_dataset, batch_size=8, epoch_num=2)  # 10s per epoch after loading the data
