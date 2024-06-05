from cn_clip.training.data import LMDBDataset, pad_dataset, DataInfo
from torch.utils.data import SequentialSampler, DataLoader


# no match for distributed only for self train
def get_dataset(db_path, resolution, batch_size, is_train, max_txt_length=64, epoch_id=0):
    assert db_path is not None

    dataset = LMDBDataset(
        db_path,
        split="train" if is_train else "val",
        max_txt_length=max_txt_length,
        use_augment=is_train,
        resolution=resolution,
    )

    # pad the dataset splits using the beginning samples in the LMDB files
    # to make the number of samples enough for a full final global batch
    pad_dataset(dataset, batch_size)

    num_samples = dataset.dataset_len
    # Update in 22.12.11: We have changed the **validation** dataset sampler during finetuning
    # from sequential to shuffled (in a determistic order between experiments and epochs).
    # This is to avoid there being one text matching multiple images (or vice versa) in a local batch
    # which will affect the correctness of computing the validation in-batch accuracy.
    sampler = SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=12,
        sampler=sampler,
    )

    dataloader.num_samples = num_samples
    assert num_samples % dataset.global_batch_size == 0
    dataloader.num_batches = num_samples // dataset.global_batch_size

    return DataInfo(dataloader, sampler, dataset, epoch_id)
