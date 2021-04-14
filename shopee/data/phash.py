import pandas as pd


def merge_groups_by_phash(data: pd.DataFrame) -> pd.DataFrame:
    """Merge label_groups containing images with same phash

    Args:
        data: DataFrame from Shopee dataset (train/test). Will be modified

    Returns:
        the modified DataFrame containing a new column 'phash_group' so that
            each unique phash belong to only one phash_group
    """

    phash_group = (
        data.groupby("image_phash").label_group.agg("unique").to_dict()
    )
    data["phash_group"] = data.image_phash.map(phash_group)

    max_before = max(len(group) for group in data.phash_group)

    data["phash_group"] = data.apply(lambda row: min(row.phash_group), axis=1)
    tmp = data.groupby("image_phash").phash_group.agg("unique").to_dict()

    max_after = max(len(group) for group in tmp.values())

    print(
        f"Max groups per phash before processing: {max_before}\n"
        + f"Max groups per phash after processing: {max_after}"
    )

    return data
