from torch.utils.data import DataLoader
from dataloaders.fundus_dataloader import Fundus
"""
将多个数据集的val_loader组合在一起
可根据自己代码的dataloader自己定义
"""
DomainA = Fundus(client_idx=0, split='train', transform = None)
DomainB = Fundus(client_idx=1, split='train', transform = None)
DomainC = Fundus(client_idx=2, split='train', transform = None)
DomainD = Fundus(client_idx=3, split='train', transform = None)
def setup_loader():
    """
    Setup Data Loaders
    """
    dataset = ['DomainA', 'DomainB', 'DomainC', 'DomainD']

    val_sets = []
    val_dataset_names = []

    for dataset_name in dataset:
        if 'DomainA' == dataset_name:
            val_set = DomainA
            val_sets.append(val_set)
            val_dataset_names.append('DomainA')

        if 'DomainB' == dataset_name:
            val_set = DomainB
            val_sets.append(val_set)
            val_dataset_names.append('DomainB')

        if 'DomainC' == dataset_name:
            val_set = DomainC
            val_sets.append(val_set)
            val_dataset_names.append('DomainC')

        if 'DomainD' == dataset_name:
            val_set = DomainD
            val_sets.append(val_set)
            val_dataset_names.append('DomainD')


    batch_size = 1

    extra_val_loader = {}
    for val_set,val_dataset_name in zip(val_sets,val_dataset_names):


        from datasets.sampler import DistributedSampler
        # val_sampler = DistributedSampler(val_set, pad=False, permutation=False, consecutive_sample=False)
        val_sampler = None

        val_loader = DataLoader(val_set, batch_size=batch_size,
                                shuffle=False, drop_last=False,
                                sampler=val_sampler)

        extra_val_loader[val_dataset_name] = val_loader

    return extra_val_loader #extra_val_loader里面包含多个数据集的val_loader