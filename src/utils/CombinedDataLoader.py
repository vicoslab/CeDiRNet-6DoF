import itertools
import torch
from torch.utils.data import DataLoader

class CombinedDataLoader:
    def __init__(self, dataset1, dataset2, sampler1, sampler2, num_workers=0, pin_memory=False, collate_fn=None):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.batch_size1 = sampler1.batch_size
        self.batch_size2 = sampler2.batch_size

        self.dataloader1 = DataLoader(dataset1,
                                      batch_sampler=sampler1,
                                      num_workers=num_workers//2,
                                      pin_memory=pin_memory,
                                      collate_fn=collate_fn)

        self.dataloader2 = DataLoader(dataset2,
                                      batch_sampler=sampler2,
                                      num_workers=num_workers//2,
                                      pin_memory=pin_memory,
                                      collate_fn=collate_fn)

        size1 = len(self.dataloader1)
        size2 = len(self.dataloader2)

        if size1 < size2:
            self.small_dataset = self.dataloader2
            self.big_dataset = self.dataloader1
            self.len = size1
        else:
            self.small_dataset = self.dataloader1
            self.big_dataset = self.dataloader2
            self.len = size2

        self.index = 0

    def __len__(self):
        return self.len*2

    def __iter__(self):
        self.cycle = itertools.cycle(self.small_dataset)
        self.bigger_dataset = iter(self.big_dataset)
        return self

    def __next__(self):
        data1 = next(self.bigger_dataset)
        data2 = next(self.cycle)

        for key, value in data1.items():
            if type(value) is torch.Tensor:
                data1[key] = torch.cat((data1[key], data2[key]), 0)
            elif key == "centerdir_groundtruth":
                for i in range(len(value)):
                    data1[key][i] = torch.cat((data1[key][i], data2[key][i]), 0)
            elif key == "im_name":
                data1[key] += data2[key]
            elif key == "im_size":
                data1[key][0] = torch.cat((data1[key][0], data2[key][0]), 0)
                data1[key][1] = torch.cat((data1[key][1], data2[key][1]), 0)

        return data1
