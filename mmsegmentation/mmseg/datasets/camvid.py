# %%
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import os.path as osp
# %%
# Camvid classes and palette acquisition
classes32 = []
palette32 = []
with open("./data/CamVid32/class_dict.csv") as class_file:
    for line in class_file:
        split_line = line.split(",")
        classes32.append(split_line[0].strip())
        palette32.append([split_line[1].strip(), split_line[2].strip(), split_line[3].strip()])
classes32 = tuple(classes32[1:])
palette32 = palette32[1:]
palette32 = [[int(int(j)) for j in i] for i in palette32]

classes11 = []
palette11 = []
with open("./data/CamVid11/class_dict.csv") as class_file:
    for line in class_file:
        split_line = line.split(",")
        classes11.append(split_line[0].strip())
        palette11.append([split_line[1].strip(), split_line[2].strip(), split_line[3].strip()])
classes11 = tuple(classes11[1:])
palette11 = palette11[1:]
palette11 = [[int(int(j)) for j in i] for i in palette11]



# %%
@DATASETS.register_module()
class Camvid32Dataset(CustomDataset):

    """CamVid dataset (original 32 classes) for semantic segmentation.

    dataset (and split) obtained from: 
    https://www.kaggle.com/datasets/carlolepelaars/camvid
    """

    CLASSES = classes32
    PALETTE = palette32
    def __init__(self, **kwargs):
        super().__init__(img_suffix='.png', seg_map_suffix='_L.png', **kwargs)
        assert osp.exists(self.img_dir) is not None
        
@DATASETS.register_module()
class Camvid11Dataset(CustomDataset):

    """CamVid dataset (11 classes (+ 1 void) as used in SegNet) for semantic segmentation.
    """

    CLASSES = classes11
    PALETTE = palette11
    def __init__(self, **kwargs):
        super().__init__(img_suffix='.png', seg_map_suffix='_L.png', **kwargs)
        assert osp.exists(self.img_dir) is not None