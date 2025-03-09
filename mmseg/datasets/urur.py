import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class ururdataset(CustomDataset):
    """DRIVE dataset.

    In segmentation map annotation for DRIVE, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '_manual1.png'.
    """
    CLASSES = ('building', 'farmland',"greenhouse","woodland","bareland","water","road","others")
    PALETTE = [[0, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [0, 255, 0], 
               [0, 0, 255], [255, 0, 0],[255, 255, 255]]

    def __init__(self, **kwargs):
        super(ururdataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir)