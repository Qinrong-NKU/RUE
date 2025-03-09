import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class cragdataset(CustomDataset):
    """DRIVE dataset.

    In segmentation map annotation for DRIVE, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '_manual1.png'.
    """
    
    CLASSES = ('class1', 'class2')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super(cragdataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_mask.png',
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir)