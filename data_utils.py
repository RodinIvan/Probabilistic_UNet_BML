"""Custom instruments for working with data"""


import warnings
import os
from pathlib import Path
from collections import namedtuple
from typing import (List,
                    Callable,
                    Any,
                    Sequence,
                    Union,
                    Iterator,
                    Tuple,
                    Dict)

import nibabel
from sklearn import model_selection
import numpy as np

import torch
from torch.utils import data
from torch.utils.data.dataloader import default_collate

import utils


class LITSDataset(data.Dataset):

    """Dataset for LITS competition"""

    def __init__(self,
                 samples: np.array,
                 masks: np.array,
                 transform: Any = None) -> None:

        super(LITSDataset, self).__init__()

        self.samples = samples
        self.masks = masks
        self.shape = samples.shape
        self.size = samples.shape[0]
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, i: int) -> Sequence[np.array]:
        return utils.scale(self.samples[i]), self.masks[i]

    @staticmethod
    def collate_fn(in_batch: Sequence) -> Sequence[torch.Tensor]:

        """Custom collate_fn for LITSDataset"""

        samples, masks = zip(*in_batch)
        return torch.Tensor(samples).unsqueeze(1), torch.Tensor(masks)


class NiiReader(object):

    """Handling with medical .nii files"""

    def __init__(self, root: str = './') -> None:

        """Keyword arguments:
                — root: path to folder with nii-files or single nii-file.
        """

        if os.path.isdir(root):
            self.root_type = 'dir'
        elif os.path.isfile(root):
            self.root_type = 'file'
        else:
            raise ValueError('No such file or directory')
        self.root = root

    def read(self,
             quantity: int = 0,
             peculiar_pathes: Union[Sequence, str]=tuple()) -> np.array:

        """Reads, concatenates and returns concatenated nii slices.

            Keyword arguments:
                — quantity: max number of files in folder to read, in ascending
                    order, if there is only one file to read, quantity = 1.
                    —— 0: all files in directory will be read,
                    —— warning: if number of nii-files in root more than
                        quantity, warning is raised.
        """

        if peculiar_pathes and self.root_type != 'file':
            _join = os.path.join
            _base = os.path.basename
            if isinstance(peculiar_pathes, str):
                pathes = [peculiar_pathes]  # type: List
            else:
                pathes = list(peculiar_pathes)
            quantity = len(pathes)
            pathes = [_join(self.root, _base(p)) for p in pathes]
            for path in pathes:
                if not os.path.isfile(path):
                    raise FileExistsError('No such file exists')
        else:
            if self.root_type == 'file':
                pathes = [self.root]
            else:
                pathes = list(Path(self.root).glob('*.nii'))
            if not pathes:
                raise ValueError('No nii-files to read in root found')
            pathes.sort()
            if quantity == 0:
                _quantity = len(pathes)
            elif quantity > len(pathes):
                warnings.warn('There are less items in root than quantity')
                _quantity = len(pathes)
            else:
                _quantity = quantity

        gathered_data = np.array([])
        for path in pathes[:_quantity]:
            tmp_data = nibabel.load(str(path)).get_data()
            tmp_data = tmp_data.swapaxes(0, 2).swapaxes(1, 2)
            if gathered_data.size > 0:
                gathered_data = np.concatenate((gathered_data, tmp_data))
            else:
                gathered_data = tmp_data
        return gathered_data

    def read_save(self, save_path: str, quantity: int = 0) -> None:

        """Read and saves concatenated slices"""

        np.save(save_path, self.read(quantity))


_Shuffles = namedtuple('_Shuffles', ['train', 'test'])
_SplitSizes = namedtuple('_SplitSizes', ['train', 'val'])
_BatchSizes = namedtuple('_BatchSizes', ['train', 'val', 'test'])


class SplitSizes(_SplitSizes):

    def __new__(cls,
                train_size: float,
                val_size: float = 0):

        if train_size < 0 and train_size > 1:
            raise ValueError('Should be in interval (0, 1].')
        if val_size < 0 and val_size > 1:
            raise ValueError('Should be in interval (0, 1]')
        return super(SplitSizes, cls).__new__(cls, train_size, val_size)

    def __setattr__(self, name: Any, value: Any) -> None:
        raise AttributeError(r"can't set attribute")


class Shuffles(_Shuffles):

    def __new__(cls,
                train_shuffle: bool = True,
                test_shuffle: bool = False):

        _shuffles = [train_shuffle, test_shuffle]
        if any(not isinstance(_s, bool) for _s in _shuffles):
            raise ValueError('Should be of type bool')

        return super(Shuffles, cls).__new__(cls, *_shuffles)

    def __setattr__(self, name: Any, value: Any) -> None:
        raise AttributeError(r"can't set attribute")


class BatchSizes(_BatchSizes):

    def __new__(cls,
                train_bs: int,
                val_bs: int = 0,
                test_bs: int = 1):

        _batch_sizes = [train_bs, val_bs, test_bs]
        if any(not isinstance(_bs, int) for _bs in _batch_sizes):
            raise ValueError('Should be of type int')

        return super(BatchSizes, cls).__new__(cls, *_batch_sizes)

    def __setattr__(self, name: Any, value: Any) -> None:
        raise AttributeError(r"can't set attribute")


def train_val_test_split(*arrays: Sequence[Sequence[np.array]],
                         split_sizes: Sequence = SplitSizes(0.8, 0.3),
                         shuffles: Sequence = Shuffles(True, False),
                         random_state: Union[float, None]=None,
                         ) -> List[np.array]:

    """Split arrays or matrices into random train, val and test subsets"""

    _fn = model_selection.train_test_split
    _fn_params = dict()
    _split_sizes = SplitSizes(*split_sizes)  # type: SplitSizes
    _shuffles = Shuffles(*shuffles)  # type: Shuffles

    _fn_params['train_size'] = _split_sizes.train
    _fn_params['shuffle'] = _shuffles.test
    _fn_params['random_state'] = random_state
    x_train, x_test, y_train, y_test = _fn(*arrays, **_fn_params)

    _fn_params['test_size'] = _split_sizes.val
    _fn_params['shuffle'] = _shuffles.train
    x_train, x_val, y_train, y_val = _fn(x_train, y_train, **_fn_params)
    return [x_train, x_val, x_test, y_train, y_val, y_test]


class DataLoaderConstructor(object):

    """Constructs train, val (?) and test loaders from given data. Always
    returns 6 loaders, if val_size is None, x_val and y_val will be None.
    """

    def __init__(self,
                 dataset: data.Dataset,
                 split_sizes: Sequence = SplitSizes(0.8, 0),
                 batch_sizes: Sequence = BatchSizes(5, 5, 1),
                 shuffles: Sequence = Shuffles(True, False),
                 random_state: Union[float, None]=None,
                 collate_fn: Callable = default_collate) -> None:

        """Arguments:
            – dataset: type of dataset will be used for dataloaders.

        Keyword arguments:
            — split_size: Sequence with train size and val_size respectively,
            — batch_sizes: Sequence with batch size for train, val, test stage,
            — shuffles: Sequence with shuffles parameters for train_test_split,
            — collate_fn: special function for dataset.
        """

        self.dataset = dataset
        self.split_sizes = SplitSizes(*split_sizes)
        self.batch_sizes = BatchSizes(*batch_sizes)
        self.shuffles = Shuffles(*shuffles)
        self.collate_fn = collate_fn
        self.random_state = random_state

    def from_numpy(self, *in_data: Sequence[np.array]) -> Sequence:

        """Creates dataloader from np.arrays"""

        if len(in_data) != 2:
            raise ValueError('in_data contains sample and mask parts only')

        _fn_params = dict()  # type: Dict[str, Any]
        _fn_params['random_state'] = self.random_state
        if self.split_sizes.val:
            _fn = train_val_test_split
            _fn_params['shuffles'] = self.shuffles
            _fn_params['split_sizes'] = self.split_sizes
            _data = _fn(*in_data, **_fn_params)
            del _fn_params['shuffles'], _fn_params['split_sizes']
        else:
            _fn = model_selection.train_test_split
            _fn_params['shuffle'] = self.shuffles.test
            _fn_params['train_size'] = self.split_sizes.train
            _data = _fn(*in_data, **_fn_params)
            _data = utils.insert_at_positions(_data, {0: None, 2: None})

        _data = zip(*np.split(np.array(_data), [3]))  # type: ignore
        loaders = []  # type: List
        for data_pair, batch_size in zip(_data, self.batch_sizes):
            if all(item is None for item in data_pair):
                loader = None
            else:
                params = {'batch_size': batch_size,
                          'collate_fn': self.collate_fn}
                loader = data.DataLoader(self.dataset(*data_pair), **params)
            loaders.append(loader)
        return loaders

    def from_nii(self,
                 *in_roots: Sequence[str],
                 quantity: int = 0) -> Sequence:

        """Creates dataloaders from nii-file or folder with nii-files"""

        if len(in_roots) != 2:
            raise ValueError('in_roots contains sample and mask parts only')

        samples_root, masks_root = in_roots
        return self.from_numpy(NiiReader(samples_root).read(quantity),
                               NiiReader(masks_root).read(quantity))
