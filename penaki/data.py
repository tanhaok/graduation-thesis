"""
    This module used to handle data in both image and video
"""
from pathlib import Path
from typing import Tuple, List
from functools import partial
import cv2
from PIL import Image
from beartype.door import is_bearable
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader as PytorchDataLoader
from torchvision import transforms as T
from einops import rearrange


def exists(val):
    """_summary_

    Args:
        val (_type_): _description_

    Returns:
        _type_: _description_
    """
    return val is not None


def identity(_t, *_, **__):
    """_summary_

    Args:
        t (_type_): _description_

    Returns:
        _type_: _description_
    """
    return _t


def pair(val):
    """_summary_

    Args:
        val (_type_): _description_

    Returns:
        _type_: _description_
    """
    return val if isinstance(val, tuple) else (val, val)


def cast_num_frames(_t, *, frames):
    """_summary_

    Args:
        _t (_type_): _description_
        frames (_type_): _description_

    Returns:
        _type_: _description_
    """
    _frames = _t.shape[1]

    if _frames == frames:
        return _t

    if _frames > frames:
        return _t[:, :frames]

    return F.pad(_t, (0, 0, 0, 0, 0, frames - _frames))


def convert_image_to_fn(img_type, image):
    """_summary_

    Args:
        img_type (_type_): _description_
        image (_type_): _description_

    Returns:
        _type_: _description_
    """
    if image.mode != img_type:
        return image.convert(img_type)
    return image


# constants: handle reading and writing gif
CHANNELS_TO_MODE = {1: 'L', 3: 'RGB', 4: 'RGBA'}


# image and video related helpers functions and dataset
class ImageDataset(Dataset):
    """_summary_

    Args:
        Dataset (_type_): _description_
    """

    def __init__(self, folder, image_size, exts=['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [
            p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')
        ]

        print(f'{len(self.paths)} training samples found at {folder}')

        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')
                     if img.mode != 'RGB' else img),
            T.Resize(image_size),
            T.RandomHorizontalFlip(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)


def seek_all_images(img, channels=3):
    """Seek all images

    Args:
        img (_type_): _description_
        channels (int, optional): _description_. Defaults to 3.

    Yields:
        _type_: _description_
    """
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1


def video_tensor_to_gif(tensor, path, duration=120, loop=0, optimize=True):
    """Convert from tensor to gif format

    Args:
        tensor (touch.tensor): tensor
        path (str): path to save file gif
        duration (int, optional): Time of gif file. Defaults to 120.
        loop (int, optional): Times to loop. Defaults to 0.
        optimize (bool, optional): . Defaults to True.

    Returns:
        map: image array
    """
    images = map(T.ToPILImage(), tensor.unbind(dim=1))
    first_img, *rest_imgs = images
    first_img.save(path,
                   save_all=True,
                   append_images=rest_imgs,
                   duration=duration,
                   loop=loop,
                   optimize=optimize)
    return images


# gif -> (channels, frame, height, width) tensor


def gif_to_tensor(path, channels=3, transform=T.ToTensor()):
    """_summary_

    Args:
        path (_type_): _description_
        channels (int, optional): _description_. Defaults to 3.
        transform (_type_, optional): _description_. Defaults to T.ToTensor().

    Returns:
        _type_: _description_
    """
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels=channels)))
    return torch.stack(tensors, dim=1)


def video_to_tensor(path: str, num_frames=-1, crop_size=None):
    """Handle reading and writing mp4

    Args:
        path (str): Path of the video to be imported
        num_frames (int, optional): Number of frames to be stored in the output tensor. 
                                    Defaults to -1.
        crop_size (_type_, optional): (width, height): Size of frame want to crop. Defaults to None.

    Returns:
        torch.tensor:  shape (1, channels, frames, height, width)
    """

    video = cv2.VideoCapture(path)

    frames = []
    check = True

    while check:
        check, frame = video.read()

        if not check:
            continue

        if exists(crop_size):
            frame = crop_center(frame, *pair(crop_size))

        frames.append(rearrange(frame, '... -> 1 ...'))

    # convert list of frames to numpy array
    frames = np.array(np.concatenate(frames[:-1], axis=0))
    frames = rearrange(frames, 'f h w c -> c f h w')

    frames_torch = torch.tensor(frames).float()

    return frames_torch[:, :num_frames, :, :]


def tensor_to_video(tensor, path: str, fps=25, video_format='MP4V'):
    """ Import the video and cut it into frames.

    Args:
        tensor (touch.tensor): Pytorch video tensor
        path (str): Path of the video to be saved
        fps (number): Frames per second for the saved video
    Returns:
        _type_: _description_
    """
    tensor = tensor.cpu()

    num_frames, height, width = tensor.shape[-3:]

    # Changes in this line can allow for different video formats.
    fourcc = cv2.VideoWriter_fourcc(*video_format)
    video = cv2.VideoWriter(path, fourcc, fps, (width, height))

    for idx in range(num_frames):
        numpy_frame = tensor[:, idx, :, :].numpy()
        numpy_frame = np.uint8(rearrange(numpy_frame, 'c h w -> h w c'))
        video.write(numpy_frame)

    video.release()

    cv2.destroyAllWindows()

    return video


def crop_center(img, cropx, cropy):
    """Function used to crop tensor

    Args:
        img (tensor): tensor to be crop
        cropx (number):  Length of the final image in the x direction.
        cropy (number):  Length of the final image in the y direction.

    Returns:
        tensor: a tensor after crop
    """
    height, width, _ = img.shape
    startx = width // 2 - cropx // 2
    starty = height // 2 - cropy // 2
    return img[starty:(starty + cropy), startx:(startx + cropx), :]


# video dataset


class VideoDataset(Dataset):
    """_summary_

    Args:
        Dataset (_type_): _description_
    """

    def __init__(self,
                 folder,
                 image_size,
                 channels=3,
                 num_frames=17,
                 horizontal_flip=False,
                 force_num_frames=True,
                 exts=['gif', 'mp4']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.channels = channels
        self.paths = [
            p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')
        ]

        self.transform = T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip()
            if horizontal_flip else T.Lambda(identity),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

        # functions to transform video path to tensor

        self.gif_to_tensor = partial(gif_to_tensor,
                                     channels=self.channels,
                                     transform=self.transform)
        self.mp4_to_tensor = partial(video_to_tensor,
                                     crop_size=self.image_size)

        self.cast_num_frames_fn = partial(
            cast_num_frames,
            frames=num_frames) if force_num_frames else identity

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        ext = path.suffix

        if ext == '.gif':
            tensor = self.gif_to_tensor(path)
        elif ext == '.mp4':
            tensor = self.mp4_to_tensor(str(path))
        else:
            raise ValueError(f'unknown extension {ext}')

        return self.cast_num_frames_fn(tensor)


def collate_tensors_and_strings(data):
    """_summary_

    Args:
        data (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if is_bearable(data, List[torch.Tensor]):
        return (torch.stack(data, dim=0), )

    data = zip(*data)
    output = []

    for datum in data:
        if is_bearable(datum, Tuple[torch.Tensor, ...]):
            datum = torch.stack(datum, dim=0)
        elif is_bearable(datum, Tuple[str, ...]):
            datum = list(datum)
        else:
            raise ValueError('detected invalid type being passed from dataset')

        output.append(datum)

    return tuple(output)


def DataLoader(*args, **kwargs):
    """ Override dataloader to be able to collate strings
    Returns:
        _type_: _description_
    """
    return PytorchDataLoader(*args,
                             collate_fn=collate_tensors_and_strings,
                             **kwargs)
