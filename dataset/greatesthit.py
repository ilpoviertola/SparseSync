from pathlib import Path
import csv
from math import ceil
import typing as tp

from einops import repeat
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch import Tensor

from dataset.dataset_utils import (
    get_audio_stream,
    get_fixed_offsets,
    get_video_and_audio,
)


EPS = 1e-9


class GreatestHitDataset(Dataset):
    def __init__(
        self,
        split: str,
        data_path: Path,
        transforms: tp.Optional[tp.Callable] = None,
        meta_path: Path = Path("./data/greatesthit.csv"),
        split_dir_path: Path = Path("./data/"),
        video_length: float = 5.0,
        sample_rate_audio: int = 24000,
        sample_rate_video: float = 25.0,
        run_additional_checks: bool = True,
        load_fixed_offsets_on_test=True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.split = split
        self.transforms = transforms
        self.max_clip_len_sec = None
        self.load_fixed_offsets_on_test = load_fixed_offsets_on_test

        split_file_path = split_dir_path / f"greatesthit_{self.split}.txt"
        assert (
            split_file_path.is_file()
        ), f"Split file {split_file_path.as_posix()} does not exist."

        self.data_path = data_path
        assert (
            data_path.is_dir()
        ), f"Data directory {data_path.as_posix()} does not exist."

        assert meta_path.is_file(), f"Meta file {meta_path.as_posix()} does not exist."

        self.dataset = []
        with open(split_file_path, encoding="utf-8") as f:
            within_split = f.read().splitlines()

        for basename in within_split:
            files = self._get_all_files_with_same_basename(basename, data_path)
            self.dataset += files

        (
            self.filename2label,
            self.filename2material,
            self.filename2motion,
        ) = self._get_filename2all(meta_path)

        unique_classes = sorted(list(set(self.filename2label.values())))
        self.label2target = {
            label: target for target, label in enumerate(unique_classes)
        }
        self.target2label = {
            target: label for label, target in self.label2target.items()
        }
        self.filename2target = {
            item[0]: self.label2target[item[1]] for item in self.filename2label.items()
        }

        self.video_len_in_samples = ceil(video_length * sample_rate_video)
        self.audio_len_in_samples = ceil(video_length * sample_rate_audio)
        self.video_len = video_length
        self.a_sr = sample_rate_audio
        self.v_sr = sample_rate_video

        if load_fixed_offsets_on_test and split in ["valid", "test"]:
            self.vid2offset_params = get_fixed_offsets(
                transforms, split, split_dir_path, "greatesthit"
            )

        if run_additional_checks:
            pass  # for now

    def __getitem__(self, index) -> dict:
        path = self.data_path / Path(self.dataset[index])
        rgb, audio, meta = self.load_media(path.as_posix())
        audio = F.pad(
            audio,
            (0, self.audio_len_in_samples - audio.shape[-1]),
            mode="constant",
            value=0,
        )
        rgb = rgb[: self.video_len_in_samples, :, :, :]
        # Calculate the number of frames to pad
        num_padding_frames = self.video_len_in_samples - rgb.shape[0]
        # If the number of frames to pad is greater than 0, pad the video tensor
        if num_padding_frames > 0:
            # Create a padding tensor of zeros with the same dimensions as the video tensor, except for the length
            padding = (0, 0, 0, 0, 0, 0, 0, num_padding_frames)
            rgb = F.pad(rgb, padding, mode="constant", value=0)
        item = self.make_datapoint(path, rgb, audio, meta)
        if self.transforms is not None:
            item = self.transforms(item)
        return item

    def __len__(self) -> int:
        return len(self.dataset)

    def load_media(self, path):
        rgb, audio, meta = get_video_and_audio(
            path, get_meta=True, max_clip_len_sec=self.max_clip_len_sec
        )
        return rgb, audio, meta

    def make_datapoint(self, path: Path, rgb: Tensor, audio: Tensor, meta: dict):
        # (Tv, 3, H, W) in [0, 225], (Ta, C) in [-1, 1]
        item: dict = {
            "video": rgb,
            "audio": audio,
            "meta": meta,
            "path": path.as_posix(),
            "targets": {
                "label": self.filename2label[path.name],
                "target": self.filename2target[path.name],
            },
            "split": self.split,
        }

        # loading the fixed offsets. COMMENT THIS IF YOU DON'T HAVE A FILE YET
        if self.load_fixed_offsets_on_test and self.split in ["valid", "test"]:
            item["targets"]["offset_sec"] = self.vid2offset_params[
                str(Path(path).stem)
            ]["offset_sec"]
            item["targets"]["v_start_i_sec"] = self.vid2offset_params[
                str(Path(path).stem)
            ]["v_start_i_sec"]

        return item

    @staticmethod
    def _get_all_files_with_same_basename(basename: str, data_dir: Path) -> list:
        all_files = data_dir.glob(f"{basename}_denoised*")
        return [
            f.name for f in list(all_files) if f.suffix == ".mp4"
        ]  # return only filenames

    @staticmethod
    def _get_filename2all(meta_path: Path) -> tp.Tuple[dict, dict, dict]:
        filename2label = {}
        filename2material = {}
        filename2motion = {}
        with open(meta_path, encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file)
            next(reader)  # skip header
            for row in reader:
                filename2label[row[0]] = row[5]
                filename2material[row[0]] = row[4]
                filename2motion[row[0]] = row[6]
        return filename2label, filename2material, filename2motion

    @staticmethod
    def _get_max_len_in_samples(filepath: Path) -> int:
        with open(filepath, encoding="utf-8") as f:
            return int(f.read().strip())

    @staticmethod
    def _get_base_file_name(filename: str) -> str:
        return "_".join(filename.split("_")[:-1])


class GreatestHitAudioOnlyDataset(GreatestHitDataset):
    def __init__(
        self,
        split: str,
        data_path: Path,
        transforms: tp.Optional[tp.Callable] = None,
        meta_path: Path = Path("./data/greatesthit.csv"),
        split_dir_path: Path = Path("./data/"),
        video_length: float = 2.0,
        sample_rate_audio: int = 16000,
        sample_rate_video: float = 25.0,
        run_additional_checks: bool = True,
        load_fixed_offsets_on_test=True,
        **kwargs,
    ) -> None:
        super().__init__(
            split,
            data_path,
            transforms,
            meta_path,
            split_dir_path,
            video_length,
            sample_rate_audio,
            sample_rate_video,
            run_additional_checks,
            load_fixed_offsets_on_test,
            **kwargs,
        )

    def __getitem__(self, index):
        path = self.data_path / Path(self.dataset[index])
        # (Ta, C) in [-1, 1]
        audio, meta = get_audio_stream(path.as_posix(), get_meta=True)

        target = self.filename2target[(path.with_suffix(".mp4")).name]
        item = {
            "audio": audio,
            "meta": meta,
            "path": path.as_posix(),
            "target": target,
            "label": self.target2label[target],
            "split": self.split,
        }

        if self.transforms is not None:
            item = self.transforms(item)

        return item
