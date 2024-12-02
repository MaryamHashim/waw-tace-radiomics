import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from dataclasses import dataclass
from IPython.display import display
from ipywidgets import widgets, interact


@dataclass
class VolumeDisplay:
    paths: list[str]
    mask_path: str | None = None
    minimum: int = -100
    maximum: int = 250

    def __post_init__(self):
        self.volumes_sitk = [sitk.ReadImage(path) for path in self.paths]
        if len(self.volumes_sitk) > 1:
            self.volumes_sitk[1:] = [
                self.resample(v, self.volumes_sitk[0])
                for v in self.volumes_sitk[1:]
            ]
        self.volumes = [sitk.GetArrayFromImage(v) for v in self.volumes_sitk]
        self.volumes = [self.preprocess_volume(v) for v in self.volumes]

        self.min_slice = 0
        self.max_slice = self.volumes[0].shape[0]

        if self.mask_path is not None:
            self.mask_sitk = sitk.ReadImage(self.mask_path)
            self.mask_sitk = self.resample(
                self.mask_sitk, self.volumes_sitk[0], is_mask=True
            )
            self.mask = sitk.GetArrayFromImage(self.mask_sitk)
        else:
            self.mask = None

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_yticklabels([])
        self.ax.set_xticklabels([])

        self.slice_widget = widgets.IntSlider(
            min=self.min_slice,
            max=self.max_slice - 1,
            step=1,
            value=0,
            description="Slice index",
        )
        self.volume_widget = widgets.Dropdown(
            value=0,
            options=[i for i in range(len(self.volumes))],
            description="Volume index",
        )
        self.mask_widget = widgets.ToggleButton(
            value=False, description="Show mask"
        )

    def preprocess_volume(self, volume: np.ndarray) -> np.ndarray:
        volume = np.where(volume < self.minimum, self.minimum, volume)
        volume = np.where(volume > self.maximum, self.maximum, volume)
        volume = (volume - volume.min()) / (volume.max() - volume.min())
        return volume

    def resample(
        self,
        sitk_image: sitk.Image,
        target_image: sitk.Image,
        is_mask: bool = False,
    ) -> sitk.Image:
        return sitk.Resample(
            sitk_image,
            target_image,
            sitk.Transform(),
            sitk.sitkNearestNeighbor if is_mask is True else sitk.sitkBSpline,
            sitk_image.GetPixelIDValue(),
        )

    def display_slice(
        self, slice_idx: int, show_mask: bool = False, volume_idx: int = 0
    ):
        s = np.stack(
            [self.volumes[volume_idx][slice_idx] for _ in range(3)], axis=-1
        )
        if self.mask is not None and show_mask is True:
            s[:, :, 0] = np.where(self.mask[slice_idx] > 0.5, 1, s[:, :, 0])
            # s[:,:, 1] = np.where(self.mask[slice_idx] > 0.5, 1, s[:,:, 1])
            # s[:,:, 2] = np.where(self.mask[slice_idx] > 0.5, 1, s[:,:, 2])
        self.ax.imshow(s)
        self.fig.canvas.draw()
        display(self.fig)

    def __call__(self):
        interact(
            self.display_slice,
            slice_idx=self.slice_widget,
            show_mask=self.mask_widget,
            volume_idx=self.volume_widget,
        )


def resample(
    sitk_image: sitk.Image, target_image: sitk.Image, is_mask: bool = False
) -> sitk.Image:
    return sitk.Resample(
        sitk_image,
        target_image,
        sitk.Transform(),
        sitk.sitkNearestNeighbor if is_mask is True else sitk.sitkBSpline,
        sitk_image.GetPixelIDValue(),
    )
