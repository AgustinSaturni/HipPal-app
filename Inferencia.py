from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
    LoadImage
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob


def segmentar():
    base_path = "/home/ggimeno/Documents/electron-app"

# data_dir = r"/content/drive/MyDrive"
    data_dir = f"{base_path}/Hip-Pal_v2"

    test_images = sorted(glob.glob(os.path.join(
        r"/home/ggimeno/Documents/electron-app/Hip-Pal_v2", "tomografias_nii", "*.nii.gz")))
    print(f"que verga hay aca adentro: {len(test_images)}")

    test_data = [{"image": image} for image in test_images]

    device = torch.device("cpu")
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=8,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)

    test_org_transforms = Compose(
        [
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=400,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image"),
        ]
    )

    test_org_ds = Dataset(data=test_data, transform=test_org_transforms)

    test_org_loader = DataLoader(test_org_ds, batch_size=1, num_workers=4)

    post_transforms = Compose(
        [
            Invertd(
                keys="pred",
                transform=test_org_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ),
            AsDiscreted(keys="pred", argmax=True, to_onehot=8),
            SaveImaged(keys="pred", meta_keys="pred_meta_dict",
                       output_dir=r"/home/ggimeno/Documents/electron-app/Hip-Pal_v2/tomografias_segmentadas", output_postfix="seg", resample=False),
        ]
    )

    # root_dir=r"/content/drive/MyDrive"
    root_dir = r"/home/ggimeno/Documents/electron-app/Hip-Pal_v2"

    model.load_state_dict(torch.load(os.path.join(
        r"/home/ggimeno/Documents/electron-app/Hip-Pal_v2", "best_metric_model.pth"), map_location=device))
    model.eval()
    loader = LoadImage()
    with torch.no_grad():
        for test_data in test_org_loader:
            test_inputs = test_data["image"].to(device)
            roi_size = (160, 160, 160)
            sw_batch_size = 4
            test_data["pred"] = sliding_window_inference(
                test_inputs, roi_size, sw_batch_size, model)

            test_data = [post_transforms(i)
                         for i in decollate_batch(test_data)]

            # uncomment the following lines to visualize the predicted results
            test_output = from_engine(["pred"])(test_data)
