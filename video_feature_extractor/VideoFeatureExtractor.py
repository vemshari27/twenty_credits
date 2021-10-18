import json
import urllib
import torch
from pytorchvideo.data.encoded_video import EncodedVideo

from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)

from PackPathway import PackPathway

class VideoFeatureExtractor(torch.nn.Module):
    def __init__(self, model_name="slowfast_r50", device="cpu"):
        super().__init__()
        self.device = device
        self.model_name = model_name
        if model_name == "slowfast_r50":
            # Pick a pretrained model 
            self.model = torch.hub.load("facebookresearch/pytorchvideo:main", model=model_name, pretrained=True)

            # Set to eval mode and move to desired device
            self.model = self.model.to(device)
            self.model = self.model.eval()

            side_size = 256
            mean = [0.45, 0.45, 0.45]
            std = [0.225, 0.225, 0.225]
            crop_size = 256
            num_frames = 32
            sampling_rate = 2
            frames_per_second = 30
            
            self.transform =  ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames),
                        Lambda(lambda x: x/255.0),
                        NormalizeVideo(mean, std),
                        ShortSideScale(
                            size=side_size
                        ),
                        CenterCropVideo(crop_size),
                        PackPathway()
                    ]
                ),
            )
            self.clip_duration = (num_frames * sampling_rate)/frames_per_second

        elif model_name == "x3d_s":
            self.model = torch.hub.load("facebookresearch/pytorchvideo:main", model_name, pretrained=True)
            # self.model = torch.hub.load("../pytorchvideo", model=model_name, source="local")
            # # checkpoint = torch.hub.load_state_dict("/home/srihari/udiva/personality_project/video_feature_extractor/models/X3D_S.pyth", map_location=device)

            # # # Unwrap the DistributedDataParallel module
            # # # module.layer -> layer
            # # state_dict = checkpoint["model_state"]
            # self.model.load_state_dict("/home/srihari/udiva/personality_project/video_feature_extractor/models/X3D_S.pyth")

            self.model = self.model.eval()
            self.model = self.model.to(device)

            mean = [0.45, 0.45, 0.45]
            std = [0.225, 0.225, 0.225]
            frames_per_second = 30
            model_transform_params  = {
                "x3d_xs": {
                    "side_size": 182,
                    "crop_size": 182,
                    "num_frames": 4,
                    "sampling_rate": 12,
                },
                "x3d_s": {
                    "side_size": 182,
                    "crop_size": 182,
                    "num_frames": 13,
                    "sampling_rate": 6,
                },
                "x3d_m": {
                    "side_size": 256,
                    "crop_size": 256,
                    "num_frames": 16,
                    "sampling_rate": 5,
                }
            }

            # Get transform parameters based on model
            transform_params = model_transform_params[model_name]

            # Note that this transform is specific to the slow_R50 model.
            self.transform =  ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(transform_params["num_frames"]),
                        Lambda(lambda x: x/255.0),
                        NormalizeVideo(mean, std),
                        ShortSideScale(size=transform_params["side_size"]),
                        CenterCropVideo(
                            crop_size=(transform_params["crop_size"], transform_params["crop_size"])
                        )
                    ]
                ),
            )

            # The duration of the input clip is also specific to the model.
            self.clip_duration = (transform_params["num_frames"] * transform_params["sampling_rate"])/frames_per_second
        
        elif model_name == "slow_r50":
            self.model = torch.hub.load('facebookresearch/pytorchvideo:main', 'slow_r50', pretrained=True)
            self.model = self.model.eval()
            self.model = self.model.to(device)

            side_size = 256
            mean = [0.45, 0.45, 0.45]
            std = [0.225, 0.225, 0.225]
            crop_size = 256
            num_frames = 8
            sampling_rate = 8
            frames_per_second = 30

            # Note that this transform is specific to the slow_R50 model.
            self.transform =  ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames),
                        Lambda(lambda x: x/255.0),
                        NormalizeVideo(mean, std),
                        ShortSideScale(
                            size=side_size
                        ),
                        CenterCropVideo(crop_size=(crop_size, crop_size))
                    ]
                ),
            )

            # The duration of the input clip is also specific to the model.
            self.clip_duration = (num_frames * sampling_rate)/frames_per_second
        # self.model = torch.nn.Sequential(*list(self.model.children())[-1])
        
    def forward(self, video_data):
        # Apply a transform to normalize the video input
        video_data = self.transform(video_data)

        # Move the inputs to the desired device
        inputs = video_data["video"]
        inputs = [i.to(self.device)[None, ...] for i in inputs]

        # Pass the input clip through the model 
        preds = self.model(inputs)

        return preds

