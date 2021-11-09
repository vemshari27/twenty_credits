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

class VideoTransformer:
    def __init__(self, model_name, device="cpu"):
        self.model_name = model_name
        self.device = device
    
    def transform(self, X):
        if self.model_name == "x3d_s":
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
            transform_params = model_transform_params[self.model_name]

            # Note that this transform is specific to the slow_R50 model.
            transform =  ApplyTransformToKey(
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
            clip_duration = (transform_params["num_frames"] * transform_params["sampling_rate"])/frames_per_second
            input_batch = []
            for video_path in X:
                video = EncodedVideo.from_path(X)
                video_data = video.get_clip(start_sec=0, end_sec=video.duration)
                video_data = transform(video_data)
                inputs = video_data["video"]
                inputs.to(self.device)
                inputs = inputs[None, ...]
                input_batch.append(inputs[0])
            return input_batch
