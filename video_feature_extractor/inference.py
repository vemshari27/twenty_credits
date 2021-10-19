import argparse
import os
import numpy as np
import pickle

from VideoFeatureExtractor import VideoFeatureExtractor

from pytorchvideo.data.encoded_video import EncodedVideo

def inference(opt, ):
    source, device, output, model_name, do_split = opt.source, opt.device, opt.output, opt.model_name, opt.do_split
    if do_split:
        model = VideoFeatureExtractor(model_name, device=device)
        print(model)

        pkfile = open(output+'/'+'video_features', 'ab')
        file_num = 0
        for dyad in os.listdir(source):
            file_num += 1
            features = {}
            print("In a dyad", dyad, file_num)
            dyad_in_path = os.path.join(source, dyad)
            dyad_out_path = os.path.join(output, dyad)
            # if not os.path.isdir(dyad_out_path):
            #     os.mkdir(dyad_out_path)

            video_features = []
            for video_file in os.listdir(dyad_in_path):
                print("In a video")
                video = EncodedVideo.from_path(dyad_in_path+'/'+video_file)
                video_name = video_file[:-4]
                start_sec = 0
                end_sec = video.duration
                clip_features = []
                while(start_sec<end_sec):
                    video_data = video.get_clip(start_sec=start_sec, end_sec=min(start_sec+3, end_sec))
                    preds = model(video_data).cpu().detach().numpy()
                    # print(preds.shape)
                    clip_features.append(preds)
                    start_sec += 6
                    # break
                clip_features = np.asarray(clip_features, dtype=np.float32)
                video_features.append(clip_features)
            features[dyad] = video_features
            pickle.dump(features, pkfile)                     
        
        pkfile.close()

        # # second version
        # model = VideoFeatureExtractor(model_name, device=device)
        # print(model)

        # for dyad in os.listdir(source):
        #     dyad_in_path = os.path.join(source, dyad)
        #     dyad_out_path = os.path.join(output, dyad)
        #     if not os.path.isdir(dyad_out_path):
        #         os.mkdir(dyad_out_path)

        #     for video_file in os.listdir(dyad_in_path):
        #         video = EncodedVideo.from_path(dyad_in_path+'/'+video_file)
        #         video_name = video_file[:-4]
        #         if not os.path.isdir(dyad_out_path+'/'+video_name):
        #             os.mkdir(dyad_out_path+'/'+video_name)
        #         start_sec = 0
        #         end_sec = video.duration
        #         # clip_features = []
        #         while(start_sec<end_sec):
        #             video_data = video.get_clip(start_sec=start_sec, end_sec=min(start_sec+3, end_sec))
        #             preds = model(video_data).cpu().detach().numpy()
        #             np.save(dyad_out_path+'/'+video_name+'/'+video_name+'_'+str(start_sec), preds)
        #             start_sec += 6
        #             # break
        #         # clip_features = np.asarray(clip_features, dtype=np.float32)
        #         # np.save(dyad_out_path+'/'+video_name, clip_features)


            
    else:
        model = VideoFeatureExtractor(model_name, device=device)
        for sub_f in os.listdir(source):
            print("Entered sub folder:", sub_f)

            if not os.path.isdir(output+'/'+sub_f):
                os.mkdir(output+'/'+sub_f)

            for video_file in os.listdir(source+'/'+sub_f):
                if not os.path.isfile(output+'/'+sub_f+'/'+video_file[:-4]+".npy"):
                    print("Processing video:", video_file)
                    video = EncodedVideo.from_path(source+'/'+sub_f+'/'+video_file)
                    video_data = video.get_clip(start_sec=0, end_sec=video.duration)
                    preds = model(video_data).cpu().detach().numpy()
                    np.save(output+'/'+sub_f+'/'+video_file[:-4]+".npy", preds)
                else:
                    print("Skipping video:", video_file)
        # for video_file in os.listdir(source):
        #     video = EncodedVideo.from_path(source+'/'+video_file)
        #     video_data = video.get_clip(start_sec=0, end_sec=video.duration)
        #     preds = model(video_data).cpu().detach().numpy()
        #     print(preds.shape)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--source", type=str, default="../data")
    parser.add_argument("--do_split", action="store_true")
    parser.add_argument("--output", type=str, default="../output")
    parser.add_argument("--model_name", type=str, default="slowfast_r50")
    opt = parser.parse_args()
    print(opt)

    inference(opt)