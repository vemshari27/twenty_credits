import pickle

f = open("../video_features_final4.pkl", "rb")
f1 = open("../videos.txt", "r")
f2 = open("../videos_new4.txt", "w")
dicts = pickle.load(f)
tmp = dicts.keys()
f.close()
for l in f1.readlines():
    video_name = l[-8:-2]
    if video_name not in tmp:
        f2.write(l)
f1.close()
f2.close()