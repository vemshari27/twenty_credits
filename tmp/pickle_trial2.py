import pickle

f = open("../video_features_final4.pkl", "rb")
end = 0
while True:
    dicts = pickle.load(f)
    print(len(dicts.keys()))
    end += 1
    print(end)
