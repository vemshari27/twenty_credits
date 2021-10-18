import pickle

final_dict = {}

f = open("../video_features_final3.pkl", "rb")
for i in range(1):
    dicts = pickle.load(f)
    for i in dicts:
        if i not in final_dict.keys():
            final_dict[i] = dicts[i]
f.close()
f = open("../video_features", "rb")
for i in range(17):
    dicts = pickle.load(f)
    for i in dicts:
        if i not in final_dict.keys():
            final_dict[i] = dicts[i]

f.close()
print(len(final_dict.keys()))

f1 = open("../video_features_final4.pkl", "ab")
pickle.dump(final_dict, f1)
f1.close()