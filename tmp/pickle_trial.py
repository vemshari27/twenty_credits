import pickle

d1 = {'red':['rose', 'blood'], 'green': ['leaf']}
# d1 = {'yellow':['banana', 'taxi'], 'red': ['ocean']}

pkfile = open("dicts", "ab")
pickle.dump(d1, pkfile)
pkfile.close()