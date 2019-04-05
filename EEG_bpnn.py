from bpnn import BPNN
from bpnn_ref import BPNNet
import pandas as pd
import numpy as np
import sys
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix  

np.set_printoptions(threshold=np.nan)

def read_feature(filename):
    # 读取EEG_feature文件
    with open(filename) as f:
        lines = f.readlines()
        data = []
        for line in lines:
            line = line.split()
            line = [float(i) for i in line]
            data.append(line)
    return np.array(data)

def read_label(filename):
    # 读取*_label文件
    with open(filename) as f:
        lines = f.readlines()
        label = []
        for line in lines:
            line = line.split()
            # line = [int(i) for i in line]
            label.append(int(line[0]))
    return np.array(label)

def load_EEG_va_train():
    # EEG-valence-arousal训练集
    train_labels = read_label("./EEG/DEAP/valence_arousal_label.txt")
    train_labels = train_labels - 1
    train_data = read_feature("./EEG/DEAP/EEG_feature.txt")
    return train_data, train_labels        

def load_EEG_va_test():
    # EEG-valence-arousal测试集
    test_labels = read_label("./EEG/MAHNOB-HCI/valence_arousal_label.txt")
    test_labels = test_labels - 1
    test_data = read_feature("./EEG/MAHNOB-HCI/EEG_feature.txt")
    return test_data, test_labels  

def load_EEG_va_data():
    # 导入EEG-valence-arousal数据
    train_data, train_labels = load_EEG_va_train()
    test_data, test_labels = load_EEG_va_test()
    return train_data, train_labels, test_data, test_labels

def load_EEG_valence_data():
    # 导入EEG-valence数据
    data1 = read_feature("./EEG/DEAP/EEG_feature.txt")
    label1 = read_label("./EEG/DEAP/valence_arousal_label.txt")
    data2 = read_feature("./EEG/MAHNOB-HCI/EEG_feature.txt")
    label2 = read_label("./EEG/MAHNOB-HCI/valence_arousal_label.txt")
    data = np.row_stack((data1, data2))
    if len(label1.shape) == 1:
        label = np.array(list(label1) + list(label2))
    else:
        label = np.row_stack((label1, label2))
    return data, label - 1

def load_EEG_emotion_data():
    # 导入EEG-emotion数据
    labels = read_label("./EEG/MAHNOB-HCI/EEG_emotion_category.txt")
    data = read_feature("./EEG/MAHNOB-HCI/EEG_feature.txt")
    return data, labels

def group_sampling(labels):
    # 分层抽样
    # 用于emotion(多分类)
    index = range(len(labels))
    df = pd.DataFrame({'index': index,
                       'label': labels})
    grouped = df.groupby('label')
    g = grouped.apply(lambda x: x.sample(frac=0.2))
    sample_index = list(g['index'])
    return sample_index

def gen_grouped_data_set(data, labels, index):
    # 按照分层抽样结果返回训练集,测试集
    # 用于emotion
    train_data, test_data = [], []
    train_label, test_label = [], []
    for i in range(len(labels)):
        if i in index:
            test_data.append(data[i])
            test_label.append(labels[i])
        else:
            train_data.append(data[i])
            train_label.append(labels[i])
    return np.array(train_data), np.array(test_data), \
        np.array(train_label), np.array(test_label)

def ref_std(data, labels):
    # 根据bpnn_ref.py的数据格式
    samples = []
    for i in range(data.shape[0]):
        dat = list(data[i])
        label = list(labels[i])
        sample = []
        sample.append(dat)
        sample.append(label)
        samples.append(sample)
    return samples

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--random-seed', type=int, default=3)
args = parser.parse_args()

seed = args.random_seed

# EEG-emotion
# data, labels = load_EEG_emotion_data()
# index = group_sampling(labels)
# train_data, test_data, train_label, test_label = gen_grouped_data_set(data, labels, index)

# EEG-valence
data, labels = load_EEG_valence_data()
index = group_sampling(labels)
train_data, test_data, train_label, test_label = gen_grouped_data_set(data, labels, index)

print(pd.value_counts(train_label))
print(pd.value_counts(test_label))

eeg_bpnn = BPNN(160, 20, 1, random_seed=seed)
eeg_bpnn.accumulate_train(samples=train_data, labels=train_label, rate=1, epochs=300)
eeg_bpnn.accumulate_train(samples=train_data, labels=train_label, rate=0.5, epochs=300)
eeg_bpnn.accumulate_train(samples=train_data, labels=train_label, rate=0.1, epochs=300)
eeg_bpnn.save("EEG" + str(seed) + ".m")
print("Training Finished!")

# predict_label = np.array(eeg_bpnn.test(test_data))
# outfile = open("EEG-test-" + str(seed), "w+")
# for i in range(len(test_label)):
#     print(test_label[i], '->', predict_label[i], file=outfile)


# sklearn.randomforest - 随机森林
# rfc = RandomForestClassifier(n_estimators=500, oob_score=True, criterion="gini", \
# max_features="log2")
# rfc.fit(train_data, train_label)
# print("Random Forest oob_score: " + str(rfc.oob_score_))
# predict_label = rfc.predict(test_data)
# matrix = confusion_matrix(test_label, predict_label)
# print(matrix)

# EEG-valence-arousal
# train_data, train_labels, test_data, test_labels = load_EEG_va_data()

# eeg_bpnn = BPNN(160, 20, 1, random_seed=seed)
# eeg_bpnn.accumulate_train(samples=train_data, labels=train_labels, rate=1, epochs=200)
# eeg_bpnn.save("EEG" + str(seed) + ".m")
# print("Training Finished!")

# predict_labels = np.array(eeg_bpnn.test(test_data))
# outfile = open("EEG-test-" + str(seed), "w+")
# for i in range(len(test_labels)):
#     print(test_labels[i], '->', predict_labels[i], file=outfile)

# bpnn-ref
# train_samples = ref_std(train_data, train_labels)
# test_samples = ref_std(test_data, test_labels)

# eeg_bpnn = BPNNet(160, 20, 1)
# eeg_bpnn.train(train_samples)
# eeg_bpnn.test(test_samples)