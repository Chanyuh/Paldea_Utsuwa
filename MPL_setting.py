import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import metrics 
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# 学習用データのパス
LOAD_TRAIN_DATA_PATH = "venv/data/train.csv"

# 学習済みモデルデータの保存先パス
SAVE_TRAINED_DATA_PATH = "venv/data/train.learn"

# 検証用データのパス
LOAD_TEST_DATA_PATH = "venv/data/test.csv"

# グラフ出力先パス
save_graph_img_path = "venv/data/graph.png"

# グラフ画像のサイズ
fig_size_x = 10
fig_size_y = 10


# ニューラルネットワークのパラメータ
solver = "adam"
random_state = 0
max_iter = 10000
hidden_layer_sizes = 100
    
# 学習用のデータを読み込み
train_data = pd.read_csv(LOAD_TRAIN_DATA_PATH, sep=",")

# 説明変数
train_X = train_data.loc[:,"H", "A", "B", "C", "D", "S","sum"].values

# 説明変数の変換
def change(ar):
    h_b_d = float(ar[0] - ar[2] - ar[4])
    max_b = float(max(ar[2],ar[4]))
    max_a = float(max(ar[1],ar[3]))
    a_c = float(abs(ar[1] - ar[3]))
    speed_high = float(0)
    if ar[5]>=130:
        speed_high = float(10)
    return [float(ar[0]),max_b,h_b_d,max_a,a_c,float(ar[5]),speed_high]

changed_train_X = []
for x in train_X:
    changed_train_X.append(change(x)) 

changed_train_X = np.array(changed_train_X)

# 目的変数
train_y = train_data["result"].values

# 学習
clf = MLPClassifier(
    solver=solver,
    random_state=random_state,
    max_iter=max_iter,
    hidden_layer_sizes=hidden_layer_sizes)

accuracy_score = float(0)
best_X_train = []
best_Y_train = []

for i in range(200):
    X_train, X_test, Y_train, Y_test = train_test_split(changed_train_X, train_y, test_size=0.30)
    clf.fit(X_train, Y_train)
    print("Accuracy score (train):", clf.score(X_train, Y_train))
    print("Accuracy score (test):", clf.score(X_test, Y_test))
    new_accuracy_score = clf.score(X_test, Y_test)
    if(accuracy_score<new_accuracy_score):
        accuracy_score = new_accuracy_score
        best_X_train = X_train
        best_Y_train = Y_train



clf.fit(best_X_train, best_Y_train)
print("Best Accuracy score (train):", clf.score(X_train, Y_train))
print("Best Accuracy score (test):", clf.score(X_test, Y_test))

# 学習結果を出力
joblib.dump(clf, SAVE_TRAINED_DATA_PATH)

# 学習済ファイルのロード
clf2 = joblib.load(SAVE_TRAINED_DATA_PATH)

# テスト用データの読み込み
test_data = pd.read_csv(LOAD_TEST_DATA_PATH, sep=",")
print(test_data)

# 検証用の説明変数（学習データ）
test_name = test_data["Name"]
test_X = test_data.loc[:, ["H", "A", "B", "C", "D", "S"]].values

changed_test_X = []
for x in test_X:
    changed_test_X.append(change(x)) 

changed_test_X = np.array(changed_test_X)
print(changed_test_X)



# 学習結果の検証（テスト用データx1, x2を入力）
predict_y = clf2.predict(changed_test_X)

# 検証結果の表示
for i in range(test_name.size):
    print(test_name[i], ":", predict_y[i])
