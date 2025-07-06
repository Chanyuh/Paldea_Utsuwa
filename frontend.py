import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neural_network import MLPClassifier
from sklearn import metrics 
import joblib
import jaconv

LOAD_POKEMON_DATA_PATH = "data/Pokemon_list.csv"
SAVE_TRAINED_DATA_PATH = "data/train.learn"

#ポケモン検索データ
pokemon_data = pd.read_csv(LOAD_POKEMON_DATA_PATH, sep=",")

pokemon_list = []
pokemon_dict = {}

for i in range(pokemon_data.shape[0]):
    pokemon_list.append(pokemon_data.loc[i,"Name"])
    pokemon_dict[pokemon_data.loc[i,"Name"]]=i+1

for i in range(pokemon_data.shape[0]):
    pokemon_list.append(jaconv.kata2hira(pokemon_data.loc[i,"Name"]))
    pokemon_dict[jaconv.kata2hira(pokemon_data.loc[i,"Name"])]=-i-1


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



def main():
    
    #ポケモン検索機能
    selected = st.selectbox("ポケモン名", pokemon_list, index = 522)

    id = pokemon_dict[selected]
    if id<0: id*=-1

    search = pokemon_data.loc[id-1]
        
    default_H = search["H"]
    default_A = search["A"]
    default_B = search["B"]
    default_C = search["C"]
    default_D = search["D"]
    default_S = search["S"]

    #HABCDS
    colH1, colH2 = st.columns([1,5])
    colH1.write("##### HP")
    with colH2:
        H: int=st.slider("HP",1, 255, default_H, 1, label_visibility="collapsed")

    colA1, colA2 = st.columns([1,5])
    colA1.write("###### 攻撃")
    with colA2:
        A: int=st.slider("攻撃",1, 255, default_A, 1, label_visibility="collapsed")

    colB1, colB2 = st.columns([1,5])
    colB1.write("###### 防御")
    with colB2:
        B: int=st.slider("防御",1, 255, default_B, 1, label_visibility="collapsed")

    colC1, colC2 = st.columns([1,5])
    colC1.write("###### 特攻")
    with colC2:
        C: int=st.slider("特攻",1, 255, default_C, 1, label_visibility="collapsed")
    
    colD1, colD2 = st.columns([1,5])
    colD1.write("###### 特防")
    with colD2:
        D: int=st.slider("特防",1, 255, default_D, 1, label_visibility="collapsed")
    
    colS1, colS2 = st.columns([1,5])
    colS1.write("###### 素早さ")
    with colS2:
        S: int=st.slider("素早さ",1, 255, default_S, 1, label_visibility="collapsed")

 
    st.write(f"##### 合計値: {H+A+B+C+D+S}")
   
   
    clf2 = joblib.load(SAVE_TRAINED_DATA_PATH)
    check = np.array([change([H,A,B,C,D,S])])
    predict = clf2.predict(check)

    st.write(" ")

    container = st.container(border=True)
    with container:
        if(predict[0]):
            st.write("##### このポケモンはパルデアの器です")
        else:
            st.write("##### このポケモンはパルデアの器ではありません")
        proba = 100 * clf2.predict_proba(check)[0][1]
        st.write(f"#### 予測確率: {round(proba,2)}%")
        
if __name__ == '__main__':
    main()