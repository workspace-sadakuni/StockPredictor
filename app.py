import jaconv
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import tensorflow as tf

# --- 再現性確保 ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

st.title("日本株 LSTM 株価予測ツール")

# --- 銘柄マスター（例：主要企業） ---
japan_stocks = {
    "トヨタ自動車": "7203.T", "ソニーグループ": "6758.T", "ソフトバンクグループ": "9984.T",
    "キーエンス": "6861.T", "東京エレクトロン": "8035.T", "任天堂": "7974.T",
    "KDDI": "9433.T", "リクルート": "6098.T", "NTT（日本電信電話）": "9432.T",
    "三菱UFJフィナンシャルG": "8306.T", "三井住友FG": "8316.T", "三菱商事": "8058.T",
    "三井物産": "8031.T", "伊藤忠商事": "8001.T", "村田製作所": "6981.T",
    "デンソー": "6902.T", "ホンダ": "7267.T", "日産自動車": "7201.T",
    "パナソニックHD": "6752.T", "TDK": "6762.T", "富士通": "6702.T", "オムロン": "6645.T",
    "アドバンテスト": "6857.T", "HOYA": "7741.T", "花王": "4452.T", "資生堂": "4911.T",
    "味の素": "2802.T", "キリンHD": "2503.T", "アサヒGHD": "2502.T", "JT（日本たばこ）": "2914.T",
    "ユニ・チャーム": "8113.T", "第一三共": "4568.T", "エーザイ": "4523.T",
    "武田薬品工業": "4502.T", "中外製薬": "4519.T", "塩野義製薬": "4507.T",
    "ENEOSホールディングス": "5020.T", "出光興産": "5019.T", "東京ガス": "9531.T",
    "関西電力": "9503.T", "ファーストリテイリング": "9983.T", "Zホールディングス": "4689.T"
}

st.subheader("銘柄の検索")
company_input = st.text_input("証券コードから検索（例：7203.T（トヨタ）、7974.T（任天堂） など）", "")

def normalize(text):
    return jaconv.hira2kata(jaconv.z2h(text.lower(), kana=True, digit=True, ascii=True))

normalized_input = normalize(company_input)

# 正規化した社名辞書
normalized_dict = {normalize(k): (k, v) for k, v in japan_stocks.items()}

# 曖昧一致で候補抽出
matched = {orig_name: code for norm_name, (orig_name, code) in normalized_dict.items() if normalized_input in norm_name}

selected_code = None

if normalized_input:
    selected_code = normalized_input
elif matched:
    selected_name = st.selectbox("候補から選択：", list(matched.keys()))
    selected_code = matched[selected_name]
    st.success(f"選択された証券コード：{selected_code}")
else:
    selected_code = "7203.T"  # トヨタでデフォルト
    st.info("デフォルトのトヨタで検索します。")

TICKER = selected_code

# --- 設定 ---
START_DATE = "2015-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
LOOKBACK = 60

@st.cache_data
def load_stock_data(ticker):
    df = yf.download(ticker, start=START_DATE, end=END_DATE)[["Close"]]
    df.rename(columns={"Close": "Stock_Close"}, inplace=True)
    return df

@st.cache_data
def load_volume_data(ticker):
    df = yf.download(ticker, start=START_DATE, end=END_DATE)[["Volume"]]
    return df

@st.cache_data
def load_fx_data():
    fx = yf.download("JPY=X", start=START_DATE, end=END_DATE)[["Close"]]
    fx.rename(columns={"Close": "USDJPY"}, inplace=True)
    return fx

# --- データ取得 ---
st.subheader("1. 株価・為替データの取得（終値）")
stock_df = load_stock_data(TICKER)
volume_df = load_volume_data(TICKER)
fx_df = load_fx_data()
df = stock_df.join(fx_df, how='inner')
df['Volume'] = volume_df["Volume"]

# --- テクニカル指標の追加 ---
df['SMA_20'] = df['Stock_Close'].rolling(window=20).mean()
delta = df['Stock_Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
df['RSI_14'] = 100 - (100 / (1 + rs))
ema12 = df['Stock_Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Stock_Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema12 - ema26
df.dropna(inplace=True)

st.dataframe(df.tail())

# --- 企業情報表示 ---
st.subheader("企業情報と株価チャート")
try:
    info = yf.Ticker(TICKER).info
    st.markdown(f"**企業名**: {info.get('longName', 'N/A')}")
    st.markdown(f"**業種**: {info.get('sector', 'N/A')}")
    st.markdown(f"**所在地**: {info.get('city', '')}, {info.get('country', '')}")
    st.markdown(f"**従業員数**: {info.get('fullTimeEmployees', 'N/A'):,}人")
    st.markdown(f"**概要**: {info.get('longBusinessSummary', 'N/A')}")

    st.markdown("**過去1年の株価チャート（終値）**")
    # チャートデータ取得
    chart_data = yf.download(TICKER, period="1y", group_by='ticker')
    # st.write("【DEBUG】chart_data:", chart_data.head())

    # 列名が複数ティッカー構造で返るので修正（例：Close["7203.T"] → Close）
    if isinstance(chart_data.columns, pd.MultiIndex):
        try:
            chart_data.columns = chart_data.columns.get_level_values(1)  # '7203.T' の部分だけにする
            chart_data = chart_data.rename(columns={TICKER: 'Close'})
        except:
            st.warning("列名の変換に失敗しました。")
    else:
        chart_data = chart_data.rename(columns={"Close": "Close"})  # 念のため

    # 描画
    if "Close" in chart_data.columns:
        st.line_chart(chart_data["Close"])
    else:
        st.warning("終値データが取得できませんでした。")
except Exception as e:
    st.warning(f"企業情報の取得中にエラーが発生しました: {e}")

# --- 予測期間の選択 ---
st.subheader("2 予測期間の選択")
forecast_options = {
    "翌営業日（1日後）": 1,
    "1週間後（7日後）": 7,
    "1ヶ月後（30日後）": 30,
    "1年後（252営業日後）": 252,
    "3年後（756営業日後）": 756,
    "5年後（1260営業日後）": 1260
}
selected_label = st.selectbox("予測したい期間を選んでください：", list(forecast_options.keys()))
target_offset = forecast_options[selected_label]
st.write(f"{target_offset} 日後の株価を予測します。")

# --- データ前処理 ---
# st.subheader("3. データ前処理")
features = ['Stock_Close', 'USDJPY', 'Volume', 'SMA_20', 'RSI_14', 'MACD']
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features])

X, y = [], []
for i in range(LOOKBACK, len(scaled_data) - target_offset):
    X.append(scaled_data[i - LOOKBACK:i])
    y.append(scaled_data[i + target_offset][0])

X, y = np.array(X), np.array(y)

# --- LSTM モデル構築・学習 ---
st.subheader("3. LSTM モデル構築と学習（終値予測）")
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
with st.spinner("学習中...（数十秒かかります）"):
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
st.success("学習完了")

# --- 予測 ---
st.subheader("4. 株価予測結果")
last_60_days = scaled_data[-LOOKBACK:]
input_data = np.expand_dims(last_60_days, axis=0)
pred_scaled = model.predict(input_data)[0][0]

# 株価予測値を全体のスケールで逆変換
last_features = scaled_data[-1].copy()
last_features[0] = pred_scaled
pred_price = scaler.inverse_transform([last_features])[0][0]

st.metric(f"予測株価（{target_offset}日後）", f"{pred_price:.2f} 円")

matplotlib.rcParams['font.family'] = 'MS Gothic'

# --- グラフ表示 ---
st.subheader("5. 株価チャート")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df.index, df['Stock_Close'], label="実株価")
ax.axvline(df.index[-1] + timedelta(days=target_offset), color='r', linestyle='--', label='予測日')
ax.scatter(df.index[-1] + timedelta(days=target_offset), pred_price, color='red', label="予測値")
ax.legend()
st.pyplot(fig)

st.caption("※ この予測は学習データに基づいた参考値です。投資判断はご自身でお願いします。")

# --- 予測誤差の可視化 ---
st.subheader("6. 検証データに対する予測精度")

# 学習・検証分割
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 検証データの予測
y_pred_scaled = model.predict(X_test)
y_pred = []
for i in range(len(y_pred_scaled)):
    last = X_test[i, -1, :].copy()
    last[0] = y_pred_scaled[i][0]
    inv = scaler.inverse_transform([last])[0][0]
    y_pred.append(inv)

# 実測もスケール逆変換
y_true = []
for i in range(len(y_test)):
    last = X_test[i, -1, :].copy()
    last[0] = y_test[i]
    inv = scaler.inverse_transform([last])[0][0]
    y_true.append(inv)

# X_test の直近日時を元に、検証開始以降の日付を算出
test_start_idx = LOOKBACK + split_index
test_dates = df.index[test_start_idx + target_offset:]

# グラフ修正（横軸を日付に）
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(test_dates[:len(y_true)], y_true, label="実測株価", color='blue')
ax2.plot(test_dates[:len(y_pred)], y_pred, label="予測株価", color='orange')
ax2.set_title("予測 vs 実測（検証データ）")
ax2.set_xlabel("日付")
ax2.set_ylabel("株価（円）")
ax2.legend()
ax2.grid(True)
fig2.autofmt_xdate()
st.pyplot(fig2)


