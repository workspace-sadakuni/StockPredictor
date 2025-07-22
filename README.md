# 株価予測ツール  
## 機能
- 証券コードまたはプルタウン選択による銘柄情報検索
- 検索した銘柄の将来の株価を予測
- 実測と予測の予測精度の可視化

## 予測データについて
- 株価データ  
以下指標を基に予測。  
  - 過去10年間の株価終値
  - 出来高：株価の勢い
  - RSI(14日)：過熱感/売られすぎの判断
  - SMA(20日)：トレンド把握
  - MACD：トレンド変換予測

## スクリーンショット
<img width="1920" height="3158" alt="Image" src="https://github.com/user-attachments/assets/54f18d80-1e40-4d63-908d-04dcb6d093a2" />

## 環境構築  
### ライブラリのインストール  
```
pip install jaconv streamlit yfinance pandas numpy scikit-learn tensorflow matplotlib requests
```
または、requirements.txtよりインストールする場合（プロトタイプなため、ライブラリのインストール方法は任意とする。）
```
pip install -r requirements.txt
```

### アプリの実行  
```
streamlit run app.py
```

## 技術スタック

| 項目 | バージョン |
|------|------------|
| Python | 3.9 |

## 備考  
個人利用を目的としたプロトタイプツールです。
予測結果は極端に乖離した数値を算出するなど、意図しない結果となることがあります。