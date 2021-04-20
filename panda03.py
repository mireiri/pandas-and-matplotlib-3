# pandasとmatplotlibをimportする。
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

# Excelファイルの複数シートから1つのデータフレームを作成する。
iter_num = [i for i in range(0, 4)]

df = pd.DataFrame()

for i in iter_num:
    each_df = pd.read_excel('2016年12月.xlsx', index_col=0, sheet_name=i)
    df = pd.concat([df, each_df], axis=0)

# 作成したデータフレームの行列数を確認する。
df.shape

# （参考）
# ひとつひとつデータフレームを作成してから結合する場合

cts = pd.read_excel('2016年12月.xlsx', sheet_name='CTS')
itm = pd.read_excel('2016年12月.xlsx', sheet_name='ITM')
fuk = pd.read_excel('2016年12月.xlsx', sheet_name='FUK')
oka = pd.read_excel('2016年12月.xlsx', sheet_name='OKA')

df = pd.concat([cts, itm, fuk, oka], axis=0)
df.set_index('日付', inplace=True)

# 日付列を降順に並び替える。
df.sort_values('日付', inplace=True)

# 各列の値を計算した結果を、新しい列として追加する。
df.loc[:, '旅客数/運航回数'] = df.loc[:, '旅客数'] / df.loc[:, '運航回数']

# 小数点以下を丸める。
df.loc[:, '旅客数/運航回数'] = df.loc[:, '旅客数/運航回数'].round()

# 各空港の旅客数の分布を箱ひげ図で可視化する。
cts = df[df.loc[:, '目的地'] == 'CTS']
itm = df[df.loc[:, '目的地'] == 'ITM']
fuk = df[df.loc[:, '目的地'] == 'FUK']
oka = df[df.loc[:, '目的地'] == 'OKA']

fig, ax = plt.subplots()

label = ['CTS', 'ITM', 'FUK', 'OKA']

ax.boxplot((cts.loc[:, '旅客数/運航回数'], itm.loc[:, '旅客数/運航回数'],
            fuk.loc[:, '旅客数/運航回数'], oka.loc[:, '旅客数/運航回数']), 
            labels=label)
ax.set_title('各空港の1運航あたりの旅客数')

plt.show()

# 各空港の貨物重量の分布を箱ひげ図で可視化する。
df.loc[:, '貨物重量/運航回数'] = df.loc[:, '貨物重量'] / df.loc[:, '運航回数']
df.loc[:, '貨物重量/運航回数'] = df.loc[:, '貨物重量/運航回数'].round()

cts = df[df.loc[:, '目的地'] == 'CTS']
itm = df[df.loc[:, '目的地'] == 'ITM']
fuk = df[df.loc[:, '目的地'] == 'FUK']
oka = df[df.loc[:, '目的地'] == 'OKA']

fig, ax = plt.subplots()

label = ['CTS', 'ITM', 'FUK', 'OKA']

ax.boxplot((cts.loc[:, '貨物重量/運航回数'], itm.loc[:, '貨物重量/運航回数'],
            fuk.loc[:, '貨物重量/運航回数'], oka.loc[:, '貨物重量/運航回数']), 
            labels=label, patch_artist=True,
            boxprops=dict(facecolor='darkgray'),
            medianprops=dict(color='darkgreen'))
ax.set_title('各空港の1運航あたりの貨物重量')

plt.show()

# 各空港の要約統計量を確認する。
cts.describe()

itm.describe()

fuk.describe()

oka.describe()

# 旅客数のヒストグラムを描画する。
fig, ax = plt.subplots()

num, bins, plotdata =  ax.hist(cts.loc[:, '旅客数'], bins=30)
ax.set_title('CTS_旅客数分布')

plt.show()

for i, j in enumerate(num):
    print('{:.1f} {:.1f} : {}'.format(bins[i], bins[i+1], j))

# 週単位のグルーピング
df_week = df.groupby(pd.Grouper(freq='W-Sun')).sum()

cts_week = cts.groupby(pd.Grouper(freq='W-Sun')).sum()
itm_week = itm.groupby(pd.Grouper(freq='W-Sun')).sum()
fuk_week = fuk.groupby(pd.Grouper(freq='W-Sun')).sum()
oka_week = oka.groupby(pd.Grouper(freq='W-Sun')).sum()

# 各空港の折れ線グラフを描画する。
fig, ax = plt.subplots()

x = [i for i in range(0, 5)]

ax.plot(x, cts_week.loc[:, '旅客数/運航回数'], label='CTS')
ax.plot(x, itm_week.loc[:, '旅客数/運航回数'], label='ITM')
ax.plot(x, fuk_week.loc[:, '旅客数/運航回数'], label='FUK')
ax.plot(x, oka_week.loc[:, '旅客数/運航回数'], label='OKA')
ax.set_title('各空港の1運航あたりの旅客数')
ax.legend(loc='best')
ax.xaxis.set_visible(False)

plt.show()

# 各データの相関係数を確認する。
df.corr()

# 散布図を描画する。
fig, ax = plt.subplots()

ax.scatter(df.loc[:, '貨物重量'], df.loc[:, '旅客数'])

plt.show()


# 2行2列のエリアに散布図を描画する。
fig, ax = plt.subplots(figsize=(9, 6), nrows=2, ncols=2)

ax[0, 0].scatter(cts.loc[:, '貨物重量'], cts.loc[:, '旅客数'], marker='*')
ax[0, 0].set_title('CTS')
ax[0, 1].scatter(itm.loc[:, '貨物重量'], itm.loc[:, '旅客数'],
                 marker='s', color='orange')
ax[0, 1].set_title('ITM')
ax[1, 0].scatter(fuk.loc[:, '貨物重量'], fuk.loc[:, '旅客数'],
                 marker='^', color='green')
ax[1, 0].set_title('FUK')
ax[1, 1].scatter(oka.loc[:, '貨物重量'], oka.loc[:, '旅客数'],
                 marker='v', color='red')
ax[1, 1].set_title('OKA')

plt.tight_layout()
plt.show()

# numpyのcorrcoefで相関係数を確認する。
import numpy as np

np.corrcoef(cts.loc[:, '貨物重量'], cts.loc[:, '旅客数'])

# クラスタリングにチャレンジ！
from sklearn.cluster import KMeans

# クラスタ数を4つにしてインスタンスを作成する。
KM = KMeans(n_clusters=4, init='random', n_init=100, random_state=0)

X = df.iloc[:, 2:]

cluster = KM.fit_predict(X)

# クラスタリングした結果のデータデータフレームに追加する。
df.loc[:, 'cluster'] = cluster

# 各クラスタのデータを確認する。
df.query('cluster == 0').head(3)

df.query('cluster == 1').head(3)

df.query('cluster == 2').head(3)

df.query('cluster == 3').head(3)

# Excelファイルに変換する。
df.to_excel('cluster.xlsx')

# ここから回帰（単回帰）予測にチャレンジ！！
# 各データの相関係数を確認する。
cts.corr()

# 説明変数と目的変数にするデータの相関を散布図で確認する。
fig, ax = plt.subplots()

ax.scatter(cts.loc[:, '旅客数'], cts.loc[:, '貨物重量'])
ax.set_xlabel('説明変数（旅客数）')
ax.set_ylabel('目的変数（貨物重量）')

plt.show()

# train_test_split()関数で、実データを学習データとテストデータに分割する。
from sklearn.model_selection import train_test_split

X = cts[['旅客数']]
y = cts['貨物重量']

X_train, X_test, y_train, y_test = train_test_split(
                                   X, y, test_size=0.3, random_state=0)

# 説明変数と目的変数の形状を確認する。
print(X.shape, y.shape)

# 分割した各データの形状を確認する。
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# 回帰予測に使うLinearRegressionをインスタンス化する。
from sklearn.linear_model import LinearRegression

LR = LinearRegression()

# 学習データを与えて予測モデルを作成し、テストデータで予測させる。
model = LR.fit(X_train, y_train)

result = model.predict(X_test)

# テストデータと予測データの相関を確認する。
np.corrcoef(y_test, result)

# テストデータと予測データの散布図で確認する。
fig, ax = plt.subplots()

ax.scatter(y_test, result)
ax.set_xlabel('テストデータ')
ax.set_ylabel('予測データ')

plt.show()

# 散布図に予測結果の回帰直線を追加する。
# 回帰直線を追加
fig, ax = plt.subplots()

ax.scatter(X_test, y_test)
ax.set_xlabel('テスト用説明変数データ')
ax.set_ylabel('テスト用目的変数データ')

ax.plot(X_test, model.predict(X_test))

plt.show()
print(model.coef_, model.intercept_)

# テストデータと予測結果のデータフレームを作成し、その差異の列を追加する。
result_df = pd.DataFrame({'test_data': y_test, 'result_data': result})

result_df.loc[:, 'diff'] = result_df.loc[:, 'test_data'] - result_df.loc[:, 'result_data']

# 日付順に並び替える。
result_df.sort_values('日付')

# 決定係数を確認する。
from sklearn.metrics import r2_score

r2_score(y_test, result)

# ここからが、まだわからない目的変数のデータ（未知のデータ）を予測する手順です。
# 予測したいファイルをデータフレームにする。
# 予測モデルに説明変数を与えて予測結果を得る。
predict_df = pd.read_excel('predict_test.xlsx', index_col=0)

X = predict_df[['旅客数']]

predict_result = model.predict(X)

# 予測結果をデータフレームに追加する。
predict_df['予測貨物重量'] = predict_result

# データがない（NaN）列を削除する。
predict_df.dropna(axis=1, inplace=True)

# 小数点以下を丸める。
predict_df.loc[:, '予測貨物重量'] = predict_df.loc[:, '予測貨物重量'].round()

# 列を並び替えする。
predict_df = predict_df.reindex(columns=['曜日', '目的地', '運航回数',
                                         '座席数', '予測貨物重量', '旅客数'])

# Excelファイルに変換する。
predict_df.to_excel('predict_test_result.xlsx')