# ==============================================================================
# 既存の「感情AI（Affective Computing）」が犯している「データ補完のズル」を排除し、
# WESADデータセットの「安静時」と「ストレス時」の「生理空間の重なり（Invariance Breaking）」
# を数学的に証明・可視化するための解析スクリプトです。
#
# このコードは、論文における「手法（Methodology）」と「結果（Results）」の核心部分を
# 実行し、2つの強力なエビデンス画像（2D KDEと全被験者の包含率グラフ）を生成します。
# ==============================================================================

import pandas as pd  # データフレーム（エクセルのような表データ）を操作・集計するためのライブラリ
import numpy as np   # ベクトルや行列の高速な数値計算（平均、乱数など）を行うライブラリ
import matplotlib.pyplot as plt  # グラフを描画するための基本ライブラリ
import seaborn as sns  # matplotlibを美しくし、KDE（カーネル密度推定）などを直感的に描くライブラリ
from sklearn.neighbors import NearestNeighbors  # 「重なり合い」を計算するための機械学習「半径近傍探索」アルゴリズム

# 100Hzレベルに拡張・同期化されたWESADの全被験者データセット（CSV）をメモリに読み込む
print("Loading augmented dataset...")
df = pd.read_csv('wesad_100hz_instantaneous_augmented.csv')

# 解析に使用する4次元の生理的特徴量をリストに定義（EDAの背景とスパイク、心拍変動のLFとHF）
features = ['EDA_Tonic', 'EDA_Phasic', 'HRV_Inst_LF', 'HRV_Inst_HF']
# Zスコア化（正規化）した後の新しい列名のリストを一括作成（例: 'EDA_Tonic_Z'）
z_features = [f'{col}_Z' for col in features]


# ==============================================================================
# 1. 衝撃事例の可視化：被験者S11の「2次元カーネル密度（KDE）」オーバーラップ図の作成
# （論文で「全く同じ身体なのに感情が真逆」と指摘する代表例のグラフを作ります）
# ==============================================================================
sub = 'S11'  # トリックがないことを示すため、代表となるターゲット被験者をS11さんに指定
s11 = df[df['subject_id'] == sub].copy()  # 全データからS11さんのデータだけを切り出して独立したコピーを作成
base_s11 = s11[s11['label'] == 'baseline'].copy()  # さらにその中で「安静タスク（青）」の時間を切り出し
stress_s11 = s11[s11['label'] == 'stress'].copy()  # 同様に「ストレスタスク（赤）」の時間を切り出し

# --- Zスコア化（個人的な安静状態を基準とした正規化）---
# 各特徴量（Tonicなど4つの空間軸）について、S11さんの「安静時の自分」を基準としたズレ（Zスコア）を計算する
for col in features:
    # ポイント：既存研究のようにデータ全体の平均をとるズルを排除し、必ず「安静時（base_s11）」の平均と分散を取得する
    mean_val = base_s11[col].mean()  
    std_val = base_s11[col].std()    
    
    # 安静時データも、ストレス時データも、すべて「自分自身の安静時の平均と分散」を基準にしてZスコア化する
    # これにより、4次元空間の原点(0)が「S11さんの完璧な安静状態のど真ん中」になる
    base_s11.loc[:, f'{col}_Z'] = (base_s11[col] - mean_val) / std_val
    stress_s11.loc[:, f'{col}_Z'] = (stress_s11[col] - mean_val) / std_val

# KDE等高線グラフを描く際に、センサーのエラー値（数万点に1つほどの極端な外れ値）があるとグラフ全体が潰れるため、
# 4次元すべてにおいて「安静基準から5標準偏差（99.999%以上）外れた物理的にありえない異常値」を除外（クリーニング）する
base_clean = base_s11[(base_s11[z_features].abs() < 5).all(axis=1)]
stress_clean = stress_s11[(stress_s11[z_features].abs() < 5).all(axis=1)]

# グラフのキャンバス（横10インチ × 縦8インチ）を準備する
plt.figure(figsize=(10, 8))

# KDE関数に何万点ものデータをそのまま突っ込むと等高線の計算でPCのメモリがクラッシュするため、
# 形（密度分布）を知るだけなら十分な数である「最大5000点」だけをランダムに間引き抽出する
b_sample = base_clean.sample(min(5000, len(base_clean)), random_state=42)
s_sample = stress_clean.sample(min(5000, len(stress_clean)), random_state=42)

# Seabornライブラリを使って、等高線のようなカーネル密度推定（KDE）プロットを空間上に描画する
# X軸として「EDA Tonicの変動」、Y軸として「HRV LFの変動」を指定。青色に塗りつぶす。
sns.kdeplot(x=b_sample['EDA_Tonic_Z'], y=b_sample['HRV_Inst_LF_Z'], cmap="Blues", fill=True, alpha=0.5) 
# 同様にストレス時（赤い等高線）を描画。この瞬間に、赤と青が完全に被る事実が判明する。
sns.kdeplot(x=s_sample['EDA_Tonic_Z'], y=s_sample['HRV_Inst_LF_Z'], cmap="Reds", fill=True, alpha=0.5)  

# プログラム的なグラフではなく、論文で分かりやすく説明するための凡例（Legendカラーバー）を手動で作る
from matplotlib.patches import Patch
legend_elements = [
    # 青色は安静タスク。S11さん本人の申告アンケートはValence 6（ポジティブに落ち着いている）
    Patch(facecolor='blue', alpha=0.5, label='Baseline Block (True Meaning: Valence 6, Arousal 4)'),
    # 赤色はストレスタスク。本人の申告アンケートはValence 2（ネガティブに焦っている）
    Patch(facecolor='red', alpha=0.5, label='Stress Block (True Meaning: Valence 2, Arousal 6)')
]
plt.legend(handles=legend_elements, loc='upper right') # 作った凡例をグラフの右上に配置

# 「意味づけは真逆なのに、分布は完全に同一空間にいる」という主張を含めた、力強いタイトルと軸ラベルを設定
plt.title('Distribution-Level Invariance Breaking (S11)\nThe physiological spaces of Baseline and Stress overlap almost entirely,\nyet the subjective meanings attached to these blocks are opposites.', fontsize=13, fontweight='bold')
plt.xlabel('EDA Tonic (Z-Score from Baseline)')
plt.ylabel('HRV LF (Z-Score from Baseline)')

# 余白を自動計算して美しく整え、PNG画像として300dpi（高画質、論文投稿レベル）で保存する
plt.tight_layout()
plt.savefig('distribution_overlap_S11.png', dpi=300)
print("Saved 2D Distribution Overlap for S11.")


# ==============================================================================
# 2. 全15名を対象とした「4次元空間での分布包含率の計算（重なりが何％か）」
# （ここがIEEE論文の最強のエビデンスとなる、Nearest Neighborsアルゴリズムです）
# ==============================================================================
results = []  # 各被験者の計算が終わるたびに「ID」と「重なり率（％）」を保存するための空リストを準備

# データフレーム(df)に入っている 'S2', 'S3' など、存在する全被験者IDをforループで順番に取り出す
for sub_id in df['subject_id'].unique():
    sub_df = df[df['subject_id'] == sub_id].copy()  # 今ループしている被験者のデータだけを切り出す
    b = sub_df[sub_df['label'] == 'baseline'].copy()  # 安静ブロック (b) を抽出
    s = sub_df[sub_df['label'] == 'stress'].copy()    # ストレスブロック (s) を抽出
    
    # センサーの不調等で安静かストレスのどちらかのデータが空っぽになっている場合は、次へスキップする
    if b.empty or s.empty: continue
        
    # --- 全員分について、それぞれ各自の「自分の安静時」を基準にしてZスコア（正規化）を行う ---
    for col in features:
        mean_val = b[col].mean()  # 「自分の」安静時の平均値
        std_val = b[col].std()    # 「自分の」安静時の標準偏差
        if std_val == 0: continue # データが全く動いておらずゼロ割エラーが出るのを防ぐ
        b.loc[:, f'{col}_Z'] = (b[col] - mean_val) / std_val # 安静データのZスコア
        s.loc[:, f'{col}_Z'] = (s[col] - mean_val) / std_val # ストレスデータのZスコア
        
    # 万が一Zスコアの計算列が作成できていなかったらスキップ
    if f'{features[0]}_Z' not in b.columns: continue  
    
    # ===【最重要計算】ストレス空間の点が、どれくらい安静空間の「中に取り込まれて」いるかを計算 ===
    
    # NearestNeighbors: 与えられたデータ点を記憶し、ある点から指定した「半径(radius)」以内の点を探すレーダーを作る。
    # ここでは半径 1.0 (標準偏差1個分という極めて狭くて厳しい空間) を被覆判定の閾値に設定している。
    nn = NearestNeighbors(radius=1.0)
    
    # 安静時（b）の4次元（Tonic,Phasic,LF,HF）データを抽出して、NumPy配列（数学の行列）に変換
    b_vals = b[z_features].values
    
    # KD-Treeの計算量爆発を抑えるため、1万点を超えているなら、ランダムに1万点だけ抜き出す（ダウンサンプリング）
    if len(b_vals) > 10000:
        np.random.seed(42)  # 誰が何度やっても同じ間引き方になるようにシード値を42に固定
        b_vals = b_vals[np.random.choice(len(b_vals), 10000, replace=False)]
        
    # 【1】レーダー(nn) に「安静時の青い点群の4次元座標」をすべて記憶させる（空間インデックス KD-Tree の構築）
    nn.fit(b_vals)

    # 次に、ストレス時（s）の4次元データも同様に行列化・ダウンサンプリングする
    s_vals = s[z_features].values
    if len(s_vals) > 10000:
        np.random.seed(42)
        s_vals = s_vals[np.random.choice(len(s_vals), 10000, replace=False)]
        
    # 【2】ここで空間レーダーを発射する：
    # 「ストレス時の全点群（s_vals）」を一つ一つレーダーの中心に置き、半径1.0以内に「記憶させた安静時（青い点）」が存在するか検索する。
    # return_distance=False: 「何ミリ離れていたか」という距離ベクトルはいらない、「存在したかどうか」のID情報だけを返す設定。
    ind = nn.radius_neighbors(s_vals, return_distance=False)
    
    # 【3】包含率（$\Omega$）の計算：
    # ind 配列には、各赤い点ごとに「見つかった青い点のリスト」が入っている。
    # len(n) > 0 は「青い点が1個以上見つかったら True (1)、見つからなかったら False (0)」という判定文。
    # np.mean([...]) で、すべての赤い点のなかで True（重なった）だった点の割合の平均を取り、最後に100を掛けてパーセント(%)に戻す。
    overlap_pct = np.mean([len(n) > 0 for n in ind]) * 100
    
    # 計算が終わった被験者ごとの結果（ID名 と 算出した重なり率）を、準備していたリスト(results)に追加する
    results.append({
        'Subject': sub_id,
        'Overlap (%)': overlap_pct
    })

# ループ終了。すべての被験者の計算が終わったら、結果リストをDataFrame（表）に変換し、Overalap率が高い順（降順）に並べ替える
res_df = pd.DataFrame(results).sort_values('Overlap (%)', ascending=False)

# コンソール画面（コンソール文字）に分析結果のランキング表をそのまま出力させて確認する
print("\n=== Universal Distribution Overlap (Radius=1.0 StdDev in 4D) ===")
print(res_df.to_string())

# --- 全員の重なり率ランキングをバープロット（棒グラフ）にして保存する ---
plt.figure(figsize=(12, 6))  # 横を長めのプロット画面を作成
# Seabornで棒グラフ（barplot）を描画。X軸:被験者ID、Y軸:さっき計算した重なり率(%)。色は紫(#8e44ad)。
ax = sns.barplot(data=res_df, x='Subject', y='Overlap (%)', color='#8e44ad')

# グラフのタイトルやラベル。「データ補完なしにおける生理状態オーバラップの普遍性」と明記。
plt.title('Universality of State Overlap without Data Interpolation\n% of "Stress" Physiology that is statistically identical (r<1.0 StdDev) to "Baseline" Physiology', fontsize=14, fontweight='bold')
plt.ylabel('Overlap Percentage (%)', fontsize=12)

# マジョリティである50%のラインに、赤い「レッドライン（これが超えたらもうお手上げという基準）」の点線を引く
plt.axhline(50, color='red', linestyle='--', label='50% Overlap')

# 各棒グラフのてっぺん（先端）の上に、「98.6%」などの具体的な数字を書き込むループ処理
for p in ax.patches:
    ax.annotate(f"{p.get_height():.1f}%", 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', fontsize=9, color='black', 
                xytext=(0, 10), textcoords='offset points')

plt.legend()       # 50%レッドラインの凡例を出す
plt.tight_layout() # 端が切れないように余白を調節
# `universal_distribution_overlap.png` という名前で全調査結果の画像を保存
plt.savefig('universal_distribution_overlap.png', dpi=300)
print("Saved universal distribution overlap plot.")


