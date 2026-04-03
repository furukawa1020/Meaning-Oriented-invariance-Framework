import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from moif.invariance.mahalanobis import calculate_mahalanobis_distance

def main():
    print("Loading WESAD raw physiological data for Phase 2 Evaluation...")
    # Load raw feature data (100Hz)
    raw_path = 'results/wesad_100hz_instantaneous_raw.csv'
    
    # We will only load S11 to quickly verify the theory on a known subject
    df_iter = pd.read_csv(raw_path, chunksize=100000)
    df_s11 = pd.concat([chunk[(chunk['subject_id'] == 'S11') & (chunk['label'].isin(['baseline', 'stress']))] for chunk in df_iter])
    
    features = ['ECG_LF', 'ECG_HF', 'EDA_Phasic', 'EDA_Tonic']
    
    print(f"Data loaded for S11. Filtering valid features... (Total rows: {len(df_s11)})")
    
    # 1. マハラノビス距離を用いたベースライン・アンカリング
    print("Step 1: Calculating Mahalanobis distance from baseline distribution...")
    df_s11['D_M'] = calculate_mahalanobis_distance(df_s11, features, baseline_label='baseline')
    
    # NaN drop (in case of division by zero or invalid transforms)
    df_s11 = df_s11.dropna(subset=['D_M'] + features)
    
    # 2. GMMによる自動状態クラスタリング（ラベルに頼らない客観的状態の定義）
    print("Step 2: Fitting Gaussian Mixture Model (GMM) on anchored features...")
    # D_M (全体の乖離度エネルギー) と EDA_Phasic (突発的な活性度) の2次元空間で状態を抽象化
    # 生理的クラスター（状態）を3つ（S0: 安静状態, S1: 中度活性, S2: 高度活性）に自己組織化させる
    cluster_features = ['D_M', 'EDA_Phasic']
    
    # 高速化のためにサンプリングして学習
    train_df = df_s11.sample(n=min(20000, len(df_s11)), random_state=42)
    
    gmm = GaussianMixture(n_components=3, random_state=42, covariance_type='tied')
    gmm.fit(train_df[cluster_features])
    
    # 推論：全データにどの生理状態S（0~2）にいるかをアサイン
    df_s11['State_S'] = gmm.predict(df_s11[cluster_features])
    
    # 3. 状態Sにおける「意味づけの分岐（Invariance Breaking）」の集計
    print("Step 3: Calculating conditional probability of subjective interpretation (P(Y|S))...")
    results = []
    
    for k in range(3):
        # 状態 k (完全に同じ生理活動空間) にいるデータ群
        state_df = df_s11[df_s11['State_S'] == k]
        if len(state_df) == 0: continue
        
        # 完全に同じ体反応なのに「ストレス」とラベリングしている割合
        p_stress = len(state_df[state_df['label'] == 'stress']) / len(state_df)
        p_baseline = 1.0 - p_stress
        
        # クラスタの重心をラベル表示用に取得
        centroid_dm = state_df['D_M'].mean()
        
        results.append({
            'State': f"S{k} (DM_avg={centroid_dm:.1f})", 
            'P_Stress': p_stress, 
            'P_Baseline': p_baseline, 
            'Count': len(state_df)
        })

    import pprint
    pprint.pprint(results)

    print("Step 4: Generating Visualization...")
    # グラフの作成
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 左：GMMによる客観的な身体状態の分布（AIはラベルを知らない）
    sns.scatterplot(x='D_M', y='EDA_Phasic', hue='State_S', data=train_df, palette='viridis', alpha=0.3, ax=ax1)
    ax1.set_title("AI Discovered Physiological States (GMM: $S \\in \\{0,1,2\\}$)", fontsize=13)
    ax1.set_xlabel("Mahalanobis Distance from True Baseline ($D_M$)")
    ax1.set_ylabel("EDA Phasic Component (Instantaneous)")

    # 右：各同一状態 (S) の中での「主観の意味づけ (Y)」の割合
    states = [r['State'] for r in results]
    p_stress_vals = [r['P_Stress'] * 100 for r in results]
    p_base_vals = [r['P_Baseline'] * 100 for r in results]

    ax2.bar(states, p_base_vals, label='Reported as "Baseline / Calm"', color='blue', alpha=0.7)
    ax2.bar(states, p_stress_vals, bottom=p_base_vals, label='Reported as "Stress"', color='red', alpha=0.7)
    ax2.set_ylabel("Subjective Interpretation Probability P(Y|S) (%)", fontsize=11)
    ax2.set_xlabel("Physiological State S", fontsize=11)
    ax2.set_title("Meaning Divergence: Identical State $\\to$ Different Feeling", fontsize=13)
    
    # 50%ラインの描画（コイン投げと同じくらい意味が分かれる＝状態から感情が判定不可能）
    ax2.axhline(50, color='black', linestyle='--', alpha=0.5)
    ax2.text(0, 52, "50% (Max Ambiguity)", color='black')
    
    ax2.legend()
    
    plt.suptitle("Phase 2 (WESAD S11): Mahalanobis Anchoring & Physiological State Clustering", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    out_path = 'results/phase2_gmm_divergence_S11.png'
    plt.savefig(out_path, dpi=300)
    print(f"Algorithm finished. Generated proof plot at: {out_path}")

if __name__ == "__main__":
    main()
