import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv

def calculate_mahalanobis_distance(df: pd.DataFrame, features: list[str], baseline_label: str = 'baseline') -> pd.Series:
    """
    Calculate the Mahalanobis distance for every point in the dataframe
    relative to the 'baseline' distribution.
    
    Args:
        df: DataFrame containing the physiological data and labels.
        features: List of column names representing the physiological feature space.
        baseline_label: The string label used to identify the baseline condition.
        
    Returns:
        A Pandas Series containing the Mahalanobis distance for each row.
    """
    # 1. 抽出: ベースライン（真の安静時）データのみを取り出す
    baseline_data = df[df['label'] == baseline_label][features]
    
    if len(baseline_data) < len(features) + 1:
        raise ValueError("Not enough baseline data to compute covariance matrix.")
        
    # 2. 計算: ベースライン空間の重心（平均ベクトル）と共分散行列を求める
    mu_base = baseline_data.mean().values
    cov_base = baseline_data.cov().values
    
    try:
        # 共分散行列の逆行列を求める（マハラノビス距離に必須）
        inv_cov_base = inv(cov_base)
    except np.linalg.LinAlgError:
        # 特異行列（特徴量が完全に相関している等）の場合のフォールバック
        # 微小なノイズ(Ridge正則化)を加えて逆行列計算を安定化させる
        cov_base += np.eye(len(features)) * 1e-6
        inv_cov_base = inv(cov_base)

    # 3. 変換: 全データに対して、ベースライン重心からのマハラノビス距離を計算
    # 巨大なデータフレームを効率的に処理するため、ベクトル演算（内積）を使用
    x_minus_mu = df[features].values - mu_base
    
    # マハラノビス距離の2乗: (x - μ)^T Σ^{-1} (x - μ)
    # np.einsum を使って行ごとに効率的に二次形式を計算
    m_dist_sq = np.einsum('ij,jk,ik->i', x_minus_mu, inv_cov_base, x_minus_mu)
    
    # 平方根をとって距離とする
    m_dist = np.sqrt(m_dist_sq)
    
    return pd.Series(m_dist, index=df.index, name='mahalanobis_dist')
