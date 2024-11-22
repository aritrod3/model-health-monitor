"""
Drift detection module.

Implements statistical tests to detect feature distribution shifts between training and production data.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def detect_drift(
    reference_data: pd.DataFrame,
    production_data: pd.DataFrame,
    threshold: float = 0.05,
    method: str = "ks"
) -> Dict[str, float]:
    """
    Detect drift in feature distributions using statistical tests.
    
    Args:
        reference_data: Baseline/reference dataframe (training data)
        production_data: Production dataframe to test
        threshold: P-value threshold for drift detection
        method: Statistical test method ("ks", "mannwhitney", "chi2")
        
    Returns:
        Dictionary mapping feature names to drift p-values
    """
    drift_scores = {}
    
    # Ensure same columns
    common_columns = set(reference_data.columns) & set(production_data.columns)
    if len(common_columns) != len(reference_data.columns):
        logger.warning(f"Column mismatch: {len(reference_data.columns)} reference vs "
                      f"{len(production_data.columns)} production columns")
    
    for col in common_columns:
        ref_col = reference_data[col].dropna()
        prod_col = production_data[col].dropna()
        
        if len(ref_col) == 0 or len(prod_col) == 0:
            logger.warning(f"Empty data for column '{col}', skipping")
            continue
        
        # Determine if column is numeric or categorical
        is_numeric = pd.api.types.is_numeric_dtype(ref_col) and pd.api.types.is_numeric_dtype(prod_col)
        
        if is_numeric:
            # Use Kolmogorov-Smirnov test or Mann-Whitney U test
            if method == "mannwhitney":
                try:
                    statistic, p_value = stats.mannwhitneyu(ref_col, prod_col, alternative='two-sided')
                    drift_scores[col] = float(p_value)
                except Exception as e:
                    logger.warning(f"Mann-Whitney test failed for '{col}': {e}")
                    drift_scores[col] = 1.0
            else:  # KS test (default)
                try:
                    statistic, p_value = stats.ks_2samp(ref_col, prod_col)
                    drift_scores[col] = float(p_value)
                except Exception as e:
                    logger.warning(f"KS test failed for '{col}': {e}")
                    drift_scores[col] = 1.0
        else:
            # Categorical data - use Chi-square test
            if method in ["ks", "mannwhitney"]:
                # For categorical, use chi2 regardless of method parameter
                method = "chi2"
            
            try:
                # Create contingency table
                ref_counts = ref_col.value_counts()
                prod_counts = prod_col.value_counts()
                
                # Get union of all categories
                all_categories = set(ref_counts.index) | set(prod_counts.index)
                
                # Create aligned arrays
                ref_array = np.array([ref_counts.get(cat, 0) for cat in all_categories])
                prod_array = np.array([prod_counts.get(cat, 0) for cat in all_categories])
                
                # Chi-square test
                contingency_table = np.array([ref_array, prod_array])
                statistic, p_value = stats.chi2_contingency(contingency_table)[:2]
                drift_scores[col] = float(p_value)
            except Exception as e:
                logger.warning(f"Chi-square test failed for '{col}': {e}")
                drift_scores[col] = 1.0
    
    return drift_scores


def get_drifted_features(
    drift_scores: Dict[str, float],
    threshold: float = 0.05
) -> List[str]:
    """
    Extract features that have drifted based on p-value threshold.
    
    Args:
        drift_scores: Dictionary of feature names to p-values
        threshold: P-value threshold for drift (lower = more drift)
        
    Returns:
        List of feature names with detected drift
    """
    drifted_features = [
        feature for feature, p_value in drift_scores.items()
        if p_value < threshold
    ]
    
    if drifted_features:
        logger.info(f"Detected drift in {len(drifted_features)} features: {drifted_features}")
    
    return drifted_features


def compute_psi(
    reference_data: pd.Series,
    production_data: pd.Series,
    bins: int = 10
) -> float:
    """
    Compute Population Stability Index (PSI) for a feature.
    
    Args:
        reference_data: Reference feature values
        production_data: Production feature values
        bins: Number of bins for discretization
        
    Returns:
        PSI score (higher indicates more drift, > 0.25 is significant)
    """
    # Remove NaN values
    ref_data = reference_data.dropna()
    prod_data = production_data.dropna()
    
    if len(ref_data) == 0 or len(prod_data) == 0:
        logger.warning("Empty data for PSI computation")
        return float('inf')
    
    # Determine if numeric or categorical
    is_numeric = pd.api.types.is_numeric_dtype(ref_data) and pd.api.types.is_numeric_dtype(prod_data)
    
    if is_numeric:
        # Create bins based on reference data
        _, bin_edges = pd.cut(ref_data, bins=bins, retbins=True, duplicates='drop')
        
        # Align bins to include all data
        all_data = pd.concat([ref_data, prod_data])
        min_val = all_data.min()
        max_val = all_data.max()
        
        # Create bins
        if len(bin_edges) < 2:
            bin_edges = np.linspace(min_val, max_val, bins + 1)
        else:
            bin_edges = np.concatenate([[min_val], bin_edges[1:-1], [max_val]])
            bin_edges = np.unique(bin_edges)
        
        # Bin the data
        ref_binned = pd.cut(ref_data, bins=bin_edges, include_lowest=True)
        prod_binned = pd.cut(prod_data, bins=bin_edges, include_lowest=True)
    else:
        # Categorical - use categories as bins
        all_categories = set(ref_data.unique()) | set(prod_data.unique())
        ref_binned = ref_data
        prod_binned = prod_data
    
    # Compute distributions
    ref_counts = ref_binned.value_counts(normalize=True)
    prod_counts = prod_binned.value_counts(normalize=True)
    
    # Get all bins/categories
    all_bins = set(ref_counts.index) | set(prod_counts.index)
    
    # Compute PSI
    psi = 0.0
    for bin_val in all_bins:
        ref_pct = ref_counts.get(bin_val, 0.0)
        prod_pct = prod_counts.get(bin_val, 0.0)
        
        # Handle zero percentages (add small epsilon)
        if ref_pct == 0:
            ref_pct = 1e-6
        if prod_pct == 0:
            prod_pct = 1e-6
        
        psi += (prod_pct - ref_pct) * np.log(prod_pct / ref_pct)
    
    return float(psi)


def compute_mmd(
    reference_data: pd.DataFrame,
    production_data: pd.DataFrame
) -> float:
    """
    Compute Maximum Mean Discrepancy (MMD) between datasets.
    
    Simplified implementation using empirical means.
    For proper MMD, would need kernel methods (scikit-learn or custom).
    
    Args:
        reference_data: Reference dataframe
        production_data: Production dataframe
        
    Returns:
        MMD score (higher indicates more drift)
    """
    # For numeric columns only
    numeric_cols = reference_data.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col in production_data.columns]
    
    if len(numeric_cols) == 0:
        logger.warning("No numeric columns for MMD computation")
        return 0.0
    
    # Compute mean of means (simplified MMD)
    mmd_scores = []
    for col in numeric_cols:
        ref_mean = reference_data[col].mean()
        prod_mean = production_data[col].mean()
        mmd_score = abs(ref_mean - prod_mean)
        mmd_scores.append(mmd_score)
    
    # Average across features
    avg_mmd = np.mean(mmd_scores) if mmd_scores else 0.0
    
    return float(avg_mmd)
