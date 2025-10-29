import pandas as pd
import numpy as np
import ta

class FeaturesGenerator:
    def __init__(self):
        pass

    def returnificate(self, df):
        df['ret_open'] = np.log(df['open'] / df['open'].shift(1))
        df['ret_high'] = np.log(df['high'] / df['high'].shift(1))
        df['ret_low'] = np.log(df['low'] / df['low'].shift(1))
        df['ret_close'] = np.log(df['close'] / df['close'].shift(1))
        return df

    def add_features(self, df):
        df.columns = df.columns.str.lower()
        df = self.add_sma(df)
        df = self.add_rsi(df)
        df = self.add_stochastic(df)
        df = self.add_bollinger(df)
        df = self.add_float(df)
        df = self.clear_data(df)
        return df

    def add_sma(self, df, window = 10):
        # SMA small moving average
        df['SMA_10'] = df['close'].rolling(window=window).mean()
        df['SMA_20'] = df['close'].rolling(window=window*2).mean()
        
        # SMA_cross_10: Close crossing SMA_10 (upward: +1, downward: -1, no cross: 0)
        cross_up_10 = (df['close'] > df['SMA_10']) & (df['close'].shift(1) <= df['SMA_10'].shift(1))
        cross_down_10 = (df['close'] < df['SMA_10']) & (df['close'].shift(1) >= df['SMA_10'].shift(1))
        df['SMA_cross_10'] = 0
        df.loc[cross_up_10, 'SMA_cross_10'] = 1
        df.loc[cross_down_10, 'SMA_cross_10'] = -1
        
        # SMA_cross: SMA_10 crossing SMA_20 (upward: +1, downward: -1, no cross: 0)
        cross_up = (df['SMA_10'] > df['SMA_20']) & (df['SMA_10'].shift(1) <= df['SMA_20'].shift(1))
        cross_down = (df['SMA_10'] < df['SMA_20']) & (df['SMA_10'].shift(1) >= df['SMA_20'].shift(1))
        df['SMA_cross'] = 0
        df.loc[cross_up, 'SMA_cross'] = 1
        df.loc[cross_down, 'SMA_cross'] = -1
        
        # DROP SMA columns
        df.drop(['SMA_10', 'SMA_20'], axis=1, inplace=True)
        return df

    def add_rsi(self, df, min = 30, max = 70):
        # Calculate RSI using ta library
        rsi_indicator = ta.momentum.RSIIndicator(close=df['close'], window=14)
        df['RSI'] = rsi_indicator.rsi()
        
        # RSI_cross_min: RSI crossing oversold threshold (upward from oversold: +1, downward to oversold: -1, no cross: 0)
        cross_up_min = (df['RSI'] > min) & (df['RSI'].shift(1) <= min)
        cross_down_min = (df['RSI'] < min) & (df['RSI'].shift(1) >= min)
        df['RSI_cross_min'] = 0
        df.loc[cross_up_min, 'RSI_cross_min'] = 1
        df.loc[cross_down_min, 'RSI_cross_min'] = -1
        
        # DROP RSI column
        df.drop(['RSI'], axis=1, inplace=True)
        return df

    def add_stochastic(self, df, min = 20, max = 80):
        # Calculate Stochastic Oscillator using ta library
        stoch_indicator = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
        df['STOCH_K'] = stoch_indicator.stoch()  # %K line
        df['STOCH_D'] = stoch_indicator.stoch_signal()  # %D line (signal)
        
        # STOCH_cross_min: Stochastic crossing oversold threshold (upward from oversold: +1, downward to oversold: -1, no cross: 0)
        cross_up_min = (df['STOCH_K'] > min) & (df['STOCH_K'].shift(1) <= min)
        cross_down_min = (df['STOCH_K'] < min) & (df['STOCH_K'].shift(1) >= min)
        df['STOCH_cross_min'] = 0
        df.loc[cross_up_min, 'STOCH_cross_min'] = 1
        df.loc[cross_down_min, 'STOCH_cross_min'] = -1
        
        # DROP STOCH columns
        df.drop(['STOCH_K', 'STOCH_D'], axis=1, inplace=True)
        return df

    def add_bollinger(self, df, window = 14):
        bollinger_indicator = ta.volatility.BollingerBands(close=df['close'], window=window)
        df['BOLLINGER_High'] = bollinger_indicator.bollinger_hband()
        df['BOLLINGER_Low'] = bollinger_indicator.bollinger_lband()
        df['BOLLINGER_Middle'] = bollinger_indicator.bollinger_mavg()
        
        # BOLLINGER_cross_mid: Close crossing middle band (upward: +1, downward: -1, no cross: 0)
        cross_up_mid = (df['close'] > df['BOLLINGER_Middle']) & (df['close'].shift(1) <= df['BOLLINGER_Middle'].shift(1))
        cross_down_mid = (df['close'] < df['BOLLINGER_Middle']) & (df['close'].shift(1) >= df['BOLLINGER_Middle'].shift(1))
        df['BOLLINGER_cross_mid'] = 0
        df.loc[cross_up_mid, 'BOLLINGER_cross_mid'] = 1
        df.loc[cross_down_mid, 'BOLLINGER_cross_mid'] = -1

        # DROP BOLLINGER columns
        df.drop(['BOLLINGER_High', 'BOLLINGER_Low', 'BOLLINGER_Middle'], axis=1, inplace=True)
        return df

    def add_float(self, df):
        # Price returns (log returns)
        df['Return_15'] = np.log(df['close'] / df['close'].shift(15))
        df['Return_15'] = df['Return_15'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        df['Return_1'] = np.log(df['close'] / df['close'].shift(1))
        df['Return_1'] = df['Return_1'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # RSI log return - log of ratio from previous period
        rsi_temp = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
        # Add small constant to avoid log(0) and ensure positive values
        df['RSI'] = np.log((rsi_temp + 1) / (rsi_temp.shift(1) + 1))
        df['RSI'] = df['RSI'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Stochastic K log return - log of ratio from previous period
        stoch_k_temp = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3).stoch()
        # Add small constant to avoid log(0) and ensure positive values
        df['STOCH_K'] = np.log((stoch_k_temp + 1) / (stoch_k_temp.shift(1) + 1))
        df['STOCH_K'] = df['STOCH_K'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Stochastic D log return - log of ratio from previous period
        stoch_d_temp = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3).stoch_signal()
        # Add small constant to avoid log(0) and ensure positive values
        df['STOCH_D'] = np.log((stoch_d_temp + 1) / (stoch_d_temp.shift(1) + 1))
        df['STOCH_D'] = df['STOCH_D'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return df

    def clear_data(self, df):
        df = df.dropna()
        return df

    def test_features(self, df, target_col='target', top_n=20):
        """
        Test and evaluate feature quality for prediction.
        
        Analyzes:
        - Feature importance (correlation with target)
        - Feature redundancy (correlation between features)
        - Missing values
        - Feature distributions
        - Mutual information scores
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with features and target
        target_col : str
            Name of target column
        top_n : int
            Number of top features to display
        
        Returns:
        --------
        dict
            Dictionary with feature analysis results
        """
        import pandas as pd
        import numpy as np
        from sklearn.feature_selection import mutual_info_classif
        
        print("="*80)
        print("FEATURE QUALITY ANALYSIS")
        print("="*80)
        
        # Separate features and target
        if target_col not in df.columns:
            print(f"Error: Target column '{target_col}' not found in dataframe")
            return None
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Remove non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        print(f"\nDataset Info:")
        print(f"  Total samples: {len(df)}")
        print(f"  Total features: {len(X.columns)}")
        print(f"  Target classes: {sorted(y.unique())}")
        print(f"  Target distribution:\n{y.value_counts().sort_index()}")
        
        # 1. Missing Values Analysis
        print("\n" + "="*80)
        print("1. MISSING VALUES ANALYSIS")
        print("="*80)
        missing = X.isnull().sum()
        missing_pct = (missing / len(X)) * 100
        missing_df = pd.DataFrame({
            'Feature': missing.index,
            'Missing Count': missing.values,
            'Missing %': missing_pct.values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
        
        if len(missing_df) > 0:
            print(f"\nFeatures with missing values: {len(missing_df)}")
            print(missing_df.to_string(index=False))
        else:
            print("\n✓ No missing values found!")
        
        # 2. Feature-Target Correlation
        print("\n" + "="*80)
        print("2. FEATURE-TARGET CORRELATION")
        print("="*80)
        
        correlations = []
        for col in X.columns:
            try:
                corr = X[col].corr(y)
                correlations.append({
                    'Feature': col,
                    'Correlation': corr,
                    'Abs_Correlation': abs(corr)
                })
            except:
                pass
        
        corr_df = pd.DataFrame(correlations).sort_values('Abs_Correlation', ascending=False)
        
        print(f"\nTop {min(top_n, len(corr_df))} features by correlation with target:")
        print(corr_df.head(top_n)[['Feature', 'Correlation']].to_string(index=False))
        
        # 3. Mutual Information Scores
        print("\n" + "="*80)
        print("3. MUTUAL INFORMATION SCORES")
        print("="*80)
        print("(Measures non-linear relationships with target)")
        
        # Handle missing values for mutual info
        X_clean = X.fillna(X.mean())
        
        try:
            mi_scores = mutual_info_classif(X_clean, y, random_state=42)
            mi_df = pd.DataFrame({
                'Feature': X.columns,
                'MI_Score': mi_scores
            }).sort_values('MI_Score', ascending=False)
            
            print(f"\nTop {min(top_n, len(mi_df))} features by mutual information:")
            print(mi_df.head(top_n).to_string(index=False))
        except Exception as e:
            print(f"\nWarning: Could not calculate mutual information: {e}")
            mi_df = None
        
        # 4. Feature Redundancy (Inter-feature correlation)
        print("\n" + "="*80)
        print("4. FEATURE REDUNDANCY ANALYSIS")
        print("="*80)
        print("(High correlation between features = redundancy)")
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.8:  # Threshold for high correlation
                    high_corr_pairs.append({
                        'Feature_1': corr_matrix.columns[i],
                        'Feature_2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })
        
        if high_corr_pairs:
            redundancy_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', ascending=False)
            print(f"\nHighly correlated feature pairs (>0.8): {len(redundancy_df)}")
            print(redundancy_df.head(20).to_string(index=False))
            print("\n⚠ Consider removing redundant features to reduce multicollinearity")
        else:
            print("\n✓ No highly correlated feature pairs found (threshold: 0.8)")
        
        # 5. Feature Statistics
        print("\n" + "="*80)
        print("5. FEATURE STATISTICS")
        print("="*80)
        
        stats_df = pd.DataFrame({
            'Feature': X.columns,
            'Mean': X.mean(),
            'Std': X.std(),
            'Min': X.min(),
            'Max': X.max(),
            'Zeros_%': ((X == 0).sum() / len(X) * 100)
        }).sort_values('Std', ascending=False)
        
        print(f"\nTop {min(10, len(stats_df))} features by standard deviation:")
        print(stats_df.head(10).to_string(index=False))
        
        # Check for constant or near-constant features
        low_variance = stats_df[stats_df['Std'] < 0.01]
        if len(low_variance) > 0:
            print(f"\n⚠ Features with very low variance (<0.01): {len(low_variance)}")
            print(low_variance[['Feature', 'Std']].to_string(index=False))
            print("Consider removing these features as they provide little information")
        
        # 6. Overall Assessment
        print("\n" + "="*80)
        print("6. OVERALL FEATURE SET ASSESSMENT")
        print("="*80)
        
        # Calculate scores
        avg_abs_corr = corr_df['Abs_Correlation'].mean()
        max_abs_corr = corr_df['Abs_Correlation'].max()
        top10_avg_corr = corr_df.head(10)['Abs_Correlation'].mean()
        
        if mi_df is not None:
            avg_mi = mi_df['MI_Score'].mean()
            max_mi = mi_df['MI_Score'].max()
        else:
            avg_mi = max_mi = 0
        
        redundancy_ratio = len(high_corr_pairs) / len(X.columns) if len(X.columns) > 0 else 0
        
        print(f"\nFeature Quality Metrics:")
        print(f"  Average |correlation| with target: {avg_abs_corr:.4f}")
        print(f"  Maximum |correlation| with target: {max_abs_corr:.4f}")
        print(f"  Top 10 avg |correlation|:          {top10_avg_corr:.4f}")
        if mi_df is not None:
            print(f"  Average mutual information:        {avg_mi:.4f}")
            print(f"  Maximum mutual information:        {max_mi:.4f}")
        print(f"  Feature redundancy ratio:          {redundancy_ratio:.4f}")
        print(f"  Features with missing values:      {len(missing_df)}")
        print(f"  Low variance features:             {len(low_variance)}")
        
        # Overall rating
        print("\n" + "-"*80)
        print("FEATURE SET RATING:")
        print("-"*80)
        
        score = 0
        max_score = 5
        
        # Criterion 1: Strong features (top 10 avg correlation)
        if top10_avg_corr > 0.15:
            print("✓ Strong predictive features (top 10 avg |corr| > 0.15)")
            score += 1
        else:
            print("✗ Weak predictive features (top 10 avg |corr| ≤ 0.15)")
        
        # Criterion 2: Best feature quality
        if max_abs_corr > 0.25:
            print("✓ At least one very strong feature (max |corr| > 0.25)")
            score += 1
        else:
            print("✗ No very strong features (max |corr| ≤ 0.25)")
        
        # Criterion 3: Low redundancy
        if redundancy_ratio < 0.1:
            print("✓ Low feature redundancy (<10%)")
            score += 1
        else:
            print("✗ High feature redundancy (≥10%)")
        
        # Criterion 4: No missing values
        if len(missing_df) == 0:
            print("✓ No missing values")
            score += 1
        else:
            print(f"✗ {len(missing_df)} features with missing values")
        
        # Criterion 5: No low variance features
        if len(low_variance) == 0:
            print("✓ All features have sufficient variance")
            score += 1
        else:
            print(f"✗ {len(low_variance)} low variance features")
        
        print(f"\nOverall Score: {score}/{max_score}")
        
        if score >= 4:
            rating = "EXCELLENT"
            color = "green"
        elif score >= 3:
            rating = "GOOD"
            color = "blue"
        elif score >= 2:
            rating = "FAIR"
            color = "yellow"
        else:
            rating = "POOR"
            color = "red"
        
        print(f"Rating: {rating}")
        
        # Recommendations
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)
        
        recommendations = []
        
        if top10_avg_corr < 0.15:
            recommendations.append("• Add more predictive features (technical indicators, price patterns)")
        
        if len(high_corr_pairs) > 0:
            recommendations.append(f"• Remove {len(high_corr_pairs)} redundant feature pairs to reduce multicollinearity")
        
        if len(low_variance) > 0:
            recommendations.append(f"• Remove {len(low_variance)} low variance features")
        
        if len(missing_df) > 0:
            recommendations.append(f"• Handle {len(missing_df)} features with missing values")
        
        if avg_abs_corr < 0.05:
            recommendations.append("• Consider feature engineering: create interaction terms, ratios, or transformations")
        
        if len(recommendations) > 0:
            for rec in recommendations:
                print(rec)
        else:
            print("✓ Feature set looks good! No major issues found.")
        
        print("\n" + "="*80)
        
        # 7. Generate Visualizations
        print("\n" + "="*80)
        print("7. GENERATING VISUALIZATIONS")
        print("="*80)
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('seaborn-v0_8-darkgrid')
            sns.set_palette("husl")
            
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # 1. Top Features by Correlation (Bar plot)
            ax1 = fig.add_subplot(gs[0, :])
            top_n_plot = min(15, len(corr_df))
            top_corr = corr_df.head(top_n_plot)
            colors = ['green' if x > 0 else 'red' for x in top_corr['Correlation']]
            ax1.barh(range(top_n_plot), top_corr['Correlation'], color=colors, alpha=0.7)
            ax1.set_yticks(range(top_n_plot))
            ax1.set_yticklabels(top_corr['Feature'])
            ax1.set_xlabel('Correlation with Target', fontsize=12, fontweight='bold')
            ax1.set_title(f'Top {top_n_plot} Features by Correlation with Target', fontsize=14, fontweight='bold')
            ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            ax1.grid(axis='x', alpha=0.3)
            ax1.invert_yaxis()
            
            # 2. Mutual Information Scores (Bar plot)
            if mi_df is not None:
                ax2 = fig.add_subplot(gs[1, 0])
                top_mi = mi_df.head(10)
                ax2.barh(range(len(top_mi)), top_mi['MI_Score'], color='steelblue', alpha=0.7)
                ax2.set_yticks(range(len(top_mi)))
                ax2.set_yticklabels(top_mi['Feature'])
                ax2.set_xlabel('Mutual Information Score', fontsize=10, fontweight='bold')
                ax2.set_title('Top 10 Features by Mutual Information', fontsize=12, fontweight='bold')
                ax2.grid(axis='x', alpha=0.3)
                ax2.invert_yaxis()
            
            # 3. Feature Variance Distribution (Histogram)
            ax3 = fig.add_subplot(gs[1, 1])
            ax3.hist(stats_df['Std'], bins=30, color='coral', alpha=0.7, edgecolor='black')
            ax3.axvline(x=0.01, color='red', linestyle='--', linewidth=2, label='Low Variance Threshold')
            ax3.set_xlabel('Standard Deviation', fontsize=10, fontweight='bold')
            ax3.set_ylabel('Number of Features', fontsize=10, fontweight='bold')
            ax3.set_title('Feature Variance Distribution', fontsize=12, fontweight='bold')
            ax3.legend()
            ax3.grid(axis='y', alpha=0.3)
            
            # 4. Correlation Distribution (Histogram)
            ax4 = fig.add_subplot(gs[1, 2])
            ax4.hist(corr_df['Abs_Correlation'], bins=30, color='purple', alpha=0.7, edgecolor='black')
            ax4.axvline(x=avg_abs_corr, color='blue', linestyle='--', linewidth=2, label=f'Mean: {avg_abs_corr:.4f}')
            ax4.axvline(x=0.15, color='green', linestyle='--', linewidth=2, label='Good Threshold: 0.15')
            ax4.set_xlabel('|Correlation| with Target', fontsize=10, fontweight='bold')
            ax4.set_ylabel('Number of Features', fontsize=10, fontweight='bold')
            ax4.set_title('Correlation Strength Distribution', fontsize=12, fontweight='bold')
            ax4.legend()
            ax4.grid(axis='y', alpha=0.3)
            
            # 5. Target Distribution (Pie chart)
            ax5 = fig.add_subplot(gs[2, 0])
            target_counts = y.value_counts().sort_index()
            colors_pie = ['#ff9999', '#66b3ff', '#99ff99']
            labels = [f'Class {int(c)}\n({count} samples)' for c, count in zip(target_counts.index, target_counts.values)]
            ax5.pie(target_counts.values, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
            ax5.set_title('Target Class Distribution', fontsize=12, fontweight='bold')
            
            # 6. Feature Quality Score (Gauge/Bar)
            ax6 = fig.add_subplot(gs[2, 1])
            criteria = ['Strong\nFeatures', 'Best\nFeature', 'Low\nRedundancy', 'No\nMissing', 'Good\nVariance']
            scores_list = [
                1 if top10_avg_corr > 0.15 else 0,
                1 if max_abs_corr > 0.25 else 0,
                1 if redundancy_ratio < 0.1 else 0,
                1 if len(missing_df) == 0 else 0,
                1 if len(low_variance) == 0 else 0
            ]
            colors_bar = ['green' if s == 1 else 'red' for s in scores_list]
            ax6.bar(criteria, scores_list, color=colors_bar, alpha=0.7, edgecolor='black')
            ax6.set_ylim(0, 1.2)
            ax6.set_ylabel('Pass (1) / Fail (0)', fontsize=10, fontweight='bold')
            ax6.set_title(f'Feature Quality Criteria\nScore: {score}/{max_score} ({rating})', fontsize=12, fontweight='bold')
            ax6.grid(axis='y', alpha=0.3)
            for i, (c, s) in enumerate(zip(criteria, scores_list)):
                ax6.text(i, s + 0.05, '✓' if s == 1 else '✗', ha='center', fontsize=16, fontweight='bold')
            
            # 7. Correlation vs Mutual Information (Scatter)
            if mi_df is not None:
                ax7 = fig.add_subplot(gs[2, 2])
                # Merge correlation and MI data
                merged = corr_df.merge(mi_df, on='Feature')
                ax7.scatter(merged['Abs_Correlation'], merged['MI_Score'], alpha=0.6, s=100, c='teal', edgecolors='black')
                ax7.set_xlabel('|Correlation|', fontsize=10, fontweight='bold')
                ax7.set_ylabel('Mutual Information', fontsize=10, fontweight='bold')
                ax7.set_title('Correlation vs Mutual Information', fontsize=12, fontweight='bold')
                ax7.grid(alpha=0.3)
                
                # Annotate top features
                top_combined = merged.nlargest(5, 'Abs_Correlation')
                for idx, row in top_combined.iterrows():
                    ax7.annotate(row['Feature'], (row['Abs_Correlation'], row['MI_Score']),
                               xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
            
            # Overall title
            fig.suptitle('Feature Quality Analysis Report', fontsize=18, fontweight='bold', y=0.995)
            
            # Save figure
            plt.savefig('feature_analysis_report.png', dpi=300, bbox_inches='tight')
            print("\n✓ Visualizations saved to: feature_analysis_report.png")
            
            # Show plot
            plt.show()
            
        except ImportError as e:
            print(f"\n⚠ Could not generate visualizations: {e}")
            print("Install matplotlib and seaborn for visualizations: pip install matplotlib seaborn")
        except Exception as e:
            print(f"\n⚠ Error generating visualizations: {e}")
        
        print("\n" + "="*80)
        
        # Return results
        results = {
            'correlation_df': corr_df,
            'mutual_info_df': mi_df,
            'redundancy_df': pd.DataFrame(high_corr_pairs) if high_corr_pairs else None,
            'missing_df': missing_df if len(missing_df) > 0 else None,
            'stats_df': stats_df,
            'score': score,
            'max_score': max_score,
            'rating': rating,
            'metrics': {
                'avg_abs_corr': avg_abs_corr,
                'max_abs_corr': max_abs_corr,
                'top10_avg_corr': top10_avg_corr,
                'avg_mi': avg_mi,
                'max_mi': max_mi,
                'redundancy_ratio': redundancy_ratio
            }
        }
        
        return results

    def add_features_otus(self, df, rolling = False):
        df.columns = df.columns.str.lower()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        #OTUS 15/16
        def add_target(df):
            df['close_next_hour'] = df['close'].shift(-1)
            df['target'] = (df['close_next_hour'] > df['close']).astype(int)
            df = df.dropna(subset=['close_next_hour'])
            return df


        def filter_invalid_targets(df):
            # Удаляем строки, где close_next_hour или target равен NaN
            return df.dropna(subset=['close_next_hour', 'target'])


        def filter_invalid_features(df_):
            df = df_.copy()
            for column_name in df.columns:
                column = df[column_name]
                if column.dtype == 'float64':
                    df[column_name] = column.fillna(df[column_name].mean())
                    if max(column) == float('inf'):
                        #df[column_name] = column.fillna(df[column_name].mean())
                        pass
            df = df.replace([np.inf, -np.inf], 0.0)
            return df.dropna()

        def create_trend_features(df, features, lag_periods):
            """
            Добавляет классические финансовые признаки: отношение к предыдущим периодам, логарифмические изменения и индикаторы трендов.
            
            df: DataFrame с исходными данными
            features: список признаков, для которых необходимо добавить индикаторы
            lag_periods: сколько периодов назад учитывать для расчетов
            
            Возвращает:
            - обновленный DataFrame с новыми фичами
            - список новых колонок, которые можно использовать как признаки
            """
            df = df.copy()  # Работаем с копией DataFrame
            new_columns = []  # Список для хранения новых колонок
            
            for feature in features:
                # Отношение текущего значения к предыдущему (лаг = 1)
                df[f'{feature}_ratio_1'] = df[feature] / df[feature].shift(1)
                new_columns.append(f'{feature}_ratio_1')
                
                # Логарифмическое изменение (логарифм отношения текущего значения к предыдущему)
                df[f'{feature}_log_diff_1'] = np.log(df[feature] / df[feature].shift(1))
                new_columns.append(f'{feature}_log_diff_1')
                
                # Momentum (разница между текущим значением и значением N периодов назад)
                df[f'{feature}_momentum_{lag_periods}'] = df[feature] - df[feature].shift(lag_periods)
                new_columns.append(f'{feature}_momentum_{lag_periods}')
                
                # Rate of Change (ROC): процентное изменение за N периодов
                df[f'{feature}_roc_{lag_periods}'] = (df[feature] - df[feature].shift(lag_periods)) / df[feature].shift(lag_periods) * 100
                new_columns.append(f'{feature}_roc_{lag_periods}')
                
                # Exponential Moving Average (EMA) с периодом N
                df[f'{feature}_ema_{lag_periods}'] = df[feature].ewm(span=lag_periods, adjust=False).mean()
                new_columns.append(f'{feature}_ema_{lag_periods}')
            
            # Удаление строк с NaN значениями, которые появились из-за сдвигов
            df = df.dropna()
            
            return df, new_columns


        def create_rolling_features(df, features, window_sizes):
            """
            Добавляет скользящие характеристики для указанных признаков и окон.
            
            df: DataFrame с исходными данными
            features: список признаков, для которых необходимо добавить скользящие характеристики
            window_sizes: список размеров окон для расчета характеристик (например, [5, 14, 30])
            
            Возвращает:
            - обновленный DataFrame с новыми фичами
            - список новых колонок, которые можно использовать как признаки
            """
            df = df.copy()  # Работаем с копией DataFrame
            new_columns = []  # Список для хранения новых колонок
            
            # Для каждого признака и для каждого окна
            for feature in features:
                for window_size in window_sizes:
                    # Скользящее среднее
                    df[f'{feature}_mean_{window_size}'] = df[feature].rolling(window=window_size).mean()
                    new_columns.append(f'{feature}_mean_{window_size}')
                    
                    # Скользящая медиана
                    df[f'{feature}_median_{window_size}'] = df[feature].rolling(window=window_size).median()
                    new_columns.append(f'{feature}_median_{window_size}')
                    
                    # Скользящий минимум
                    df[f'{feature}_min_{window_size}'] = df[feature].rolling(window=window_size).min()
                    new_columns.append(f'{feature}_min_{window_size}')
                    
                    # Скользящий максимум
                    df[f'{feature}_max_{window_size}'] = df[feature].rolling(window=window_size).max()
                    new_columns.append(f'{feature}_max_{window_size}')
                    
                    # Скользящее стандартное отклонение
                    df[f'{feature}_std_{window_size}'] = df[feature].rolling(window=window_size).std()
                    new_columns.append(f'{feature}_std_{window_size}')
                    
                    # Скользящий размах (макс - мин)
                    df[f'{feature}_range_{window_size}'] = df[f'{feature}_max_{window_size}'] - df[f'{feature}_min_{window_size}']
                    new_columns.append(f'{feature}_range_{window_size}')
                    
                    # Скользящее абсолютное отклонение от медианы (mad)
                    df[f'{feature}_mad_{window_size}'] = df[feature].rolling(window=window_size).apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
                    new_columns.append(f'{feature}_mad_{window_size}')
            
            # Удаление строк с NaN значениями, которые появляются из-за сдвигов
            df = df.dropna()
            
            return df, new_columns


        def create_macd(df, feature, short_window=12, long_window=26):
            """
            Добавляет индикатор MACD (разница между краткосрочным и долгосрочным EMA).
            
            df: DataFrame с исходными данными
            feature: признак, для которого необходимо рассчитать MACD
            short_window: окно для краткосрочного EMA (по умолчанию 12)
            long_window: окно для долгосрочного EMA (по умолчанию 26)
            
            Возвращает:
            - обновленный DataFrame с MACD
            - название новой колонки с MACD
            """
            df = df.copy()
            
            # Рассчитываем краткосрочное и долгосрочное EMA
            ema_short = df[feature].ewm(span=short_window, adjust=False).mean()
            ema_long = df[feature].ewm(span=long_window, adjust=False).mean()
            
            # Разница между краткосрочным и долгосрочным EMA (MACD)
            df[f'{feature}_macd'] = ema_short - ema_long
            
            return df, f'{feature}_macd'

        data = filter_invalid_targets(add_target(df))
        data = filter_invalid_features(data)

        if rolling:
            window_sizes = [5, 14, 30]
            features_to_rolling = ['open', 'high', 'low', 'close', 'volume']
            data_with_rolling, new_rolling_features = create_rolling_features(data, features_to_rolling, window_sizes)
            feature_cols = features_to_rolling + new_rolling_features
        else:
            lag_periods = 3  # Например, 3 периода назад
            features_to_trend = ['open', 'high', 'low', 'close', 'volume']
            # Создаем трендовые признаки
            data_with_trend, new_trend_features = create_trend_features(data, features_to_trend, lag_periods)
            # Добавляем MACD для признака 'close'
            data_with_trend, macd_column = create_macd(data_with_trend, 'close')
            # Добавляем название колонки с MACD в список новых фичей
            new_trend_features.append(macd_column)
            feature_cols = features_to_trend + new_trend_features
            data = data_with_trend[new_trend_features + ['target', 'timestamp']]
            # Определяем дату начала тестовой выборки (последний месяц)
            test_start_date = data['timestamp'].max() - pd.DateOffset(months=1)
            # Определяем дату начала валидационной выборки (предпоследний месяц)
            val_start_date = data['timestamp'].max() - pd.DateOffset(months=2)
            # Разделение данных на тренировочную, валидационную и тестовую выборки по времени
            train_data = data[data['timestamp'] < val_start_date]  # все, что до предпоследнего месяца
            val_data = data[(data['timestamp'] >= val_start_date) & (data['timestamp'] < test_start_date)]  # предпоследний месяц
            test_data = data[data['timestamp'] >= test_start_date]  # последний месяц
            # Признаки, которые будем использовать
            features = new_trend_features  # Убедись, что 'new_trend_features' определены
            # Разделение на признаки (X) и целевую переменную (y) для каждой выборки
            X_train = filter_invalid_features(train_data[new_trend_features])
            y_train = train_data['target']

            X_val = filter_invalid_features(val_data[new_trend_features])
            y_val = val_data['target']

            X_test = filter_invalid_features(test_data[new_trend_features])
            y_test = test_data['target']
            return X_train, y_train, X_val, y_val, X_test, y_test, new_trend_features

    def add_crypto_features(self, df):
        pass