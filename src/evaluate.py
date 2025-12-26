import pandas as pd
import os
import src.config as config

def evaluate_coverage(df_results):
    try:
        true_df = pd.read_csv(config.SAMPLE_SUB)
        df_results['id_clean'] = df_results['id'].apply(lambda x: os.path.splitext(str(x))[0])
        true_df['id_clean'] = true_df['id'].apply(lambda x: os.path.splitext(str(x))[0])
        
        compare = df_results.merge(true_df, on='id_clean', suffixes=('_pred', '_true'))
        
        def check_cov(row):
            sets = [s.strip().lower() for s in str(row['prediction_set']).split('|')]
            return str(row['label_true']).strip().lower() in sets

        coverage = compare.apply(check_cov, axis=1).mean()
        print("\n" + "="*30)
        print(f"Coverage thực tế: {coverage:.2%}")
        print(f"Kích thước Set TB: {df_results['set_size'].mean():.2f}")
        print("="*30)
    except Exception as e:
        print(f"[!] Lỗi khi đánh giá: {e}")