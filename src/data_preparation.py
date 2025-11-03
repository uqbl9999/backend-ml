"""
Data Preparation Module
Handles data loading, cleaning, and preprocessing for mental health screening predictions
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')


class DataPreparation:
    """Class to handle all data preparation steps"""

    def __init__(self, data_path: str):
        """
        Initialize the data preparation class

        Parameters:
        -----------
        data_path : str
            Path to the raw CSV file
        """
        self.data_path = data_path
        self.df = None
        self.df_pivot = None
        self.df_clean = None
        self.df_encoded = None

    def load_data(self) -> pd.DataFrame:
        """Load raw data from CSV file"""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path, sep=';', encoding='latin1')
        print(f"✅ Dataset loaded: {self.df.shape[0]:,} rows x {self.df.shape[1]} columns")
        return self.df

    def calculate_positivity_rate(self) -> pd.DataFrame:
        """
        Calculate the positivity rate (Tasa_Positividad) by merging
        total screenings and positive cases
        """
        print("\nCalculating positivity rate...")

        # Convert 'Casos' to numeric if needed
        if self.df['Casos'].dtype == 'object':
            self.df['Casos'] = self.df['Casos'].astype(str).str.replace(',', '').str.replace(' ', '')
            self.df['Casos'] = pd.to_numeric(self.df['Casos'], errors='coerce')
            self.df['Casos'] = self.df['Casos'].fillna(0)

        # Separate TOTAL and POSITIVOS records
        df_total = self.df[self.df['GrupoTamizaje'].str.contains('TOTAL', case=False, na=False)].copy()
        df_positivos = self.df[self.df['GrupoTamizaje'].str.contains('POSITIVOS', case=False, na=False)].copy()

        print(f"  Total records: {len(df_total):,}")
        print(f"  Positive records: {len(df_positivos):,}")

        # Rename columns
        df_total = df_total.rename(columns={'Casos': 'Total'})
        df_positivos = df_positivos.rename(columns={'Casos': 'Positivos'})

        # Drop GrupoTamizaje column
        df_total = df_total.drop(columns=['GrupoTamizaje'])
        df_positivos = df_positivos.drop(columns=['GrupoTamizaje'])

        # Merge datasets
        join_columns = ['Anio', 'NroMes', 'ubigeo', 'Departamento', 'Provincia',
                       'Distrito', 'Sexo', 'Etapa', 'DetalleTamizaje']

        self.df_pivot = df_total.merge(
            df_positivos[join_columns + ['Positivos']],
            on=join_columns,
            how='left'
        )

        # Fill NaN and calculate rate
        self.df_pivot['Positivos'] = self.df_pivot['Positivos'].fillna(0)
        self.df_pivot['Tasa_Positividad'] = np.where(
            self.df_pivot['Total'] > 0,
            (self.df_pivot['Positivos'] / self.df_pivot['Total']) * 100,
            0
        )

        # Remove rows where Total = 0
        self.df_pivot = self.df_pivot[self.df_pivot['Total'] > 0].copy()

        print(f"✅ Positivity rate calculated: {self.df_pivot.shape[0]:,} rows")
        print(f"✅ Average rate: {self.df_pivot['Tasa_Positividad'].mean():.2f}%")

        return self.df_pivot

    def clean_data(self) -> pd.DataFrame:
        """
        Remove impossible values (Tasa_Positividad > 100%)
        """
        print("\nCleaning data...")

        before_count = len(self.df_pivot)
        before_avg = self.df_pivot['Tasa_Positividad'].mean()

        # Remove impossible rates
        self.df_clean = self.df_pivot[self.df_pivot['Tasa_Positividad'] <= 100].copy()

        removed_count = before_count - len(self.df_clean)
        after_avg = self.df_clean['Tasa_Positividad'].mean()

        print(f"  Removed {removed_count:,} records with rate > 100%")
        print(f"  Before: {before_count:,} rows, avg rate: {before_avg:.2f}%")
        print(f"  After: {len(self.df_clean):,} rows, avg rate: {after_avg:.2f}%")

        return self.df_clean

    def feature_engineering(self) -> pd.DataFrame:
        """
        Apply one-hot encoding to categorical variables
        and drop unnecessary columns
        """
        print("\nApplying feature engineering...")

        # One-hot encoding
        multi_cat_cols = ["Departamento", "Sexo", "DetalleTamizaje", "Etapa"]
        self.df_encoded = pd.get_dummies(self.df_clean, columns=multi_cat_cols, dtype=int)

        # Drop unnecessary columns
        self.df_encoded = self.df_encoded.drop(columns=['Provincia', 'Distrito'])

        print(f"✅ Features encoded: {self.df_encoded.shape}")

        return self.df_encoded

    def balance_data(self, zero_ratio: float = 0.3, oversample_factor: float = 2) -> tuple:
        """
        Balance the dataset using undersampling for zeros and
        SMOTE-like oversampling for positives

        Parameters:
        -----------
        zero_ratio : float
            Desired proportion of zeros in final dataset
        oversample_factor : float
            Factor for oversampling positive cases

        Returns:
        --------
        tuple : (X_balanced, y_balanced)
        """
        print("\nBalancing data...")

        y = self.df_encoded['Tasa_Positividad']
        X = self.df_encoded.drop(columns=['Tasa_Positividad', 'Total', 'Positivos'])

        zero_mask = (y == 0)
        non_zero_mask = ~zero_mask

        X_zero = X[zero_mask]
        y_zero = y[zero_mask]
        X_non_zero = X[non_zero_mask]
        y_non_zero = y[non_zero_mask]

        print(f"  Original - Zeros: {len(y_zero):,} | Positives: {len(y_non_zero):,}")

        # STEP 1: Undersample zeros
        n_zeros_keep = int(len(y_non_zero) / (1 - zero_ratio) * zero_ratio)
        if n_zeros_keep < len(y_zero):
            zero_indices = np.random.choice(len(y_zero), n_zeros_keep, replace=False)
            X_zero_sampled = X_zero.iloc[zero_indices]
            y_zero_sampled = y_zero.iloc[zero_indices]
        else:
            X_zero_sampled = X_zero
            y_zero_sampled = y_zero

        print(f"  Undersampled - Zeros kept: {len(y_zero_sampled):,}")

        # STEP 2: Oversample positives (SMOTE-like)
        n_synthetic = int(len(X_non_zero) * (oversample_factor - 1))
        k = min(5, len(X_non_zero) - 1)

        if k < 1:
            print("⚠️  Not enough positives for oversampling")
            X_res = pd.concat([X_zero_sampled, X_non_zero], ignore_index=True)
            y_res = pd.concat([y_zero_sampled, y_non_zero], ignore_index=True)
            return X_res, y_res

        knn = NearestNeighbors(n_neighbors=k + 1)
        knn.fit(X_non_zero)

        synthetic_X = []
        synthetic_y = []

        np.random.seed(42)
        for _ in range(n_synthetic):
            idx = np.random.randint(0, len(X_non_zero))
            sample_X = X_non_zero.iloc[idx].values
            sample_y = y_non_zero.iloc[idx]

            neighbors_idx = knn.kneighbors([sample_X], return_distance=False)[0]
            neighbor_idx = np.random.choice(neighbors_idx[1:])

            alpha = np.random.random()
            neighbor_X = X_non_zero.iloc[neighbor_idx].values
            neighbor_y = y_non_zero.iloc[neighbor_idx]

            synthetic_sample_X = sample_X + alpha * (neighbor_X - sample_X)
            synthetic_sample_y = sample_y + alpha * (neighbor_y - sample_y)

            synthetic_X.append(synthetic_sample_X)
            synthetic_y.append(synthetic_sample_y)

        # Combine all data
        synthetic_X_df = pd.DataFrame(synthetic_X, columns=X.columns)
        synthetic_y_series = pd.Series(synthetic_y, name='Tasa_Positividad')

        X_res = pd.concat([X_zero_sampled, X_non_zero, synthetic_X_df], ignore_index=True)
        y_res = pd.concat([y_zero_sampled, y_non_zero, synthetic_y_series], ignore_index=True)

        print(f"  Balanced - Zeros: {(y_res == 0).sum():,} | Positives: {(y_res > 0).sum():,}")
        print(f"  Zero proportion: {(y_res == 0).sum() / len(y_res):.2%}")

        return X_res, y_res

    def prepare_full_pipeline(self, save_intermediate: bool = True) -> tuple:
        """
        Run the complete data preparation pipeline

        Parameters:
        -----------
        save_intermediate : bool
            Whether to save intermediate CSV files

        Returns:
        --------
        tuple : (X_balanced, y_balanced)
        """
        # Load data
        self.load_data()

        # Calculate positivity rate
        self.calculate_positivity_rate()

        # Clean data
        self.clean_data()
        if save_intermediate:
            self.df_clean.to_csv('data/dataset_limpio.csv', index=False, encoding='utf-8-sig')
            print(f"💾 Saved: data/dataset_limpio.csv")

        # Feature engineering
        self.feature_engineering()
        if save_intermediate:
            self.df_encoded.to_csv('data/df_clean_to_model.csv', index=False, encoding='utf-8-sig')
            print(f"💾 Saved: data/df_clean_to_model.csv")

        # Balance data
        X_balanced, y_balanced = self.balance_data(zero_ratio=0.3, oversample_factor=2)

        if save_intermediate:
            df_balanced = X_balanced.copy()
            df_balanced['Tasa_Positividad'] = y_balanced.values
            df_balanced.to_csv('data/dataset_balanceado.csv', index=False, encoding='utf-8-sig')
            print(f"💾 Saved: data/dataset_balanceado.csv")

        print("\n✅ Data preparation pipeline completed!")

        return X_balanced, y_balanced
