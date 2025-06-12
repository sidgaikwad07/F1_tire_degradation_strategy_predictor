"""
Created on Thu Jun 12 10:24:22 2025

@author: sid
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class F1FlexibleTireEngineer:
    def __init__(self, base_path):
        self.base_path = base_path
        self.yearly_data = {}
        self.combined_dataset = None
        self.feature_columns = []
        
    def detect_data_type(self, year):
        """Detect which type of data files are available for a year"""
        # Check both main directory and yearly_data subdirectory
        possible_paths = [
            self.base_path,  # Main directory
            os.path.join(self.base_path, "yearly_data")  # Subdirectory
        ]
        
        for base_dir in possible_paths:
            if not os.path.exists(base_dir):
                continue
                
            print(f"Checking directory: {base_dir}")
            
            # Check for processed comprehensive file
            comprehensive_file = os.path.join(base_dir, f"f1_comprehensive_tire_data_{year}.csv")
            
            # Check for raw files
            laps_file = os.path.join(base_dir, f"{year}_all_laps.csv")
            spanish_laps_file = os.path.join(base_dir, f"{year}_all_laps_through_spanish.csv")
            results_file = os.path.join(base_dir, f"{year}_all_results.csv")
            spanish_results_file = os.path.join(base_dir, f"{year}_all_results_through_spanish.csv")
            
            # Debug: list what files are actually there
            if os.path.exists(base_dir):
                available_files = [f for f in os.listdir(base_dir) if f.endswith('.csv') and str(year) in f]
                if available_files:
                    print(f"Available {year} files: {available_files}")
            
            if os.path.exists(comprehensive_file):
                print(f"Found comprehensive file: {comprehensive_file}")
                return "comprehensive", comprehensive_file
            elif os.path.exists(spanish_laps_file) and os.path.exists(spanish_results_file):
                print(f"Found Spanish raw files: {spanish_laps_file}, {spanish_results_file}")
                return "raw_spanish", (spanish_laps_file, spanish_results_file)
            elif os.path.exists(laps_file) and os.path.exists(results_file):
                print(f"Found full raw files: {laps_file}, {results_file}")
                return "raw_full", (laps_file, results_file)
        
        print(f"No data files found for year {year}")
        return "none", None
    
    def load_comprehensive_data(self, file_path, year):
        """Load pre-processed comprehensive tire data"""
        try:
            data = pd.read_csv(file_path)
            data['Year'] = year
            print(f"Loaded {year} comprehensive: {len(data)} laps, {len(data.columns)} columns")
            return data
        except Exception as e:
            print(f"Error loading comprehensive {year}: {str(e)}")
            return None
    
    def load_raw_data(self, files, year):
        """Load raw laps and results data"""
        laps_file, results_file = files
        
        try:
            laps = pd.read_csv(laps_file)
            results = pd.read_csv(results_file)
            print(f"Loaded {year} raw: {len(laps)} laps, {len(results)} results")
            return {'laps': laps, 'results': results}
        except Exception as e:
            print(f"Error loading raw {year}: {str(e)}")
            return None
    
    def process_comprehensive_data(self, data, year):
        """Process pre-cleaned comprehensive data"""
        df = data.copy()
        
        print(f"Processing comprehensive data for {year}...")
        
        # === BASIC CLEANING ===
        # Remove invalid laps
        df = df[
            (df['LapTime'].notna()) &
            (df['LapTime'] > 60) &
            (df['LapTime'] < 300) &
            (df['TyreLife'].notna()) &
            (df['TyreLife'] > 0)
        ].copy()
        
        # === TIRE FEATURES ===
        # Standardize compound names
        df['Compound_clean'] = df['Compound'].str.upper().str.strip()
        
        compound_mapping = {
            'SOFT': 'SOFT', 'S': 'SOFT', 'C5': 'SOFT', 'C4': 'SOFT',
            'MEDIUM': 'MEDIUM', 'M': 'MEDIUM', 'C3': 'MEDIUM',
            'HARD': 'HARD', 'H': 'HARD', 'C2': 'HARD', 'C1': 'HARD',
            'INTERMEDIATE': 'INTERMEDIATE', 'I': 'INTERMEDIATE', 'INT': 'INTERMEDIATE',
            'WET': 'WET', 'W': 'WET'
        }
        df['Compound_standard'] = df['Compound_clean'].map(compound_mapping).fillna(df['Compound_clean'])
        
        # Compound hardness
        hardness_map = {'SOFT': 1, 'MEDIUM': 2, 'HARD': 3, 'INTERMEDIATE': 4, 'WET': 5}
        df['CompoundHardness'] = df['Compound_standard'].map(hardness_map).fillna(2)
        
        # Tire age features
        df['TyreLife_sqrt'] = np.sqrt(df['TyreLife'])
        df['TyreLife_log'] = np.log1p(df['TyreLife'])
        df['TyreLife_squared'] = df['TyreLife'] ** 2
        
        # === ENVIRONMENTAL FEATURES ===
        if 'TrackTemp' in df.columns and 'AirTemp' in df.columns:
            df['TempDelta'] = df['TrackTemp'] - df['AirTemp']
            df['TempDelta_abs'] = abs(df['TempDelta'])
            df['TempRatio'] = df['TrackTemp'] / (df['AirTemp'] + 1)
            
            # Temperature categories
            df['TrackTemp_hot'] = (df['TrackTemp'] > 45).astype(int)
            df['TrackTemp_cold'] = (df['TrackTemp'] < 30).astype(int)
            df['TrackTemp_optimal'] = ((df['TrackTemp'] >= 30) & (df['TrackTemp'] <= 45)).astype(int)
        
        if 'Humidity' in df.columns:
            df['Humidity_norm'] = df['Humidity'] / 100.0
            df['Humidity_high'] = (df['Humidity'] > 70).astype(int)
        
        if 'IsWet' in df.columns:
            df['IsWet_int'] = df['IsWet'].astype(int)
        
        # === CIRCUIT & SESSION FEATURES ===
        # Circuit encoding
        unique_circuits = sorted(df['circuit'].unique())
        circuit_encoding = {circuit: idx for idx, circuit in enumerate(unique_circuits)}
        df['Circuit_encoded'] = df['circuit'].map(circuit_encoding)
        
        # Session encoding
        session_mapping = {
            'Practice 1': 1, 'Practice 2': 2, 'Practice 3': 3,
            'Sprint Shootout': 3.5, 'Sprint': 4, 'Qualifying': 4.5, 'Race': 5
        }
        df['SessionType_encoded'] = df['session_type'].map(session_mapping).fillna(3)
        
        # Season progression (already included in comprehensive data)
        if 'SeasonProgress' not in df.columns:
            if 'RaceNumber' in df.columns:
                df['SeasonProgress'] = df['RaceNumber'] / df['RaceNumber'].max()
            else:
                df['SeasonProgress'] = 0.5
        
        # === PERFORMANCE FEATURES ===
        # Speed features
        speed_cols = ['SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST']
        for col in speed_cols:
            if col in df.columns and df[col].notna().sum() > 0:
                max_speed = df[col].max()
                if max_speed > 0:
                    df[f'{col}_norm'] = df[col] / max_speed
                    df[f'{col}_relative'] = df[col] / df.groupby('Driver')[col].transform('max')
        
        # Sector features
        if all(f'Sector{i}Time' in df.columns for i in [1, 2, 3]):
            total_sectors = df['Sector1Time'] + df['Sector2Time'] + df['Sector3Time']
            df['Sector1_proportion'] = df['Sector1Time'] / total_sectors
            df['Sector2_proportion'] = df['Sector2Time'] / total_sectors
            df['Sector3_proportion'] = df['Sector3Time'] / total_sectors
        
        # === DRIVER & TEAM FEATURES ===
        unique_drivers = sorted(df['Driver'].unique())
        unique_teams = sorted(df['Team'].unique())
        
        driver_encoding = {driver: idx for idx, driver in enumerate(unique_drivers)}
        team_encoding = {team: idx for idx, team in enumerate(unique_teams)}
        
        df['Driver_encoded'] = df['Driver'].map(driver_encoding)
        df['Team_encoded'] = df['Team'].map(team_encoding)
        
        # Driver performance baseline
        driver_avg_pace = df.groupby('Driver')['LapTime'].mean()
        df['DriverAvgPace'] = df['Driver'].map(driver_avg_pace)
        df['PaceVsDriverAvg'] = df['LapTime'] - df['DriverAvgPace']
        
        # === STINT PERFORMANCE ===
        stint_col = 'StintNumber' if 'StintNumber' in df.columns else 'Stint'
        if stint_col in df.columns:
            df = df.groupby(['Driver', 'circuit', 'session_type', stint_col]).apply(
                self.calculate_stint_metrics).reset_index(drop=True)
        
        # === INTERACTION FEATURES ===
        df['TyreLife_x_TrackTemp'] = df['TyreLife'] * df.get('TrackTemp', 40) / 40
        df['Compound_x_TempDelta'] = df['CompoundHardness'] * df.get('TempDelta', 0)
        df['TyreLife_x_Circuit'] = df['TyreLife'] * df['Circuit_encoded']
        df['Session_x_TyreLife'] = df['SessionType_encoded'] * df['TyreLife']
        
        # === ADVANCED FEATURES ===
        if 'EarlySeasonAdjustment' in df.columns:
            df['SeasonAdjusted_TyreLife'] = df['TyreLife'] * (1 + df['EarlySeasonAdjustment'])
        
        # Car generation effects
        if 'CarGeneration' in df.columns:
            gen_encoding = {gen: idx for idx, gen in enumerate(df['CarGeneration'].unique())}
            df['CarGeneration_encoded'] = df['CarGeneration'].map(gen_encoding)
        
        if 'TireGeneration' in df.columns:
            tire_gen_encoding = {gen: idx for idx, gen in enumerate(df['TireGeneration'].unique())}
            df['TireGeneration_encoded'] = df['TireGeneration'].map(tire_gen_encoding)
        
        # === TARGET VARIABLES ===
        stint_col = 'StintNumber' if 'StintNumber' in df.columns else 'Stint'
        if stint_col in df.columns:
            df = df.groupby(['Driver', 'circuit', 'session_type', stint_col]).apply(
                self.create_target_variables).reset_index(drop=True)
        
        print(f"Comprehensive {year} processed: {len(df)} laps, {len(df.columns)} features")
        return df
    
    def process_raw_data(self, raw_data, year):
        """Process raw laps and results data"""
        laps = raw_data['laps'].copy()
        results = raw_data['results'].copy()
        
        print(f"Processing raw data for {year}...")
        print(f"Initial laps: {len(laps)}")
        
        # Debug laptime format
        print(f"LapTime column sample values:")
        sample_laptimes = laps['LapTime'].dropna().head(10).tolist()
        print(f"  {sample_laptimes}")
        print(f"LapTime dtype: {laps['LapTime'].dtype}")
        
        # Merge laps with results
        merge_cols = ['DriverNumber', 'circuit', 'session_type']
        if year == 2025:
            merge_cols = [col for col in merge_cols if col in laps.columns and col in results.columns]
        
        if merge_cols:
            # Add results data to laps
            results_subset = results[['DriverNumber', 'TeamName', 'Position', 'GridPosition', 'Q1', 'Q2', 'Q3'] + 
                                   [col for col in ['circuit', 'session_type'] if col in results.columns]].copy()
            
            laps = pd.merge(laps, results_subset, on=merge_cols, how='left', suffixes=('', '_result'))
        
        # Convert lap times
        print(f"Converting lap times...")
        laps['LapTime_seconds'] = laps['LapTime'].apply(self.laptime_to_seconds)
        valid_laptimes = laps['LapTime_seconds'].notna().sum()
        print(f"Valid lap times after conversion: {valid_laptimes}")
        
        # If conversion failed, try different approaches
        if valid_laptimes == 0:
            print("Laptime conversion failed! Trying alternative methods...")
            
            # Check if LapTime is already in seconds (numeric)
            if laps['LapTime'].dtype in ['float64', 'int64']:
                print("LapTime appears to be numeric, using directly...")
                laps['LapTime_seconds'] = pd.to_numeric(laps['LapTime'], errors='coerce')
                valid_laptimes = laps['LapTime_seconds'].notna().sum()
                print(f"Valid lap times after numeric conversion: {valid_laptimes}")
            
            # If still failed, try pandas to_timedelta
            if valid_laptimes == 0:
                print("Trying pandas timedelta conversion...")
                try:
                    # Convert to timedelta and extract total seconds
                    laps['LapTime_seconds'] = pd.to_timedelta(laps['LapTime']).dt.total_seconds()
                    valid_laptimes = laps['LapTime_seconds'].notna().sum()
                    print(f"Valid lap times after timedelta conversion: {valid_laptimes}")
                except:
                    print("Timedelta conversion also failed!")
        
        # Debug data cleaning step by step
        print(f"Before cleaning: {len(laps)} laps")
        
        # Check each filter individually
        mask1 = laps['LapTime_seconds'].notna()
        print(f"After removing null lap times: {mask1.sum()} laps")
        
        if mask1.sum() > 0:
            mask2 = mask1 & (laps['LapTime_seconds'] > 60)
            print(f"After removing lap times <= 60s: {mask2.sum()} laps")
            
            mask3 = mask2 & (laps['LapTime_seconds'] < 300)
            print(f"After removing lap times >= 300s: {mask3.sum()} laps")
            
            mask4 = mask3 & laps['TyreLife'].notna()
            print(f"After removing null TyreLife: {mask4.sum()} laps")
            
            mask5 = mask4 & (laps['TyreLife'] > 0)
            print(f"After removing TyreLife <= 0: {mask5.sum()} laps")
            
            # Handle Deleted column more flexibly
            if 'Deleted' in laps.columns:
                # Check data type and values
                print(f"Deleted column dtype: {laps['Deleted'].dtype}")
                print(f"Deleted column unique values: {laps['Deleted'].unique()}")
                
                # Convert to boolean more flexibly
                if laps['Deleted'].dtype == 'object':
                    laps['Deleted'] = laps['Deleted'].astype(str).str.lower().isin(['true', '1', 'yes'])
                elif laps['Deleted'].dtype in ['int64', 'float64']:
                    laps['Deleted'] = (laps['Deleted'] == 1) | (laps['Deleted'] == True)
                
                mask6 = mask5 & (laps['Deleted'] == False)
                print(f"After removing deleted laps: {mask6.sum()} laps")
            else:
                mask6 = mask5
                print(f"No 'Deleted' column found, keeping all: {mask6.sum()} laps")
            
            # Apply the final filter
            laps = laps[mask6].copy()
        else:
            print("No valid lap times found, returning empty dataset")
            return None
            
        print(f"Final cleaned laps: {len(laps)}")
        
        if len(laps) == 0:
            print(f"WARNING: No laps remaining after cleaning for year {year}!")
            return None
        
        # Rename for consistency
        laps['LapTime'] = laps['LapTime_seconds']
        laps['Year'] = year
        
        # Add missing weather columns (will be filled with defaults)
        if 'TrackTemp' not in laps.columns:
            laps['TrackTemp'] = 40  # Default track temp
        if 'AirTemp' not in laps.columns:
            laps['AirTemp'] = 25   # Default air temp
        if 'Humidity' not in laps.columns:
            laps['Humidity'] = 50  # Default humidity
        
        # Process similar to comprehensive data
        df = self.process_comprehensive_data(laps, year)
        
        return df
    
    def calculate_stint_metrics(self, stint_group):
        """Calculate stint performance metrics"""
        stint_group = stint_group.sort_values('TyreLife').copy()
        
        if len(stint_group) < 2:
            return stint_group
        
        # Stint context
        stint_group['StintLength'] = len(stint_group)
        stint_group['StintProgression'] = stint_group['TyreLife'] / stint_group['TyreLife'].max()
        
        # Performance metrics
        best_lap = stint_group['LapTime'].min()
        stint_group['LapTimeVsBest'] = stint_group['LapTime'] - best_lap
        
        # Rolling averages
        for window in [3, 5]:
            stint_group[f'LapTime_rolling_{window}'] = stint_group['LapTime'].rolling(window, min_periods=1).mean()
            stint_group[f'LapTime_std_{window}'] = stint_group['LapTime'].rolling(window, min_periods=1).std().fillna(0)
        
        # Trend analysis
        stint_group['LapTime_trend'] = stint_group['LapTime'].diff().fillna(0)
        
        return stint_group
    
    def create_target_variables(self, stint_group):
        """Create comprehensive target variables"""
        stint_group = stint_group.sort_values('TyreLife').copy()
        
        if len(stint_group) < 3:
            stint_group['DegradationRate'] = 0
            stint_group['PaceDropOff'] = 0
            stint_group['DegradationPercent'] = 0
            stint_group['DegradationCategory'] = 'Low'
            return stint_group
        
        # Baseline performance
        baseline_time = stint_group['LapTime'].min()
        
        # Primary targets
        stint_group['DegradationRate'] = (stint_group['LapTime'] - baseline_time) / (stint_group['TyreLife'] + 1)
        stint_group['PaceDropOff'] = stint_group['LapTime'] - baseline_time
        stint_group['DegradationPercent'] = ((stint_group['LapTime'] - baseline_time) / baseline_time * 100).fillna(0)
        
        # Categorical target
        degradation_75th = stint_group['DegradationRate'].quantile(0.75)
        degradation_25th = stint_group['DegradationRate'].quantile(0.25)
        
        conditions = [
            stint_group['DegradationRate'] <= degradation_25th,
            stint_group['DegradationRate'] <= degradation_75th,
            stint_group['DegradationRate'] > degradation_75th
        ]
        choices = ['Low', 'Medium', 'High']
        stint_group['DegradationCategory'] = np.select(conditions, choices, default='Medium')
        
        return stint_group
    
    def laptime_to_seconds(self, laptime_str):
        """Convert laptime string to seconds with better error handling"""
        if pd.isna(laptime_str) or laptime_str == '' or laptime_str is None:
            return np.nan
        
        try:
            # Convert to string first
            laptime_str = str(laptime_str).strip()
            
            if laptime_str == '' or laptime_str.lower() == 'nan':
                return np.nan
            
            # Handle time format (MM:SS.mmm or M:SS.mmm)
            if ':' in laptime_str:
                parts = laptime_str.split(':')
                if len(parts) == 2:
                    minutes = int(parts[0])
                    seconds = float(parts[1])
                    total_seconds = minutes * 60 + seconds
                    
                    # Sanity check
                    if 50 <= total_seconds <= 400:  # Reasonable F1 lap time range
                        return total_seconds
                    else:
                        return np.nan
                else:
                    return np.nan
            else:
                # Try to convert directly to float (already in seconds)
                total_seconds = float(laptime_str)
                
                # Sanity check
                if 50 <= total_seconds <= 400:
                    return total_seconds
                else:
                    return np.nan
                    
        except (ValueError, TypeError, AttributeError):
            return np.nan
    
    def process_all_years(self, years=[2022, 2023, 2024, 2025], use_comprehensive_when_available=True):
        """Process all years using the best available data format"""
        print(f"Processing tire degradation data for years: {years}")
        print(f"Base path: {self.base_path}")
        
        for year in years:
            print(f"\n=== Processing Year {year} ===")
            
            # Detect available data type
            data_type, file_info = self.detect_data_type(year)
            
            if data_type == "none":
                print(f"No data found for {year}")
                continue
            
            print(f"Data type detected: {data_type}")
            
            # Load and process based on data type
            if data_type == "comprehensive" and use_comprehensive_when_available:
                raw_data = self.load_comprehensive_data(file_info, year)
                if raw_data is not None:
                    processed_data = self.process_comprehensive_data(raw_data, year)
                else:
                    continue
            else:
                # Use raw data
                raw_data = self.load_raw_data(file_info, year)
                if raw_data is not None:
                    processed_data = self.process_raw_data(raw_data, year)
                else:
                    continue
            
            if processed_data is not None and len(processed_data) > 0:
                self.yearly_data[year] = processed_data
                circuits = processed_data['circuit'].unique() if 'circuit' in processed_data.columns else ['Unknown']
                sessions = processed_data['session_type'].unique() if 'session_type' in processed_data.columns else ['Unknown']
                print(f"✓ Year {year}: {len(processed_data)} laps, {len(circuits)} circuits, {len(sessions)} sessions")
            else:
                print(f"✗ Year {year}: No valid data processed")
        
        if not self.yearly_data:
            print("Warning: No data was successfully processed for any year!")
        
        return self.yearly_data
    
    def combine_all_years(self):
        """Combine all processed years"""
        if not self.yearly_data:
            print("No yearly data to combine!")
            return None
        
        print(f"\n=== Combining {len(self.yearly_data)} years ===")
        
        all_data = []
        for year, data in self.yearly_data.items():
            print(f"Year {year}: {len(data)} laps")
            all_data.append(data)
        
        # Combine all years
        print(f"Concatenating {len(all_data)} datasets...")
        self.combined_dataset = pd.concat(all_data, ignore_index=True)
        print(f"After concatenation: {len(self.combined_dataset)} laps")
        
        # Check years before cleaning
        years_before = sorted(self.combined_dataset['Year'].unique())
        print(f"Years before cleaning: {years_before}")
        
        # Final cleaning
        print(f"Starting final cleaning...")
        before_cleaning = len(self.combined_dataset)
        self.combined_dataset = self.final_cleaning(self.combined_dataset)
        after_cleaning = len(self.combined_dataset)
        print(f"Final cleaning: {before_cleaning} → {after_cleaning} laps ({before_cleaning - after_cleaning} removed)")
        
        # Check years after cleaning
        years_after = sorted(self.combined_dataset['Year'].unique())
        print(f"Years after cleaning: {years_after}")
        
        # Warning if we lost entire years
        if len(years_after) < len(years_before):
            lost_years = [y for y in years_before if y not in years_after]
            print(f"⚠️  WARNING: Lost entire years during cleaning: {lost_years}")
            print(f"This suggests outlier removal was too aggressive!")
        
        # Define feature columns
        exclude_cols = ['Year', 'Driver', 'Team', 'circuit', 'session_type', 'LapTime', 'Season',
                       'DriverFullName', 'TeamCode', 'Circuit', 'Race', 'EventName', 'Session',
                       'RaceDate', 'Timestamp', 'DegradationRate', 'PaceDropOff', 
                       'DegradationPercent', 'DegradationCategory']
        self.feature_columns = [col for col in self.combined_dataset.columns 
                               if col not in exclude_cols and not col.endswith('_result')]
        
        print(f"\nFinal Combined Dataset:")
        print(f"Total laps: {len(self.combined_dataset):,}")
        print(f"Features: {len(self.feature_columns)}")
        print(f"Years: {sorted(self.combined_dataset['Year'].unique())}")
        
        if 'circuit' in self.combined_dataset.columns:
            print(f"Circuits: {len(self.combined_dataset['circuit'].unique())}")
        
        return self.combined_dataset
    
    def final_cleaning(self, df):
        """Final data cleaning and outlier removal (much less aggressive)"""
        print(f"Starting final cleaning with {len(df)} rows")
        
        initial_rows = len(df)
        
        # Only remove extreme outliers from key performance metrics
        critical_cols = ['LapTime', 'DegradationRate', 'PaceDropOff']
        
        for col in critical_cols:
            if col in df.columns:
                before = len(df)
                # Only remove extreme outliers (0.1% and 99.9% - very conservative)
                Q1 = df[col].quantile(0.001)
                Q99 = df[col].quantile(0.999)
                df = df[(df[col] >= Q1) & (df[col] <= Q99)]
                after = len(df)
                
                if before != after:
                    print(f"  {col}: removed {before - after} extreme outliers ({before} → {after})")
        
        # Don't remove outliers from other columns - they might be valid data
        # Just fill NaN values
        
        print(f"Conservative outlier removal: {initial_rows} → {len(df)} rows")
        
        # Fill any remaining NaN values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        nan_before = df.isnull().sum().sum()
        
        for col in numeric_cols:
            if df[col].isna().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        nan_after = df.isnull().sum().sum()
        if nan_before > 0:
            print(f"Filled {nan_before} NaN values")
        
        print(f"Final cleaning complete: {len(df)} rows remaining")
        return df
    
    def get_dataset_summary(self):
        """Comprehensive dataset summary"""
        if self.combined_dataset is None:
            print("No combined dataset available!")
            return
        
        df = self.combined_dataset
        
        print(f"\n{'='*70}")
        print(f"F1 TIRE DEGRADATION PREDICTION DATASET - SUMMARY")
        print(f"{'='*70}")
        
        print(f"Total laps: {len(df):,}")
        print(f"Total features: {len(self.feature_columns)}")
        print(f"Years: {sorted(df['Year'].unique())}")
        
        if 'circuit' in df.columns:
            circuits = sorted(df['circuit'].unique())
            print(f"Circuits ({len(circuits)}): {circuits}")
        
        if 'session_type' in df.columns:
            sessions = sorted(df['session_type'].unique())
            print(f"Session types: {sessions}")
        
        print(f"Drivers: {len(df['Driver'].unique())}")
        print(f"Teams: {len(df['Team'].unique())}")
        
        # Data distribution by year
        print(f"\nData distribution by year:")
        year_counts = df['Year'].value_counts().sort_index()
        for year, count in year_counts.items():
            print(f"  {year}: {count:,} laps")
        
        # Target variable summary
        if 'DegradationRate' in df.columns:
            print(f"\nTarget Variables:")
            targets = ['DegradationRate', 'PaceDropOff', 'DegradationPercent']
            for target in targets:
                if target in df.columns:
                    print(f"  {target}: mean={df[target].mean():.4f}, std={df[target].std():.4f}")
            
            if 'DegradationCategory' in df.columns:
                print(f"  DegradationCategory: {dict(df['DegradationCategory'].value_counts())}")
        
        return df.describe()
    
    def save_processed_data(self, output_dir=None, filename_prefix="f1_tire_degradation"):
        """Save the processed dataset and related files for ML training"""
        if self.combined_dataset is None:
            print("No processed dataset to save!")
            return None
        
        # Create output directory
        if output_dir is None:
            output_dir = os.path.join(self.base_path, "processed_ml_data")
        
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        
        # 1. Save main dataset
        main_file = os.path.join(output_dir, f"{filename_prefix}_complete_dataset.csv")
        self.combined_dataset.to_csv(main_file, index=False)
        saved_files['main_dataset'] = main_file
        print(f"✓ Saved main dataset: {main_file}")
        print(f"  {len(self.combined_dataset):,} rows × {len(self.combined_dataset.columns)} columns")
        
        # 2. Save feature-only dataset (for easy model training)
        if self.feature_columns:
            features_file = os.path.join(output_dir, f"{filename_prefix}_features_only.csv")
            feature_data = self.combined_dataset[self.feature_columns + ['DegradationRate', 'DegradationCategory']]
            feature_data.to_csv(features_file, index=False)
            saved_files['features_dataset'] = features_file
            print(f"✓ Saved features dataset: {features_file}")
            print(f"  {len(self.feature_columns)} features + 2 targets")
        
        # 3. Save feature list
        features_list_file = os.path.join(output_dir, f"{filename_prefix}_feature_list.txt")
        with open(features_list_file, 'w') as f:
            f.write("F1 Tire Degradation Prediction - Feature List\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Features: {len(self.feature_columns)}\n\n")
            
            # Categorize features
            feature_categories = {
                'Tire Features': [col for col in self.feature_columns if any(x in col.lower() for x in ['tyre', 'tire', 'compound', 'fresh'])],
                'Environmental': [col for col in self.feature_columns if any(x in col.lower() for x in ['temp', 'humidity', 'wet'])],
                'Performance': [col for col in self.feature_columns if any(x in col.lower() for x in ['laptime', 'speed', 'pace', 'sector'])],
                'Context': [col for col in self.feature_columns if any(x in col.lower() for x in ['session', 'circuit', 'driver', 'team', 'stint'])],
                'Advanced': [col for col in self.feature_columns if any(x in col.lower() for x in ['rolling', 'trend', 'interaction', 'x_'])]
            }
            
            for category, features in feature_categories.items():
                if features:
                    f.write(f"{category} ({len(features)} features):\n")
                    for feature in sorted(features):
                        f.write(f"  - {feature}\n")
                    f.write("\n")
        
        saved_files['feature_list'] = features_list_file
        print(f"✓ Saved feature list: {features_list_file}")
        
        # 4. Save dataset summary and metadata
        summary_file = os.path.join(output_dir, f"{filename_prefix}_dataset_info.txt")
        with open(summary_file, 'w') as f:
            f.write("F1 Tire Degradation Prediction Dataset - Summary\n")
            f.write("=" * 60 + "\n\n")
            
            df = self.combined_dataset
            f.write(f"Dataset Size: {len(df):,} laps\n")
            f.write(f"Total Features: {len(self.feature_columns)}\n")
            f.write(f"Years Covered: {sorted(df['Year'].unique())}\n")
            
            if 'circuit' in df.columns:
                circuits = sorted(df['circuit'].unique())
                f.write(f"Circuits ({len(circuits)}): {circuits}\n")
            
            if 'session_type' in df.columns:
                sessions = sorted(df['session_type'].unique())
                f.write(f"Session Types: {sessions}\n")
            
            f.write(f"Drivers: {len(df['Driver'].unique())}\n")
            f.write(f"Teams: {len(df['Team'].unique())}\n\n")
            
            # Data distribution
            f.write("Data Distribution by Year:\n")
            year_counts = df['Year'].value_counts().sort_index()
            for year, count in year_counts.items():
                f.write(f"  {year}: {count:,} laps\n")
            
            # Target variables info
            if 'DegradationRate' in df.columns:
                f.write(f"\nTarget Variables:\n")
                targets = ['DegradationRate', 'PaceDropOff', 'DegradationPercent']
                for target in targets:
                    if target in df.columns:
                        f.write(f"  {target}: mean={df[target].mean():.4f}, std={df[target].std():.4f}\n")
                
                if 'DegradationCategory' in df.columns:
                    f.write(f"  DegradationCategory distribution: {dict(df['DegradationCategory'].value_counts())}\n")
            
            # Recommended train/test splits
            f.write(f"\nRecommended Train/Test Splits:\n")
            f.write(f"  Time-based: Train on 2022-2024, Test on 2025\n")
            f.write(f"  Random: 80/20 split with stratification on DegradationCategory\n")
            f.write(f"  Cross-validation: 5-fold with grouping by Driver or Circuit\n")
        
        saved_files['summary'] = summary_file
        print(f"✓ Saved dataset summary: {summary_file}")
        
        # 5. Save separate train/test files (time-based split)
        if len(df['Year'].unique()) > 1:
            years = sorted(df['Year'].unique())
            test_year = max(years)
            train_years = [y for y in years if y != test_year]
            
            # Training data (all years except the latest)
            train_data = df[df['Year'].isin(train_years)]
            train_file = os.path.join(output_dir, f"{filename_prefix}_train_data.csv")
            train_data.to_csv(train_file, index=False)
            saved_files['train_data'] = train_file
            print(f"✓ Saved training data: {train_file}")
            print(f"  {len(train_data):,} laps from years {train_years}")
            
            # Test data (latest year)
            test_data = df[df['Year'] == test_year]
            test_file = os.path.join(output_dir, f"{filename_prefix}_test_data.csv")
            test_data.to_csv(test_file, index=False)
            saved_files['test_data'] = test_file
            print(f"✓ Saved test data: {test_file}")
            print(f"  {len(test_data):,} laps from year {test_year}")
        
        # 6. Save data dictionary
        data_dict_file = os.path.join(output_dir, f"{filename_prefix}_data_dictionary.csv")
        
        # Create data dictionary
        data_dict = []
        for col in self.combined_dataset.columns:
            col_info = {
                'Column': col,
                'Type': str(self.combined_dataset[col].dtype),
                'Non_Null_Count': self.combined_dataset[col].count(),
                'Null_Count': self.combined_dataset[col].isnull().sum(),
                'Unique_Values': self.combined_dataset[col].nunique(),
                'Description': self.get_column_description(col)
            }
            
            if self.combined_dataset[col].dtype in ['int64', 'float64']:
                col_info.update({
                    'Min': self.combined_dataset[col].min(),
                    'Max': self.combined_dataset[col].max(),
                    'Mean': self.combined_dataset[col].mean(),
                    'Std': self.combined_dataset[col].std()
                })
            
            data_dict.append(col_info)
        
        pd.DataFrame(data_dict).to_csv(data_dict_file, index=False)
        saved_files['data_dictionary'] = data_dict_file
        print(f"✓ Saved data dictionary: {data_dict_file}")
        
        # Summary of saved files
        print(f"\n📁 All files saved to: {output_dir}")
        print(f"📊 Files ready for ML training:")
        for file_type, file_path in saved_files.items():
            print(f"  - {file_type}: {os.path.basename(file_path)}")
        
        return saved_files
    
    def get_column_description(self, col):
        """Get description for a column"""
        descriptions = {
            'LapTime': 'Lap time in seconds',
            'TyreLife': 'Number of laps completed on current tire set',
            'Compound': 'Tire compound type (SOFT/MEDIUM/HARD/etc.)',
            'TrackTemp': 'Track surface temperature in Celsius',
            'AirTemp': 'Ambient air temperature in Celsius',
            'DegradationRate': 'Target: Tire degradation rate (seconds per lap)',
            'DegradationCategory': 'Target: Categorical degradation level (Low/Medium/High)',
            'Circuit_encoded': 'Numeric encoding of circuit/track',
            'Driver_encoded': 'Numeric encoding of driver',
            'Team_encoded': 'Numeric encoding of team',
            'CompoundHardness': 'Tire compound hardness (1=Soft, 2=Medium, 3=Hard)',
            'TempDelta': 'Track temperature minus air temperature',
            'SessionType_encoded': 'Numeric encoding of session type'
        }
        
        # Pattern-based descriptions
        if 'encoded' in col:
            return 'Numeric encoding of categorical variable'
        elif 'norm' in col:
            return 'Normalized feature (0-1 scale)'
        elif 'rolling' in col:
            return 'Rolling window average'
        elif 'x_' in col:
            return 'Interaction feature (product of two variables)'
        elif 'Speed' in col:
            return 'Speed measurement at specific track location'
        elif 'Sector' in col:
            return 'Sector time or proportion'
        elif 'Stint' in col:
            return 'Stint-related metric'
        
        return descriptions.get(col, 'Feature engineered from F1 telemetry data')

# Main execution function
def process_f1_tire_data(base_path, years=[2022, 2023, 2024, 2025], use_comprehensive=True, save_data=True, output_dir=None):
    """Main function to process F1 tire degradation data"""
    
    engineer = F1FlexibleTireEngineer(base_path)
    
    # Process all years
    engineer.process_all_years(years, use_comprehensive_when_available=use_comprehensive)
    
    # Combine all years
    final_dataset = engineer.combine_all_years()
    
    # Get summary
    if final_dataset is not None:
        engineer.get_dataset_summary()
        
        # Save processed data
        if save_data:
            print(f"\n💾 Saving processed data...")
            saved_files = engineer.save_processed_data(output_dir=output_dir)
            return engineer, final_dataset, saved_files
    else:
        print("No data available for summary.")
        return engineer, None, None
    
    return engineer, final_dataset, None

# Example usage:
if __name__ == "__main__":
    base_path = "/Users/sid/Downloads/F1_tire_degradation_strategy_predictor"
    
    # Process all available years
    print(f"\n🏎️ Starting F1 tire degradation data processing...")
    engineer, dataset, saved_files = process_f1_tire_data(
        base_path, 
        years=[2022, 2023, 2024, 2025],
        use_comprehensive=True,  # Use processed files when available
        save_data=True,  # Save the processed data
        output_dir=None  # Will create processed_ml_data folder
    )
    
    print(f"\n🏁 Processing complete!")
    print(f"Final dataset: {len(dataset):,} laps ready for ML training!")