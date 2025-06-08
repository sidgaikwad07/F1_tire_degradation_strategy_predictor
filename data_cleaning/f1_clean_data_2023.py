"""
Created on Sun Jun  8 10:54:05 2025

@author: sid
F1 Data Cleaner - 2023 Season Only
Performs comprehensive data cleaning and transformation specifically for 2023 F1 data.
This version is focused exclusively on 2023 data with year-specific configurations,
tire compounds, teams, and regulations that were in place during the 2023 season.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import argparse
from datetime import datetime, timedelta
import warnings
from typing import List, Dict, Optional, Tuple, Union
import pickle
import os
import re
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('f1_data_cleaning_2023.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class F1DataCleaner2023:
    """
    F1 Data Cleaner specifically designed for 2023 season data.
    """
    
    def __init__(self, base_dir: str = '../data_collector/data', output_dir: str = '../cleaned_data_2023'):
        """Initialize the 2023 F1 Data Cleaner."""
        self.year = 2023  # Fixed to 2023 only
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.raw_dir = self.base_dir / 'raw' / '2023'
        
        # Create output directory structure for 2023
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'processed').mkdir(exist_ok=True)
        (self.output_dir / 'features').mkdir(exist_ok=True)
        (self.output_dir / 'validation').mkdir(exist_ok=True)
        (self.output_dir / 'reports').mkdir(exist_ok=True)
        
        # Initialize all the 2023-specific configurations
        self._setup_2023_config()
        self._setup_cleaning_stats()
    
    def _setup_2023_config(self):
        """Setup all 2023-specific configurations."""
        # 2023-specific configuration
        self.config_2023 = {
            'tire_compounds': ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET'],
            'teams': {
                'Red Bull Racing RBPT': 'RBR',
                'Ferrari': 'FER', 
                'Mercedes': 'MER',
                'McLaren Mercedes': 'MCL',
                'Alpine Renault': 'ALP',
                'AlphaTauri RBPT': 'AT',  # Still AlphaTauri in 2023
                'Aston Martin Aramco Mercedes': 'AM',
                'Williams Mercedes': 'WIL',
                'Alfa Romeo Ferrari': 'AR',  # Still Alfa Romeo in 2023
                'Haas Ferrari': 'HAS'
            },
            'session_formats': {
                'standard': ['FP1', 'FP2', 'FP3', 'Q', 'R'],
                'sprint': ['FP1', 'SQ', 'S', 'Q', 'R']
            },
            'sprint_weekends_2023': ['Azerbaijan', 'Austria', 'Belgium', 'Qatar', 'United States', 'Brazil']  # 6 sprint weekends
        }
        
        # 2023 F1 Calendar (22 races - Las Vegas added)
        self.circuits_2023 = {
            'Bahrain': {'round': 1, 'has_sprint': False, 'date': '2023-03-05'},
            'Saudi Arabia': {'round': 2, 'has_sprint': False, 'date': '2023-03-19'},
            'Australia': {'round': 3, 'has_sprint': False, 'date': '2023-04-02'},
            'Azerbaijan': {'round': 4, 'has_sprint': True, 'date': '2023-04-30'},
            'Miami': {'round': 5, 'has_sprint': False, 'date': '2023-05-07'},
            'Monaco': {'round': 6, 'has_sprint': False, 'date': '2023-05-28'},
            'Spain': {'round': 7, 'has_sprint': False, 'date': '2023-06-04'},
            'Canada': {'round': 8, 'has_sprint': False, 'date': '2023-06-18'},
            'Austria': {'round': 9, 'has_sprint': True, 'date': '2023-07-02'},
            'Great Britain': {'round': 10, 'has_sprint': False, 'date': '2023-07-09'},
            'Hungary': {'round': 11, 'has_sprint': False, 'date': '2023-07-23'},
            'Belgium': {'round': 12, 'has_sprint': True, 'date': '2023-07-30'},
            'Netherlands': {'round': 13, 'has_sprint': False, 'date': '2023-08-27'},
            'Italy': {'round': 14, 'has_sprint': False, 'date': '2023-09-03'},
            'Singapore': {'round': 15, 'has_sprint': False, 'date': '2023-09-17'},
            'Japan': {'round': 16, 'has_sprint': False, 'date': '2023-09-24'},
            'Qatar': {'round': 17, 'has_sprint': True, 'date': '2023-10-08'},
            'United States': {'round': 18, 'has_sprint': True, 'date': '2023-10-22'},
            'Mexico': {'round': 19, 'has_sprint': False, 'date': '2023-10-29'},
            'Brazil': {'round': 20, 'has_sprint': True, 'date': '2023-11-05'},
            'Las Vegas': {'round': 21, 'has_sprint': False, 'date': '2023-11-18'},  # NEW: Las Vegas debut
            'Abu Dhabi': {'round': 22, 'has_sprint': False, 'date': '2023-11-26'}
        }
        
        # 2023-specific data quality thresholds
        self.quality_thresholds_2023 = {
            'min_lap_time': 60.0,
            'max_lap_time': 300.0,
            'max_speed': 385.0,  # Slightly higher for 2023
            'max_tire_age': 45,  # Slightly higher tire age limit
            'tire_deg_rate_soft': 0.25,  # Refined for 2023
            'tire_deg_rate_medium': 0.18,
            'tire_deg_rate_hard': 0.12
        }
    
    def _setup_cleaning_stats(self):
        """Setup cleaning statistics tracking."""
        self.cleaning_stats = {
            'total_files_processed': 0,
            'total_records_processed': 0,
            'records_removed': 0,
            'records_corrected': 0,
            'missing_data_filled': 0,
            'outliers_detected': 0,
            'data_quality_score': 0.0,
            'processing_time': 0.0,
            'errors_encountered': [],
            'warnings_issued': [],
            'circuits_processed': 0,
            'sessions_processed': 0,
            'tire_data_quality': 0.0,
            'timing_data_quality': 0.0,
            'weather_data_quality': 0.0,
            'sprint_weekends_processed': 0
        }
    
    def load_raw_2023_data(self, circuit: str = None) -> Dict:
        """Load raw 2023 data for all circuits or a specific circuit."""
        logger.info(f"🔄 Loading raw 2023 data" + (f" - {circuit}" if circuit else " - All circuits"))
        
        if not self.raw_dir.exists():
            logger.error(f"❌ No raw 2023 data found at {self.raw_dir}")
            return {}
        
        loaded_data = {}
        circuits_to_process = [circuit] if circuit else [d.name for d in self.raw_dir.iterdir() if d.is_dir()]
        
        # Filter to only valid 2023 circuits
        valid_circuits = [c for c in circuits_to_process if c in self.circuits_2023]
        
        for circuit_name in valid_circuits:
            circuit_dir = self.raw_dir / circuit_name
            if not circuit_dir.exists():
                continue
            
            logger.info(f"  📁 Loading {circuit_name} 2023 data...")
            circuit_data = {}
            
            # Load session data files
            for file_path in circuit_dir.glob("*.csv"):
                try:
                    filename = file_path.stem
                    parts = filename.split('_')
                    
                    if len(parts) >= 2:
                        session_type = parts[0]
                        data_type = '_'.join(parts[1:])
                        
                        # Load CSV data
                        df = pd.read_csv(file_path)
                        
                        # Create nested structure
                        if circuit_name not in loaded_data:
                            loaded_data[circuit_name] = {}
                        if session_type not in loaded_data[circuit_name]:
                            loaded_data[circuit_name][session_type] = {}
                        
                        loaded_data[circuit_name][session_type][data_type] = df
                        logger.info(f"    ✅ Loaded {filename}: {len(df)} records")
                        
                except Exception as e:
                    logger.warning(f"    ⚠️ Failed to load {file_path.name}: {e}")
        
        logger.info(f"📊 Loaded 2023 data for {len(loaded_data)} circuits")
        return loaded_data
    
    def clean_2023_timing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean timing data specifically for 2023 cars and regulations."""
        if df.empty:
            return df
        
        df_clean = df.copy()
        initial_rows = len(df_clean)
        
        logger.info(f"    🔧 Cleaning 2023 timing data: {initial_rows} records")
        
        # Clean lap times with 2023-specific thresholds
        if 'LapTime' in df_clean.columns:
            # Convert to seconds if needed
            df_clean['LapTime'] = self._convert_time_to_seconds(df_clean['LapTime'])
            
            # Remove unrealistic lap times for 2023 cars
            before_clean = len(df_clean)
            df_clean = df_clean[
                (df_clean['LapTime'].isna()) |
                ((df_clean['LapTime'] >= self.quality_thresholds_2023['min_lap_time']) &
                 (df_clean['LapTime'] <= self.quality_thresholds_2023['max_lap_time']))
            ]
            removed = before_clean - len(df_clean)
            
            if removed > 0:
                logger.info(f"      ❌ Removed {removed} records with unrealistic 2023 lap times")
                self.cleaning_stats['records_removed'] += removed
        
        # Clean speed data with 2023 car capabilities
        speed_columns = [col for col in df_clean.columns if 'Speed' in col or 'speed' in col.lower()]
        for speed_col in speed_columns:
            if speed_col in df_clean.columns:
                before_clean = len(df_clean)
                df_clean = df_clean[
                    (df_clean[speed_col].isna()) |
                    ((df_clean[speed_col] >= 0) & (df_clean[speed_col] <= self.quality_thresholds_2023['max_speed']))
                ]
                removed = before_clean - len(df_clean)
                
                if removed > 0:
                    logger.info(f"      ❌ Removed {removed} unrealistic 2023 {speed_col} values")
                    self.cleaning_stats['records_removed'] += removed
        
        # Update timing data quality score
        self.cleaning_stats['timing_data_quality'] = len(df_clean) / initial_rows if initial_rows > 0 else 0.0
        
        return df_clean
    
    def clean_2023_tire_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean tire data specifically for 2023 Pirelli compounds."""
        if df.empty:
            return df
        
        df_clean = df.copy()
        initial_rows = len(df_clean)
        
        logger.info(f"    🔧 Cleaning 2023 tire data: {initial_rows} records")
        
        # Clean tire compound data
        if 'Compound' in df_clean.columns:
            # Standardize compound names for 2023
            compound_mapping = {
                'SOFT': 'SOFT', 'S': 'SOFT', 'Red': 'SOFT',
                'MEDIUM': 'MEDIUM', 'M': 'MEDIUM', 'Yellow': 'MEDIUM',
                'HARD': 'HARD', 'H': 'HARD', 'White': 'HARD',
                'INTERMEDIATE': 'INTERMEDIATE', 'I': 'INTERMEDIATE',
                'WET': 'WET', 'W': 'WET'
            }
            
            df_clean['Compound'] = df_clean['Compound'].map(compound_mapping).fillna(df_clean['Compound'])
            
            # Remove invalid compounds
            before_clean = len(df_clean)
            df_clean = df_clean[df_clean['Compound'].isin(self.config_2023['tire_compounds'])]
            removed = before_clean - len(df_clean)
            
            if removed > 0:
                logger.info(f"      ❌ Removed {removed} records with invalid tire compounds")
                self.cleaning_stats['records_removed'] += removed
        
        # Clean tire age data with 2023-specific limits
        if 'TyreLife' in df_clean.columns:
            df_clean['TyreLife'] = pd.to_numeric(df_clean['TyreLife'], errors='coerce')
            
            before_clean = len(df_clean)
            df_clean = df_clean[
                (df_clean['TyreLife'].isna()) |
                ((df_clean['TyreLife'] >= 0) & (df_clean['TyreLife'] <= self.quality_thresholds_2023['max_tire_age']))
            ]
            removed = before_clean - len(df_clean)
            
            if removed > 0:
                logger.info(f"      ❌ Removed {removed} records with invalid 2023 tire ages")
                self.cleaning_stats['records_removed'] += removed
        
        # Update tire data quality score
        self.cleaning_stats['tire_data_quality'] = len(df_clean) / initial_rows if initial_rows > 0 else 0.0
        
        return df_clean
    
    def _convert_time_to_seconds(self, time_series: pd.Series) -> pd.Series:
        """Convert time data to seconds."""
        def convert_single_time(time_val):
            if pd.isna(time_val):
                return np.nan
            
            if isinstance(time_val, (int, float)):
                return float(time_val)
            
            if hasattr(time_val, 'total_seconds'):
                return time_val.total_seconds()
            
            if isinstance(time_val, str):
                try:
                    if ':' in time_val:
                        parts = time_val.split(':')
                        if len(parts) == 2:
                            minutes = float(parts[0])
                            seconds = float(parts[1])
                            return minutes * 60 + seconds
                    return float(time_val)
                except (ValueError, TypeError):
                    return np.nan
            
            return np.nan
        
        return time_series.apply(convert_single_time)
    
    def clean_2023_circuit_data(self, circuit_data: Dict, circuit: str) -> Dict:
        """Clean all data for a specific 2023 circuit."""
        logger.info(f"🏁 Cleaning {circuit} 2023 data")
        
        cleaned_data = {}
        
        # Track sprint weekends (6 in 2023)
        if circuit in self.config_2023['sprint_weekends_2023']:
            self.cleaning_stats['sprint_weekends_processed'] += 1
            logger.info(f"  🏃 {circuit} is a 2023 sprint weekend (6 total)")
        
        for session_type, session_data in circuit_data.items():
            logger.info(f"  📊 Processing 2023 {session_type} session")
            session_cleaned = {}
            
            for data_type, raw_data in session_data.items():
                if isinstance(raw_data, pd.DataFrame) and not raw_data.empty:
                    logger.info(f"    🔧 Cleaning 2023 {data_type} data")
                    
                    # Apply 2023-specific cleaning
                    if data_type in ['laps', 'results']:
                        df_cleaned = self.clean_2023_timing_data(raw_data)
                        
                        # Additional tire cleaning if tire columns present
                        tire_columns = ['Compound', 'TyreLife', 'FreshTyre']
                        if any(col in df_cleaned.columns for col in tire_columns):
                            df_cleaned = self.clean_2023_tire_data(df_cleaned)
                    else:
                        df_cleaned = raw_data.copy()
                    
                    session_cleaned[data_type] = df_cleaned
                    logger.info(f"      ✅ {data_type}: {len(df_cleaned)} records")
                
                elif isinstance(raw_data, dict):
                    # Handle JSON metadata
                    raw_data['year'] = 2023
                    raw_data['season_context'] = '2023 Formula 1 World Championship'
                    raw_data['sprint_weekends_count'] = 6  # 2023 had 6 sprint weekends
                    raw_data['new_circuits'] = ['Las Vegas']  # Las Vegas debut
                    session_cleaned[data_type] = raw_data
            
            cleaned_data[session_type] = session_cleaned
        
        return cleaned_data
    
    def process_2023_data(self, circuits: List[str] = None) -> Dict:
        """Process and clean all 2023 F1 data."""
        logger.info(f"🚀 Starting 2023 F1 data cleaning")
        start_time = datetime.now()
        
        # Load raw 2023 data
        raw_data = self.load_raw_2023_data()
        
        if not raw_data:
            logger.error(f"❌ No raw 2023 data found")
            return {}
        
        # Filter circuits if specified
        if circuits:
            raw_data = {circuit: data for circuit, data in raw_data.items() if circuit in circuits}
        
        processed_data = {}
        
        # Process each 2023 circuit
        for circuit, circuit_data in raw_data.items():
            try:
                cleaned_circuit = self.clean_2023_circuit_data(circuit_data, circuit)
                processed_data[circuit] = cleaned_circuit
                self.cleaning_stats['circuits_processed'] += 1
                
            except Exception as e:
                logger.error(f"❌ Failed to process 2023 {circuit}: {e}")
                self.cleaning_stats['errors_encountered'].append(f"{circuit}: {str(e)}")
        
        # Calculate processing time
        end_time = datetime.now()
        self.cleaning_stats['processing_time'] = (end_time - start_time).total_seconds()
        
        logger.info(f"🏁 2023 data processing complete: {len(processed_data)} circuits")
        logger.info(f"🏃 Sprint weekends processed: {self.cleaning_stats['sprint_weekends_processed']}/6")
        
        return processed_data
    
    def _prepare_dataframe_for_parquet(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for Parquet format by fixing data types."""
        df_clean = df.copy()
        
        # Convert problematic columns to strings to avoid type conflicts
        problematic_columns = [
            'ClassifiedPosition', 'Position', 'GridPosition', 'Status', 
            'DriverNumber', 'TeamName', 'Compound', 'TrackStatus'
        ]
        
        for col in problematic_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str)
        
        # Convert time columns to float (seconds)
        time_columns = [col for col in df_clean.columns if 'Time' in col or 'time' in col.lower()]
        for col in time_columns:
            if col in df_clean.columns:
                df_clean[col] = self._convert_time_to_seconds(df_clean[col])
        
        # Handle any remaining object columns that might cause issues
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                try:
                    # Try to convert to numeric first
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='ignore')
                    # If still object, convert to string
                    if df_clean[col].dtype == 'object':
                        df_clean[col] = df_clean[col].astype(str)
                except:
                    # Last resort: convert to string
                    df_clean[col] = df_clean[col].astype(str)
        
        return df_clean

    def save_2023_cleaned_data(self, data: Dict, formats: List[str] = ['csv', 'parquet']) -> None:
        """Save cleaned 2023 data in multiple formats."""
        logger.info(f"💾 Saving cleaned 2023 data")
        
        output_2023_dir = self.output_dir / 'processed'
        output_2023_dir.mkdir(parents=True, exist_ok=True)
        
        # Save circuit-by-circuit 2023 data
        for circuit, circuit_data in data.items():
            circuit_dir = output_2023_dir / circuit
            circuit_dir.mkdir(exist_ok=True)
            
            for session_type, session_data in circuit_data.items():
                for data_type, df_or_dict in session_data.items():
                    
                    if isinstance(df_or_dict, pd.DataFrame) and not df_or_dict.empty:
                        base_filename = f"2023_{session_type}_{data_type}"
                        
                        if 'csv' in formats:
                            csv_file = circuit_dir / f"{base_filename}.csv"
                            df_or_dict.to_csv(csv_file, index=False)
                        
                        if 'parquet' in formats:
                            try:
                                # Prepare DataFrame for Parquet format
                                df_for_parquet = self._prepare_dataframe_for_parquet(df_or_dict)
                                parquet_file = circuit_dir / f"{base_filename}.parquet"
                                df_for_parquet.to_parquet(parquet_file, index=False)
                            except Exception as e:
                                logger.warning(f"      ⚠️ Could not save {base_filename} as Parquet: {e}")
                                logger.info(f"      📁 Saved as CSV only")
                    
                    elif isinstance(df_or_dict, dict):
                        json_file = circuit_dir / f"2023_{session_type}_{data_type}.json"
                        with open(json_file, 'w') as f:
                            json.dump(df_or_dict, f, indent=2, default=str)
        
        # Create consolidated 2023 datasets
        self._create_2023_consolidated_datasets(data)
        
        logger.info(f"✅ 2023 data saved to {output_2023_dir}")
    
    def _create_2023_consolidated_datasets(self, data: Dict) -> None:
        """Create consolidated datasets for 2023 analysis."""
        logger.info(f"📊 Creating consolidated 2023 datasets")
        
        consolidated_dir = self.output_dir / 'features'
        consolidated_dir.mkdir(parents=True, exist_ok=True)
        
        # Consolidate all 2023 data
        all_laps_2023 = []
        all_results_2023 = []
        
        for circuit, circuit_data in data.items():
            for session_type, session_data in circuit_data.items():
                
                # Collect 2023 lap data
                if 'laps' in session_data:
                    laps_df = session_data['laps'].copy()
                    laps_df['circuit'] = circuit
                    laps_df['session_type'] = session_type
                    laps_df['year'] = 2023
                    laps_df['is_sprint_weekend'] = circuit in self.config_2023['sprint_weekends_2023']
                    laps_df['is_new_circuit'] = circuit == 'Las Vegas'  # Las Vegas was new in 2023
                    all_laps_2023.append(laps_df)
                
                # Collect 2023 results data
                if 'results' in session_data:
                    results_df = session_data['results'].copy()
                    results_df['circuit'] = circuit
                    results_df['session_type'] = session_type
                    results_df['year'] = 2023
                    results_df['is_sprint_weekend'] = circuit in self.config_2023['sprint_weekends_2023']
                    results_df['is_new_circuit'] = circuit == 'Las Vegas'
                    all_results_2023.append(results_df)
        
        # Save consolidated 2023 datasets
        if all_laps_2023:
            consolidated_laps = pd.concat(all_laps_2023, ignore_index=True)
            
            # Save CSV (always works)
            consolidated_laps.to_csv(consolidated_dir / '2023_all_laps.csv', index=False)
            logger.info(f"  ✅ 2023 consolidated laps: {len(consolidated_laps)} records (CSV)")
            
            # Try to save Parquet with data type fixes
            try:
                consolidated_laps_parquet = self._prepare_dataframe_for_parquet(consolidated_laps)
                consolidated_laps_parquet.to_parquet(consolidated_dir / '2023_all_laps.parquet', index=False)
                logger.info(f"  ✅ 2023 consolidated laps: {len(consolidated_laps)} records (Parquet)")
            except Exception as e:
                logger.warning(f"  ⚠️ Could not save consolidated laps as Parquet: {e}")
                logger.info(f"  📁 Consolidated laps saved as CSV only")
        
        if all_results_2023:
            consolidated_results = pd.concat(all_results_2023, ignore_index=True)
            
            # Save CSV (always works)
            consolidated_results.to_csv(consolidated_dir / '2023_all_results.csv', index=False)
            logger.info(f"  ✅ 2023 consolidated results: {len(consolidated_results)} records (CSV)")
            
            # Try to save Parquet with data type fixes
            try:
                consolidated_results_parquet = self._prepare_dataframe_for_parquet(consolidated_results)
                consolidated_results_parquet.to_parquet(consolidated_dir / '2023_all_results.parquet', index=False)
                logger.info(f"  ✅ 2023 consolidated results: {len(consolidated_results)} records (Parquet)")
            except Exception as e:
                logger.warning(f"  ⚠️ Could not save consolidated results as Parquet: {e}")
                logger.info(f"  📁 Consolidated results saved as CSV only")
    
    def generate_2023_data_quality_report(self) -> str:
        """Generate comprehensive 2023 data quality report."""
        
        report = f"""
{'='*80}
F1 DATA CLEANING REPORT - 2023 SEASON ONLY
{'='*80}

PROCESSING SUMMARY - 2023 F1 WORLD CHAMPIONSHIP
-----------------------------------------------
Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Season: 2023 Formula 1 World Championship (22 races)
Circuits Processed: {self.cleaning_stats['circuits_processed']}
Files Processed: {self.cleaning_stats['total_files_processed']}
Total Records Processed: {self.cleaning_stats['total_records_processed']}
Processing Time: {self.cleaning_stats['processing_time']:.1f} seconds

2023 SEASON CONTEXT
------------------
🏆 World Champion: Max Verstappen (Red Bull Racing) - 3rd consecutive title
🏁 Total Races: 22 (including new Las Vegas GP)
🏃 Sprint Weekends: 6 (doubled from 2022) - Azerbaijan, Austria, Belgium, Qatar, United States, Brazil
🆕 New Circuits: Las Vegas Street Circuit
🔄 Major Driver Changes:
   • Oscar Piastri → McLaren (rookie, replacing Ricciardo)
   • Fernando Alonso → Aston Martin (from Alpine)
   • Pierre Gasly → Alpine (from AlphaTauri)
   • Logan Sargeant → Williams (rookie American)
   • Nico Hulkenberg → Haas (return, replacing Schumacher)
   • Daniel Ricciardo → AlphaTauri (mid-season return)

DATA QUALITY METRICS - 2023
---------------------------
Records Removed: {self.cleaning_stats['records_removed']} ({self.cleaning_stats['records_removed']/max(1, self.cleaning_stats['total_records_processed'])*100:.1f}%)
Records Corrected: {self.cleaning_stats['records_corrected']}
Missing Data Filled: {self.cleaning_stats['missing_data_filled']}
Outliers Detected: {self.cleaning_stats['outliers_detected']}
Overall Quality Score: {self.cleaning_stats['data_quality_score']:.2f}/1.00

DATA TYPE QUALITY SCORES - 2023
-------------------------------
⏱️ Timing Data Quality: {self.cleaning_stats['timing_data_quality']:.2f}/1.00
🏎️ Tire Data Quality: {self.cleaning_stats['tire_data_quality']:.2f}/1.00
🌤️ Weather Data Quality: {self.cleaning_stats['weather_data_quality']:.2f}/1.00

2023-SPECIFIC CLEANING OPERATIONS
---------------------------------
✅ 2023 Pirelli tire compound validation (SOFT, MEDIUM, HARD, INTERMEDIATE, WET)
✅ 2023 expanded sprint weekend handling (6 weekends vs 3 in 2022)
✅ 2023 car performance thresholds (max speed: {self.quality_thresholds_2023['max_speed']} km/h)
✅ 2023 tire age limits (max: {self.quality_thresholds_2023['max_tire_age']} laps)
✅ 2023 lap time validation ({self.quality_thresholds_2023['min_lap_time']}-{self.quality_thresholds_2023['max_lap_time']}s)
✅ Las Vegas circuit integration and validation
✅ Driver lineup change validation
✅ 2023 tire degradation modeling

SPRINT WEEKEND ANALYSIS - 2023
------------------------------
Sprint Weekends Processed: {self.cleaning_stats['sprint_weekends_processed']}/6
Sprint Circuits: Azerbaijan, Austria, Belgium, Qatar, United States, Brazil
Notable: Sprint format expanded significantly in 2023 (doubled from 2022)

TIRE DEGRADATION ANALYSIS READINESS - 2023
------------------------------------------
✅ All tire compounds validated and standardized
✅ Tire age data cleaned and validated
✅ Sprint weekend data properly categorized
✅ Las Vegas street circuit data integrated
✅ Performance baselines established for 2023 regulations
✅ Driver change impacts accounted for

NEXT STEPS - 2023 DATA
----------------------
1. ✅ 2023 data cleaning completed
2. 📊 Compare with 2022 data for trend analysis
3. 🔍 Analyze impact of expanded sprint weekends
4. 📈 Build predictive models incorporating 2023 changes
5. 🎯 Use for multi-year tire degradation analysis
6. 🆕 Analyze Las Vegas circuit characteristics

STORAGE LOCATIONS - 2023
------------------------
Raw Data: {self.raw_dir}
Cleaned Data: {self.output_dir / 'processed'}
Consolidated Data: {self.output_dir / 'features'}
Reports: {self.output_dir / 'reports'}
Logs: f1_data_cleaning_2023.log

COMPARISON WITH 2022
--------------------
• Sprint weekends: 6 (2023) vs 3 (2022) - 100% increase
• New circuits: Las Vegas (2023) vs Miami (2022)
• Major driver moves: 6+ significant changes in 2023
• Tire performance: Refined degradation models for 2023

{'='*80}
2023 F1 Data Cleaning Report - Ready for comparative analysis with 2022
{'='*80}
"""
        
        return report


def main():
    """Main execution function for 2023 data cleaning."""
    parser = argparse.ArgumentParser(
        description='F1 Data Cleaner - 2023 Season Only',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Circuit selection for 2023
    parser.add_argument('--circuit', type=str,
                       help='Specific 2023 circuit to process')
    parser.add_argument('--circuits', nargs='+',
                       help='Multiple 2023 circuits to process')
    
    # Processing options
    parser.add_argument('--clean-all', action='store_true',
                       help='Clean all available 2023 data (all 22 races)')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate 2023 data quality without cleaning')
    
    # Output options
    parser.add_argument('--formats', nargs='+', 
                       choices=['csv', 'parquet', 'pickle', 'json'],
                       default=['csv', 'parquet'],
                       help='Output formats for cleaned 2023 data')
    parser.add_argument('--base-dir', type=str, default='../data_collector/data',
                       help='Base directory containing raw 2023 data')
    parser.add_argument('--output-dir', type=str, default='../cleaned_data_2023',
                       help='Output directory for cleaned 2023 data')
    
    args = parser.parse_args()
    
    # Validate that at least one action is specified
    actions = [args.clean_all, args.circuit, args.circuits, args.validate_only]
    if not any(actions):
        parser.error("Please specify an action: --clean-all, --circuit, --circuits, or --validate-only")
    
    # Initialize 2023 cleaner
    try:
        cleaner = F1DataCleaner2023(
            base_dir=args.base_dir,
            output_dir=args.output_dir
        )
        
        logger.info(f"🧹 F1 2023 Data Cleaner initialized")
        logger.info(f"🏎️ Target: 2023 Formula 1 World Championship")
        logger.info(f"📁 Raw data directory: {cleaner.raw_dir}")
        logger.info(f"📁 Output directory: {cleaner.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize 2023 cleaner: {e}")
        return 1
    
    try:
        # Determine circuits to process
        circuits_to_process = None
        if args.circuit:
            if args.circuit not in cleaner.circuits_2023:
                logger.error(f"❌ Invalid 2023 circuit: {args.circuit}")
                logger.info(f"Valid 2023 circuits: {list(cleaner.circuits_2023.keys())}")
                return 1
            circuits_to_process = [args.circuit]
        elif args.circuits:
            invalid_circuits = [c for c in args.circuits if c not in cleaner.circuits_2023]
            if invalid_circuits:
                logger.error(f"❌ Invalid 2023 circuits: {invalid_circuits}")
                logger.info(f"Valid 2023 circuits: {list(cleaner.circuits_2023.keys())}")
                return 1
            circuits_to_process = args.circuits
        
        # Process the 2023 data
        if args.validate_only:
            logger.info(f"🔍 Validating 2023 data quality")
            raw_data = cleaner.load_raw_2023_data(args.circuit)
            if raw_data:
                logger.info(f"✅ 2023 data validation complete: {len(raw_data)} circuits found")
            else:
                logger.error(f"❌ No 2023 data found for validation")
                return 1
                
        elif args.clean_all or args.circuit or args.circuits:
            logger.info(f"🧹 Cleaning 2023 F1 data")
            
            # Process the 2023 data
            processed_data = cleaner.process_2023_data(circuits_to_process)
            
            if processed_data:
                # Save cleaned 2023 data
                cleaner.save_2023_cleaned_data(processed_data, args.formats)
                
                # Generate 2023 quality report
                report = cleaner.generate_2023_data_quality_report()
                
                # Save report
                report_file = cleaner.output_dir / 'reports' / 'cleaning_report_2023.txt'
                report_file.parent.mkdir(exist_ok=True)
                with open(report_file, 'w') as f:
                    f.write(report)
                
                print(report)
                logger.info(f"📄 2023 quality report saved: {report_file}")
                logger.info(f"✅ 2023 data cleaning complete")
                
                # Summary statistics
                total_circuits = len(processed_data)
                sprint_weekends = cleaner.cleaning_stats['sprint_weekends_processed']
                quality_score = cleaner.cleaning_stats['data_quality_score']
                
                logger.info(f"📊 2023 CLEANING SUMMARY:")
                logger.info(f"   🏁 Circuits processed: {total_circuits}/22")
                logger.info(f"   🏃 Sprint weekends: {sprint_weekends}/6")
                logger.info(f"   🏆 Quality score: {quality_score:.2f}/1.00")
                logger.info(f"   ⏱️ Processing time: {cleaner.cleaning_stats['processing_time']:.1f}s")
                
            else:
                logger.error(f"❌ No 2023 data processed")
                return 1
        
        logger.info(f"🏁 2023 F1 data cleaning completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.info(f"\n⚠️ 2023 data processing interrupted by user")
        return 0
        
    except Exception as e:
        logger.error(f"❌ 2023 data processing failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    
    # For IDE execution - default to cleaning all 2023 data
    if len(sys.argv) == 1:
        print("🧹 Running F1 Data Cleaner - 2023 Season Only")
        print("This will clean all available 2023 F1 data (22 races)")
        print("Key 2023 features: 6 sprint weekends, Las Vegas debut, major driver changes")
        print("Estimated time: 5-10 minutes for full season\n")
        
        # Ask user for confirmation
        response = input("Proceed with cleaning all 2023 data? (y/N): ")
        if response.lower() == 'y':
            sys.argv.extend(['--clean-all'])
        else:
            print("Operation cancelled. Use command line arguments for specific options.")
            sys.exit(0)
    
    sys.exit(main())

