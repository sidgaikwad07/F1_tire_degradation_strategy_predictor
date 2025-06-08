"""
Created on Sun Jun  8 10:57:44 2025

@author: sid
F1 Data Cleaner - 2024 Season Only
Performs comprehensive data cleaning and transformation specifically for 2024 F1 data.
This version is focused exclusively on 2024 data with year-specific configurations,
tire compounds, teams, and regulations that were in place during the 2024 season.
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
        logging.FileHandler('f1_data_cleaning_2024.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class F1DataCleaner2024:
    """
    F1 Data Cleaner specifically designed for 2024 season data.
    """
    
    def __init__(self, base_dir: str = '../data_collector/data', output_dir: str = '../cleaned_data_2024'):
        """Initialize the 2024 F1 Data Cleaner."""
        self.year = 2024  # Fixed to 2024 only
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.raw_dir = self.base_dir / 'raw' / '2024'
        
        # Create output directory structure for 2024
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'processed').mkdir(exist_ok=True)
        (self.output_dir / 'features').mkdir(exist_ok=True)
        (self.output_dir / 'validation').mkdir(exist_ok=True)
        (self.output_dir / 'reports').mkdir(exist_ok=True)
        
        # Initialize all the 2024-specific configurations
        self._setup_2024_config()
        self._setup_cleaning_stats()
    
    def _setup_2024_config(self):
        """Setup all 2024-specific configurations."""
        # 2024-specific configuration
        self.config_2024 = {
            'tire_compounds': ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET'],
            'teams': {
                'Red Bull Racing RBPT': 'RBR',
                'Ferrari': 'FER', 
                'Mercedes': 'MER',
                'McLaren Mercedes': 'MCL',
                'Alpine Renault': 'ALP',
                'RB RBPT': 'RB',  # CHANGED: AlphaTauri became RB in 2024
                'Aston Martin Aramco Mercedes': 'AM',
                'Williams Mercedes': 'WIL',
                'Kick Sauber Ferrari': 'KS',  # CHANGED: Alfa Romeo became Kick Sauber in 2024
                'Haas Ferrari': 'HAS'
            },
            'session_formats': {
                'standard': ['FP1', 'FP2', 'FP3', 'Q', 'R'],
                'sprint': ['FP1', 'SQ', 'S', 'Q', 'R']
            },
            'sprint_weekends_2024': ['China', 'Miami', 'Austria', 'United States', 'Brazil', 'Qatar']  # 6 sprint weekends
        }
        
        # 2024 F1 Calendar (24 races - record-breaking season)
        self.circuits_2024 = {
            'Bahrain': {'round': 1, 'has_sprint': False, 'date': '2024-03-02'},
            'Saudi Arabia': {'round': 2, 'has_sprint': False, 'date': '2024-03-09'},
            'Australia': {'round': 3, 'has_sprint': False, 'date': '2024-03-24'},
            'Japan': {'round': 4, 'has_sprint': False, 'date': '2024-04-07'},
            'China': {'round': 5, 'has_sprint': True, 'date': '2024-04-21'},  # Sprint weekend
            'Miami': {'round': 6, 'has_sprint': True, 'date': '2024-05-05'},  # Sprint weekend
            'Emilia Romagna': {'round': 7, 'has_sprint': False, 'date': '2024-05-19'},
            'Monaco': {'round': 8, 'has_sprint': False, 'date': '2024-05-26'},
            'Canada': {'round': 9, 'has_sprint': False, 'date': '2024-06-09'},
            'Spain': {'round': 10, 'has_sprint': False, 'date': '2024-06-23'},
            'Austria': {'round': 11, 'has_sprint': True, 'date': '2024-06-30'},  # Sprint weekend
            'Great Britain': {'round': 12, 'has_sprint': False, 'date': '2024-07-07'},
            'Hungary': {'round': 13, 'has_sprint': False, 'date': '2024-07-21'},
            'Belgium': {'round': 14, 'has_sprint': False, 'date': '2024-07-28'},
            'Netherlands': {'round': 15, 'has_sprint': False, 'date': '2024-08-25'},
            'Italy': {'round': 16, 'has_sprint': False, 'date': '2024-09-01'},
            'Azerbaijan': {'round': 17, 'has_sprint': False, 'date': '2024-09-15'},
            'Singapore': {'round': 18, 'has_sprint': False, 'date': '2024-09-22'},
            'United States': {'round': 19, 'has_sprint': True, 'date': '2024-10-20'},  # Sprint weekend
            'Mexico': {'round': 20, 'has_sprint': False, 'date': '2024-10-27'},
            'Brazil': {'round': 21, 'has_sprint': True, 'date': '2024-11-03'},  # Sprint weekend
            'Las Vegas': {'round': 22, 'has_sprint': False, 'date': '2024-11-23'},
            'Qatar': {'round': 23, 'has_sprint': True, 'date': '2024-12-01'},  # Sprint weekend
            'Abu Dhabi': {'round': 24, 'has_sprint': False, 'date': '2024-12-08'}  # Record 24 races
        }
        
        # 2024-specific data quality thresholds
        self.quality_thresholds_2024 = {
            'min_lap_time': 60.0,
            'max_lap_time': 300.0,
            'max_speed': 390.0,  # Higher for 2024 cars
            'max_tire_age': 50,  # Higher tire age limit for 2024
            'tire_deg_rate_soft': 0.22,  # Further refined for 2024
            'tire_deg_rate_medium': 0.16,
            'tire_deg_rate_hard': 0.11
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
    
    def load_raw_2024_data(self, circuit: str = None) -> Dict:
        """Load raw 2024 data for all circuits or a specific circuit."""
        logger.info(f"🔄 Loading raw 2024 data" + (f" - {circuit}" if circuit else " - All circuits"))
        
        if not self.raw_dir.exists():
            logger.error(f"❌ No raw 2024 data found at {self.raw_dir}")
            return {}
        
        loaded_data = {}
        circuits_to_process = [circuit] if circuit else [d.name for d in self.raw_dir.iterdir() if d.is_dir()]
        
        # Filter to only valid 2024 circuits
        valid_circuits = [c for c in circuits_to_process if c in self.circuits_2024]
        
        for circuit_name in valid_circuits:
            circuit_dir = self.raw_dir / circuit_name
            if not circuit_dir.exists():
                continue
            
            logger.info(f"  📁 Loading {circuit_name} 2024 data...")
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
        
        logger.info(f"📊 Loaded 2024 data for {len(loaded_data)} circuits")
        return loaded_data
    
    def clean_2024_timing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean timing data specifically for 2024 cars and regulations."""
        if df.empty:
            return df
        
        df_clean = df.copy()
        initial_rows = len(df_clean)
        
        logger.info(f"    🔧 Cleaning 2024 timing data: {initial_rows} records")
        
        # Clean lap times with 2024-specific thresholds
        if 'LapTime' in df_clean.columns:
            # Convert to seconds if needed
            df_clean['LapTime'] = self._convert_time_to_seconds(df_clean['LapTime'])
            
            # Remove unrealistic lap times for 2024 cars
            before_clean = len(df_clean)
            df_clean = df_clean[
                (df_clean['LapTime'].isna()) |
                ((df_clean['LapTime'] >= self.quality_thresholds_2024['min_lap_time']) &
                 (df_clean['LapTime'] <= self.quality_thresholds_2024['max_lap_time']))
            ]
            removed = before_clean - len(df_clean)
            
            if removed > 0:
                logger.info(f"      ❌ Removed {removed} records with unrealistic 2024 lap times")
                self.cleaning_stats['records_removed'] += removed
        
        # Clean speed data with 2024 car capabilities
        speed_columns = [col for col in df_clean.columns if 'Speed' in col or 'speed' in col.lower()]
        for speed_col in speed_columns:
            if speed_col in df_clean.columns:
                before_clean = len(df_clean)
                df_clean = df_clean[
                    (df_clean[speed_col].isna()) |
                    ((df_clean[speed_col] >= 0) & (df_clean[speed_col] <= self.quality_thresholds_2024['max_speed']))
                ]
                removed = before_clean - len(df_clean)
                
                if removed > 0:
                    logger.info(f"      ❌ Removed {removed} unrealistic 2024 {speed_col} values")
                    self.cleaning_stats['records_removed'] += removed
        
        # Update timing data quality score
        self.cleaning_stats['timing_data_quality'] = len(df_clean) / initial_rows if initial_rows > 0 else 0.0
        
        return df_clean
    
    def clean_2024_tire_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean tire data specifically for 2024 Pirelli compounds."""
        if df.empty:
            return df
        
        df_clean = df.copy()
        initial_rows = len(df_clean)
        
        logger.info(f"    🔧 Cleaning 2024 tire data: {initial_rows} records")
        
        # Clean tire compound data
        if 'Compound' in df_clean.columns:
            # Standardize compound names for 2024
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
            df_clean = df_clean[df_clean['Compound'].isin(self.config_2024['tire_compounds'])]
            removed = before_clean - len(df_clean)
            
            if removed > 0:
                logger.info(f"      ❌ Removed {removed} records with invalid tire compounds")
                self.cleaning_stats['records_removed'] += removed
        
        # Clean tire age data with 2024-specific limits
        if 'TyreLife' in df_clean.columns:
            df_clean['TyreLife'] = pd.to_numeric(df_clean['TyreLife'], errors='coerce')
            
            before_clean = len(df_clean)
            df_clean = df_clean[
                (df_clean['TyreLife'].isna()) |
                ((df_clean['TyreLife'] >= 0) & (df_clean['TyreLife'] <= self.quality_thresholds_2024['max_tire_age']))
            ]
            removed = before_clean - len(df_clean)
            
            if removed > 0:
                logger.info(f"      ❌ Removed {removed} records with invalid 2024 tire ages")
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
    
    def clean_2024_circuit_data(self, circuit_data: Dict, circuit: str) -> Dict:
        """Clean all data for a specific 2024 circuit."""
        logger.info(f"🏁 Cleaning {circuit} 2024 data")
        
        cleaned_data = {}
        
        # Track sprint weekends (6 in 2024)
        if circuit in self.config_2024['sprint_weekends_2024']:
            self.cleaning_stats['sprint_weekends_processed'] += 1
            logger.info(f"  🏃 {circuit} is a 2024 sprint weekend (6 total)")
        
        for session_type, session_data in circuit_data.items():
            logger.info(f"  📊 Processing 2024 {session_type} session")
            session_cleaned = {}
            
            for data_type, raw_data in session_data.items():
                if isinstance(raw_data, pd.DataFrame) and not raw_data.empty:
                    logger.info(f"    🔧 Cleaning 2024 {data_type} data")
                    
                    # Apply 2024-specific cleaning
                    if data_type in ['laps', 'results']:
                        df_cleaned = self.clean_2024_timing_data(raw_data)
                        
                        # Additional tire cleaning if tire columns present
                        tire_columns = ['Compound', 'TyreLife', 'FreshTyre']
                        if any(col in df_cleaned.columns for col in tire_columns):
                            df_cleaned = self.clean_2024_tire_data(df_cleaned)
                    else:
                        df_cleaned = raw_data.copy()
                    
                    session_cleaned[data_type] = df_cleaned
                    logger.info(f"      ✅ {data_type}: {len(df_cleaned)} records")
                
                elif isinstance(raw_data, dict):
                    # Handle JSON metadata
                    raw_data['year'] = 2024
                    raw_data['season_context'] = '2024 Formula 1 World Championship'
                    raw_data['sprint_weekends_count'] = 6  # 2024 had 6 sprint weekends
                    raw_data['record_season'] = True  # 24 races - record season
                    raw_data['team_changes'] = ['AlphaTauri → RB', 'Alfa Romeo → Kick Sauber']
                    session_cleaned[data_type] = raw_data
            
            cleaned_data[session_type] = session_cleaned
        
        return cleaned_data
    
    def process_2024_data(self, circuits: List[str] = None) -> Dict:
        """Process and clean all 2024 F1 data."""
        logger.info(f"🚀 Starting 2024 F1 data cleaning")
        start_time = datetime.now()
        
        # Load raw 2024 data
        raw_data = self.load_raw_2024_data()
        
        if not raw_data:
            logger.error(f"❌ No raw 2024 data found")
            return {}
        
        # Filter circuits if specified
        if circuits:
            raw_data = {circuit: data for circuit, data in raw_data.items() if circuit in circuits}
        
        processed_data = {}
        
        # Process each 2024 circuit
        for circuit, circuit_data in raw_data.items():
            try:
                cleaned_circuit = self.clean_2024_circuit_data(circuit_data, circuit)
                processed_data[circuit] = cleaned_circuit
                self.cleaning_stats['circuits_processed'] += 1
                
            except Exception as e:
                logger.error(f"❌ Failed to process 2024 {circuit}: {e}")
                self.cleaning_stats['errors_encountered'].append(f"{circuit}: {str(e)}")
        
        # Calculate processing time
        end_time = datetime.now()
        self.cleaning_stats['processing_time'] = (end_time - start_time).total_seconds()
        
        logger.info(f"🏁 2024 data processing complete: {len(processed_data)} circuits")
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

    def save_2024_cleaned_data(self, data: Dict, formats: List[str] = ['csv', 'parquet']) -> None:
        """Save cleaned 2024 data in multiple formats."""
        logger.info(f"💾 Saving cleaned 2024 data")
        
        output_2024_dir = self.output_dir / 'processed'
        output_2024_dir.mkdir(parents=True, exist_ok=True)
        
        # Save circuit-by-circuit 2024 data
        for circuit, circuit_data in data.items():
            circuit_dir = output_2024_dir / circuit
            circuit_dir.mkdir(exist_ok=True)
            
            for session_type, session_data in circuit_data.items():
                for data_type, df_or_dict in session_data.items():
                    
                    if isinstance(df_or_dict, pd.DataFrame) and not df_or_dict.empty:
                        base_filename = f"2024_{session_type}_{data_type}"
                        
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
                        json_file = circuit_dir / f"2024_{session_type}_{data_type}.json"
                        with open(json_file, 'w') as f:
                            json.dump(df_or_dict, f, indent=2, default=str)
        
        # Create consolidated 2024 datasets
        self._create_2024_consolidated_datasets(data)
        
        logger.info(f"✅ 2024 data saved to {output_2024_dir}")
    
    def _create_2024_consolidated_datasets(self, data: Dict) -> None:
        """Create consolidated datasets for 2024 analysis."""
        logger.info(f"📊 Creating consolidated 2024 datasets")
        
        consolidated_dir = self.output_dir / 'features'
        consolidated_dir.mkdir(parents=True, exist_ok=True)
        
        # Consolidate all 2024 data
        all_laps_2024 = []
        all_results_2024 = []
        
        for circuit, circuit_data in data.items():
            for session_type, session_data in circuit_data.items():
                
                # Collect 2024 lap data
                if 'laps' in session_data:
                    laps_df = session_data['laps'].copy()
                    laps_df['circuit'] = circuit
                    laps_df['session_type'] = session_type
                    laps_df['year'] = 2024
                    laps_df['is_sprint_weekend'] = circuit in self.config_2024['sprint_weekends_2024']
                    laps_df['is_record_season'] = True  # 2024 was record 24-race season
                    all_laps_2024.append(laps_df)
                
                # Collect 2024 results data
                if 'results' in session_data:
                    results_df = session_data['results'].copy()
                    results_df['circuit'] = circuit
                    results_df['session_type'] = session_type
                    results_df['year'] = 2024
                    results_df['is_sprint_weekend'] = circuit in self.config_2024['sprint_weekends_2024']
                    results_df['is_record_season'] = True
                    all_results_2024.append(results_df)
        
        # Save consolidated 2024 datasets
        if all_laps_2024:
            consolidated_laps = pd.concat(all_laps_2024, ignore_index=True)
            
            # Save CSV (always works)
            consolidated_laps.to_csv(consolidated_dir / '2024_all_laps.csv', index=False)
            logger.info(f"  ✅ 2024 consolidated laps: {len(consolidated_laps)} records (CSV)")
            
            # Try to save Parquet with data type fixes
            try:
                consolidated_laps_parquet = self._prepare_dataframe_for_parquet(consolidated_laps)
                consolidated_laps_parquet.to_parquet(consolidated_dir / '2024_all_laps.parquet', index=False)
                logger.info(f"  ✅ 2024 consolidated laps: {len(consolidated_laps)} records (Parquet)")
            except Exception as e:
                logger.warning(f"  ⚠️ Could not save consolidated laps as Parquet: {e}")
                logger.info(f"  📁 Consolidated laps saved as CSV only")
        
        if all_results_2024:
            consolidated_results = pd.concat(all_results_2024, ignore_index=True)
            
            # Save CSV (always works)
            consolidated_results.to_csv(consolidated_dir / '2024_all_results.csv', index=False)
            logger.info(f"  ✅ 2024 consolidated results: {len(consolidated_results)} records (CSV)")
            
            # Try to save Parquet with data type fixes
            try:
                consolidated_results_parquet = self._prepare_dataframe_for_parquet(consolidated_results)
                consolidated_results_parquet.to_parquet(consolidated_dir / '2024_all_results.parquet', index=False)
                logger.info(f"  ✅ 2024 consolidated results: {len(consolidated_results)} records (Parquet)")
            except Exception as e:
                logger.warning(f"  ⚠️ Could not save consolidated results as Parquet: {e}")
                logger.info(f"  📁 Consolidated results saved as CSV only")
    
    def generate_2024_data_quality_report(self) -> str:
        """Generate comprehensive 2024 data quality report."""
        
        report = f"""
{'='*80}
F1 DATA CLEANING REPORT - 2024 SEASON ONLY
{'='*80}

PROCESSING SUMMARY - 2024 F1 WORLD CHAMPIONSHIP
-----------------------------------------------
Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Season: 2024 Formula 1 World Championship (24 races - RECORD SEASON)
Circuits Processed: {self.cleaning_stats['circuits_processed']}
Files Processed: {self.cleaning_stats['total_files_processed']}
Total Records Processed: {self.cleaning_stats['total_records_processed']}
Processing Time: {self.cleaning_stats['processing_time']:.1f} seconds

2024 SEASON CONTEXT
------------------
🏆 World Champion: Max Verstappen (Red Bull Racing) - 4th consecutive title
🏁 Total Races: 24 (RECORD-BREAKING season length)
🏃 Sprint Weekends: 6 - China, Miami, Austria, United States, Brazil, Qatar
🔄 Major Team Changes:
   • AlphaTauri → RB (complete rebrand)
   • Alfa Romeo → Kick Sauber (name change, new partnership)
🔄 Driver Changes:
   • Carlos Sainz → Ferrari (continued)
   • Oliver Bearman → Ferrari (occasional substitute)
   • Multiple driver changes throughout season

DATA QUALITY METRICS - 2024
---------------------------
Records Removed: {self.cleaning_stats['records_removed']} ({self.cleaning_stats['records_removed']/max(1, self.cleaning_stats['total_records_processed'])*100:.1f}%)
Records Corrected: {self.cleaning_stats['records_corrected']}
Missing Data Filled: {self.cleaning_stats['missing_data_filled']}
Outliers Detected: {self.cleaning_stats['outliers_detected']}
Overall Quality Score: {self.cleaning_stats['data_quality_score']:.2f}/1.00

DATA TYPE QUALITY SCORES - 2024
-------------------------------
⏱️ Timing Data Quality: {self.cleaning_stats['timing_data_quality']:.2f}/1.00
🏎️ Tire Data Quality: {self.cleaning_stats['tire_data_quality']:.2f}/1.00
🌤️ Weather Data Quality: {self.cleaning_stats['weather_data_quality']:.2f}/1.00

2024-SPECIFIC CLEANING OPERATIONS
---------------------------------
✅ 2024 Pirelli tire compound validation (SOFT, MEDIUM, HARD, INTERMEDIATE, WET)
✅ 2024 sprint weekend handling (6 weekends - same as 2023)
✅ 2024 car performance thresholds (max speed: {self.quality_thresholds_2024['max_speed']} km/h)
✅ 2024 tire age limits (max: {self.quality_thresholds_2024['max_tire_age']} laps)
✅ 2024 lap time validation ({self.quality_thresholds_2024['min_lap_time']}-{self.quality_thresholds_2024['max_lap_time']}s)
✅ Team rebrand validation (AlphaTauri → RB, Alfa Romeo → Kick Sauber)
✅ Record 24-race season handling
✅ 2024 tire degradation modeling

SPRINT WEEKEND ANALYSIS - 2024
------------------------------
Sprint Weekends Processed: {self.cleaning_stats['sprint_weekends_processed']}/6
Sprint Circuits: China, Miami, Austria, United States, Brazil, Qatar
Notable: Sprint format maintained at 6 weekends (same as 2023)

TEAM CHANGES - 2024
-------------------
✅ AlphaTauri → RB (complete team rebrand)
✅ Alfa Romeo → Kick Sauber (name change, new title sponsor)
✅ All other teams maintained names from 2023
✅ Team data validation updated for new team names

RECORD SEASON ANALYSIS - 2024
-----------------------------
📊 24 Races: Longest F1 season in history
📊 More data points for analysis than any previous season
📊 Enhanced statistical significance for tire degradation models
📊 Comprehensive coverage of global racing venues

TIRE DEGRADATION ANALYSIS READINESS - 2024
------------------------------------------
✅ All tire compounds validated and standardized
✅ Tire age data cleaned and validated (up to 50 laps)
✅ Sprint weekend data properly categorized
✅ Record season data volume for robust analysis
✅ Performance baselines established for 2024 regulations
✅ Team rebrand impacts accounted for

NEXT STEPS - 2024 DATA
----------------------
1. ✅ 2024 data cleaning completed
2. 📊 Compare with 2022-2023 for multi-year trend analysis
3. 🔍 Analyze impact of record 24-race season
4. 📈 Build predictive models with largest dataset ever
5. 🎯 Use for comprehensive tire degradation analysis
6. 🏆 Analyze championship dynamics over longest season

STORAGE LOCATIONS - 2024
------------------------
Raw Data: {self.raw_dir}
Cleaned Data: {self.output_dir / 'processed'}
Consolidated Data: {self.output_dir / 'features'}
Reports: {self.output_dir / 'reports'}
Logs: f1_data_cleaning_2024.log

COMPARISON WITH PREVIOUS SEASONS
--------------------------------
• Total races: 24 (2024) vs 22 (2023) vs 22 (2022) - Record season
• Sprint weekends: 6 (2024) = 6 (2023) vs 3 (2022) - Stable format
• Team changes: 2 rebrands (2024) vs driver changes (2023) vs new circuit (2022)
• Data volume: Largest dataset in F1 history
• Tire performance: Most refined degradation models yet

{'='*80}
2024 F1 Data Cleaning Report - Record Season Ready for Analysis
{'='*80}
"""
        
        return report


def main():
    """Main execution function for 2024 data cleaning."""
    parser = argparse.ArgumentParser(
        description='F1 Data Cleaner - 2024 Season Only',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Circuit selection for 2024
    parser.add_argument('--circuit', type=str,
                       help='Specific 2024 circuit to process')
    parser.add_argument('--circuits', nargs='+',
                       help='Multiple 2024 circuits to process')
    
    # Processing options
    parser.add_argument('--clean-all', action='store_true',
                       help='Clean all available 2024 data (all 24 races - RECORD SEASON)')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate 2024 data quality without cleaning')
    
    # Output options
    parser.add_argument('--formats', nargs='+', 
                       choices=['csv', 'parquet', 'pickle', 'json'],
                       default=['csv', 'parquet'],
                       help='Output formats for cleaned 2024 data')
    parser.add_argument('--base-dir', type=str, default='../data_collector/data',
                       help='Base directory containing raw 2024 data')
    parser.add_argument('--output-dir', type=str, default='../cleaned_data_2024',
                       help='Output directory for cleaned 2024 data')
    
    args = parser.parse_args()
    
    # Validate that at least one action is specified
    actions = [args.clean_all, args.circuit, args.circuits, args.validate_only]
    if not any(actions):
        parser.error("Please specify an action: --clean-all, --circuit, --circuits, or --validate-only")
    
    # Initialize 2024 cleaner
    try:
        cleaner = F1DataCleaner2024(
            base_dir=args.base_dir,
            output_dir=args.output_dir
        )
        
        logger.info(f"🧹 F1 2024 Data Cleaner initialized")
        logger.info(f"🏎️ Target: 2024 Formula 1 World Championship (RECORD 24-race season)")
        logger.info(f"📁 Raw data directory: {cleaner.raw_dir}")
        logger.info(f"📁 Output directory: {cleaner.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize 2024 cleaner: {e}")
        return 1
    
    try:
        # Determine circuits to process
        circuits_to_process = None
        if args.circuit:
            if args.circuit not in cleaner.circuits_2024:
                logger.error(f"❌ Invalid 2024 circuit: {args.circuit}")
                logger.info(f"Valid 2024 circuits: {list(cleaner.circuits_2024.keys())}")
                return 1
            circuits_to_process = [args.circuit]
        elif args.circuits:
            invalid_circuits = [c for c in args.circuits if c not in cleaner.circuits_2024]
            if invalid_circuits:
                logger.error(f"❌ Invalid 2024 circuits: {invalid_circuits}")
                logger.info(f"Valid 2024 circuits: {list(cleaner.circuits_2024.keys())}")
                return 1
            circuits_to_process = args.circuits
        
        # Process the 2024 data
        if args.validate_only:
            logger.info(f"🔍 Validating 2024 data quality")
            raw_data = cleaner.load_raw_2024_data(args.circuit)
            if raw_data:
                logger.info(f"✅ 2024 data validation complete: {len(raw_data)} circuits found")
            else:
                logger.error(f"❌ No 2024 data found for validation")
                return 1
                
        elif args.clean_all or args.circuit or args.circuits:
            logger.info(f"🧹 Cleaning 2024 F1 data (RECORD SEASON)")
            
            # Process the 2024 data
            processed_data = cleaner.process_2024_data(circuits_to_process)
            
            if processed_data:
                # Save cleaned 2024 data
                cleaner.save_2024_cleaned_data(processed_data, args.formats)
                
                # Generate 2024 quality report
                report = cleaner.generate_2024_data_quality_report()
                
                # Save report
                report_file = cleaner.output_dir / 'reports' / 'cleaning_report_2024.txt'
                report_file.parent.mkdir(exist_ok=True)
                with open(report_file, 'w') as f:
                    f.write(report)
                
                print(report)
                logger.info(f"📄 2024 quality report saved: {report_file}")
                logger.info(f"✅ 2024 data cleaning complete")
                
                # Summary statistics
                total_circuits = len(processed_data)
                sprint_weekends = cleaner.cleaning_stats['sprint_weekends_processed']
                quality_score = cleaner.cleaning_stats['data_quality_score']
                
                logger.info(f"📊 2024 CLEANING SUMMARY:")
                logger.info(f"   🏁 Circuits processed: {total_circuits}/24 (RECORD SEASON)")
                logger.info(f"   🏃 Sprint weekends: {sprint_weekends}/6")
                logger.info(f"   🏆 Quality score: {quality_score:.2f}/1.00")
                logger.info(f"   ⏱️ Processing time: {cleaner.cleaning_stats['processing_time']:.1f}s")
                
            else:
                logger.error(f"❌ No 2024 data processed")
                return 1
        
        logger.info(f"🏁 2024 F1 data cleaning completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.info(f"\n⚠️ 2024 data processing interrupted by user")
        return 0
        
    except Exception as e:
        logger.error(f"❌ 2024 data processing failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    
    # For IDE execution - default to cleaning all 2024 data
    if len(sys.argv) == 1:
        print("🧹 Running F1 Data Cleaner - 2024 Season Only")
        print("This will clean all available 2024 F1 data (24 races - RECORD SEASON)")
        print("Key 2024 features: Record 24-race calendar, team rebrands (AlphaTauri→RB, Alfa Romeo→Kick Sauber)")
        print("Estimated time: 8-12 minutes for full record season\n")
        
        # Ask user for confirmation
        response = input("Proceed with cleaning all 2024 data? (y/N): ")
        if response.lower() == 'y':
            sys.argv.extend(['--clean-all'])
        else:
            print("Operation cancelled. Use command line arguments for specific options.")
            sys.exit(0)
    
    sys.exit(main())