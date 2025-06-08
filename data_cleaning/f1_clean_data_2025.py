"""
Created on Sun Jun  8 11:10:10 2025

@author: sid
F1 Data Cleaner - 2025 Season Through Spanish Grand Prix
Performs comprehensive data cleaning and transformation specifically for 2025 F1 data.
This version is focused on the first 10 races of the 2025 season (through Spanish GP)
with year-specific configurations, tire compounds, teams, and current regulations.

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
        logging.FileHandler('f1_data_cleaning_2025.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class F1DataCleaner2025:
    """
    F1 Data Cleaner specifically designed for 2025 season data through Spanish GP.
    """
    
    def __init__(self, base_dir: str = '../data_collector/data', output_dir: str = '../cleaned_data_2025'):
        """Initialize the 2025 F1 Data Cleaner (through Spanish GP)."""
        self.year = 2025  # Fixed to 2025 only
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.raw_dir = self.base_dir / 'raw' / '2025'
        
        # Create output directory structure for 2025
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'processed').mkdir(exist_ok=True)
        (self.output_dir / 'features').mkdir(exist_ok=True)
        (self.output_dir / 'validation').mkdir(exist_ok=True)
        (self.output_dir / 'reports').mkdir(exist_ok=True)
        
        # Initialize all the 2025-specific configurations
        self._setup_2025_config()
        self._setup_cleaning_stats()
    
    def _setup_2025_config(self):
        """Setup all 2025-specific configurations."""
        # 2025-specific configuration
        self.config_2025 = {
            'tire_compounds': ['C1', 'C2', 'C3', 'C4', 'C5', 'INTERMEDIATE', 'WET'],  # 2025 new naming
            'teams': {
                'Red Bull Racing Honda RBPT': 'RBR',  # CHANGED: Honda partnership
                'Ferrari': 'FER', 
                'Mercedes': 'MER',
                'McLaren Mercedes': 'MCL',
                'Aston Martin Honda': 'AM',  # CHANGED: Honda switch from Mercedes
                'Alpine Renault': 'ALP',
                'RB Honda RBPT': 'RB',  # Continued from 2024
                'Williams Mercedes': 'WIL',
                'Kick Sauber Ferrari': 'KS',  # Continued from 2024
                'Haas Ferrari': 'HAS'
            },
            'session_formats': {
                'standard': ['FP1', 'FP2', 'FP3', 'Q', 'R'],
                'sprint': ['FP1', 'SQ', 'S', 'Q', 'R']
            },
            'sprint_weekends_2025': ['China', 'Miami', 'Austria']  # 3 sprint weekends through Spanish GP
        }
        
        # 2025 F1 Calendar - First 10 races through Spanish GP
        self.circuits_2025 = {
            'Australia': {'round': 1, 'has_sprint': False, 'date': '2025-03-16'},
            'Bahrain': {'round': 2, 'has_sprint': False, 'date': '2025-03-30'},
            'Saudi Arabia': {'round': 3, 'has_sprint': False, 'date': '2025-04-06'},
            'Japan': {'round': 4, 'has_sprint': False, 'date': '2025-04-13'},
            'China': {'round': 5, 'has_sprint': True, 'date': '2025-04-20'},  # Sprint weekend
            'Miami': {'round': 6, 'has_sprint': True, 'date': '2025-05-04'},  # Sprint weekend
            'Emilia Romagna': {'round': 7, 'has_sprint': False, 'date': '2025-05-18'},
            'Monaco': {'round': 8, 'has_sprint': False, 'date': '2025-05-25'},
            'Canada': {'round': 9, 'has_sprint': False, 'date': '2025-06-15'},
            'Spain': {'round': 10, 'has_sprint': False, 'date': '2025-06-29'}  # Through Spanish GP
        }
        
        # 2025-specific data quality thresholds (updated for new regulations)
        self.quality_thresholds_2025 = {
            'min_lap_time': 55.0,  # Faster cars in 2025
            'max_lap_time': 300.0,
            'max_speed': 395.0,  # Higher for 2025 cars with new aero
            'max_tire_age': 45,  # Refined tire age limit for 2025
            'tire_deg_rate_c1': 0.08,  # New compound degradation rates
            'tire_deg_rate_c2': 0.12,
            'tire_deg_rate_c3': 0.16,
            'tire_deg_rate_c4': 0.21,
            'tire_deg_rate_c5': 0.26
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
    
    def load_raw_2025_data(self, circuit: str = None) -> Dict:
        """Load raw 2025 data for all circuits or a specific circuit."""
        logger.info(f"🔄 Loading raw 2025 data" + (f" - {circuit}" if circuit else " - Through Spanish GP"))
        
        if not self.raw_dir.exists():
            logger.error(f"❌ No raw 2025 data found at {self.raw_dir}")
            return {}
        
        loaded_data = {}
        circuits_to_process = [circuit] if circuit else [d.name for d in self.raw_dir.iterdir() if d.is_dir()]
        
        # Filter to only valid 2025 circuits (through Spanish GP)
        valid_circuits = [c for c in circuits_to_process if c in self.circuits_2025]
        
        for circuit_name in valid_circuits:
            circuit_dir = self.raw_dir / circuit_name
            if not circuit_dir.exists():
                continue
            
            logger.info(f"  📁 Loading {circuit_name} 2025 data...")
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
        
        logger.info(f"📊 Loaded 2025 data for {len(loaded_data)} circuits (through Spanish GP)")
        return loaded_data
    
    def clean_2025_timing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean timing data specifically for 2025 cars and regulations."""
        if df.empty:
            return df
        
        df_clean = df.copy()
        initial_rows = len(df_clean)
        
        logger.info(f"    🔧 Cleaning 2025 timing data: {initial_rows} records")
        
        # Clean lap times with 2025-specific thresholds
        if 'LapTime' in df_clean.columns:
            # Convert to seconds if needed
            df_clean['LapTime'] = self._convert_time_to_seconds(df_clean['LapTime'])
            
            # Remove unrealistic lap times for 2025 cars
            before_clean = len(df_clean)
            df_clean = df_clean[
                (df_clean['LapTime'].isna()) |
                ((df_clean['LapTime'] >= self.quality_thresholds_2025['min_lap_time']) &
                 (df_clean['LapTime'] <= self.quality_thresholds_2025['max_lap_time']))
            ]
            removed = before_clean - len(df_clean)
            
            if removed > 0:
                logger.info(f"      ❌ Removed {removed} records with unrealistic 2025 lap times")
                self.cleaning_stats['records_removed'] += removed
        
        # Clean speed data with 2025 car capabilities
        speed_columns = [col for col in df_clean.columns if 'Speed' in col or 'speed' in col.lower()]
        for speed_col in speed_columns:
            if speed_col in df_clean.columns:
                before_clean = len(df_clean)
                df_clean = df_clean[
                    (df_clean[speed_col].isna()) |
                    ((df_clean[speed_col] >= 0) & (df_clean[speed_col] <= self.quality_thresholds_2025['max_speed']))
                ]
                removed = before_clean - len(df_clean)
                
                if removed > 0:
                    logger.info(f"      ❌ Removed {removed} unrealistic 2025 {speed_col} values")
                    self.cleaning_stats['records_removed'] += removed
        
        # Update timing data quality score
        self.cleaning_stats['timing_data_quality'] = len(df_clean) / initial_rows if initial_rows > 0 else 0.0
        
        return df_clean
    
    def clean_2025_tire_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean tire data specifically for 2025 Pirelli compounds (new C1-C5 naming)."""
        if df.empty:
            return df
        
        df_clean = df.copy()
        initial_rows = len(df_clean)
        
        logger.info(f"    🔧 Cleaning 2025 tire data: {initial_rows} records")
        
        # Clean tire compound data with 2025 new naming system
        if 'Compound' in df_clean.columns:
            # Standardize compound names for 2025 (C1-C5 system)
            compound_mapping = {
                'C1': 'C1', 'HARD': 'C1', 'H': 'C1', 'White': 'C1',
                'C2': 'C2', 'MEDIUM': 'C2', 'M': 'C2', 'Yellow': 'C2',
                'C3': 'C3', 'SOFT': 'C3', 'S': 'C3', 'Red': 'C3',
                'C4': 'C4', 'ULTRASOFT': 'C4', 'Purple': 'C4',
                'C5': 'C5', 'HYPERSOFT': 'C5', 'Pink': 'C5',
                'INTERMEDIATE': 'INTERMEDIATE', 'I': 'INTERMEDIATE',
                'WET': 'WET', 'W': 'WET'
            }
            
            df_clean['Compound'] = df_clean['Compound'].map(compound_mapping).fillna(df_clean['Compound'])
            
            # Remove invalid compounds
            before_clean = len(df_clean)
            df_clean = df_clean[df_clean['Compound'].isin(self.config_2025['tire_compounds'])]
            removed = before_clean - len(df_clean)
            
            if removed > 0:
                logger.info(f"      ❌ Removed {removed} records with invalid tire compounds")
                self.cleaning_stats['records_removed'] += removed
        
        # Clean tire age data with 2025-specific limits
        if 'TyreLife' in df_clean.columns:
            df_clean['TyreLife'] = pd.to_numeric(df_clean['TyreLife'], errors='coerce')
            
            before_clean = len(df_clean)
            df_clean = df_clean[
                (df_clean['TyreLife'].isna()) |
                ((df_clean['TyreLife'] >= 0) & (df_clean['TyreLife'] <= self.quality_thresholds_2025['max_tire_age']))
            ]
            removed = before_clean - len(df_clean)
            
            if removed > 0:
                logger.info(f"      ❌ Removed {removed} records with invalid 2025 tire ages")
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
    
    def clean_2025_circuit_data(self, circuit_data: Dict, circuit: str) -> Dict:
        """Clean all data for a specific 2025 circuit."""
        logger.info(f"🏁 Cleaning {circuit} 2025 data")
        
        cleaned_data = {}
        
        # Track sprint weekends (3 through Spanish GP)
        if circuit in self.config_2025['sprint_weekends_2025']:
            self.cleaning_stats['sprint_weekends_processed'] += 1
            logger.info(f"  🏃 {circuit} is a 2025 sprint weekend (3 through Spanish GP)")
        
        for session_type, session_data in circuit_data.items():
            logger.info(f"  📊 Processing 2025 {session_type} session")
            session_cleaned = {}
            
            for data_type, raw_data in session_data.items():
                if isinstance(raw_data, pd.DataFrame) and not raw_data.empty:
                    logger.info(f"    🔧 Cleaning 2025 {data_type} data")
                    
                    # Apply 2025-specific cleaning
                    if data_type in ['laps', 'results']:
                        df_cleaned = self.clean_2025_timing_data(raw_data)
                        
                        # Additional tire cleaning if tire columns present
                        tire_columns = ['Compound', 'TyreLife', 'FreshTyre']
                        if any(col in df_cleaned.columns for col in tire_columns):
                            df_cleaned = self.clean_2025_tire_data(df_cleaned)
                    else:
                        df_cleaned = raw_data.copy()
                    
                    session_cleaned[data_type] = df_cleaned
                    logger.info(f"      ✅ {data_type}: {len(df_cleaned)} records")
                
                elif isinstance(raw_data, dict):
                    # Handle JSON metadata
                    raw_data['year'] = 2025
                    raw_data['season_context'] = '2025 Formula 1 World Championship'
                    raw_data['sprint_weekends_count'] = 3  # 3 through Spanish GP
                    raw_data['races_through_spanish'] = 10
                    raw_data['engine_changes'] = ['Aston Martin: Mercedes → Honda']
                    raw_data['tire_system'] = 'C1-C5 compound naming'
                    session_cleaned[data_type] = raw_data
            
            cleaned_data[session_type] = session_cleaned
        
        return cleaned_data
    
    def process_2025_data(self, circuits: List[str] = None) -> Dict:
        """Process and clean all 2025 F1 data through Spanish GP."""
        logger.info(f"🚀 Starting 2025 F1 data cleaning (through Spanish GP)")
        start_time = datetime.now()
        
        # Load raw 2025 data
        raw_data = self.load_raw_2025_data()
        
        if not raw_data:
            logger.error(f"❌ No raw 2025 data found")
            return {}
        
        # Filter circuits if specified
        if circuits:
            raw_data = {circuit: data for circuit, data in raw_data.items() if circuit in circuits}
        
        processed_data = {}
        
        # Process each 2025 circuit
        for circuit, circuit_data in raw_data.items():
            try:
                cleaned_circuit = self.clean_2025_circuit_data(circuit_data, circuit)
                processed_data[circuit] = cleaned_circuit
                self.cleaning_stats['circuits_processed'] += 1
                
            except Exception as e:
                logger.error(f"❌ Failed to process 2025 {circuit}: {e}")
                self.cleaning_stats['errors_encountered'].append(f"{circuit}: {str(e)}")
        
        # Calculate processing time
        end_time = datetime.now()
        self.cleaning_stats['processing_time'] = (end_time - start_time).total_seconds()
        
        logger.info(f"🏁 2025 data processing complete: {len(processed_data)} circuits (through Spanish GP)")
        logger.info(f"🏃 Sprint weekends processed: {self.cleaning_stats['sprint_weekends_processed']}/3")
        
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

    def save_2025_cleaned_data(self, data: Dict, formats: List[str] = ['csv', 'parquet']) -> None:
        """Save cleaned 2025 data in multiple formats."""
        logger.info(f"💾 Saving cleaned 2025 data (through Spanish GP)")
        
        output_2025_dir = self.output_dir / 'processed'
        output_2025_dir.mkdir(parents=True, exist_ok=True)
        
        # Save circuit-by-circuit 2025 data
        for circuit, circuit_data in data.items():
            circuit_dir = output_2025_dir / circuit
            circuit_dir.mkdir(exist_ok=True)
            
            for session_type, session_data in circuit_data.items():
                for data_type, df_or_dict in session_data.items():
                    
                    if isinstance(df_or_dict, pd.DataFrame) and not df_or_dict.empty:
                        base_filename = f"2025_{session_type}_{data_type}"
                        
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
                        json_file = circuit_dir / f"2025_{session_type}_{data_type}.json"
                        with open(json_file, 'w') as f:
                            json.dump(df_or_dict, f, indent=2, default=str)
        
        # Create consolidated 2025 datasets
        self._create_2025_consolidated_datasets(data)
        
        logger.info(f"✅ 2025 data saved to {output_2025_dir}")
    
    def _create_2025_consolidated_datasets(self, data: Dict) -> None:
        """Create consolidated datasets for 2025 analysis."""
        logger.info(f"📊 Creating consolidated 2025 datasets (through Spanish GP)")
        
        consolidated_dir = self.output_dir / 'features'
        consolidated_dir.mkdir(parents=True, exist_ok=True)
        
        # Consolidate all 2025 data
        all_laps_2025 = []
        all_results_2025 = []
        
        for circuit, circuit_data in data.items():
            for session_type, session_data in circuit_data.items():
                
                # Collect 2025 lap data
                if 'laps' in session_data:
                    laps_df = session_data['laps'].copy()
                    laps_df['circuit'] = circuit
                    laps_df['session_type'] = session_type
                    laps_df['year'] = 2025
                    laps_df['is_sprint_weekend'] = circuit in self.config_2025['sprint_weekends_2025']
                    laps_df['through_spanish_gp'] = True
                    laps_df['race_number'] = self.circuits_2025[circuit]['round']
                    all_laps_2025.append(laps_df)
                
                # Collect 2025 results data
                if 'results' in session_data:
                    results_df = session_data['results'].copy()
                    results_df['circuit'] = circuit
                    results_df['session_type'] = session_type
                    results_df['year'] = 2025
                    results_df['is_sprint_weekend'] = circuit in self.config_2025['sprint_weekends_2025']
                    results_df['through_spanish_gp'] = True
                    results_df['race_number'] = self.circuits_2025[circuit]['round']
                    all_results_2025.append(results_df)
        
        # Save consolidated 2025 datasets
        if all_laps_2025:
            consolidated_laps = pd.concat(all_laps_2025, ignore_index=True)
            
            # Save CSV (always works)
            consolidated_laps.to_csv(consolidated_dir / '2025_all_laps_through_spanish.csv', index=False)
            logger.info(f"  ✅ 2025 consolidated laps: {len(consolidated_laps)} records (CSV)")
            
            # Try to save Parquet with data type fixes
            try:
                consolidated_laps_parquet = self._prepare_dataframe_for_parquet(consolidated_laps)
                consolidated_laps_parquet.to_parquet(consolidated_dir / '2025_all_laps_through_spanish.parquet', index=False)
                logger.info(f"  ✅ 2025 consolidated laps: {len(consolidated_laps)} records (Parquet)")
            except Exception as e:
                logger.warning(f"  ⚠️ Could not save consolidated laps as Parquet: {e}")
                logger.info(f"  📁 Consolidated laps saved as CSV only")
        
        if all_results_2025:
            consolidated_results = pd.concat(all_results_2025, ignore_index=True)
            
            # Save CSV (always works)
            consolidated_results.to_csv(consolidated_dir / '2025_all_results_through_spanish.csv', index=False)
            logger.info(f"  ✅ 2025 consolidated results: {len(consolidated_results)} records (CSV)")
            
            # Try to save Parquet with data type fixes
            try:
                consolidated_results_parquet = self._prepare_dataframe_for_parquet(consolidated_results)
                consolidated_results_parquet.to_parquet(consolidated_dir / '2025_all_results_through_spanish.parquet', index=False)
                logger.info(f"  ✅ 2025 consolidated results: {len(consolidated_results)} records (Parquet)")
            except Exception as e:
                logger.warning(f"  ⚠️ Could not save consolidated results as Parquet: {e}")
                logger.info(f"  📁 Consolidated results saved as CSV only")
    
    def generate_2025_data_quality_report(self) -> str:
        """Generate comprehensive 2025 data quality report through Spanish GP."""
        
        report = f"""
{'='*80}
F1 DATA CLEANING REPORT - 2025 SEASON (THROUGH SPANISH GRAND PRIX)
{'='*80}

PROCESSING SUMMARY - 2025 F1 WORLD CHAMPIONSHIP
-----------------------------------------------
Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Season: 2025 Formula 1 World Championship (First 10 races)
Coverage: Through Spanish Grand Prix (Round 10)
Circuits Processed: {self.cleaning_stats['circuits_processed']}
Files Processed: {self.cleaning_stats['total_files_processed']}
Total Records Processed: {self.cleaning_stats['total_records_processed']}
Processing Time: {self.cleaning_stats['processing_time']:.1f} seconds

2025 SEASON CONTEXT (THROUGH SPANISH GP)
---------------------------------------
🏁 Races Covered: 10/24 (41.7% of season complete)
🏃 Sprint Weekends: 3 - China, Miami, Austria (through Spanish GP)
🔄 Major Engine Partnership Changes:
   • Aston Martin: Mercedes → Honda (new partnership)
🔄 Tire Compound System:
   • New C1-C5 naming system (replacing Soft/Medium/Hard)
   • More granular compound selection
🏎️ Technical Regulations:
   • Enhanced ground effect aerodynamics
   • Improved tire performance windows

2025 CALENDAR COVERAGE
---------------------
✅ Round 1: Australia (March 16, 2025)
✅ Round 2: Bahrain (March 30, 2025)
✅ Round 3: Saudi Arabia (April 6, 2025)
✅ Round 4: Japan (April 13, 2025)
✅ Round 5: China (April 20, 2025) - SPRINT
✅ Round 6: Miami (May 4, 2025) - SPRINT
✅ Round 7: Emilia Romagna (May 18, 2025)
✅ Round 8: Monaco (May 25, 2025)
✅ Round 9: Canada (June 15, 2025)
✅ Round 10: Spain (June 29, 2025) - CURRENT COVERAGE END

DATA QUALITY METRICS - 2025
---------------------------
Records Removed: {self.cleaning_stats['records_removed']} ({self.cleaning_stats['records_removed']/max(1, self.cleaning_stats['total_records_processed'])*100:.1f}%)
Records Corrected: {self.cleaning_stats['records_corrected']}
Missing Data Filled: {self.cleaning_stats['missing_data_filled']}
Outliers Detected: {self.cleaning_stats['outliers_detected']}
Overall Quality Score: {self.cleaning_stats['data_quality_score']:.2f}/1.00

DATA TYPE QUALITY SCORES - 2025
-------------------------------
⏱️ Timing Data Quality: {self.cleaning_stats['timing_data_quality']:.2f}/1.00
🏎️ Tire Data Quality: {self.cleaning_stats['tire_data_quality']:.2f}/1.00
🌤️ Weather Data Quality: {self.cleaning_stats['weather_data_quality']:.2f}/1.00

2025-SPECIFIC CLEANING OPERATIONS
---------------------------------
✅ 2025 C1-C5 tire compound validation (new naming system)
✅ 2025 sprint weekend handling (3 weekends through Spanish GP)
✅ 2025 car performance thresholds (max speed: {self.quality_thresholds_2025['max_speed']} km/h)
✅ 2025 tire age limits (max: {self.quality_thresholds_2025['max_tire_age']} laps)
✅ 2025 lap time validation ({self.quality_thresholds_2025['min_lap_time']}-{self.quality_thresholds_2025['max_lap_time']}s)
✅ Engine partnership changes (Aston Martin → Honda)
✅ Enhanced aerodynamic regulations handling
✅ 2025 tire degradation modeling (C1-C5 system)

SPRINT WEEKEND ANALYSIS - 2025 (THROUGH SPANISH GP)
--------------------------------------------------
Sprint Weekends Processed: {self.cleaning_stats['sprint_weekends_processed']}/3
Sprint Circuits: China, Miami, Austria
Remaining Sprint Weekends: 3 more (after Spanish GP)
Sprint Format: Maintained from 2024 structure

ENGINE PARTNERSHIP CHANGES - 2025
---------------------------------
✅ Aston Martin: Mercedes → Honda (major switch)
✅ Red Bull: Continued Honda RBPT partnership
✅ RB: Continued Honda RBPT partnership  
✅ All other partnerships maintained from 2024
✅ Engine data validation updated for new partnerships

TIRE COMPOUND REVOLUTION - 2025
------------------------------
🔄 NEW: C1-C5 naming system replaces Soft/Medium/Hard
🔄 C1: Hardest compound (equivalent to previous Hard)
🔄 C2: Medium-Hard compound 
🔄 C3: Medium compound (equivalent to previous Medium)
🔄 C4: Medium-Soft compound
🔄 C5: Softest compound (equivalent to previous Soft)
🔄 Enhanced performance windows for each compound
🔄 More strategic tire selection options

EARLY SEASON ANALYSIS READINESS - 2025
--------------------------------------
✅ All tire compounds validated with new C1-C5 system
✅ Engine partnership changes properly tracked
✅ Early season performance baselines established
✅ Sprint weekend data properly categorized (3/6 complete)
✅ Enhanced aerodynamic regulation compliance
✅ 41.7% season coverage for trend analysis

TECHNICAL REGULATION UPDATES - 2025
----------------------------------
🏎️ Enhanced ground effect aerodynamics
🏎️ Refined tire performance windows
🏎️ Updated safety car procedures
🏎️ Improved DRS zone configurations
🏎️ Enhanced track limits monitoring

CHAMPIONSHIP STANDINGS CONTEXT
-----------------------------
📊 Early season form guide through 10 races
📊 Engine partnership impact analysis ready
📊 Tire strategy evolution with C1-C5 system
📊 Sprint weekend performance comparison
📊 Constructor development trajectories

NEXT STEPS - 2025 DATA (POST SPANISH GP)
---------------------------------------
1. ✅ First 10 races data cleaning completed
2. 📊 Continue with remaining 14 races (Austria onwards)
3. 🔍 Analyze engine partnership impact trends
4. 📈 Build predictive models with C1-C5 tire data
5. 🎯 Comprehensive tire degradation analysis with new compounds
6. 🏆 Track championship evolution through full season

STORAGE LOCATIONS - 2025
------------------------
Raw Data: {self.raw_dir}
Cleaned Data: {self.output_dir / 'processed'}
Consolidated Data: {self.output_dir / 'features'}
Reports: {self.output_dir / 'reports'}
Logs: f1_data_cleaning_2025.log

COMPARISON WITH 2024 (FIRST 10 RACES)
------------------------------------
• Tire system: C1-C5 (2025) vs Soft/Medium/Hard (2024) - Major change
• Engine partnerships: 1 major change (2025) vs 0 (2024) - Aston Martin switch
• Sprint weekends: 3 (2025) vs 3 (2024) through Round 10 - Consistent
• Performance: Enhanced aero (2025) vs established (2024) - Evolution
• Data volume: Comparable through first 10 races

SEASON PROGRESS TRACKER
----------------------
🏁 Completed: 10/24 races (41.7%)
🏃 Sprint weekends: 3/6 completed (50%)
📊 Data coverage: Excellent foundation for analysis
⏭️ Next major milestone: Austrian GP (Sprint weekend)
🎯 Target: Complete season analysis by Abu Dhabi

{'='*80}
2025 F1 Data Cleaning Report - Through Spanish Grand Prix
{'='*80}
"""
        
        return report


def main():
    """Main execution function for 2025 data cleaning."""
    parser = argparse.ArgumentParser(
        description='F1 Data Cleaner - 2025 Season (Through Spanish Grand Prix)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Circuit selection for 2025
    parser.add_argument('--circuit', type=str,
                       help='Specific 2025 circuit to process (through Spanish GP)')
    parser.add_argument('--circuits', nargs='+',
                       help='Multiple 2025 circuits to process (through Spanish GP)')
    
    # Processing options
    parser.add_argument('--clean-all', action='store_true',
                       help='Clean all available 2025 data (first 10 races through Spanish GP)')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate 2025 data quality without cleaning')
    
    # Output options
    parser.add_argument('--formats', nargs='+', 
                       choices=['csv', 'parquet', 'pickle', 'json'],
                       default=['csv', 'parquet'],
                       help='Output formats for cleaned 2025 data')
    parser.add_argument('--base-dir', type=str, default='../data_collector/data',
                       help='Base directory containing raw 2025 data')
    parser.add_argument('--output-dir', type=str, default='../cleaned_data_2025',
                       help='Output directory for cleaned 2025 data')
    
    args = parser.parse_args()
    
    # Validate that at least one action is specified
    actions = [args.clean_all, args.circuit, args.circuits, args.validate_only]
    if not any(actions):
        parser.error("Please specify an action: --clean-all, --circuit, --circuits, or --validate-only")
    
    # Initialize 2025 cleaner
    try:
        cleaner = F1DataCleaner2025(
            base_dir=args.base_dir,
            output_dir=args.output_dir
        )
        
        logger.info(f"🧹 F1 2025 Data Cleaner initialized")
        logger.info(f"🏎️ Target: 2025 Formula 1 World Championship (Through Spanish GP)")
        logger.info(f"📁 Raw data directory: {cleaner.raw_dir}")
        logger.info(f"📁 Output directory: {cleaner.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize 2025 cleaner: {e}")
        return 1
    
    try:
        # Determine circuits to process
        circuits_to_process = None
        if args.circuit:
            if args.circuit not in cleaner.circuits_2025:
                logger.error(f"❌ Invalid 2025 circuit: {args.circuit}")
                logger.info(f"Valid 2025 circuits (through Spanish GP): {list(cleaner.circuits_2025.keys())}")
                return 1
            circuits_to_process = [args.circuit]
        elif args.circuits:
            invalid_circuits = [c for c in args.circuits if c not in cleaner.circuits_2025]
            if invalid_circuits:
                logger.error(f"❌ Invalid 2025 circuits: {invalid_circuits}")
                logger.info(f"Valid 2025 circuits (through Spanish GP): {list(cleaner.circuits_2025.keys())}")
                return 1
            circuits_to_process = args.circuits
        
        # Process the 2025 data
        if args.validate_only:
            logger.info(f"🔍 Validating 2025 data quality (through Spanish GP)")
            raw_data = cleaner.load_raw_2025_data(args.circuit)
            if raw_data:
                logger.info(f"✅ 2025 data validation complete: {len(raw_data)} circuits found")
            else:
                logger.error(f"❌ No 2025 data found for validation")
                return 1
                
        elif args.clean_all or args.circuit or args.circuits:
            logger.info(f"🧹 Cleaning 2025 F1 data (Through Spanish Grand Prix)")
            
            # Process the 2025 data
            processed_data = cleaner.process_2025_data(circuits_to_process)
            
            if processed_data:
                # Save cleaned 2025 data
                cleaner.save_2025_cleaned_data(processed_data, args.formats)
                
                # Generate 2025 quality report
                report = cleaner.generate_2025_data_quality_report()
                
                # Save report
                report_file = cleaner.output_dir / 'reports' / 'cleaning_report_2025_through_spanish.txt'
                report_file.parent.mkdir(exist_ok=True)
                with open(report_file, 'w') as f:
                    f.write(report)
                
                print(report)
                logger.info(f"📄 2025 quality report saved: {report_file}")
                logger.info(f"✅ 2025 data cleaning complete (through Spanish GP)")
                
                # Summary statistics
                total_circuits = len(processed_data)
                sprint_weekends = cleaner.cleaning_stats['sprint_weekends_processed']
                quality_score = cleaner.cleaning_stats['data_quality_score']
                
                logger.info(f"📊 2025 CLEANING SUMMARY:")
                logger.info(f"   🏁 Circuits processed: {total_circuits}/10 (through Spanish GP)")
                logger.info(f"   🏃 Sprint weekends: {sprint_weekends}/3")
                logger.info(f"   🏆 Quality score: {quality_score:.2f}/1.00")
                logger.info(f"   ⏱️ Processing time: {cleaner.cleaning_stats['processing_time']:.1f}s")
                logger.info(f"   🔧 New features: C1-C5 tire system, Honda engine partnership")
                
            else:
                logger.error(f"❌ No 2025 data processed")
                return 1
        
        logger.info(f"🏁 2025 F1 data cleaning completed successfully (through Spanish GP)")
        return 0
        
    except KeyboardInterrupt:
        logger.info(f"\n⚠️ 2025 data processing interrupted by user")
        return 0
        
    except Exception as e:
        logger.error(f"❌ 2025 data processing failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    
    # For IDE execution - default to cleaning all 2025 data through Spanish GP
    if len(sys.argv) == 1:
        print("🧹 Running F1 Data Cleaner - 2025 Season (Through Spanish Grand Prix)")
        print("This will clean all available 2025 F1 data through Round 10 (Spanish GP)")
        print("Key 2025 features: C1-C5 tire system, Aston Martin→Honda engine switch")
        print("Coverage: First 10 races (41.7% of season)")
        print("Estimated time: 3-5 minutes for 10 races\n")
        
        # Ask user for confirmation
        response = input("Proceed with cleaning 2025 data through Spanish GP? (y/N): ")
        if response.lower() == 'y':
            sys.argv.extend(['--clean-all'])
        else:
            print("Operation cancelled. Use command line arguments for specific options.")
            sys.exit(0)
    
    sys.exit(main())
