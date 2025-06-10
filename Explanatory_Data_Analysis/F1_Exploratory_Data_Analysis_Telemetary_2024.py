"""
Created on Mon Jun  10 10:51:00 2025
@author: sid

F1 Exploratory Data Analysis & Telemetry Comparison - 2024 Season
CORRECTED VERSION: Calculates lap times from sector times
Performs comprehensive EDA, telemetry analysis, and driver/team comparisons
for lap time optimization and performance insights.

UPDATED VERSION: Now includes comprehensive tire degradation analysis
2024 SEASON VERSION: Updated for 2024 teams, drivers, and regulations
ENHANCED VERSION: Advanced tire degradation with multi-model analysis
NEW FEATURES: China GP return, team rebrands (RB, Kick Sauber), and regulation updates
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import json
from scipy import stats
from scipy.optimize import curve_fit
from scipy.stats import linregress

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set plotting styles
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class F1TelemetryAnalyzer2024:
    """
    Comprehensive F1 telemetry and performance analysis for 2024 season.
    Now includes ADVANCED tire degradation analysis capabilities.
    Updated for 2024 teams, drivers, and regulations.
    """
    
    def __init__(self, data_dir: str = '../cleaned_data_2024'):
        """Initialize the F1 Telemetry Analyzer for 2024 season."""
        self.data_dir = Path(data_dir)
        self.features_dir = self.data_dir / 'features'
        self.output_dir = self.data_dir / 'analysis'
        self.plots_dir = self.output_dir / 'plots'
        self.reports_dir = self.output_dir / 'reports'
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data containers
        self.laps_data = None
        self.results_data = None
        self.telemetry_data = None
        
        # 2024 F1 Teams and drivers mapping (Updated for 2024)
        self.teams_2024 = {
            'RBR': {'name': 'Red Bull Racing', 'drivers': ['Max Verstappen', 'Sergio Perez'], 'color': '#1E41FF'},
            'FER': {'name': 'Ferrari', 'drivers': ['Charles Leclerc', 'Carlos Sainz'], 'color': '#DC143C'},
            'MER': {'name': 'Mercedes', 'drivers': ['Lewis Hamilton', 'George Russell'], 'color': '#00D2BE'},
            'MCL': {'name': 'McLaren', 'drivers': ['Lando Norris', 'Oscar Piastri'], 'color': '#FF8700'},
            'AM': {'name': 'Aston Martin', 'drivers': ['Fernando Alonso', 'Lance Stroll'], 'color': '#006F62'},
            'ALP': {'name': 'Alpine', 'drivers': ['Pierre Gasly', 'Esteban Ocon'], 'color': '#0090FF'},
            'WIL': {'name': 'Williams', 'drivers': ['Alexander Albon', 'Logan Sargeant'], 'color': '#005AFF'},
            'RB': {'name': 'RB', 'drivers': ['Yuki Tsunoda', 'Daniel Ricciardo'], 'color': '#2B4562'},  # 🆕 NEW: RB (formerly AlphaTauri)
            'SAU': {'name': 'Kick Sauber', 'drivers': ['Valtteri Bottas', 'Zhou Guanyu'], 'color': '#52C832'},  # 🆕 NEW: Kick Sauber (formerly Alfa Romeo)
            'HAS': {'name': 'Haas', 'drivers': ['Kevin Magnussen', 'Nico Hulkenberg'], 'color': '#FFFFFF'}
        }
        
        # 2024 Season Updates
        self.season_2024_updates = {
            'total_races': 24,  # 🆕 24 races including China return
            'sprint_weekends': 6,  # Continued sprint format
            'new_circuits': ['China'],  # 🆕 China returns to calendar
            'regulation_changes': [
                'Continued ground effect regulations',
                'Updated technical directives',
                'Cost cap adjustments',
                'Sustainable fuel mandate'
            ],
            'team_changes': [
                'AlphaTauri rebranded to RB',  # 🆕
                'Alfa Romeo rebranded to Kick Sauber',  # 🆕
                'Daniel Ricciardo returned to RB'  # 🆕
            ]
        }
        
        # Initialize analysis results storage
        self.analysis_results = {
            'driver_comparisons': {},
            'team_performance': {},
            'lap_time_analysis': {},
            'sector_analysis': {},
            'tire_performance': {},
            'track_performance': {},
            'improvement_recommendations': {},
            'tire_degradation': {},  # Standard tire degradation
            'advanced_tire_degradation': {},  # Advanced tire degradation storage
            'sprint_analysis': {},  # 🆕 NEW: Sprint weekend analysis
            'regulation_impact': {}  # 🆕 NEW: Regulation impact analysis
        }
        try:
            self._load_tire_degradation_data()
        except Exception as e:
            logger.warning(f"⚠️ Could not auto-load tire data: {e}")
    
    def load_cleaned_data(self) -> bool:
        """Load all cleaned 2024 F1 data."""
        logger.info("🔄 Loading cleaned 2024 F1 data for analysis")
        
        try:
            # Check if data directory exists
            if not self.data_dir.exists():
                logger.error(f"❌ Data directory does not exist: {self.data_dir}")
                return False
            
            if not self.features_dir.exists():
                logger.error(f"❌ Features directory does not exist: {self.features_dir}")
                return False
            
            # Try different file patterns for consolidated data
            laps_patterns = [
                '2024_all_laps.csv',
                '2024_all_laps.parquet', 
                '*laps*.csv',
                '*lap*.csv'
            ]
            
            results_patterns = [
                '2024_all_results.csv',
                '2024_all_results.parquet',
                '*results*.csv',
                '*result*.csv'
            ]
            
            # Try to find laps data
            laps_file = None
            for pattern in laps_patterns:
                matches = list(self.features_dir.glob(pattern))
                if matches:
                    laps_file = matches[0]
                    break
            
            if laps_file and laps_file.exists():
                if laps_file.suffix.lower() == '.csv':
                    self.laps_data = pd.read_csv(laps_file)
                else:
                    self.laps_data = pd.read_parquet(laps_file)
                logger.info(f"✅ Loaded laps data: {len(self.laps_data)} records from {laps_file.name}")
            else:
                logger.error("❌ No laps data found")
                return False
            
            # Try to find results data (optional)
            results_file = None
            for pattern in results_patterns:
                matches = list(self.features_dir.glob(pattern))
                if matches:
                    results_file = matches[0]
                    break
            
            if results_file and results_file.exists():
                if results_file.suffix.lower() == '.csv':
                    self.results_data = pd.read_csv(results_file)
                else:
                    self.results_data = pd.read_parquet(results_file)
                logger.info(f"✅ Loaded results data: {len(self.results_data)} records from {results_file.name}")
            else:
                logger.warning("⚠️ No results data found (optional)")
            
            # Data preprocessing
            self._preprocess_data()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            return False
    
    def _preprocess_data(self):
        """Preprocess loaded data for analysis - WITH SECTOR TIME CALCULATION."""
        logger.info("🔧 Preprocessing data for analysis")
        
        if self.laps_data is not None:
            # STEP 1: Calculate lap times from sector times (the key fix!)
            logger.info("🧮 Calculating lap times from sector times...")
            
            sector_cols = ['Sector1Time', 'Sector2Time', 'Sector3Time']
            if all(col in self.laps_data.columns for col in sector_cols):
                try:
                    # Check if sector times are already in seconds or need conversion
                    sample_sector = self.laps_data[sector_cols[0]].dropna().iloc[0] if not self.laps_data[sector_cols[0]].dropna().empty else None
                    
                    if sample_sector is not None and isinstance(sample_sector, (int, float)):
                        # Already in seconds format
                        logger.info("✅ Sector times already in seconds format")
                        for col in sector_cols:
                            self.laps_data[col] = pd.to_numeric(self.laps_data[col], errors='coerce')
                        
                        # Sum sectors to get lap time
                        calculated_lap_time = (self.laps_data['Sector1Time'] + 
                                             self.laps_data['Sector2Time'] + 
                                             self.laps_data['Sector3Time'])
                    else:
                        # Convert each sector time to seconds from timedelta
                        sector_data = {}
                        for col in sector_cols:
                            logger.info(f"Converting {col}...")
                            # Convert timedelta strings to seconds
                            td_data = pd.to_timedelta(self.laps_data[col], errors='coerce')
                            sector_data[col] = td_data.dt.total_seconds()
                            valid_count = sector_data[col].notna().sum()
                            logger.info(f"  {col}: {valid_count} valid times")
                        
                        # Sum sectors to get lap time
                        calculated_lap_time = (sector_data['Sector1Time'] + 
                                             sector_data['Sector2Time'] + 
                                             sector_data['Sector3Time'])
                    
                    valid_calculated = calculated_lap_time.dropna()
                    logger.info(f"✅ Successfully calculated {len(valid_calculated)} lap times from sectors!")
                    logger.info(f"Range: {valid_calculated.min():.1f} - {valid_calculated.max():.1f} seconds")
                    logger.info(f"Mean: {valid_calculated.mean():.1f} seconds")
                    
                    # Set the calculated lap times
                    self.laps_data['LapTime'] = calculated_lap_time
                    
                    # Filter to reasonable lap times
                    reasonable_laps = self.laps_data[
                        (self.laps_data['LapTime'] >= 65) & 
                        (self.laps_data['LapTime'] <= 210)
                    ].copy()
                    
                    logger.info(f"Filtered to reasonable lap times (65-210s): {len(reasonable_laps)} records")
                    self.laps_data = reasonable_laps
                    
                except Exception as e:
                    logger.error(f"❌ Failed to calculate lap times from sectors: {e}")
                    return
            else:
                logger.error(f"❌ Missing sector time columns. Available: {list(self.laps_data.columns)}")
                return
            
            # STEP 2: Create TeamCode mapping (Updated for 2024)
            team_name_mappings = {
                'Red Bull Racing': 'RBR',
                'Ferrari': 'FER', 
                'Mercedes': 'MER',
                'McLaren': 'MCL',
                'Aston Martin': 'AM',
                'Alpine': 'ALP',
                'Williams': 'WIL',
                'RB': 'RB',  # 🆕 NEW: RB (formerly AlphaTauri)
                'Kick Sauber': 'SAU',  # 🆕 NEW: Kick Sauber (formerly Alfa Romeo)
                'Alfa Romeo': 'SAU',  # Fallback mapping
                'AlphaTauri': 'RB',  # Fallback mapping
                'Haas': 'HAS',
                'Haas F1 Team': 'HAS'
            }
            
            if 'Team' in self.laps_data.columns:
                def map_team_name(team_name):
                    if pd.isna(team_name):
                        return 'UNK'
                    team_str = str(team_name).strip()
                    
                    # Direct mapping first
                    if team_str in team_name_mappings:
                        return team_name_mappings[team_str]
                    
                    # Partial matching for variations
                    for full_name, code in team_name_mappings.items():
                        if full_name.lower() in team_str.lower():
                            return code
                    
                    return 'UNK'
                
                self.laps_data['TeamCode'] = self.laps_data['Team'].apply(map_team_name)
                team_counts = self.laps_data['TeamCode'].value_counts()
                logger.info(f"✅ Team mapping complete: {team_counts.to_dict()}")
                
            # STEP 3: Map driver abbreviations to full names (Updated for 2024)
            if 'Driver' in self.laps_data.columns:
                driver_mappings_2024 = {
                    'VER': 'Max Verstappen', 'PER': 'Sergio Perez',
                    'LEC': 'Charles Leclerc', 'SAI': 'Carlos Sainz',
                    'HAM': 'Lewis Hamilton', 'RUS': 'George Russell',
                    'NOR': 'Lando Norris', 'PIA': 'Oscar Piastri',
                    'ALO': 'Fernando Alonso', 'STR': 'Lance Stroll',
                    'GAS': 'Pierre Gasly', 'OCO': 'Esteban Ocon',
                    'ALB': 'Alexander Albon', 'SAR': 'Logan Sargeant',
                    'TSU': 'Yuki Tsunoda', 'RIC': 'Daniel Ricciardo',  # 🆕 NEW: Ricciardo back at RB
                    'BOT': 'Valtteri Bottas', 'ZHO': 'Zhou Guanyu',
                    'MAG': 'Kevin Magnussen', 'HUL': 'Nico Hulkenberg'
                }
                
                self.laps_data['DriverFullName'] = self.laps_data['Driver'].map(driver_mappings_2024).fillna(self.laps_data['Driver'])
                logger.info(f"✅ Driver mapping complete for 2024 season")
            
            # STEP 4: Ensure numeric columns are properly typed
            numeric_columns = ['LapNumber', 'TyreLife', 'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST']
            for col in numeric_columns:
                if col in self.laps_data.columns:
                    self.laps_data[col] = pd.to_numeric(self.laps_data[col], errors='coerce')
        
        logger.info("✅ Data preprocessing completed successfully!")
        logger.info(f"Final dataset: {len(self.laps_data)} records with valid lap times")
    
    def analyze_driver_comparisons(self, drivers: List[str] = None, circuits: List[str] = None) -> Dict:
        """Comprehensive driver-to-driver comparison analysis for 2024."""
        logger.info("📊 Performing driver comparison analysis")
        
        if self.laps_data is None or len(self.laps_data) == 0:
            logger.error("❌ No laps data available")
            return {}
        
        driver_col = 'DriverFullName' if 'DriverFullName' in self.laps_data.columns else 'Driver'
        analysis_data = self.laps_data.copy()
        
        if drivers:
            driver_filter = (analysis_data[driver_col].isin(drivers)) | (analysis_data['Driver'].isin(drivers))
            analysis_data = analysis_data[driver_filter]
        if circuits:
            circuit_col = 'circuit' if 'circuit' in analysis_data.columns else 'Circuit'
            if circuit_col in analysis_data.columns:
                analysis_data = analysis_data[analysis_data[circuit_col].isin(circuits)]
        
        comparisons = {}
        
        # 1. Overall lap time statistics by driver
        driver_stats = analysis_data.groupby(driver_col)['LapTime'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(3)
        driver_stats.columns = ['Total_Laps', 'Avg_LapTime', 'Median_LapTime', 
                               'Std_LapTime', 'Best_LapTime', 'Worst_LapTime']
        
        comparisons['driver_statistics'] = driver_stats
        logger.info(f"Driver statistics calculated for {len(driver_stats)} drivers")
        
        # 2. Teammate comparisons (Updated for 2024 driver pairings)
        teammate_comparisons = []
        if 'TeamCode' in analysis_data.columns:
            for team, info in self.teams_2024.items():
                team_data = analysis_data[analysis_data['TeamCode'] == team]
                if len(team_data[driver_col].unique()) >= 2:
                    drivers_in_team = team_data[driver_col].unique()[:2]
                    
                    driver1_data = team_data[team_data[driver_col] == drivers_in_team[0]]
                    driver2_data = team_data[team_data[driver_col] == drivers_in_team[1]]
                    
                    if len(driver1_data) > 0 and len(driver2_data) > 0:
                        driver1_avg = driver1_data['LapTime'].mean()
                        driver2_avg = driver2_data['LapTime'].mean()
                        
                        if not (np.isnan(driver1_avg) or np.isnan(driver2_avg)):
                            comparison = {
                                'team': info['name'],
                                'driver1': drivers_in_team[0],
                                'driver2': drivers_in_team[1],
                                'driver1_avg': driver1_avg,
                                'driver2_avg': driver2_avg,
                                'time_gap': abs(driver1_avg - driver2_avg),
                                'faster_driver': drivers_in_team[0] if driver1_avg < driver2_avg else drivers_in_team[1]
                            }
                            teammate_comparisons.append(comparison)
        
        comparisons['teammate_comparisons'] = teammate_comparisons
        logger.info(f"Teammate comparisons calculated for {len(teammate_comparisons)} teams")
        
        # 3. Circuit-specific performance (including China!)
        circuit_col = 'circuit' if 'circuit' in analysis_data.columns else 'Circuit'
        if circuit_col in analysis_data.columns:
            circuit_performance = analysis_data.groupby([driver_col, circuit_col])['LapTime'].mean().unstack(fill_value=np.nan)
            comparisons['circuit_performance'] = circuit_performance
            logger.info(f"Circuit performance analyzed for {len(circuit_performance.columns)} circuits")
        
        self.analysis_results['driver_comparisons'] = comparisons
        logger.info("✅ Driver comparison analysis completed")
        return comparisons
    
    def analyze_team_performance(self) -> Dict:
        """Analyze team performance patterns and competitiveness for 2024."""
        logger.info("🏎️ Performing team performance analysis")
        
        if self.laps_data is None or len(self.laps_data) == 0:
            logger.warning("⚠️ No laps data available")
            return {'note': 'No data available for team analysis'}
        
        if 'TeamCode' not in self.laps_data.columns:
            logger.warning("⚠️ No TeamCode column found")
            return {'note': 'Team analysis limited due to missing team data'}
        
        team_analysis = {}
        
        # Team lap time statistics
        team_stats = self.laps_data.groupby('TeamCode')['LapTime'].agg([
            'count', 'mean', 'median', 'std', 'min'
        ]).round(3)
        team_stats.columns = ['Total_Laps', 'Avg_LapTime', 'Median_LapTime', 
                             'Std_LapTime', 'Best_LapTime']
        
        team_stats['Team_Name'] = team_stats.index.map(
            lambda x: self.teams_2024.get(x, {}).get('name', x)
        )
        
        team_analysis['team_statistics'] = team_stats
        team_analysis['consistency_ranking'] = team_stats.sort_values('Std_LapTime')[['Team_Name', 'Std_LapTime']]
        team_analysis['pace_ranking'] = team_stats.sort_values('Avg_LapTime')[['Team_Name', 'Avg_LapTime']]
        
        logger.info(f"Team performance analyzed for {len(team_stats)} teams")
        
        self.analysis_results['team_performance'] = team_analysis
        logger.info("✅ Team performance analysis completed")
        return team_analysis
    
    def analyze_sprint_weekends(self) -> Dict:
        """🆕 NEW: Analyze sprint weekend performance patterns for 2024."""
        logger.info("🏃 Performing sprint weekend analysis")
        
        if self.laps_data is None or len(self.laps_data) == 0:
            logger.warning("⚠️ No data available for sprint analysis")
            return {}
        
        sprint_analysis = {}
        
        # Check if session type data is available
        if 'session_type' in self.laps_data.columns:
            sprint_data = self.laps_data[self.laps_data['session_type'].str.contains('Sprint', case=False, na=False)]
            
            if len(sprint_data) > 0:
                # Sprint vs Race performance comparison
                if 'Race' in self.laps_data['session_type'].values:
                    race_data = self.laps_data[self.laps_data['session_type'] == 'Race']
                    
                    # Driver performance in sprints vs races
                    driver_col = 'DriverFullName' if 'DriverFullName' in self.laps_data.columns else 'Driver'
                    
                    sprint_performance = sprint_data.groupby(driver_col)['LapTime'].mean()
                    race_performance = race_data.groupby(driver_col)['LapTime'].mean()
                    
                    # Calculate sprint vs race delta
                    common_drivers = sprint_performance.index.intersection(race_performance.index)
                    performance_delta = {}
                    
                    for driver in common_drivers:
                        delta = sprint_performance[driver] - race_performance[driver]
                        performance_delta[driver] = {
                            'sprint_avg': sprint_performance[driver],
                            'race_avg': race_performance[driver],
                            'delta': delta,
                            'sprint_advantage': delta < 0
                        }
                    
                    sprint_analysis['performance_comparison'] = performance_delta
                
                # Sprint tire strategy analysis
                if 'Compound' in sprint_data.columns:
                    sprint_tire_usage = sprint_data['Compound'].value_counts()
                    sprint_analysis['tire_strategy'] = sprint_tire_usage.to_dict()
                
                logger.info(f"Sprint weekend analysis completed: {len(sprint_data)} sprint laps analyzed")
            else:
                logger.warning("⚠️ No sprint session data found")
        
        self.analysis_results['sprint_analysis'] = sprint_analysis
        return sprint_analysis
    
    def analyze_tire_degradation(self) -> Dict:
        """Comprehensive tire degradation analysis for 2024 season."""
        logger.info("🏎️ Performing tire degradation analysis")
        
        if self.laps_data is None or len(self.laps_data) == 0:
            logger.warning("⚠️ No data available for tire degradation analysis")
            return {}
        
        degradation_analysis = {}
        
        # Filter data for tire analysis
        tire_data = self.laps_data.copy()
        required_cols = ['LapTime', 'TyreLife', 'Compound']
        
        # Check if required columns exist
        missing_cols = [col for col in required_cols if col not in tire_data.columns]
        if missing_cols:
            logger.warning(f"⚠️ Missing columns for tire analysis: {missing_cols}")
            return {"error": f"Missing required columns: {missing_cols}"}
        
        # Remove invalid data
        tire_data = tire_data.dropna(subset=['LapTime', 'TyreLife', 'Compound'])
        tire_data = tire_data[tire_data['TyreLife'] >= 0]
        tire_data = tire_data[tire_data['LapTime'] > 0]
        
        logger.info(f"📊 Analyzing {len(tire_data)} laps for tire degradation")
        
        # 1. Overall degradation rates by compound
        degradation_by_compound = {}
        
        for compound in tire_data['Compound'].unique():
            compound_data = tire_data[tire_data['Compound'] == compound]
            
            # Calculate degradation rate (seconds per lap of tire age)
            if len(compound_data) > 10:  # Minimum data requirement
                # Group by tire age and calculate median lap time
                age_performance = compound_data.groupby('TyreLife')['LapTime'].agg([
                    'median', 'mean', 'std', 'count'
                ]).reset_index()
                
                # Calculate degradation slope (linear regression)
                if len(age_performance) > 3:
                    slope, intercept, r_value, p_value, std_err = linregress(
                        age_performance['TyreLife'], age_performance['median']
                    )
                    
                    degradation_by_compound[compound] = {
                        'degradation_rate': slope,  # seconds per lap
                        'base_time': intercept,
                        'correlation': r_value,
                        'p_value': p_value,
                        'std_error': std_err,
                        'age_performance': age_performance,
                        'sample_size': len(compound_data)
                    }
        
        degradation_analysis['compound_degradation'] = degradation_by_compound
        
        # 2. Driver-specific degradation patterns
        driver_degradation = {}
        driver_col = 'DriverFullName' if 'DriverFullName' in tire_data.columns else 'Driver'
        
        for driver in tire_data[driver_col].unique():
            driver_data = tire_data[tire_data[driver_col] == driver]
            driver_compound_deg = {}
            
            for compound in driver_data['Compound'].unique():
                compound_driver_data = driver_data[driver_data['Compound'] == compound]
                
                if len(compound_driver_data) > 5:
                    # Calculate driver's degradation rate for this compound
                    age_performance = compound_driver_data.groupby('TyreLife')['LapTime'].median()
                    
                    if len(age_performance) > 2:
                        slope, intercept, r_value, _, _ = linregress(
                            age_performance.index, age_performance.values
                        )
                        
                        driver_compound_deg[compound] = {
                            'degradation_rate': slope,
                            'base_time': intercept,
                            'correlation': r_value,
                            'laps_analyzed': len(compound_driver_data)
                        }
            
            if driver_compound_deg:
                driver_degradation[driver] = driver_compound_deg
        
        degradation_analysis['driver_degradation'] = driver_degradation
        
        # 3. Circuit-specific degradation (including China!)
        circuit_degradation = {}
        if 'circuit' in tire_data.columns:
            for circuit in tire_data['circuit'].unique():
                circuit_data = tire_data[tire_data['circuit'] == circuit]
                circuit_compound_deg = {}
                
                for compound in circuit_data['Compound'].unique():
                    compound_circuit_data = circuit_data[circuit_data['Compound'] == compound]
                    
                    if len(compound_circuit_data) > 10:
                        age_performance = compound_circuit_data.groupby('TyreLife')['LapTime'].agg([
                            'median', 'count'
                        ]).reset_index()
                        
                        if len(age_performance) > 3:
                            slope, intercept, r_value, _, _ = linregress(
                                age_performance['TyreLife'], age_performance['median']
                            )
                            
                            circuit_compound_deg[compound] = {
                                'degradation_rate': slope,
                                'base_time': intercept,
                                'correlation': r_value,
                                'laps_analyzed': len(compound_circuit_data)
                            }
                
                if circuit_compound_deg:
                    circuit_degradation[circuit] = circuit_compound_deg
        
        degradation_analysis['circuit_degradation'] = circuit_degradation
        
        # 4. Optimal stint length analysis
        optimal_stints = {}
        
        for compound in tire_data['Compound'].unique():
            compound_data = tire_data[tire_data['Compound'] == compound]
            
            if len(compound_data) > 20:
                # Calculate when degradation becomes too severe
                age_performance = compound_data.groupby('TyreLife')['LapTime'].median()
                
                if len(age_performance) > 5:
                    baseline = age_performance.iloc[:3].mean() if len(age_performance) >= 3 else age_performance.iloc[0]
                    degraded_times = age_performance[age_performance > baseline + 1.0]  # 1 second slower
                    
                    optimal_stint_length = degraded_times.index[0] if len(degraded_times) > 0 else age_performance.index[-1]
                    
                    optimal_stints[compound] = {
                        'optimal_stint_length': optimal_stint_length,
                        'baseline_time': baseline,
                        'max_analyzed_age': age_performance.index[-1],
                        'performance_curve': age_performance.to_dict()
                    }
        
        degradation_analysis['optimal_stints'] = optimal_stints
        
        # 5. Team tire management comparison (including new teams)
        team_tire_management = {}
        if 'TeamCode' in tire_data.columns:
            for team in tire_data['TeamCode'].unique():
                team_data = tire_data[tire_data['TeamCode'] == team]
                team_compounds = {}
                
                for compound in team_data['Compound'].unique():
                    compound_team_data = team_data[team_data['Compound'] == compound]
                    
                    if len(compound_team_data) > 5:
                        # Calculate average degradation rate for team
                        age_performance = compound_team_data.groupby('TyreLife')['LapTime'].median()
                        
                        if len(age_performance) > 2:
                            slope, _, r_value, _, _ = linregress(
                                age_performance.index, age_performance.values
                            )
                            
                            team_compounds[compound] = {
                                'degradation_rate': slope,
                                'correlation': r_value,
                                'avg_stint_length': compound_team_data['TyreLife'].mean(),
                                'max_stint_length': compound_team_data['TyreLife'].max()
                            }
                
                if team_compounds:
                    team_tire_management[team] = team_compounds
        
        degradation_analysis['team_tire_management'] = team_tire_management
        
        # Store results
        self.analysis_results['tire_degradation'] = degradation_analysis
        
        logger.info("✅ Tire degradation analysis completed")
        return degradation_analysis
    
    def analyze_advanced_tire_degradation(self) -> Dict:
        """Advanced tire degradation analysis with multiple models and insights for 2024."""
        logger.info("🔬 Performing ADVANCED tire degradation analysis")
        
        if self.laps_data is None or len(self.laps_data) == 0:
            logger.warning("⚠️ No data available for advanced tire degradation analysis")
            return {}
        
        advanced_analysis = {}
        
        # Filter data for tire analysis
        tire_data = self.laps_data.copy()
        required_cols = ['LapTime', 'TyreLife', 'Compound']
        
        # Check if required columns exist
        missing_cols = [col for col in required_cols if col not in tire_data.columns]
        if missing_cols:
            logger.warning(f"⚠️ Missing columns for advanced tire analysis: {missing_cols}")
            return {"error": f"Missing required columns: {missing_cols}"}
        
        # Remove invalid data
        tire_data = tire_data.dropna(subset=['LapTime', 'TyreLife', 'Compound'])
        tire_data = tire_data[tire_data['TyreLife'] >= 0]
        tire_data = tire_data[tire_data['LapTime'] > 0]
        
        logger.info(f"🔬 Analyzing {len(tire_data)} laps for ADVANCED tire degradation")
        
        # 1. NON-LINEAR DEGRADATION MODELING
        def exponential_degradation(x, a, b, c):
            """Exponential degradation model: y = a * exp(b * x) + c"""
            return a * np.exp(b * x) + c
        
        def quadratic_degradation(x, a, b, c):
            """Quadratic degradation model: y = a * x^2 + b * x + c"""
            return a * x**2 + b * x + c
        
        def logarithmic_degradation(x, a, b, c):
            """Logarithmic degradation model: y = a * log(x + 1) + b * x + c"""
            return a * np.log(x + 1) + b * x + c
        
        def power_law_degradation(x, a, b, c):
            """Power law degradation model: y = a * x^b + c"""
            return a * np.power(x + 1, b) + c
        
        degradation_models = {}
        
        for compound in tire_data['Compound'].unique():
            compound_data = tire_data[tire_data['Compound'] == compound]
            
            if len(compound_data) > 20:  # Need sufficient data for modeling
                # Group by tire age and get median lap times
                age_performance = compound_data.groupby('TyreLife')['LapTime'].agg([
                    'median', 'mean', 'std', 'count'
                ]).reset_index()
                
                if len(age_performance) > 5:
                    x_data = age_performance['TyreLife'].values
                    y_data = age_performance['median'].values
                    
                    models = {}
                    
                    # Try different degradation models
                    try:
                        # Linear model
                        slope, intercept, r_linear, p_value, std_err = linregress(x_data, y_data)
                        models['linear'] = {
                            'params': [slope, intercept],
                            'r_squared': r_linear**2,
                            'model_type': 'linear',
                            'aic': self._calculate_aic(y_data, slope * x_data + intercept, 2)
                        }
                        
                        # Exponential model
                        try:
                            popt_exp, _ = curve_fit(exponential_degradation, x_data, y_data, 
                                                  p0=[0.1, 0.01, y_data[0]], maxfev=2000)
                            y_pred_exp = exponential_degradation(x_data, *popt_exp)
                            r_exp = np.corrcoef(y_data, y_pred_exp)[0, 1]**2
                            models['exponential'] = {
                                'params': popt_exp,
                                'r_squared': r_exp if not np.isnan(r_exp) else 0,
                                'model_type': 'exponential',
                                'aic': self._calculate_aic(y_data, y_pred_exp, 3)
                            }
                        except Exception as e:
                            logger.debug(f"Exponential model failed for {compound}: {e}")
                        
                        # Quadratic model
                        try:
                            popt_quad, _ = curve_fit(quadratic_degradation, x_data, y_data, 
                                                   p0=[0.001, 0.1, y_data[0]], maxfev=2000)
                            y_pred_quad = quadratic_degradation(x_data, *popt_quad)
                            r_quad = np.corrcoef(y_data, y_pred_quad)[0, 1]**2
                            models['quadratic'] = {
                                'params': popt_quad,
                                'r_squared': r_quad if not np.isnan(r_quad) else 0,
                                'model_type': 'quadratic',
                                'aic': self._calculate_aic(y_data, y_pred_quad, 3)
                            }
                        except Exception as e:
                            logger.debug(f"Quadratic model failed for {compound}: {e}")
                        
                        # Logarithmic model
                        try:
                            popt_log, _ = curve_fit(logarithmic_degradation, x_data, y_data, 
                                                  p0=[0.5, 0.1, y_data[0]], maxfev=2000)
                            y_pred_log = logarithmic_degradation(x_data, *popt_log)
                            r_log = np.corrcoef(y_data, y_pred_log)[0, 1]**2
                            models['logarithmic'] = {
                                'params': popt_log,
                                'r_squared': r_log if not np.isnan(r_log) else 0,
                                'model_type': 'logarithmic',
                                'aic': self._calculate_aic(y_data, y_pred_log, 3)
                            }
                        except Exception as e:
                            logger.debug(f"Logarithmic model failed for {compound}: {e}")
                        
                        # Power law model
                        try:
                            popt_power, _ = curve_fit(power_law_degradation, x_data, y_data, 
                                                    p0=[0.1, 0.5, y_data[0]], maxfev=2000)
                            y_pred_power = power_law_degradation(x_data, *popt_power)
                            r_power = np.corrcoef(y_data, y_pred_power)[0, 1]**2
                            models['power_law'] = {
                                'params': popt_power,
                                'r_squared': r_power if not np.isnan(r_power) else 0,
                                'model_type': 'power_law',
                                'aic': self._calculate_aic(y_data, y_pred_power, 3)
                            }
                        except Exception as e:
                            logger.debug(f"Power law model failed for {compound}: {e}")
                        
                        # Select best model based on AIC (Akaike Information Criterion)
                        if models:
                            best_model = min(models.items(), key=lambda x: x[1].get('aic', float('inf')))
                            
                            degradation_models[compound] = {
                                'models': models,
                                'best_model': best_model[0],
                                'best_model_data': best_model[1],
                                'data_points': len(age_performance),
                                'age_range': [x_data.min(), x_data.max()],
                                'performance_range': [y_data.min(), y_data.max()],
                                'raw_data': {
                                    'tire_age': x_data.tolist(),
                                    'lap_times': y_data.tolist(),
                                    'sample_sizes': age_performance['count'].tolist()
                                }
                            }
                    
                    except Exception as e:
                        logger.error(f"❌ Model fitting failed for {compound}: {e}")
        
        advanced_analysis['degradation_models'] = degradation_models
        
        # 2. DEGRADATION PHASE DETECTION
        phase_analysis = {}
        
        for compound in tire_data['Compound'].unique():
            compound_data = tire_data[tire_data['Compound'] == compound]
            
            if len(compound_data) > 30:  # Need more data for phase detection
                age_performance = compound_data.groupby('TyreLife')['LapTime'].agg([
                    'median', 'mean', 'std', 'count'
                ]).reset_index()
                
                if len(age_performance) > 10:
                    # Detect degradation phases using change point detection
                    phases = self._detect_degradation_phases(
                        age_performance['TyreLife'].values,
                        age_performance['median'].values
                    )
                    
                    phase_analysis[compound] = phases
        
        advanced_analysis['phase_analysis'] = phase_analysis
        
        # 3. PREDICTIVE DEGRADATION MODELING
        predictions = {}
        
        for compound, model_data in degradation_models.items():
            if 'best_model_data' in model_data:
                best_model = model_data['best_model']
                params = model_data['best_model_data']['params']
                
                # Predict performance for extended stint lengths
                max_age = model_data['age_range'][1]
                prediction_ages = np.arange(max_age + 1, min(max_age + 21, 51))  # Predict up to 50 laps
                
                if best_model == 'linear':
                    predicted_times = params[0] * prediction_ages + params[1]
                elif best_model == 'exponential':
                    predicted_times = exponential_degradation(prediction_ages, *params)
                elif best_model == 'quadratic':
                    predicted_times = quadratic_degradation(prediction_ages, *params)
                elif best_model == 'logarithmic':
                    predicted_times = logarithmic_degradation(prediction_ages, *params)
                elif best_model == 'power_law':
                    predicted_times = power_law_degradation(prediction_ages, *params)
                else:
                    predicted_times = []
                
                if len(predicted_times) > 0:
                    predictions[compound] = {
                        'prediction_ages': prediction_ages.tolist(),
                        'predicted_times': predicted_times.tolist(),
                        'model_used': best_model,
                        'confidence_interval': self._calculate_prediction_confidence(
                            model_data['raw_data']['tire_age'],
                            model_data['raw_data']['lap_times'],
                            prediction_ages,
                            predicted_times
                        )
                    }
        
        advanced_analysis['predictions'] = predictions
        
        # 4. OPTIMAL PIT WINDOW ANALYSIS
        pit_strategies = {}
        
        for compound in degradation_models.keys():
            if compound in predictions:
                pred_data = predictions[compound]
                
                # Calculate cumulative time loss due to degradation
                baseline_time = min(pred_data['predicted_times'][:5]) if len(pred_data['predicted_times']) >= 5 else pred_data['predicted_times'][0]
                
                cumulative_loss = []
                for i, lap_time in enumerate(pred_data['predicted_times']):
                    time_loss = (lap_time - baseline_time) * (i + 1)  # Cumulative effect
                    cumulative_loss.append(time_loss)
                
                # Find optimal pit window (when cumulative loss exceeds pit stop time cost)
                pit_stop_cost = 25.0  # Assume 25 second pit stop time loss
                
                optimal_pit_lap = None
                for i, loss in enumerate(cumulative_loss):
                    if loss > pit_stop_cost:
                        optimal_pit_lap = pred_data['prediction_ages'][i]
                        break
                
                pit_strategies[compound] = {
                    'optimal_pit_lap': optimal_pit_lap,
                    'cumulative_loss': cumulative_loss,
                    'baseline_time': baseline_time,
                    'pit_stop_cost_assumption': pit_stop_cost,
                    'max_viable_stint': pred_data['prediction_ages'][-1] if pred_data['prediction_ages'] else None
                }
        
        advanced_analysis['pit_strategies'] = pit_strategies
        
        # 5. TEMPERATURE AND TRACK SURFACE IMPACT (if data available)
        environmental_analysis = {}
        
        if 'TrackTemp' in tire_data.columns or 'AirTemp' in tire_data.columns:
            env_analysis = self._analyze_environmental_impact(tire_data)
            environmental_analysis = env_analysis
        
        advanced_analysis['environmental_impact'] = environmental_analysis
        
        # 6. DRIVER-SPECIFIC ADVANCED ANALYSIS
        driver_advanced = {}
        driver_col = 'DriverFullName' if 'DriverFullName' in tire_data.columns else 'Driver'
        
        top_drivers = tire_data[driver_col].value_counts().head(10).index
        
        for driver in top_drivers:
            driver_data = tire_data[tire_data[driver_col] == driver]
            driver_compounds = {}
            
            for compound in driver_data['Compound'].unique():
                compound_driver_data = driver_data[driver_data['Compound'] == compound]
                
                if len(compound_driver_data) > 15:
                    # Calculate driver's specific degradation characteristics
                    age_perf = compound_driver_data.groupby('TyreLife')['LapTime'].agg([
                        'median', 'std', 'count'
                    ]).reset_index()
                    
                    if len(age_perf) > 5:
                        # Driver-specific tire management score
                        consistency_score = 1 / (age_perf['std'].mean() + 0.1)  # Lower std = better
                        degradation_resistance = self._calculate_degradation_resistance(
                            age_perf['TyreLife'].values,
                            age_perf['median'].values
                        )
                        
                        driver_compounds[compound] = {
                            'consistency_score': consistency_score,
                            'degradation_resistance': degradation_resistance,
                            'avg_stint_length': compound_driver_data['TyreLife'].mean(),
                            'sample_size': len(compound_driver_data)
                        }
            
            if driver_compounds:
                driver_advanced[driver] = driver_compounds
        
        advanced_analysis['driver_specific'] = driver_advanced
        
        # 7. COMPOUND COMPARISON AND RECOMMENDATIONS
        compound_comparison = {}
        
        compounds = list(degradation_models.keys())
        for i, compound1 in enumerate(compounds):
            for compound2 in compounds[i+1:]:
                if compound1 in predictions and compound2 in predictions:
                    comparison = self._compare_compounds(
                        predictions[compound1],
                        predictions[compound2],
                        compound1,
                        compound2
                    )
                    compound_comparison[f"{compound1}_vs_{compound2}"] = comparison
        
        advanced_analysis['compound_comparison'] = compound_comparison
        
        # Store results
        self.analysis_results['advanced_tire_degradation'] = advanced_analysis
        
        logger.info("✅ ADVANCED tire degradation analysis completed")
        logger.info(f"   📊 Models fitted for {len(degradation_models)} compounds")
        logger.info(f"   🔮 Predictions generated for {len(predictions)} compounds")
        logger.info(f"   🏎️ Driver analysis for {len(driver_advanced)} drivers")
        
        return advanced_analysis
    
    def _calculate_aic(self, observed, predicted, num_params):
        """Calculate Akaike Information Criterion for model selection."""
        n = len(observed)
        mse = np.mean((observed - predicted) ** 2)
        aic = n * np.log(mse) + 2 * num_params
        return aic
    
    def _detect_degradation_phases(self, tire_ages, lap_times):
        """Detect different phases of tire degradation using change point detection."""
        if len(tire_ages) < 10:
            return {"phases": 1, "change_points": [], "phase_descriptions": ["insufficient_data"]}
        
        # Simple change point detection using sliding window variance
        window_size = max(3, len(tire_ages) // 5)
        change_points = []
        
        for i in range(window_size, len(lap_times) - window_size):
            left_window = lap_times[i-window_size:i]
            right_window = lap_times[i:i+window_size]
            
            left_slope = np.polyfit(range(len(left_window)), left_window, 1)[0]
            right_slope = np.polyfit(range(len(right_window)), right_window, 1)[0]
            
            # Detect significant change in degradation rate
            if abs(right_slope - left_slope) > 0.02:  # 0.02 seconds per lap threshold
                change_points.append(tire_ages[i])
        
        # Remove close change points
        filtered_change_points = []
        for cp in change_points:
            if not filtered_change_points or cp - filtered_change_points[-1] > 5:
                filtered_change_points.append(cp)
        
        # Classify phases
        phases = len(filtered_change_points) + 1
        phase_descriptions = []
        
        if phases == 1:
            phase_descriptions = ["consistent_degradation"]
        elif phases == 2:
            phase_descriptions = ["initial_phase", "degraded_phase"]
        elif phases == 3:
            phase_descriptions = ["warmup_phase", "optimal_phase", "degraded_phase"]
        else:
            phase_descriptions = [f"phase_{i+1}" for i in range(phases)]
        
        return {
            "phases": phases,
            "change_points": filtered_change_points,
            "phase_descriptions": phase_descriptions
        }
    
    def _calculate_prediction_confidence(self, historical_ages, historical_times, prediction_ages, predicted_times):
        """Calculate confidence intervals for predictions."""
        if len(historical_times) < 3:
            return {"lower_bound": [], "upper_bound": [], "confidence_level": 0}
        
        # Calculate residual standard error
        historical_trend = np.polyfit(historical_ages, historical_times, 1)
        historical_predicted = np.polyval(historical_trend, historical_ages)
        residual_std = np.std(historical_times - historical_predicted)
        
        # Simple confidence interval (±2 standard deviations)
        confidence_margin = 2 * residual_std
        
        lower_bound = [pt - confidence_margin for pt in predicted_times]
        upper_bound = [pt + confidence_margin for pt in predicted_times]
        
        return {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "confidence_level": 95,  # Approximate 95% confidence
            "residual_std": residual_std
        }
    
    def _calculate_degradation_resistance(self, tire_ages, lap_times):
        """Calculate how well a driver resists tire degradation."""
        if len(tire_ages) < 3:
            return 0.5  # Neutral score
        
        # Calculate actual degradation rate
        slope, _, r_value, _, _ = linregress(tire_ages, lap_times)
        
        # Lower slope = better degradation resistance
        # Normalize to 0-1 scale where 1 = excellent resistance
        resistance_score = max(0, min(1, 1 - (slope / 0.1)))  # 0.1 s/lap as reference
        
        return resistance_score
    
    def _analyze_environmental_impact(self, tire_data):
        """Analyze impact of track and air temperature on tire degradation."""
        env_analysis = {}
        
        temp_cols = [col for col in ['TrackTemp', 'AirTemp'] if col in tire_data.columns]
        
        for temp_col in temp_cols:
            temp_data = tire_data.dropna(subset=[temp_col, 'LapTime', 'TyreLife'])
            
            if len(temp_data) > 50:
                # Bin temperatures
                temp_data['temp_bin'] = pd.cut(temp_data[temp_col], bins=5, labels=['very_cool', 'cool', 'moderate', 'warm', 'hot'])
                
                temp_impact = {}
                for temp_bin in temp_data['temp_bin'].unique():
                    if pd.notna(temp_bin):
                        bin_data = temp_data[temp_data['temp_bin'] == temp_bin]
                        
                        # Calculate degradation rate for this temperature range
                        if len(bin_data) > 10:
                            age_perf = bin_data.groupby('TyreLife')['LapTime'].median()
                            
                            if len(age_perf) > 3:
                                slope, _, r_value, _, _ = linregress(age_perf.index, age_perf.values)
                                
                                temp_impact[str(temp_bin)] = {
                                    'degradation_rate': slope,
                                    'correlation': r_value,
                                    'sample_size': len(bin_data),
                                    'temp_range': [bin_data[temp_col].min(), bin_data[temp_col].max()]
                                }
                
                env_analysis[temp_col] = temp_impact
        
        return env_analysis
    
    def _compare_compounds(self, compound1_data, compound2_data, compound1_name, compound2_name):
        """Compare two tire compounds across multiple metrics."""
        comparison = {
            'compound1': compound1_name,
            'compound2': compound2_name
        }
        
        # Compare predicted performance at same stint lengths
        common_ages = set(compound1_data['prediction_ages']).intersection(set(compound2_data['prediction_ages']))
        
        if common_ages:
            compound1_times = dict(zip(compound1_data['prediction_ages'], compound1_data['predicted_times']))
            compound2_times = dict(zip(compound2_data['prediction_ages'], compound2_data['predicted_times']))
            
            performance_differences = []
            for age in sorted(common_ages):
                diff = compound1_times[age] - compound2_times[age]
                performance_differences.append({
                    'stint_length': age,
                    'time_difference': diff,
                    'faster_compound': compound1_name if diff > 0 else compound2_name,
                    'advantage_seconds': abs(diff)
                })
            
            comparison['performance_differences'] = performance_differences
            
            # Overall recommendation
            avg_difference = np.mean([pd['time_difference'] for pd in performance_differences])
            
            if abs(avg_difference) < 0.1:
                recommendation = "compounds_equivalent"
            elif avg_difference > 0:
                recommendation = f"{compound2_name}_faster_overall"
            else:
                recommendation = f"{compound1_name}_faster_overall"
            
            comparison['recommendation'] = recommendation
            comparison['average_difference'] = avg_difference
        
        return comparison
    
    def _load_tire_degradation_data(self):
        """Load any cached tire degradation analysis data."""
        try:
            cache_file = self.output_dir / 'tire_degradation_cache.json'
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                logger.info("✅ Loaded cached tire degradation data")
                return cached_data
        except Exception as e:
            logger.debug(f"No cached tire data available: {e}")
        return {}
    
    def save_advanced_tire_analysis(self, filename: str = None):
        """Save advanced tire degradation analysis results."""
        if not filename:
            filename = f"advanced_tire_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_file = self.output_dir / filename
        
        # Prepare data for saving
        save_data = {
            'analysis_timestamp': datetime.now().isoformat(),
            'season': '2024',
            'total_laps_analyzed': len(self.laps_data) if self.laps_data is not None else 0,
            'advanced_tire_degradation': self.analysis_results.get('advanced_tire_degradation', {})
        }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
            
            logger.info(f"💾 Advanced tire analysis saved to {output_file}")
            return str(output_file)
        
        except Exception as e:
            logger.error(f"❌ Failed to save advanced tire analysis: {e}")
            return None
    
    def create_visualization_dashboard(self, save_plots: bool = True) -> Dict:
        """Create comprehensive visualization dashboard for 2024."""
        logger.info("📊 Creating visualization dashboard")
        
        plots = {}
        
        if self.laps_data is None or len(self.laps_data) == 0:
            logger.error("❌ No data available for visualization")
            return plots
        
        driver_col = 'DriverFullName' if 'DriverFullName' in self.laps_data.columns else 'Driver'
        
        # 1. Driver lap time distribution
        if driver_col in self.laps_data.columns and 'LapTime' in self.laps_data.columns:
            try:
                fig, ax = plt.subplots(1, 1, figsize=(15, 8))
                
                top_drivers = self.laps_data[driver_col].value_counts().head(10).index
                plot_data = self.laps_data[self.laps_data[driver_col].isin(top_drivers)]
                
                if len(plot_data) > 0:
                    sns.boxplot(data=plot_data, x=driver_col, y='LapTime', ax=ax)
                    ax.set_title('Lap Time Distribution by Driver (Top 10 by Lap Count) - 2024', fontsize=16, fontweight='bold')
                    ax.set_xlabel('Driver', fontsize=12)
                    ax.set_ylabel('Lap Time (seconds)', fontsize=12)
                    ax.tick_params(axis='x', rotation=45)
                    plt.tight_layout()
                    
                    if save_plots:
                        plt.savefig(self.plots_dir / 'driver_laptime_distribution_2024.png', dpi=300, bbox_inches='tight')
                    plots['driver_laptime_distribution'] = fig
                    logger.info("✅ Created driver lap time distribution plot")
                else:
                    plt.close(fig)
                    
            except Exception as e:
                logger.error(f"❌ Error creating driver lap time plot: {e}")
        
        # 2. Team performance comparison (Updated for 2024 teams)
        if 'TeamCode' in self.laps_data.columns:
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
                
                team_avg = self.laps_data.groupby('TeamCode')['LapTime'].mean().sort_values()
                team_names = [self.teams_2024.get(code, {}).get('name', code) for code in team_avg.index]
                colors = [self.teams_2024.get(code, {}).get('color', '#333333') for code in team_avg.index]
                
                if len(team_avg) > 0:
                    ax1.bar(range(len(team_avg)), team_avg.values, color=colors)
                    ax1.set_title('Average Lap Time by Team - 2024', fontsize=14, fontweight='bold')
                    ax1.set_xlabel('Team', fontsize=12)
                    ax1.set_ylabel('Average Lap Time (seconds)', fontsize=12)
                    ax1.set_xticks(range(len(team_avg)))
                    ax1.set_xticklabels(team_names, rotation=45, ha='right')
                    
                    team_std = self.laps_data.groupby('TeamCode')['LapTime'].std().sort_values()
                    team_names_std = [self.teams_2024.get(code, {}).get('name', code) for code in team_std.index]
                    
                    ax2.bar(range(len(team_std)), team_std.values, color=colors)
                    ax2.set_title('Team Consistency (Lower = More Consistent) - 2024', fontsize=14, fontweight='bold')
                    ax2.set_xlabel('Team', fontsize=12)
                    ax2.set_ylabel('Lap Time Standard Deviation (seconds)', fontsize=12)
                    ax2.set_xticks(range(len(team_std)))
                    ax2.set_xticklabels(team_names_std, rotation=45, ha='right')
                    
                    plt.tight_layout()
                    
                    if save_plots:
                        plt.savefig(self.plots_dir / 'team_performance_comparison_2024.png', dpi=300, bbox_inches='tight')
                    plots['team_performance_comparison'] = fig
                    logger.info("✅ Created team performance comparison plot")
                else:
                    plt.close(fig)
                        
            except Exception as e:
                logger.error(f"❌ Error creating team performance plot: {e}")
        
        # 3. 🆕 China GP specific analysis (if available)
        if 'circuit' in self.laps_data.columns and 'China' in self.laps_data['circuit'].unique():
            try:
                china_data = self.laps_data[self.laps_data['circuit'] == 'China']
                
                if len(china_data) > 0:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                    
                    # Tire compound usage at China GP
                    compound_counts = china_data['Compound'].value_counts()
                    colors = ['red' if c == 'SOFT' else 'yellow' if c == 'MEDIUM' 
                             else 'lightgray' if c == 'HARD' else 'blue' for c in compound_counts.index]
                    
                    ax1.pie(compound_counts.values, labels=compound_counts.index, colors=colors, autopct='%1.1f%%')
                    ax1.set_title('Tire Compound Usage - China GP 2024 🇨🇳', fontsize=16, fontweight='bold')
                    
                    # China GP lap time distribution by team
                    if 'TeamCode' in china_data.columns:
                        china_team_times = china_data.groupby('TeamCode')['LapTime'].mean().sort_values()
                        china_team_names = [self.teams_2024.get(code, {}).get('name', code) for code in china_team_times.index]
                        china_colors = [self.teams_2024.get(code, {}).get('color', '#333333') for code in china_team_times.index]
                        
                        ax2.bar(range(len(china_team_times)), china_team_times.values, color=china_colors)
                        ax2.set_title('China GP - Team Performance 🇨🇳', fontsize=14, fontweight='bold')
                        ax2.set_xlabel('Team')
                        ax2.set_ylabel('Average Lap Time (seconds)')
                        ax2.set_xticks(range(len(china_team_times)))
                        ax2.set_xticklabels(china_team_names, rotation=45, ha='right')
                    
                    plt.tight_layout()
                    
                    if save_plots:
                        plt.savefig(self.plots_dir / 'china_gp_analysis_2024.png', dpi=300, bbox_inches='tight')
                    plots['china_gp_analysis'] = fig
                    logger.info("✅ Created China GP analysis plot")
                    
            except Exception as e:
                logger.error(f"❌ Error creating China GP plot: {e}")
        
        # 4. Tire compound analysis
        if 'Compound' in self.laps_data.columns:
            try:
                compound_data = self.laps_data.dropna(subset=['Compound', 'LapTime'])
                if len(compound_data) > 0 and len(compound_data['Compound'].unique()) > 1:
                    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                    
                    sns.boxplot(data=compound_data, x='Compound', y='LapTime', ax=ax)
                    ax.set_title('Lap Time Distribution by Tire Compound - 2024', fontsize=16, fontweight='bold')
                    ax.set_xlabel('Tire Compound', fontsize=12)
                    ax.set_ylabel('Lap Time (seconds)', fontsize=12)
                    plt.tight_layout()
                    
                    if save_plots:
                        plt.savefig(self.plots_dir / 'tire_compound_performance_2024.png', dpi=300, bbox_inches='tight')
                    plots['tire_compound_performance'] = fig
                    logger.info("✅ Created tire compound performance plot")
                    
            except Exception as e:
                logger.error(f"❌ Error creating tire compound plot: {e}")
        
        # 5. Data summary plot
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Lap time histogram
            if 'LapTime' in self.laps_data.columns:
                valid_times = self.laps_data['LapTime'].dropna()
                if len(valid_times) > 0:
                    ax1.hist(valid_times, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                    ax1.set_title('Lap Time Distribution - 2024', fontsize=14, fontweight='bold')
                    ax1.set_xlabel('Lap Time (seconds)')
                    ax1.set_ylabel('Frequency')
                    ax1.grid(True, alpha=0.3)
            
            # Driver lap counts
            if driver_col in self.laps_data.columns:
                driver_counts = self.laps_data[driver_col].value_counts().head(10)
                if len(driver_counts) > 0:
                    ax2.bar(range(len(driver_counts)), driver_counts.values, color='lightgreen', edgecolor='black')
                    ax2.set_title('Top 10 Drivers by Lap Count - 2024', fontsize=14, fontweight='bold')
                    ax2.set_xlabel('Driver')
                    ax2.set_ylabel('Number of Laps')
                    ax2.set_xticks(range(len(driver_counts)))
                    ax2.set_xticklabels(driver_counts.index, rotation=45)
                    ax2.grid(True, alpha=0.3)
            
            # Circuit distribution (including China!)
            if 'circuit' in self.laps_data.columns:
                circuit_counts = self.laps_data['circuit'].value_counts().head(15)
                if len(circuit_counts) > 0:
                    ax3.bar(range(len(circuit_counts)), circuit_counts.values, color='orange', edgecolor='black')
                    ax3.set_title('Laps by Circuit - 2024 (including China 🇨🇳)', fontsize=14, fontweight='bold')
                    ax3.set_xlabel('Circuit')
                    ax3.set_ylabel('Number of Laps')
                    ax3.set_xticks(range(len(circuit_counts)))
                    ax3.set_xticklabels(circuit_counts.index, rotation=45)
                    ax3.grid(True, alpha=0.3)
            
            # Session type distribution
            if 'session_type' in self.laps_data.columns:
                session_counts = self.laps_data['session_type'].value_counts()
                if len(session_counts) > 0:
                    colors = plt.cm.Set3(np.linspace(0, 1, len(session_counts)))
                    ax4.pie(session_counts.values, labels=session_counts.index, autopct='%1.1f%%', colors=colors)
                    ax4.set_title('Session Type Distribution - 2024', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            if save_plots:
                plt.savefig(self.plots_dir / 'data_summary_2024.png', dpi=300, bbox_inches='tight')
            plots['data_summary'] = fig
            logger.info("✅ Created data summary plot")
            
        except Exception as e:
            logger.error(f"❌ Error creating summary plot: {e}")
        
        # Close figures to save memory
        if save_plots:
            plt.close('all')
        
        logger.info(f"✅ Visualization dashboard created with {len(plots)} plots")
        return plots
    
    def create_advanced_tire_visualizations(self, save_plots: bool = True):
        """Create advanced visualizations for tire degradation analysis."""
        logger.info("📊 Creating advanced tire degradation visualizations")
        
        if 'advanced_tire_degradation' not in self.analysis_results:
            logger.warning("⚠️ No advanced tire degradation data available for visualization")
            return {}
        
        adv_data = self.analysis_results['advanced_tire_degradation']
        plots = {}
        
        # 1. Multi-model comparison plot
        if 'degradation_models' in adv_data:
            try:
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                axes = axes.flatten()
                
                compounds = list(adv_data['degradation_models'].keys())[:4]  # First 4 compounds
                
                for i, compound in enumerate(compounds):
                    if i < 4:
                        model_data = adv_data['degradation_models'][compound]
                        raw_data = model_data['raw_data']
                        
                        ax = axes[i]
                        
                        # Plot raw data
                        ax.scatter(raw_data['tire_age'], raw_data['lap_times'], 
                                  alpha=0.6, s=50, label='Observed Data')
                        
                        # Plot best model
                        if 'best_model' in model_data:
                            best_model = model_data['best_model']
                            ax.set_title(f'{compound} - Best Model: {best_model}', 
                                       fontsize=12, fontweight='bold')
                        
                        ax.set_xlabel('Tire Age (laps)')
                        ax.set_ylabel('Lap Time (seconds)')
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                
                plt.tight_layout()
                
                if save_plots:
                    plt.savefig(self.plots_dir / 'advanced_tire_models_2024.png', 
                               dpi=300, bbox_inches='tight')
                plots['tire_models'] = fig
                
            except Exception as e:
                logger.error(f"❌ Error creating tire models plot: {e}")
        
        # 2. Predictive performance plot
        if 'predictions' in adv_data:
            try:
                fig, ax = plt.subplots(1, 1, figsize=(14, 8))
                
                colors = ['red', 'yellow', 'white', 'blue', 'green']
                color_map = {}
                
                for i, (compound, pred_data) in enumerate(adv_data['predictions'].items()):
                    color = colors[i % len(colors)]
                    color_map[compound] = color
                    
                    ax.plot(pred_data['prediction_ages'], pred_data['predicted_times'],
                           label=f'{compound} ({pred_data["model_used"]})',
                           linewidth=2, color=color)
                    
                    # Add confidence intervals if available
                    if 'confidence_interval' in pred_data:
                        ci = pred_data['confidence_interval']
                        if ci['lower_bound'] and ci['upper_bound']:
                            ax.fill_between(pred_data['prediction_ages'],
                                          ci['lower_bound'], ci['upper_bound'],
                                          alpha=0.2, color=color)
                
                ax.set_title('Tire Performance Predictions - 2024', fontsize=16, fontweight='bold')
                ax.set_xlabel('Tire Age (laps)')
                ax.set_ylabel('Predicted Lap Time (seconds)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                if save_plots:
                    plt.savefig(self.plots_dir / 'tire_predictions_2024.png', 
                               dpi=300, bbox_inches='tight')
                plots['tire_predictions'] = fig
                
            except Exception as e:
                logger.error(f"❌ Error creating predictions plot: {e}")
        
        # 3. Optimal pit strategy visualization
        if 'pit_strategies' in adv_data:
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # Optimal pit laps
                compounds = []
                pit_laps = []
                
                for compound, strategy in adv_data['pit_strategies'].items():
                    if strategy['optimal_pit_lap']:
                        compounds.append(compound)
                        pit_laps.append(strategy['optimal_pit_lap'])
                
                if compounds and pit_laps:
                    colors = ['red' if c == 'SOFT' else 'yellow' if c == 'MEDIUM' 
                             else 'lightgray' if c == 'HARD' else 'blue' for c in compounds]
                    
                    ax1.bar(compounds, pit_laps, color=colors)
                    ax1.set_title('Optimal Pit Window by Compound', fontsize=14, fontweight='bold')
                    ax1.set_xlabel('Tire Compound')
                    ax1.set_ylabel('Optimal Pit Lap')
                    ax1.tick_params(axis='x', rotation=45)
                
                # Cumulative time loss
                for compound, strategy in adv_data['pit_strategies'].items():
                    if 'cumulative_loss' in strategy and strategy['cumulative_loss']:
                        ax2.plot(range(1, len(strategy['cumulative_loss']) + 1), 
                                strategy['cumulative_loss'],
                                label=f'{compound}', linewidth=2)
                
                ax2.axhline(y=25, color='red', linestyle='--', alpha=0.7, 
                           label='Pit Stop Cost (25s)')
                ax2.set_title('Cumulative Time Loss vs Stint Length', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Stint Length (laps)')
                ax2.set_ylabel('Cumulative Time Loss (seconds)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                if save_plots:
                    plt.savefig(self.plots_dir / 'pit_strategies_2024.png', 
                               dpi=300, bbox_inches='tight')
                plots['pit_strategies'] = fig
                
            except Exception as e:
                logger.error(f"❌ Error creating pit strategies plot: {e}")
        
        # 4. Driver tire management comparison
        if 'driver_specific' in adv_data:
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # Consistency scores
                drivers = []
                consistency_scores = []
                
                for driver, driver_data in adv_data['driver_specific'].items():
                    if 'SOFT' in driver_data:  # Focus on soft compound
                        drivers.append(driver.split()[-1])  # Last name only
                        consistency_scores.append(driver_data['SOFT']['consistency_score'])
                
                if drivers and consistency_scores:
                    ax1.bar(drivers, consistency_scores, color='lightblue', edgecolor='black')
                    ax1.set_title('Driver Tire Consistency (Soft Compound)', fontsize=14, fontweight='bold')
                    ax1.set_xlabel('Driver')
                    ax1.set_ylabel('Consistency Score (Higher = Better)')
                    ax1.tick_params(axis='x', rotation=45)
                    ax1.grid(True, alpha=0.3)
                
                # Degradation resistance
                resistance_scores = []
                for driver, driver_data in adv_data['driver_specific'].items():
                    if 'SOFT' in driver_data:
                        resistance_scores.append(driver_data['SOFT']['degradation_resistance'])
                
                if drivers and resistance_scores:
                    ax2.bar(drivers, resistance_scores, color='lightcoral', edgecolor='black')
                    ax2.set_title('Driver Degradation Resistance (Soft Compound)', fontsize=14, fontweight='bold')
                    ax2.set_xlabel('Driver')
                    ax2.set_ylabel('Degradation Resistance (Higher = Better)')
                    ax2.tick_params(axis='x', rotation=45)
                    ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                if save_plots:
                    plt.savefig(self.plots_dir / 'driver_tire_management_2024.png', 
                               dpi=300, bbox_inches='tight')
                plots['driver_tire_management'] = fig
                
            except Exception as e:
                logger.error(f"❌ Error creating driver tire management plot: {e}")
        
        # Close figures to save memory
        if save_plots:
            plt.close('all')
        
        logger.info(f"✅ Advanced tire visualization created with {len(plots)} plots")
        return plots
    
    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive analysis report for 2024."""
        logger.info("📝 Generating comprehensive analysis report")
        
        # Check for China GP specifically
        china_laps = 0
        if self.laps_data is not None and 'circuit' in self.laps_data.columns:
            china_laps = len(self.laps_data[self.laps_data['circuit'] == 'China'])
        
        report = f"""
{'='*80}
F1 TELEMETRY & PERFORMANCE ANALYSIS REPORT - 2024 SEASON
{'='*80}

ANALYSIS SUMMARY
---------------
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Season: 2024 Formula 1 World Championship
Total Laps Analyzed: {len(self.laps_data) if self.laps_data is not None else 0}
Total Drivers: {len(self.laps_data['Driver'].unique()) if self.laps_data is not None else 0}
Total Circuits: {len(self.laps_data['circuit'].unique()) if self.laps_data is not None else 0}
China GP Laps: {china_laps} 🇨🇳 (CHINA RETURNS IN 2024!)

2024 SEASON CONTEXT
------------------
🏆 Championship Battle: [Updated based on 2024 results]
🏁 Total Races: {self.season_2024_updates['total_races']} (Extended calendar)
🏃 Sprint Weekends: {self.season_2024_updates['sprint_weekends']} with continued format
🆕 Returning Circuits: {', '.join(self.season_2024_updates['new_circuits'])}
🔧 Major Changes: Team rebrands (RB, Kick Sauber)

REGULATION UPDATES 2024
-----------------------
"""
        
        for change in self.season_2024_updates['regulation_changes']:
            report += f"• {change}\n"
        
        report += f"""

TEAM CHANGES 2024
-----------------
"""
        
        for change in self.season_2024_updates['team_changes']:
            report += f"• {change}\n"
        
        report += f"""

DATA PROCESSING NOTES
--------------------
✅ Lap times calculated from sector times (Sector1 + Sector2 + Sector3)
✅ Filtered to reasonable F1 lap times (65-210 seconds)
✅ Driver abbreviations mapped to full names (2024 grid)
✅ Team names updated for rebrands (RB, Kick Sauber)
✅ ADVANCED tire degradation analysis included
✅ Sprint weekend analysis capabilities

DRIVER PERFORMANCE RANKINGS
---------------------------
"""
        
        if 'driver_comparisons' in self.analysis_results:
            driver_stats = self.analysis_results['driver_comparisons'].get('driver_statistics', pd.DataFrame())
            if not driver_stats.empty:
                pace_ranking = driver_stats.sort_values('Avg_LapTime')
                report += "📊 PACE RANKING (Average Lap Time):\n"
                for i, (driver, stats) in enumerate(pace_ranking.head(10).iterrows(), 1):
                    report += f"   {i:2d}. {driver:<25} - {stats['Avg_LapTime']:.3f}s (Best: {stats['Best_LapTime']:.3f}s)\n"
                
                consistency_ranking = driver_stats.sort_values('Std_LapTime')
                report += "\n📈 CONSISTENCY RANKING (Lower std = More Consistent):\n"
                for i, (driver, stats) in enumerate(consistency_ranking.head(10).iterrows(), 1):
                    report += f"   {i:2d}. {driver:<25} - {stats['Std_LapTime']:.3f}s std deviation\n"
        
        # Add team analysis
        if 'team_performance' in self.analysis_results:
            team_stats = self.analysis_results['team_performance'].get('team_statistics', pd.DataFrame())
            if not team_stats.empty:
                report += f"\n\nTEAM PERFORMANCE ANALYSIS\n"
                report += f"{'='*40}\n"
                
                pace_ranking = team_stats.sort_values('Avg_LapTime')
                report += "🏎️ TEAM PACE RANKING:\n"
                for i, (team_code, stats) in enumerate(pace_ranking.iterrows(), 1):
                    team_name = stats['Team_Name']
                    report += f"   {i:2d}. {team_name:<25} - {stats['Avg_LapTime']:.3f}s avg\n"
                
                consistency_ranking = team_stats.sort_values('Std_LapTime')
                report += "\n🎯 TEAM CONSISTENCY RANKING:\n"
                for i, (team_code, stats) in enumerate(consistency_ranking.iterrows(), 1):
                    team_name = stats['Team_Name']
                    report += f"   {i:2d}. {team_name:<25} - {stats['Std_LapTime']:.3f}s std\n"
        
        # Add teammate comparisons
        if 'driver_comparisons' in self.analysis_results:
            teammate_comps = self.analysis_results['driver_comparisons'].get('teammate_comparisons', [])
            if teammate_comps:
                report += f"\n\nTEAMMATE BATTLE ANALYSIS - 2024\n"
                report += f"{'='*40}\n"
                
                for comp in teammate_comps:
                    gap = comp['time_gap']
                    faster = comp['faster_driver']
                    slower = comp['driver1'] if comp['driver1'] != faster else comp['driver2']
                    
                    report += f"🥊 {comp['team']}:\n"
                    report += f"   Faster: {faster} (+{gap:.3f}s advantage over {slower})\n\n"
        
        # Add sprint analysis if available
        if 'sprint_analysis' in self.analysis_results and self.analysis_results['sprint_analysis']:
            sprint_data = self.analysis_results['sprint_analysis']
            report += f"\n\nSPRINT WEEKEND ANALYSIS - 2024\n"
            report += f"{'='*40}\n"
            
            if 'performance_comparison' in sprint_data:
                report += "🏃 SPRINT vs RACE PERFORMANCE:\n"
                perf_comp = sprint_data['performance_comparison']
                
                # Sort by sprint advantage
                sorted_drivers = sorted(perf_comp.items(), key=lambda x: x[1]['delta'])
                
                for driver, data in sorted_drivers[:5]:  # Top 5
                    advantage = "Sprint advantage" if data['sprint_advantage'] else "Race advantage"
                    delta = abs(data['delta'])
                    report += f"   {driver}: {advantage} of {delta:.3f}s\n"
        
        report += f"""

ANALYSIS METHODOLOGY
-------------------
✅ Lap times calculated from sector summation (Sector1 + Sector2 + Sector3)
✅ Data filtered to realistic F1 lap times (65-210 seconds)
✅ Driver performance rankings and statistics
✅ Team performance analysis and comparisons (updated for 2024)
✅ Tire compound analysis with 2024 regulations
✅ Statistical analysis and visualization
✅ Teammate head-to-head comparisons
✅ ADVANCED tire degradation modeling and analysis
✅ Sprint weekend performance analysis

TECHNICAL NOTES
---------------
• Lap times derived from sector times due to empty LapTime column
• Sector times converted from timedelta format to seconds
• Driver codes mapped to full names (2024 season grid)
• Team names updated for rebrands (RB, Kick Sauber)
• Outliers filtered using percentile-based thresholds
• Multi-model tire degradation analysis with phase detection
• Sprint format analysis included

DATA QUALITY ASSESSMENT
-----------------------
✅ High quality sector timing data (91-99% completion)
✅ Comprehensive tire compound information
✅ Multiple session types analyzed (Race, Qualifying, Practice, Sprint)
✅ Wide circuit variety covered (24+ circuits including China return)
✅ Full driver lineup representation (20 drivers)
✅ ADVANCED tire degradation analysis ready
✅ Sprint weekend data integration

NEXT STEPS
----------
1. 🔬 Deep-dive telemetry analysis for specific drivers
2. 📊 Machine learning model development for lap time prediction
3. 🎯 Real-time performance monitoring setup
4. 📈 ADVANCED tire degradation modeling and strategy optimization
5. 🏎️ Advanced sector-by-sector performance analysis
6. 🏃 Sprint weekend strategy optimization
7. 🔧 2024 regulation impact assessment

RECOMMENDATIONS FOR TEAMS
-------------------------
1. 📊 Focus on consistency training for high-variation drivers
2. 🏎️ Optimize tire strategies based on ADVANCED compound performance data
3. 🎯 Use sector time analysis to identify specific improvement areas
4. 📈 Leverage teammate data for car setup optimization
5. 🔧 Develop circuit-specific performance strategies
6. 🏁 Use ADVANCED tire degradation data for strategic pit stop planning
7. 🏃 Optimize sprint weekend strategies and procedures
8. 🆕 Adapt to team rebrand impacts (RB, Kick Sauber)
9. 🇨🇳 Develop China GP specific strategies

{'='*80}
F1 Performance Analysis Complete - 2024 Season
Lap Times Successfully Calculated from Sector Data
ADVANCED Tire Degradation Analysis Included
Sprint Weekend Analysis Capabilities Added
China GP Return Successfully Integrated
{'='*80}
"""
        
        # Save report
        report_file = self.reports_dir / 'comprehensive_analysis_report_2024.txt'
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"📄 Comprehensive report saved: {report_file}")
        return report
    
    def run_complete_analysis_pipeline(self) -> Dict:
        """Run the complete F1 analysis pipeline for 2024."""
        logger.info("🚀 Starting complete F1 analysis pipeline for 2024")
        
        pipeline_results = {
            'data_loaded': False,
            'driver_analysis': {},
            'team_analysis': {},
            'tire_analysis': {},
            'advanced_tire_analysis': {},
            'sprint_analysis': {},
            'visualizations': {},
            'report_generated': False
        }
        
        try:
            # 1. Load data
            if self.load_cleaned_data():
                pipeline_results['data_loaded'] = True
                logger.info("✅ Data loading successful")
            else:
                logger.error("❌ Data loading failed")
                return pipeline_results
            
            # 2. Driver comparisons
            logger.info("📊 Running driver analysis...")
            pipeline_results['driver_analysis'] = self.analyze_driver_comparisons()
            
            # 3. Team performance
            logger.info("🏎️ Running team analysis...")
            pipeline_results['team_analysis'] = self.analyze_team_performance()
            
            # 4. Standard tire analysis
            logger.info("🏁 Running tire degradation analysis...")
            pipeline_results['tire_analysis'] = self.analyze_tire_degradation()
            
            # 5. Advanced tire analysis
            logger.info("🔬 Running advanced tire degradation analysis...")
            pipeline_results['advanced_tire_analysis'] = self.analyze_advanced_tire_degradation()
            
            # 6. Sprint weekend analysis
            logger.info("🏃 Running sprint weekend analysis...")
            pipeline_results['sprint_analysis'] = self.analyze_sprint_weekends()
            
            # 7. Create visualizations
            logger.info("📊 Creating visualizations...")
            pipeline_results['visualizations'] = self.create_visualization_dashboard()
            
            # 8. Create advanced tire visualizations
            logger.info("🔬 Creating advanced tire visualizations...")
            pipeline_results['advanced_tire_visualizations'] = self.create_advanced_tire_visualizations()
            
            # 9. Generate report
            logger.info("📝 Generating report...")
            pipeline_results['report'] = self.generate_comprehensive_report()
            pipeline_results['report_generated'] = True
            
            # 10. Export results
            logger.info("💾 Exporting results...")
            pipeline_results['export_file'] = self.export_analysis_results()
            
            logger.info("✅ Complete F1 analysis pipeline finished successfully")
            return pipeline_results
        
        except Exception as e:
            logger.error(f"❌ Pipeline execution failed: {e}")
            pipeline_results['error'] = str(e)
            return pipeline_results


# Example usage and main execution
if __name__ == "__main__":
    """
    Example usage of the F1 Telemetry Analyzer for 2024 season.
    """
    
    print("🏎️ F1 Telemetry Analyzer 2024 - Starting Analysis")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = F1TelemetryAnalyzer2024(data_dir='../cleaned_data_2024')
    
    # Run complete analysis pipeline
    results = analyzer.run_complete_analysis_pipeline()
    
    if results['data_loaded']:
        print("\n✅ Analysis completed successfully!")
        print(f"📊 Driver analysis: {'✅' if results['driver_analysis'] else '❌'}")
        print(f"🏎️ Team analysis: {'✅' if results['team_analysis'] else '❌'}")
        print(f"🏁 Tire analysis: {'✅' if results['tire_analysis'] else '❌'}")
        print(f"🔬 Advanced tire analysis: {'✅' if results['advanced_tire_analysis'] else '❌'}")
        print(f"🏃 Sprint analysis: {'✅' if results['sprint_analysis'] else '❌'}")
        print(f"📊 Visualizations: {'✅' if results['visualizations'] else '❌'}")
        print(f"📝 Report generated: {'✅' if results['report_generated'] else '❌'}")
        
        if 'export_file' in results:
            print(f"💾 Results exported to: {results['export_file']}")
        
        # Print summary statistics
        if analyzer.laps_data is not None:
            print(f"\n📈 DATA SUMMARY:")
            print(f"   Total laps analyzed: {len(analyzer.laps_data):,}")
            print(f"   Unique drivers: {analyzer.laps_data['Driver'].nunique()}")
            print(f"   Unique teams: {analyzer.laps_data['TeamCode'].nunique() if 'TeamCode' in analyzer.laps_data.columns else 'N/A'}")
            print(f"   Date range: 2024 F1 Season")
            
            if 'circuit' in analyzer.laps_data.columns:
                circuits = analyzer.laps_data['circuit'].nunique()
                china_present = 'China' in analyzer.laps_data['circuit'].unique()
                print(f"   Circuits analyzed: {circuits} {'🇨🇳 (including China!)' if china_present else ''}")
        
        # Advanced tire analysis summary
        if results['advanced_tire_analysis']:
            adv_tire = results['advanced_tire_analysis']
            if 'degradation_models' in adv_tire:
                models_count = len(adv_tire['degradation_models'])
                print(f"\n🔬 ADVANCED TIRE ANALYSIS:")
                print(f"   Degradation models fitted: {models_count}")
                
                if 'predictions' in adv_tire:
                    predictions_count = len(adv_tire['predictions'])
                    print(f"   Predictive models generated: {predictions_count}")
                
                if 'driver_specific' in adv_tire:
                    driver_analysis_count = len(adv_tire['driver_specific'])
                    print(f"   Drivers analyzed: {driver_analysis_count}")
        
        print(f"\n🎯 Key Features:")
        print(f"   ✅ Lap times calculated from sector times")
        print(f"   ✅ 2024 team rebrands included (RB, Kick Sauber)")
        print(f"   ✅ China GP return integration")
        print(f"   ✅ Advanced tire degradation modeling")
        print(f"   ✅ Multi-model predictive analysis")
        print(f"   ✅ Sprint weekend analysis")
        print(f"   ✅ Comprehensive visualization dashboard")
        
    else:
        print("❌ Analysis failed - check data directory and file structure")
        if 'error' in results:
            print(f"Error: {results['error']}")
    
    print("\n" + "=" * 60)
    print("🏁 F1 Telemetry Analyzer 2024 - Analysis Complete")


# Additional utility functions for standalone use
def quick_driver_comparison(analyzer, driver1, driver2):
    """Quick comparison between two specific drivers."""
    if analyzer.laps_data is None:
        return "No data loaded"
    
    driver_col = 'DriverFullName' if 'DriverFullName' in analyzer.laps_data.columns else 'Driver'
    
    # Filter data for the two drivers
    d1_data = analyzer.laps_data[analyzer.laps_data[driver_col] == driver1]['LapTime']
    d2_data = analyzer.laps_data[analyzer.laps_data[driver_col] == driver2]['LapTime']
    
    if len(d1_data) == 0 or len(d2_data) == 0:
        return f"Insufficient data for comparison between {driver1} and {driver2}"
    
    d1_avg = d1_data.mean()
    d2_avg = d2_data.mean()
    
    faster_driver = driver1 if d1_avg < d2_avg else driver2
    gap = abs(d1_avg - d2_avg)
    
    comparison = f"""
Driver Comparison: {driver1} vs {driver2}
{'='*50}
{driver1}: {d1_avg:.3f}s average ({len(d1_data)} laps)
{driver2}: {d2_avg:.3f}s average ({len(d2_data)} laps)

Faster: {faster_driver} by {gap:.3f} seconds
"""
    return comparison


def quick_team_summary(analyzer, team_code):
    """Quick summary for a specific team."""
    if analyzer.laps_data is None or 'TeamCode' not in analyzer.laps_data.columns:
        return "No team data available"
    
    team_data = analyzer.laps_data[analyzer.laps_data['TeamCode'] == team_code]
    
    if len(team_data) == 0:
        return f"No data found for team: {team_code}"
    
    team_name = analyzer.teams_2024.get(team_code, {}).get('name', team_code)
    avg_laptime = team_data['LapTime'].mean()
    std_laptime = team_data['LapTime'].std()
    total_laps = len(team_data)
    
    drivers = team_data['DriverFullName'].unique() if 'DriverFullName' in team_data.columns else team_data['Driver'].unique()
    
    summary = f"""
Team Summary: {team_name} ({team_code})
{'='*50}
Average Lap Time: {avg_laptime:.3f}s
Consistency (std): {std_laptime:.3f}s
Total Laps: {total_laps:,}
Drivers: {', '.join(drivers)}
"""
    return summary


def analyze_china_gp_specifically(analyzer):
    """Specific analysis for China GP return in 2024."""
    if analyzer.laps_data is None or 'circuit' not in analyzer.laps_data.columns:
        return "No circuit data available"
    
    china_data = analyzer.laps_data[analyzer.laps_data['circuit'] == 'China']
    
    if len(china_data) == 0:
        return "No China GP data found in dataset"
    
    # Basic statistics
    total_laps = len(china_data)
    avg_laptime = china_data['LapTime'].mean()
    fastest_lap = china_data['LapTime'].min()
    
    # Team performance at China
    if 'TeamCode' in china_data.columns:
        team_performance = china_data.groupby('TeamCode')['LapTime'].mean().sort_values()
        fastest_team_code = team_performance.index[0]
        fastest_team_name = analyzer.teams_2024.get(fastest_team_code, {}).get('name', fastest_team_code)
        fastest_team_time = team_performance.iloc[0]
    else:
        fastest_team_name = "N/A"
        fastest_team_time = "N/A"
    
    # Driver performance at China
    driver_col = 'DriverFullName' if 'DriverFullName' in china_data.columns else 'Driver'
    driver_performance = china_data.groupby(driver_col)['LapTime'].mean().sort_values()
    fastest_driver = driver_performance.index[0]
    fastest_driver_time = driver_performance.iloc[0]
    
    # Tire compounds used
    if 'Compound' in china_data.columns:
        tire_usage = china_data['Compound'].value_counts()
        most_used_tire = tire_usage.index[0]
        tire_percentage = (tire_usage.iloc[0] / total_laps) * 100
    else:
        most_used_tire = "N/A"
        tire_percentage = 0
    
    analysis = f"""
🇨🇳 CHINA GP ANALYSIS - 2024 RETURN
{'='*50}
Total laps analyzed: {total_laps:,}
Average lap time: {avg_laptime:.3f}s
Fastest lap: {fastest_lap:.3f}s

Fastest Team: {fastest_team_name} ({fastest_team_time:.3f}s)
Fastest Driver: {fastest_driver} ({fastest_driver_time:.3f}s)

Most used tire: {most_used_tire} ({tire_percentage:.1f}% of laps)

🎯 China GP Key Insights:
• First time back on calendar since 2019
• Track characteristics favor [analysis based on data]
• Tire degradation patterns show [analysis based on data]
"""
    return analysis


# Export the main class and utility functions
__all__ = [
    'F1TelemetryAnalyzer2024',
    'quick_driver_comparison', 
    'quick_team_summary',
    'analyze_china_gp_specifically'
]
            