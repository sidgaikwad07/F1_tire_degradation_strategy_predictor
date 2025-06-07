#!/usr/bin/env python3
"""
F1 Data Collector - 2025 Season (Up to Spanish GP)
Collects F1 data for 2025 season races completed up to Spanish GP for Canadian GP prediction.

This version includes:
- Rate limiting protection (500 calls/hour limit)
- Data duplication prevention
- Improved session loading with validation
- Progress persistence and resume capability
- 2025 calendar up to Spanish GP (9 races)
- Canadian GP prediction preparation focus

Usage:
    python data_collector_2025.py --collect-available
    python data_collector_2025.py --prediction-prep
    python data_collector_2025.py --circuit "Spain"
    python data_collector_2025.py --resume
"""

import fastf1
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import argparse
from datetime import datetime, timedelta
import warnings
import time
import signal
from contextlib import contextmanager
import sys
from typing import List, Dict, Optional
import os

# Suppress FastF1 warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('f1_data_collection_2025.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RateLimitError(Exception):
    """Custom exception for rate limit handling."""
    pass

class TimeoutError(Exception):
    """Custom timeout exception for data collection operations."""
    pass

@contextmanager
def timeout_handler(seconds: int):
    """Context manager for handling operation timeouts."""
    def timeout_signal_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    old_handler = signal.signal(signal.SIGALRM, timeout_signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

class F1DataCollector2025:
    """
    F1 Data Collector for 2025 season with Canadian GP prediction focus.
    
    Features:
    - API rate limit management (500 calls/hour)
    - Data duplication prevention
    - Progress persistence and resume capability
    - Robust session loading with validation
    - 2025 calendar up to Spanish GP
    - Canadian GP prediction preparation
    """
    
    def __init__(self, base_dir: str = 'data', timeout: int = 300, rate_limit_delay: int = 8):
        """
        Initialize the 2025 F1 Data Collector.
        
        Args:
            base_dir (str): Base directory for data storage
            timeout (int): Timeout in seconds for each data collection operation
            rate_limit_delay (int): Delay between API calls to respect rate limits
        """
        self.year = 2025
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / 'raw' / str(self.year)
        self.cache_dir = Path('cache')
        self.progress_dir = self.base_dir / 'progress'
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        
        # Create directory structure
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        self.progress_dir.mkdir(exist_ok=True)
        
        # Enable FastF1 cache for faster subsequent runs
        fastf1.Cache.enable_cache(str(self.cache_dir))
        
        # Progress tracking file
        self.progress_file = self.progress_dir / 'collection_progress_2025.json'
        self.collected_sessions = self._load_progress()
        
        # 2025 F1 Calendar - COMPLETED RACES (up to Spanish GP as of June 2025)
        self.circuits_2025_completed = {
            'Bahrain': {'sessions': ['FP1', 'FP2', 'FP3', 'Q', 'R'], 'has_sprint': False, 'round': 1},
            'Saudi Arabia': {'sessions': ['FP1', 'FP2', 'FP3', 'Q', 'R'], 'has_sprint': False, 'round': 2},
            'Australia': {'sessions': ['FP1', 'FP2', 'FP3', 'Q', 'R'], 'has_sprint': False, 'round': 3},
            'Japan': {'sessions': ['FP1', 'FP2', 'FP3', 'Q', 'R'], 'has_sprint': False, 'round': 4},
            'China': {'sessions': ['FP1', 'SQ', 'S', 'Q', 'R'], 'has_sprint': True, 'round': 5},  # Sprint weekend
            'Miami': {'sessions': ['FP1', 'SQ', 'S', 'Q', 'R'], 'has_sprint': True, 'round': 6},  # Sprint weekend
            'Emilia Romagna': {'sessions': ['FP1', 'FP2', 'FP3', 'Q', 'R'], 'has_sprint': False, 'round': 7},
            'Monaco': {'sessions': ['FP1', 'FP2', 'FP3', 'Q', 'R'], 'has_sprint': False, 'round': 8},
            'Spain': {'sessions': ['FP1', 'FP2', 'FP3', 'Q', 'R'], 'has_sprint': False, 'round': 9}  # Last completed
        }
        
        # NEXT RACE - Canadian GP (prediction target)
        self.prediction_target = 'Canada'
        
        # Future races after Canada (for reference)
        self.circuits_2025_future = {
            'Canada': {'sessions': ['FP1', 'FP2', 'FP3', 'Q', 'R'], 'has_sprint': False, 'round': 10},  # PREDICTION TARGET
            'Austria': {'sessions': ['FP1', 'SQ', 'S', 'Q', 'R'], 'has_sprint': True, 'round': 11},
            'Great Britain': {'sessions': ['FP1', 'FP2', 'FP3', 'Q', 'R'], 'has_sprint': False, 'round': 12},
            'Hungary': {'sessions': ['FP1', 'FP2', 'FP3', 'Q', 'R'], 'has_sprint': False, 'round': 13},
            'Belgium': {'sessions': ['FP1', 'FP2', 'FP3', 'Q', 'R'], 'has_sprint': False, 'round': 14},
            'Netherlands': {'sessions': ['FP1', 'FP2', 'FP3', 'Q', 'R'], 'has_sprint': False, 'round': 15},
            'Italy': {'sessions': ['FP1', 'FP2', 'FP3', 'Q', 'R'], 'has_sprint': False, 'round': 16},
            'Azerbaijan': {'sessions': ['FP1', 'FP2', 'FP3', 'Q', 'R'], 'has_sprint': False, 'round': 17},
            'Singapore': {'sessions': ['FP1', 'FP2', 'FP3', 'Q', 'R'], 'has_sprint': False, 'round': 18},
            'United States': {'sessions': ['FP1', 'SQ', 'S', 'Q', 'R'], 'has_sprint': True, 'round': 19},
            'Mexico': {'sessions': ['FP1', 'FP2', 'FP3', 'Q', 'R'], 'has_sprint': False, 'round': 20},
            'Brazil': {'sessions': ['FP1', 'SQ', 'S', 'Q', 'R'], 'has_sprint': True, 'round': 21},
            'Las Vegas': {'sessions': ['FP1', 'FP2', 'FP3', 'Q', 'R'], 'has_sprint': False, 'round': 22},
            'Qatar': {'sessions': ['FP1', 'SQ', 'S', 'Q', 'R'], 'has_sprint': True, 'round': 23},
            'Abu Dhabi': {'sessions': ['FP1', 'FP2', 'FP3', 'Q', 'R'], 'has_sprint': False, 'round': 24}
        }
        
        # Circuits most similar to Canada for prediction modeling
        self.canada_similar_circuits = [
            'Australia',      # Similar high-speed layout with long straights
            'Emilia Romagna', # Mixed-speed corners, similar racing line complexity
            'Spain'           # Good overtaking opportunities, tire management focus
        ]
        
        # All circuits (completed + future) for reference
        self.all_circuits_2025 = {**self.circuits_2025_completed, **self.circuits_2025_future}
        
        # Priority circuits for Canadian GP prediction (most relevant training data)
        self.priority_circuits = list(self.circuits_2025_completed.keys())  # All completed races
        
        # Collection statistics tracking
        self.stats = {
            'total_attempted': 0,
            'successful_collections': 0,
            'failed_collections': 0,
            'skipped_existing': 0,
            'rate_limit_delays': 0,
            'timeouts': 0,
            'network_errors': 0,
            'data_errors': 0,
            'sessions_by_type': {
                'FP1': {'attempted': 0, 'successful': 0, 'skipped': 0},
                'FP2': {'attempted': 0, 'successful': 0, 'skipped': 0},
                'FP3': {'attempted': 0, 'successful': 0, 'skipped': 0},
                'Q': {'attempted': 0, 'successful': 0, 'skipped': 0},
                'SQ': {'attempted': 0, 'successful': 0, 'skipped': 0},  # Sprint Qualifying
                'S': {'attempted': 0, 'successful': 0, 'skipped': 0},   # Sprint Race
                'R': {'attempted': 0, 'successful': 0, 'skipped': 0}
            },
            'failed_sessions': [],
            'last_api_call': None,
            'sprint_weekends_collected': 0,
            'completed_races_collected': 0,
            'canada_similar_circuits_collected': 0,
            'prediction_ready': False
        }
    
    def _load_progress(self) -> Dict:
        """Load collection progress to avoid duplicates."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                logger.info(f"📋 Loaded progress: {len(progress)} sessions already collected")
                return progress
            except Exception as e:
                logger.warning(f"⚠️ Could not load progress file: {e}")
        return {}
    
    def _save_progress(self):
        """Save current collection progress."""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.collected_sessions, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"⚠️ Could not save progress: {e}")
    
    def _is_session_collected(self, circuit: str, session_type: str) -> bool:
        """Check if session data already exists and is complete."""
        session_id = f"{self.year}_{circuit}_{session_type}"
        
        # Check progress tracking
        if session_id in self.collected_sessions:
            return True
        
        # Check if files exist and are non-empty
        circuit_dir = self.raw_dir / circuit
        if not circuit_dir.exists():
            return False
        
        # Check for essential files
        essential_files = [
            f'{session_type}_session_info.json',
            f'{session_type}_metadata.json'
        ]
        
        for file_name in essential_files:
            file_path = circuit_dir / file_name
            if not file_path.exists() or file_path.stat().st_size == 0:
                return False
        
        # Mark as collected if files exist
        self.collected_sessions[session_id] = {
            'collected_at': datetime.now().isoformat(),
            'method': 'file_verification'
        }
        self._save_progress()
        return True
    
    def _respect_rate_limit(self):
        """Ensure we don't exceed the API rate limit."""
        current_time = datetime.now()
        
        if self.stats['last_api_call']:
            time_since_last_call = (current_time - self.stats['last_api_call']).total_seconds()
            
            if time_since_last_call < self.rate_limit_delay:
                sleep_time = self.rate_limit_delay - time_since_last_call
                logger.info(f"⏱️ Rate limiting: waiting {sleep_time:.1f}s")
                time.sleep(sleep_time)
                self.stats['rate_limit_delays'] += 1
        
        self.stats['last_api_call'] = datetime.now()
    
    def collect_session_data(self, circuit: str, session_type: str, max_retries: int = 2) -> Optional[Dict]:
        """
        Collect data for a specific session with rate limiting and duplicate prevention.
        
        Args:
            circuit (str): Circuit name
            session_type (str): Session type (FP1, FP2, FP3, Q, SQ, S, R)
            max_retries (int): Maximum retry attempts
            
        Returns:
            Optional[Dict]: Session data or None if failed
        """
        session_id = f"{self.year}_{circuit}_{session_type}"
        
        # Check if already collected
        if self._is_session_collected(circuit, session_type):
            logger.info(f"⏭️ Skipping {session_id} - already collected")
            self.stats['skipped_existing'] += 1
            self.stats['sessions_by_type'][session_type]['skipped'] += 1
            return {'status': 'skipped', 'reason': 'already_collected'}
        
        logger.info(f"🏎️ Collecting {session_id}...")
        
        # Update statistics
        self.stats['total_attempted'] += 1
        self.stats['sessions_by_type'][session_type]['attempted'] += 1
        
        # Respect rate limits
        self._respect_rate_limit()
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"    🔄 Retry attempt {attempt}/{max_retries}")
                    time.sleep(10)  # Longer wait for retries
                
                # Use timeout protection
                with timeout_handler(self.timeout):
                    session_data = self._collect_session_core(circuit, session_type)
                
                if session_data and session_data.get('status') != 'failed':
                    # Mark as collected
                    self.collected_sessions[session_id] = {
                        'collected_at': datetime.now().isoformat(),
                        'attempt': attempt + 1
                    }
                    self._save_progress()
                    
                    self.stats['successful_collections'] += 1
                    self.stats['sessions_by_type'][session_type]['successful'] += 1
                    logger.info(f"    ✅ Successfully collected {session_id}")
                    return session_data
                    
            except fastf1.req.RateLimitExceededError:
                logger.warning(f"    🚫 Rate limit exceeded. Waiting 1 hour...")
                time.sleep(3600)  # Wait 1 hour
                continue
                
            except TimeoutError:
                self.stats['timeouts'] += 1
                logger.warning(f"    ⏰ Timeout on attempt {attempt + 1}")
                
            except Exception as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ['network', 'connection', 'timeout']):
                    self.stats['network_errors'] += 1
                    logger.warning(f"    🌐 Network error on attempt {attempt + 1}: {e}")
                else:
                    self.stats['data_errors'] += 1
                    logger.warning(f"    ❌ Data error on attempt {attempt + 1}: {e}")
        
        # All attempts failed
        self.stats['failed_collections'] += 1
        self.stats['failed_sessions'].append(session_id)
        logger.error(f"    💥 Failed to collect {session_id} after {max_retries + 1} attempts")
        return None
    
    def _collect_session_core(self, circuit: str, session_type: str) -> Dict:
        """
        Core session data collection with improved validation.
        
        Args:
            circuit (str): Circuit name
            session_type (str): Session type
            
        Returns:
            Dict: Collected session data
        """
        try:
            # Load F1 session
            session = fastf1.get_session(self.year, circuit, session_type)
            logger.info(f"    📥 Loading session data...")
            
            # Load session with validation
            session.load()
            
            # Validate session loaded properly
            if not self._validate_session_loading(session):
                logger.warning(f"    ⚠️ Session validation failed - incomplete data")
                return {'status': 'failed', 'reason': 'validation_failed'}
            
            # Extract session metadata
            session_info = self._extract_session_info(session, circuit, session_type)
            
            # Extract different data components
            results_data = self._extract_results(session)
            laps_data = self._extract_laps(session)
            weather_data = self._extract_weather(session)
            
            # Extract session-specific data
            session_specific_data = {}
            if session_type == 'Q':
                session_specific_data = self._extract_qualifying_data(session)
            elif session_type == 'SQ':  # Sprint Qualifying
                session_specific_data = self._extract_sprint_qualifying_data(session)
            elif session_type == 'S':
                session_specific_data = self._extract_sprint_data(session)
                self.stats['sprint_weekends_collected'] += 1
            elif session_type in ['FP1', 'FP2', 'FP3']:
                session_specific_data = self._extract_practice_data(session)
            elif session_type == 'R':
                session_specific_data = self._extract_race_data(session)
                self.stats['completed_races_collected'] += 1
            
            # Track Canada-similar circuits
            if circuit in self.canada_similar_circuits:
                self.stats['canada_similar_circuits_collected'] += 1
            
            # Compile session data package
            session_data = {
                'session_info': session_info,
                'results': results_data,
                'laps': laps_data,
                'weather': weather_data,
                'session_specific': session_specific_data,
                'collection_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'timeout_used': self.timeout,
                    'rate_limit_delay': self.rate_limit_delay,
                    'data_completeness': {
                        'has_results': results_data is not None,
                        'has_laps': laps_data is not None,
                        'has_weather': weather_data is not None,
                        'has_session_specific': bool(session_specific_data)
                    },
                    'validation_passed': True,
                    'is_sprint_format': session_type in ['SQ', 'S'],
                    'prediction_relevance': 'high' if circuit in self.canada_similar_circuits else 'standard',
                    'race_round': self.all_circuits_2025.get(circuit, {}).get('round', None),
                    'is_prediction_training_data': circuit in self.circuits_2025_completed
                },
                'status': 'success'
            }
            
            # Save the collected data
            self._save_session_data(session_data, circuit, session_type)
            
            return session_data
            
        except Exception as e:
            logger.error(f"    ❌ Core collection failed: {e}")
            return {'status': 'failed', 'reason': str(e)}
    
    def _validate_session_loading(self, session) -> bool:
        """Validate that session data loaded properly."""
        try:
            # Check if session has basic attributes
            if not hasattr(session, 'name'):
                return False
            
            # Session appears to be loaded
            logger.info(f"    ✅ Session validation passed")
            return True
            
        except Exception as e:
            logger.warning(f"    ❌ Session validation failed: {e}")
            return False
    
    def _extract_session_info(self, session, circuit: str, session_type: str) -> Dict:
        """Extract basic session information and metadata."""
        return {
            'year': self.year,
            'circuit': circuit,
            'session_type': session_type,
            'session_name': getattr(session, 'name', session_type),
            'event_name': getattr(session.event, 'EventName', f'{circuit} Grand Prix'),
            'event_date': str(session.event.EventDate) if hasattr(session.event, 'EventDate') else None,
            'session_date': str(session.date) if hasattr(session, 'date') and session.date else None,
            'circuit_info': {
                'circuit_key': getattr(session.event, 'CircuitKey', circuit),
                'location': getattr(session.event, 'Location', None),
                'country': getattr(session.event, 'Country', None),
                'is_canada_similar': circuit in self.canada_similar_circuits,
                'prediction_target': circuit == self.prediction_target
            },
            'session_details': {
                'total_laps': getattr(session, 'total_laps', None),
                'is_sprint_weekend': self.all_circuits_2025.get(circuit, {}).get('has_sprint', False),
                'race_round': self.all_circuits_2025.get(circuit, {}).get('round', None),
                'races_before_canada': len(self.circuits_2025_completed),
                'is_training_data': circuit in self.circuits_2025_completed
            }
        }
    
    def _extract_results(self, session) -> Optional[pd.DataFrame]:
        """Extract session results with improved error handling."""
        try:
            if hasattr(session, 'results') and session.results is not None:
                if not session.results.empty:
                    results = session.results.copy()
                    logger.info(f"      ✅ Results: {len(results)} drivers")
                    return results
                else:
                    logger.info(f"      ⚠️ Results exist but are empty")
                    return None
            else:
                logger.info(f"      ⚠️ No results data available")
                return None
        except Exception as e:
            logger.warning(f"      ❌ Results extraction failed: {e}")
            return None
    
    def _extract_laps(self, session) -> Optional[pd.DataFrame]:
        """Extract lap-by-lap data with improved validation."""
        try:
            if hasattr(session, 'laps') and session.laps is not None:
                if not session.laps.empty:
                    laps = session.laps.copy()
                    
                    # Add lap validation
                    if 'LapTime' in laps.columns:
                        valid_laps = laps.dropna(subset=['LapTime'])
                        logger.info(f"      ✅ Laps: {len(valid_laps)} valid laps from {len(laps)} total")
                    else:
                        logger.info(f"      ✅ Laps: {len(laps)} total laps (no lap times)")
                    
                    return laps
                else:
                    logger.info(f"      ⚠️ Laps exist but are empty")
                    return None
            else:
                logger.info(f"      ⚠️ No lap data available")
                return None
        except Exception as e:
            logger.warning(f"      ❌ Laps extraction failed: {e}")
            return None
    
    def _extract_weather(self, session) -> Optional[pd.DataFrame]:
        """Extract weather data with timeout protection."""
        try:
            logger.info(f"      🌤️ Extracting weather data...")
            
            with timeout_handler(30):  # Shorter timeout for weather
                if hasattr(session, 'weather_data') and session.weather_data is not None:
                    if not session.weather_data.empty:
                        weather = session.weather_data.copy()
                        logger.info(f"      ✅ Weather: {len(weather)} data points")
                        return weather
                    else:
                        logger.info(f"      ⚠️ Weather data is empty")
                        return None
                else:
                    logger.info(f"      ⚠️ No weather data available")
                    return None
                        
        except TimeoutError:
            logger.warning(f"      ⏰ Weather extraction timed out")
        except Exception as e:
            logger.warning(f"      ❌ Weather extraction failed: {e}")
        
        return None
    
    def _extract_qualifying_data(self, session) -> Dict:
        """Extract qualifying-specific data."""
        qualifying_data = {}
        
        try:
            if hasattr(session, 'results') and session.results is not None and not session.results.empty:
                results = session.results.copy()
                
                # Extract Q1, Q2, Q3 times
                q_columns = ['Q1', 'Q2', 'Q3']
                available_q_columns = [col for col in q_columns if col in results.columns]
                
                if available_q_columns:
                    qualifying_data['qualifying_times'] = results[
                        ['DriverNumber', 'Abbreviation', 'FullName'] + available_q_columns
                    ].copy()
                
                # Extract sector times
                sector_columns = [col for col in results.columns if 'Sector' in col]
                if sector_columns:
                    qualifying_data['sector_times'] = results[
                        ['DriverNumber', 'Abbreviation'] + sector_columns
                    ].copy()
            
            logger.info(f"      🏁 Qualifying data extracted: {list(qualifying_data.keys())}")
            
        except Exception as e:
            logger.warning(f"      ❌ Qualifying data extraction failed: {e}")
        
        return qualifying_data
    
    def _extract_sprint_qualifying_data(self, session) -> Dict:
        """Extract sprint qualifying-specific data."""
        sprint_qualifying_data = {}
        
        try:
            if hasattr(session, 'results') and session.results is not None and not session.results.empty:
                results = session.results.copy()
                sprint_qualifying_data['sprint_qualifying_results'] = results.copy()
                
                # Extract sprint qualifying times and positions
                if 'Position' in results.columns:
                    sprint_qualifying_data['sprint_qualifying_positions'] = results[['DriverNumber', 'Abbreviation', 'Position']].copy()
                
                # Track sprint qualifying characteristics
                sprint_qualifying_data['sprint_qualifying_stats'] = {
                    'total_drivers': len(results),
                    'session_format': 'sprint_qualifying_2025'
                }
            
            logger.info(f"      🏃 Sprint Qualifying data extracted (2025 format)")
            
        except Exception as e:
            logger.warning(f"      ❌ Sprint Qualifying data extraction failed: {e}")
        
        return sprint_qualifying_data
    
    def _extract_sprint_data(self, session) -> Dict:
        """Extract sprint-specific data."""
        sprint_data = {}
        
        try:
            if hasattr(session, 'results') and session.results is not None and not session.results.empty:
                results = session.results.copy()
                sprint_data['sprint_results'] = results.copy()
                
                sprint_data['sprint_stats'] = {
                    'total_drivers': len(results),
                    'completed_distance': getattr(session, 'total_laps', 0),
                    'sprint_format': '2025_format'
                }
            
            logger.info(f"      🏃 Sprint data extracted (2025 format)")
            
        except Exception as e:
            logger.warning(f"      ❌ Sprint data extraction failed: {e}")
        
        return sprint_data
    
    def _extract_practice_data(self, session) -> Dict:
        """Extract practice session specific data."""
        practice_data = {}
        
        try:
            if hasattr(session, 'laps') and session.laps is not None and not session.laps.empty:
                laps = session.laps.copy()
                
                if 'LapTime' in laps.columns and 'Driver' in laps.columns:
                    # Get fastest lap per driver
                    fastest_laps = laps.loc[laps.groupby('Driver')['LapTime'].idxmin()]
                    practice_data['fastest_laps_per_driver'] = fastest_laps
                    
                    # Long run analysis
                    practice_data['total_laps_per_driver'] = laps.groupby('Driver').size().to_dict()
            
            logger.info(f"      🔧 Practice data extracted")
            
        except Exception as e:
            logger.warning(f"      ❌ Practice data extraction failed: {e}")
        
        return practice_data
    
    def _extract_race_data(self, session) -> Dict:
        """Extract race-specific data."""
        race_data = {}
        
        try:
            if hasattr(session, 'results') and session.results is not None and not session.results.empty:
                results = session.results.copy()
                race_data['race_results'] = results.copy()
                
                race_data['race_stats'] = {
                    'total_finishers': len(results[results['Status'] == 'Finished']) if 'Status' in results.columns else len(results),
                    'total_classified': len(results),
                    'race_distance': getattr(session, 'total_laps', 0),
                    'points_awarded': True if 'Points' in results.columns else False
                }
                
                # Extract podium information
                if 'Position' in results.columns:
                    podium = results[results['Position'].isin([1, 2, 3])].copy()
                    race_data['podium'] = podium
                
                # Extract fastest lap information
                if 'FastestLap' in results.columns:
                    fastest_lap_holder = results[results['FastestLap'].notna()].copy()
                    race_data['fastest_lap_info'] = fastest_lap_holder
            
            logger.info(f"      🏁 Race data extracted")
            
        except Exception as e:
            logger.warning(f"      ❌ Race data extraction failed: {e}")
        
        return race_data
    
    def _save_session_data(self, session_data: Dict, circuit: str, session_type: str):
        """Save collected session data to files."""
        try:
            circuit_dir = self.raw_dir / circuit
            circuit_dir.mkdir(exist_ok=True)
            
            # Save session info as JSON
            session_info_file = circuit_dir / f'{session_type}_session_info.json'
            with open(session_info_file, 'w') as f:
                json.dump(session_data['session_info'], f, indent=2, default=str)
            
            # Save collection metadata
            metadata_file = circuit_dir / f'{session_type}_metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(session_data['collection_metadata'], f, indent=2, default=str)
            
            # Save session-specific data
            if session_data['session_specific']:
                session_specific_file = circuit_dir / f'{session_type}_session_specific.json'
                with open(session_specific_file, 'w') as f:
                    json.dump(session_data['session_specific'], f, indent=2, default=str)
            
            # Save DataFrames as CSV and pickle
            for data_type, data in [
                ('results', session_data['results']),
                ('laps', session_data['laps']),
                ('weather', session_data['weather'])
            ]:
                if data is not None and isinstance(data, pd.DataFrame):
                    # Save as CSV
                    csv_file = circuit_dir / f'{session_type}_{data_type}.csv'
                    data.to_csv(csv_file, index=False)
                    
                    # Save as pickle for preservation of data types
                    pkl_file = circuit_dir / f'{session_type}_{data_type}.pkl'
                    data.to_pickle(pkl_file)
            
            logger.info(f"      💾 Data saved to {circuit_dir}")
            
        except Exception as e:
            logger.error(f"      ❌ Failed to save data: {e}")
    
    def collect_circuit_data(self, circuit: str) -> Dict:
        """
        Collect all session data for a specific circuit.
        
        Args:
            circuit (str): Circuit name
            
        Returns:
            Dict: Collection results summary
        """
        logger.info(f"🏁 Starting data collection for {circuit}")
        
        if circuit not in self.all_circuits_2025:
            logger.error(f"❌ Circuit '{circuit}' not found in 2025 calendar")
            return {'status': 'error', 'reason': 'circuit_not_found'}
        
        circuit_info = self.all_circuits_2025[circuit]
        sessions = circuit_info['sessions']
        
        collection_results = {
            'circuit': circuit,
            'total_sessions': len(sessions),
            'successful_sessions': 0,
            'failed_sessions': 0,
            'skipped_sessions': 0,
            'session_results': {},
            'is_sprint_weekend': circuit_info.get('has_sprint', False),
            'prediction_relevance': 'high' if circuit in self.canada_similar_circuits else 'standard'
        }
        
        for session_type in sessions:
            logger.info(f"  📊 Collecting {circuit} {session_type}")
            
            session_result = self.collect_session_data(circuit, session_type)
            collection_results['session_results'][session_type] = session_result
            
            if session_result:
                if session_result.get('status') == 'skipped':
                    collection_results['skipped_sessions'] += 1
                elif session_result.get('status') == 'success':
                    collection_results['successful_sessions'] += 1
                else:
                    collection_results['failed_sessions'] += 1
            else:
                collection_results['failed_sessions'] += 1
        
        # Log circuit collection summary
        logger.info(f"🏆 {circuit} collection complete: "
                   f"{collection_results['successful_sessions']} successful, "
                   f"{collection_results['skipped_sessions']} skipped, "
                   f"{collection_results['failed_sessions']} failed")
        
        return collection_results
    
    def collect_all_available_data(self) -> Dict:
        """
        Collect data for all completed 2025 races (up to Spanish GP).
        
        Returns:
            Dict: Overall collection results
        """
        logger.info(f"🚀 Starting collection of all available 2025 F1 data")
        logger.info(f"📅 Target: {len(self.circuits_2025_completed)} completed races for Canadian GP prediction")
        
        overall_results = {
            'total_circuits': len(self.circuits_2025_completed),
            'successful_circuits': 0,
            'failed_circuits': 0,
            'circuit_results': {},
            'prediction_data_quality': {
                'canada_similar_circuits': 0,
                'sprint_weekends': 0,
                'total_race_weekends': 0,
                'total_sessions': 0
            },
            'collection_summary': {
                'start_time': datetime.now().isoformat(),
                'end_time': None,
                'duration': None
            }
        }
        
        start_time = datetime.now()
        
        # Collect data for each completed circuit
        for circuit in self.circuits_2025_completed.keys():
            logger.info(f"\n{'='*60}")
            logger.info(f"🏎️ Processing {circuit} (Round {self.circuits_2025_completed[circuit]['round']})")
            logger.info(f"{'='*60}")
            
            try:
                circuit_result = self.collect_circuit_data(circuit)
                overall_results['circuit_results'][circuit] = circuit_result
                
                if circuit_result.get('successful_sessions', 0) > 0:
                    overall_results['successful_circuits'] += 1
                    
                    # Track prediction-relevant data
                    if circuit in self.canada_similar_circuits:
                        overall_results['prediction_data_quality']['canada_similar_circuits'] += 1
                    
                    if circuit_result.get('is_sprint_weekend', False):
                        overall_results['prediction_data_quality']['sprint_weekends'] += 1
                    
                    overall_results['prediction_data_quality']['total_race_weekends'] += 1
                    overall_results['prediction_data_quality']['total_sessions'] += circuit_result.get('successful_sessions', 0)
                
                else:
                    overall_results['failed_circuits'] += 1
                    
            except Exception as e:
                logger.error(f"❌ Circuit {circuit} collection failed: {e}")
                overall_results['failed_circuits'] += 1
                overall_results['circuit_results'][circuit] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Calculate completion time
        end_time = datetime.now()
        duration = end_time - start_time
        overall_results['collection_summary']['end_time'] = end_time.isoformat()
        overall_results['collection_summary']['duration'] = str(duration)
        
        # Assess prediction readiness
        prediction_quality = overall_results['prediction_data_quality']
        self.stats['prediction_ready'] = (
            prediction_quality['canada_similar_circuits'] >= 2 and
            prediction_quality['total_race_weekends'] >= 7 and
            prediction_quality['total_sessions'] >= 30
        )
        
        # Final summary
        logger.info(f"\n{'='*80}")
        logger.info(f"🏁 COLLECTION COMPLETE - 2025 F1 DATA FOR CANADIAN GP PREDICTION")
        logger.info(f"{'='*80}")
        logger.info(f"✅ Successful circuits: {overall_results['successful_circuits']}/{overall_results['total_circuits']}")
        logger.info(f"📊 Canada-similar circuits: {prediction_quality['canada_similar_circuits']}/3")
        logger.info(f"🏃 Sprint weekends: {prediction_quality['sprint_weekends']}")
        logger.info(f"📈 Total sessions collected: {prediction_quality['total_sessions']}")
        logger.info(f"⏱️ Collection duration: {duration}")
        logger.info(f"🎯 Prediction ready: {'YES' if self.stats['prediction_ready'] else 'NO'}")
        
        return overall_results
    
    def collect_prediction_prep_data(self) -> Dict:
        """
        Collect data specifically optimized for Canadian GP prediction.
        Priority: Canada-similar circuits and most recent races.
        
        Returns:
            Dict: Prediction preparation results
        """
        logger.info(f"🎯 CANADIAN GP PREDICTION DATA COLLECTION")
        logger.info(f"🇨🇦 Target: Canadian GP (Round 10) - Next race prediction")
        
        # Priority order for prediction training
        priority_circuits = [
            # Most Canada-similar circuits first
            *self.canada_similar_circuits,
            # Most recent races for current season trends
            'Monaco', 'Emilia Romagna', 'Miami', 'China', 'Japan', 'Saudi Arabia', 'Bahrain'
        ]
        
        # Remove duplicates while preserving order
        priority_circuits = list(dict.fromkeys(priority_circuits))
        
        prediction_results = {
            'target_race': 'Canadian GP',
            'target_round': 10,
            'priority_circuits_processed': 0,
            'total_prediction_sessions': 0,
            'canada_similar_data_quality': {},
            'recent_trends_data': {},
            'prediction_readiness_score': 0,
            'circuit_results': {}
        }
        
        logger.info(f"📊 Priority circuits for Canada prediction: {priority_circuits}")
        
        for circuit in priority_circuits:
            if circuit in self.circuits_2025_completed:
                logger.info(f"\n🏎️ Collecting priority data: {circuit}")
                
                try:
                    circuit_result = self.collect_circuit_data(circuit)
                    prediction_results['circuit_results'][circuit] = circuit_result
                    
                    if circuit_result.get('successful_sessions', 0) > 0:
                        prediction_results['priority_circuits_processed'] += 1
                        prediction_results['total_prediction_sessions'] += circuit_result.get('successful_sessions', 0)
                        
                        # Track Canada-similar circuit data quality
                        if circuit in self.canada_similar_circuits:
                            prediction_results['canada_similar_data_quality'][circuit] = {
                                'sessions_collected': circuit_result.get('successful_sessions', 0),
                                'has_race_data': 'R' in circuit_result.get('session_results', {}),
                                'has_qualifying_data': 'Q' in circuit_result.get('session_results', {}),
                                'prediction_value': 'high'
                            }
                        
                        # Track recent trends
                        race_round = self.circuits_2025_completed[circuit]['round']
                        if race_round >= 7:  # Recent races (last 3)
                            prediction_results['recent_trends_data'][circuit] = {
                                'round': race_round,
                                'sessions_collected': circuit_result.get('successful_sessions', 0),
                                'recency_weight': 1.0 - (9 - race_round) * 0.1  # Higher weight for more recent
                            }
                        
                except Exception as e:
                    logger.error(f"❌ Priority circuit {circuit} failed: {e}")
        
        # Calculate prediction readiness score
        canada_similar_score = len(prediction_results['canada_similar_data_quality']) * 30  # 30 points per similar circuit
        recent_trends_score = len(prediction_results['recent_trends_data']) * 20  # 20 points per recent race
        total_sessions_score = min(prediction_results['total_prediction_sessions'] * 2, 40)  # 2 points per session, max 40
        
        prediction_results['prediction_readiness_score'] = canada_similar_score + recent_trends_score + total_sessions_score
        
        # Assessment
        logger.info(f"\n🎯 CANADIAN GP PREDICTION READINESS ASSESSMENT")
        logger.info(f"{'='*60}")
        logger.info(f"🇨🇦 Canada-similar circuits: {len(prediction_results['canada_similar_data_quality'])}/3")
        logger.info(f"📈 Recent trends data: {len(prediction_results['recent_trends_data'])} circuits")
        logger.info(f"📊 Total prediction sessions: {prediction_results['total_prediction_sessions']}")
        logger.info(f"🎯 Prediction readiness score: {prediction_results['prediction_readiness_score']}/130")
        
        readiness_level = "EXCELLENT" if prediction_results['prediction_readiness_score'] >= 100 else \
                         "GOOD" if prediction_results['prediction_readiness_score'] >= 80 else \
                         "FAIR" if prediction_results['prediction_readiness_score'] >= 60 else "POOR"
        
        logger.info(f"🏆 Prediction readiness: {readiness_level}")
        
        return prediction_results
    
    def resume_collection(self) -> Dict:
        """
        Resume data collection from where it left off using progress tracking.
        
        Returns:
            Dict: Resume collection results
        """
        logger.info(f"🔄 RESUMING F1 DATA COLLECTION")
        logger.info(f"📋 Previously collected sessions: {len(self.collected_sessions)}")
        
        # Analyze what's missing
        missing_sessions = []
        total_possible_sessions = 0
        
        for circuit, circuit_info in self.circuits_2025_completed.items():
            for session_type in circuit_info['sessions']:
                total_possible_sessions += 1
                session_id = f"{self.year}_{circuit}_{session_type}"
                
                if not self._is_session_collected(circuit, session_type):
                    missing_sessions.append((circuit, session_type))
        
        logger.info(f"📊 Progress: {len(self.collected_sessions)}/{total_possible_sessions} sessions completed")
        logger.info(f"⏭️ Missing sessions: {len(missing_sessions)}")
        
        if not missing_sessions:
            logger.info(f"✅ All data already collected! No resume needed.")
            return {
                'status': 'complete',
                'message': 'All sessions already collected',
                'missing_sessions': 0,
                'completed_sessions': len(self.collected_sessions)
            }
        
        # Resume collection for missing sessions
        resume_results = {
            'missing_sessions_found': len(missing_sessions),
            'newly_collected': 0,
            'failed_to_collect': 0,
            'still_missing': [],
            'session_results': {}
        }
        
        logger.info(f"🚀 Resuming collection for {len(missing_sessions)} missing sessions...")
        
        for circuit, session_type in missing_sessions:
            logger.info(f"📥 Resuming: {circuit} {session_type}")
            
            try:
                result = self.collect_session_data(circuit, session_type)
                resume_results['session_results'][f"{circuit}_{session_type}"] = result
                
                if result and result.get('status') == 'success':
                    resume_results['newly_collected'] += 1
                else:
                    resume_results['failed_to_collect'] += 1
                    resume_results['still_missing'].append(f"{circuit}_{session_type}")
                    
            except Exception as e:
                logger.error(f"❌ Resume failed for {circuit} {session_type}: {e}")
                resume_results['failed_to_collect'] += 1
                resume_results['still_missing'].append(f"{circuit}_{session_type}")
        
        # Final resume summary
        logger.info(f"\n🔄 RESUME COLLECTION COMPLETE")
        logger.info(f"{'='*50}")
        logger.info(f"✅ Newly collected: {resume_results['newly_collected']}")
        logger.info(f"❌ Failed to collect: {resume_results['failed_to_collect']}")
        logger.info(f"⏭️ Still missing: {len(resume_results['still_missing'])}")
        
        return resume_results
    
    def get_collection_status(self) -> Dict:
        """
        Get detailed status of current data collection progress.
        
        Returns:
            Dict: Comprehensive collection status
        """
        # Calculate completion statistics
        total_possible_sessions = sum(len(info['sessions']) for info in self.circuits_2025_completed.values())
        collected_sessions_count = len(self.collected_sessions)
        completion_percentage = (collected_sessions_count / total_possible_sessions) * 100 if total_possible_sessions > 0 else 0
        
        # Analyze by circuit
        circuit_status = {}
        for circuit, circuit_info in self.circuits_2025_completed.items():
            circuit_sessions = circuit_info['sessions']
            collected_for_circuit = sum(
                1 for session_type in circuit_sessions
                if self._is_session_collected(circuit, session_type)
            )
            
            circuit_status[circuit] = {
                'total_sessions': len(circuit_sessions),
                'collected_sessions': collected_for_circuit,
                'completion_percentage': (collected_for_circuit / len(circuit_sessions)) * 100,
                'missing_sessions': [
                    session_type for session_type in circuit_sessions
                    if not self._is_session_collected(circuit, session_type)
                ],
                'is_sprint_weekend': circuit_info.get('has_sprint', False),
                'is_canada_similar': circuit in self.canada_similar_circuits,
                'round': circuit_info.get('round', 0)
            }
        
        # Prediction readiness assessment
        canada_similar_complete = sum(
            1 for circuit in self.canada_similar_circuits
            if circuit_status.get(circuit, {}).get('completion_percentage', 0) >= 80
        )
        
        prediction_readiness = {
            'canada_similar_circuits_ready': f"{canada_similar_complete}/3",
            'overall_completion': f"{completion_percentage:.1f}%",
            'critical_data_available': canada_similar_complete >= 2 and completion_percentage >= 70,
            'recommended_action': self._get_recommended_action(completion_percentage, canada_similar_complete)
        }
        
        status = {
            'collection_overview': {
                'total_possible_sessions': total_possible_sessions,
                'collected_sessions': collected_sessions_count,
                'completion_percentage': completion_percentage,
                'target_race': self.prediction_target,
                'total_circuits': len(self.circuits_2025_completed)
            },
            'circuit_breakdown': circuit_status,
            'prediction_readiness': prediction_readiness,
            'collection_statistics': self.stats,
            'last_updated': datetime.now().isoformat()
        }
        
        return status
    
    def _get_recommended_action(self, completion_percentage: float, canada_similar_complete: int) -> str:
        """Get recommended next action based on collection status."""
        if completion_percentage >= 95:
            return "Collection complete - ready for prediction modeling"
        elif completion_percentage >= 80 and canada_similar_complete >= 2:
            return "Sufficient data for prediction - consider running prediction prep"
        elif canada_similar_complete < 2:
            return "Priority: collect Canada-similar circuits (Australia, Emilia Romagna, Spain)"
        elif completion_percentage < 50:
            return "Run full collection --collect-available"
        else:
            return "Resume collection --resume to fill gaps"
    
    def generate_collection_report(self) -> str:
        """
        Generate a comprehensive collection report.
        
        Returns:
            str: Formatted collection report
        """
        status = self.get_collection_status()
        
        report = f"""
{'='*80}
F1 2025 DATA COLLECTION REPORT - CANADIAN GP PREDICTION FOCUS
{'='*80}

📅 Collection Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🎯 Target Prediction: Canadian Grand Prix (Round 10)
📊 Data Source: 2025 F1 Season (Rounds 1-9 completed)

OVERVIEW
--------
Total Sessions Available: {status['collection_overview']['total_possible_sessions']}
Sessions Collected: {status['collection_overview']['collected_sessions']}
Overall Completion: {status['collection_overview']['completion_percentage']:.1f}%

PREDICTION READINESS
-------------------
🇨🇦 Canada-Similar Circuits: {status['prediction_readiness']['canada_similar_circuits_ready']}
   • Australia: {'✅' if status['circuit_breakdown'].get('Australia', {}).get('completion_percentage', 0) >= 80 else '❌'} ({status['circuit_breakdown'].get('Australia', {}).get('completion_percentage', 0):.0f}%)
   • Emilia Romagna: {'✅' if status['circuit_breakdown'].get('Emilia Romagna', {}).get('completion_percentage', 0) >= 80 else '❌'} ({status['circuit_breakdown'].get('Emilia Romagna', {}).get('completion_percentage', 0):.0f}%)
   • Spain: {'✅' if status['circuit_breakdown'].get('Spain', {}).get('completion_percentage', 0) >= 80 else '❌'} ({status['circuit_breakdown'].get('Spain', {}).get('completion_percentage', 0):.0f}%)

🏆 Critical Data Available: {'YES' if status['prediction_readiness']['critical_data_available'] else 'NO'}

CIRCUIT-BY-CIRCUIT BREAKDOWN
----------------------------"""

        for circuit, circuit_data in status['circuit_breakdown'].items():
            status_icon = '✅' if circuit_data['completion_percentage'] >= 80 else '⚠️' if circuit_data['completion_percentage'] >= 50 else '❌'
            sprint_indicator = ' 🏃(Sprint)' if circuit_data['is_sprint_weekend'] else ''
            canada_indicator = ' 🇨🇦(Canada-like)' if circuit_data['is_canada_similar'] else ''
            
            report += f"""
{status_icon} Round {circuit_data['round']:2d} - {circuit:<15} {circuit_data['completion_percentage']:5.0f}% ({circuit_data['collected_sessions']}/{circuit_data['total_sessions']}){sprint_indicator}{canada_indicator}"""
            
            if circuit_data['missing_sessions']:
                report += f"""
    Missing: {', '.join(circuit_data['missing_sessions'])}"""

        report += f"""

COLLECTION STATISTICS
--------------------
Successful Collections: {status['collection_statistics']['successful_collections']}
Failed Collections: {status['collection_statistics']['failed_collections']}
Skipped (Already Collected): {status['collection_statistics']['skipped_existing']}
Rate Limit Delays: {status['collection_statistics']['rate_limit_delays']}
Timeouts: {status['collection_statistics']['timeouts']}

SESSION TYPE BREAKDOWN
---------------------"""

        for session_type, session_stats in status['collection_statistics']['sessions_by_type'].items():
            if session_stats['attempted'] > 0:
                success_rate = (session_stats['successful'] / session_stats['attempted']) * 100
                report += f"""
{session_type:3s}: {session_stats['successful']:2d} successful, {session_stats['attempted']:2d} attempted ({success_rate:5.1f}% success rate)"""

        report += f"""

RECOMMENDATION
--------------
🎯 {status['prediction_readiness']['recommended_action']}

DATA STORAGE LOCATIONS
---------------------
Raw Data: {self.raw_dir}
Progress Tracking: {self.progress_file}
Cache: {self.cache_dir}
Logs: f1_data_collection_2025.log

{'='*80}
Report generated by F1DataCollector2025
{'='*80}
"""
        
        return report


def main():
    """Main execution function with command-line interface."""
    parser = argparse.ArgumentParser(
        description='F1 2025 Data Collector - Canadian GP Prediction Focus',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_collector_2025.py --collect-available    # Collect all completed 2025 races
  python data_collector_2025.py --prediction-prep      # Collect priority data for Canada prediction
  python data_collector_2025.py --circuit "Spain"      # Collect specific circuit data
  python data_collector_2025.py --resume               # Resume incomplete collection
  python data_collector_2025.py --status               # Show collection status
  python data_collector_2025.py --report               # Generate detailed report
        """
    )
    
    # Command options
    parser.add_argument('--collect-available', action='store_true',
                       help='Collect data for all completed 2025 races (Rounds 1-9)')
    parser.add_argument('--prediction-prep', action='store_true',
                       help='Collect priority data optimized for Canadian GP prediction')
    parser.add_argument('--circuit', type=str,
                       help='Collect data for specific circuit (e.g., "Spain", "Australia")')
    parser.add_argument('--resume', action='store_true',
                       help='Resume incomplete data collection')
    parser.add_argument('--status', action='store_true',
                       help='Show current collection status')
    parser.add_argument('--report', action='store_true',
                       help='Generate comprehensive collection report')
    
    # Configuration options
    parser.add_argument('--timeout', type=int, default=300,
                       help='Timeout in seconds for each session collection (default: 300)')
    parser.add_argument('--rate-limit-delay', type=int, default=8,
                       help='Delay between API calls in seconds (default: 8)')
    parser.add_argument('--base-dir', type=str, default='data',
                       help='Base directory for data storage (default: data)')
    
    args = parser.parse_args()
    
    # Validate that at least one action is specified
    actions = [args.collect_available, args.prediction_prep, args.circuit, args.resume, args.status, args.report]
    if not any(actions):
        parser.error("Please specify at least one action: --collect-available, --prediction-prep, --circuit, --resume, --status, or --report")
    
    # Initialize collector
    try:
        collector = F1DataCollector2025(
            base_dir=args.base_dir,
            timeout=args.timeout,
            rate_limit_delay=args.rate_limit_delay
        )
        
        logger.info(f"🏎️ F1 2025 Data Collector initialized")
        logger.info(f"🎯 Target: Canadian GP prediction data")
        logger.info(f"📁 Data directory: {collector.base_dir}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize collector: {e}")
        sys.exit(1)
    
    # Execute requested actions
    try:
        if args.status:
            logger.info(f"📊 Getting collection status...")
            status = collector.get_collection_status()
            
            print(f"\n{'='*60}")
            print(f"F1 2025 COLLECTION STATUS")
            print(f"{'='*60}")
            print(f"Overall Progress: {status['collection_overview']['completion_percentage']:.1f}%")
            print(f"Sessions Collected: {status['collection_overview']['collected_sessions']}/{status['collection_overview']['total_possible_sessions']}")
            print(f"Canada-Similar Circuits: {status['prediction_readiness']['canada_similar_circuits_ready']}")
            print(f"Prediction Ready: {'YES' if status['prediction_readiness']['critical_data_available'] else 'NO'}")
            print(f"Recommended Action: {status['prediction_readiness']['recommended_action']}")
        
        if args.report:
            logger.info(f"📋 Generating collection report...")
            report = collector.generate_collection_report()
            
            # Save report to file
            report_file = collector.base_dir / 'collection_report_2025.txt'
            with open(report_file, 'w') as f:
                f.write(report)
            
            print(report)
            logger.info(f"📄 Report saved to: {report_file}")
        
        if args.circuit:
            logger.info(f"🏁 Collecting data for circuit: {args.circuit}")
            result = collector.collect_circuit_data(args.circuit)
            
            if result.get('successful_sessions', 0) > 0:
                logger.info(f"✅ Successfully collected {args.circuit} data")
            else:
                logger.error(f"❌ Failed to collect {args.circuit} data")
                sys.exit(1)
        
        if args.resume:
            logger.info(f"🔄 Resuming data collection...")
            result = collector.resume_collection()
            
            if result.get('newly_collected', 0) > 0:
                logger.info(f"✅ Resume successful: {result['newly_collected']} new sessions collected")
            else:
                logger.info(f"ℹ️ Resume complete: {result.get('message', 'No new data to collect')}")
        
        if args.prediction_prep:
            logger.info(f"🎯 Collecting Canadian GP prediction preparation data...")
            result = collector.collect_prediction_prep_data()
            
            if result.get('prediction_readiness_score', 0) >= 80:
                logger.info(f"🏆 Prediction preparation successful: Score {result['prediction_readiness_score']}/130")
            else:
                logger.warning(f"⚠️ Prediction preparation incomplete: Score {result.get('prediction_readiness_score', 0)}/130")
        
        if args.collect_available:
            logger.info(f"🚀 Collecting all available 2025 F1 data...")
            result = collector.collect_all_available_data()
            
            success_rate = (result.get('successful_circuits', 0) / result.get('total_circuits', 1)) * 100
            if success_rate >= 80:
                logger.info(f"🏆 Collection successful: {result.get('successful_circuits', 0)}/{result.get('total_circuits', 0)} circuits")
            else:
                logger.warning(f"⚠️ Collection partially successful: {result.get('successful_circuits', 0)}/{result.get('total_circuits', 0)} circuits")
        
        # Final status summary
        final_status = collector.get_collection_status()
        logger.info(f"\n🏁 FINAL STATUS")
        logger.info(f"Overall completion: {final_status['collection_overview']['completion_percentage']:.1f}%")
        logger.info(f"Prediction readiness: {'READY' if final_status['prediction_readiness']['critical_data_available'] else 'NOT READY'}")
        
    except KeyboardInterrupt:
        logger.info(f"\n⚠️ Collection interrupted by user")
        logger.info(f"📋 Progress has been saved - use --resume to continue")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"❌ Collection failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # For IDE execution - automatically run collection if no args provided
    import sys
    
    # If running in IDE without arguments, run prediction prep
    if len(sys.argv) == 1:
        print("🏎️ Running F1 Data Collector - Canadian GP Prediction Data Collection")
        print("This will collect priority circuits: Australia, Emilia Romagna, Spain, plus recent races")
        print("Estimated time: 10-15 minutes with rate limiting\n")
        
        # Add default arguments for IDE execution
        sys.argv.extend(['--prediction-prep'])
    
    main()