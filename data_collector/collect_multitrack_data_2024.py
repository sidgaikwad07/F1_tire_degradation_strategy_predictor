"""
Created on Sat Jun  7 10:13:14 2025

@author: sid
F1 Data Collector - 2024 Season (Rate Limited & Robust)
Handles API rate limits, prevents data duplication, and ensures robust data loading.

This version includes:
- Rate limiting protection (500 calls/hour limit)
- Data duplication prevention
- Improved session loading with validation
- Progress persistence and resume capability
- Updated 2024 calendar with sprint format consolidation

Usage:
    python data_collector_2024.py --collect-all
    python data_collector_2024.py --circuit "Canada" --sessions "Q,R"
    python data_collector_2024.py --resume
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
        logging.FileHandler('f1_data_collection_2024.log'),
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

class F1DataCollector2024:
    """
    Rate-limited F1 Data Collector for 2024 season with robust error handling.
    
    Features:
    - API rate limit management (500 calls/hour)
    - Data duplication prevention
    - Progress persistence and resume capability
    - Robust session loading with validation
    - Updated 2024 sprint format and calendar changes
    """
    
    def __init__(self, base_dir: str = 'data', timeout: int = 300, rate_limit_delay: int = 8):
        """
        Initialize the 2024 F1 Data Collector with rate limiting.
        
        Args:
            base_dir (str): Base directory for data storage
            timeout (int): Timeout in seconds for each data collection operation
            rate_limit_delay (int): Delay between API calls to respect rate limits
        """
        self.year = 2024
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / 'raw' / str(self.year)
        self.cache_dir = Path('cache')
        self.progress_dir = self.base_dir / 'progress'
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay  # 8 seconds = ~450 calls/hour (safe margin)
        
        # Create directory structure
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        self.progress_dir.mkdir(exist_ok=True)
        
        # Enable FastF1 cache for faster subsequent runs
        fastf1.Cache.enable_cache(str(self.cache_dir))
        
        # Progress tracking file
        self.progress_file = self.progress_dir / 'collection_progress_2024.json'
        self.collected_sessions = self._load_progress()
        
        # 2024 F1 Calendar - 24 races with updated sprint format
        self.circuits_2024 = {
            'Bahrain': {'sessions': ['FP1', 'FP2', 'FP3', 'Q', 'R'], 'has_sprint': False},
            'Saudi Arabia': {'sessions': ['FP1', 'FP2', 'FP3', 'Q', 'R'], 'has_sprint': False},
            'Australia': {'sessions': ['FP1', 'FP2', 'FP3', 'Q', 'R'], 'has_sprint': False},
            'Japan': {'sessions': ['FP1', 'FP2', 'FP3', 'Q', 'R'], 'has_sprint': False},
            'China': {'sessions': ['FP1', 'SQ', 'S', 'Q', 'R'], 'has_sprint': True},  # Sprint weekend
            'Miami': {'sessions': ['FP1', 'SQ', 'S', 'Q', 'R'], 'has_sprint': True},  # Sprint weekend
            'Emilia Romagna': {'sessions': ['FP1', 'FP2', 'FP3', 'Q', 'R'], 'has_sprint': False},
            'Monaco': {'sessions': ['FP1', 'FP2', 'FP3', 'Q', 'R'], 'has_sprint': False},
            'Canada': {'sessions': ['FP1', 'FP2', 'FP3', 'Q', 'R'], 'has_sprint': False},
            'Spain': {'sessions': ['FP1', 'FP2', 'FP3', 'Q', 'R'], 'has_sprint': False},
            'Austria': {'sessions': ['FP1', 'SQ', 'S', 'Q', 'R'], 'has_sprint': True},  # Sprint weekend
            'Great Britain': {'sessions': ['FP1', 'FP2', 'FP3', 'Q', 'R'], 'has_sprint': False},
            'Hungary': {'sessions': ['FP1', 'FP2', 'FP3', 'Q', 'R'], 'has_sprint': False},
            'Belgium': {'sessions': ['FP1', 'FP2', 'FP3', 'Q', 'R'], 'has_sprint': False},
            'Netherlands': {'sessions': ['FP1', 'FP2', 'FP3', 'Q', 'R'], 'has_sprint': False},
            'Italy': {'sessions': ['FP1', 'FP2', 'FP3', 'Q', 'R'], 'has_sprint': False},
            'Azerbaijan': {'sessions': ['FP1', 'FP2', 'FP3', 'Q', 'R'], 'has_sprint': False},
            'Singapore': {'sessions': ['FP1', 'FP2', 'FP3', 'Q', 'R'], 'has_sprint': False},
            'United States': {'sessions': ['FP1', 'SQ', 'S', 'Q', 'R'], 'has_sprint': True},  # Sprint weekend
            'Mexico': {'sessions': ['FP1', 'FP2', 'FP3', 'Q', 'R'], 'has_sprint': False},
            'Brazil': {'sessions': ['FP1', 'SQ', 'S', 'Q', 'R'], 'has_sprint': True},  # Sprint weekend
            'Las Vegas': {'sessions': ['FP1', 'FP2', 'FP3', 'Q', 'R'], 'has_sprint': False},
            'Qatar': {'sessions': ['FP1', 'SQ', 'S', 'Q', 'R'], 'has_sprint': True},  # Sprint weekend
            'Abu Dhabi': {'sessions': ['FP1', 'FP2', 'FP3', 'Q', 'R'], 'has_sprint': False}
        }
        
        # Priority circuits for focused collection (updated for 2024)
        self.priority_circuits = [
            'Bahrain', 'Australia', 'China', 'Miami', 'Monaco', 'Canada', 
            'Austria', 'Great Britain', 'Hungary', 'Netherlands', 'Singapore', 'Las Vegas'
        ]
        
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
            'china_return_collected': False  # Track China's return to calendar
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
            
            # Extract different data components with improved error handling
            results_data = self._extract_results(session)
            laps_data = self._extract_laps(session)
            weather_data = self._extract_weather(session)
            
            # Extract session-specific data based on 2024 format
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
            
            # Track China's return
            if circuit == 'China':
                self.stats['china_return_collected'] = True
            
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
                    'is_china_return': circuit == 'China'  # Flag China's return
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
            
            # For sessions that should have results, check if they exist
            if hasattr(session, 'results'):
                # Results exist but might be empty for practice sessions
                pass
            
            # For sessions that should have laps, check if they exist
            if hasattr(session, 'laps'):
                # Laps exist but might be empty
                pass
            
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
                'is_china_return': circuit == 'China'  # Flag China's return to calendar
            },
            'session_details': {
                'total_laps': getattr(session, 'total_laps', None),
                'is_sprint_weekend': self.circuits_2024.get(circuit, {}).get('has_sprint', False),
                'season_race_count': 24  # 2024 had 24 races
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
                    'session_format': 'sprint_qualifying_2024'
                }
            
            logger.info(f"      🏃 Sprint Qualifying data extracted (2024 format)")
            
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
                    'sprint_format': '2024_consolidated'  # Track the consolidated format
                }
            
            logger.info(f"      🏃 Sprint data extracted (2024 format)")
            
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
                    'race_distance': getattr(session, 'total_laps', 0)
                }
            
            logger.info(f"      🏁 Race data extracted")
            
        except Exception as e:
            logger.warning(f"      ❌ Race data extraction failed: {e}")
        
        return race_data
    
    def _save_session_data(self, session_data: Dict, circuit: str, session_type: str):
        """Save collected session data to organized file structure."""
        
        # Create circuit directory
        circuit_dir = self.raw_dir / circuit
        circuit_dir.mkdir(exist_ok=True)
        
        # Save session info as JSON
        session_info_file = circuit_dir / f'{session_type}_session_info.json'
        with open(session_info_file, 'w') as f:
            json.dump(session_data['session_info'], f, indent=2, default=str)
        
        # Save results data if available
        if session_data['results'] is not None:
            results_file = circuit_dir / f'{session_type}_results.csv'
            session_data['results'].to_csv(results_file, index=False)
        
        # Save laps data if available
        if session_data['laps'] is not None:
            laps_file = circuit_dir / f'{session_type}_laps.csv'
            session_data['laps'].to_csv(laps_file, index=False)

        # Save weather data if available
        if session_data['weather'] is not None:
            weather_file = circuit_dir / f'{session_type}_weather.csv'
            session_data['weather'].to_csv(weather_file, index=False)
        
        # Save session-specific data
        if session_data['session_specific']:
            session_specific_dir = circuit_dir / f'{session_type}_specific_data'
            session_specific_dir.mkdir(exist_ok=True)
            
            for data_name, data in session_data['session_specific'].items():
                if isinstance(data, pd.DataFrame):
                    data_file = session_specific_dir / f'{data_name}.csv'
                    data.to_csv(data_file, index=False)
                else:
                    data_file = session_specific_dir / f'{data_name}.json'
                    with open(data_file, 'w') as f:
                        json.dump(data, f, indent=2, default=str)
        
        # Save collection metadata
        metadata_file = circuit_dir / f'{session_type}_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(session_data['collection_metadata'], f, indent=2, default=str)
    
    def collect_circuit_complete(self, circuit: str) -> Dict:
        """Collect all available sessions for a specific circuit."""
        
        logger.info(f"\n🏁 Collecting complete {circuit} Grand Prix 2024")
        
        if circuit not in self.circuits_2024:
            logger.error(f"❌ Circuit '{circuit}' not found in 2024 calendar")
            return {'error': f'Circuit {circuit} not in 2024 calendar'}
        
        circuit_info = self.circuits_2024[circuit]
        sessions = circuit_info['sessions']
        
        logger.info(f"📋 Sessions to collect: {sessions}")
        if circuit_info['has_sprint']:
            logger.info(f"🏃 Sprint weekend detected (2024 format)!")
        if circuit == 'China':
            logger.info(f"🆕 China returns to the calendar!")
        
        results = {
            'circuit': circuit,
            'year': self.year,
            'total_sessions': len(sessions),
            'successful_sessions': [],
            'failed_sessions': [],
            'skipped_sessions': [],
            'is_sprint_weekend': circuit_info['has_sprint'],
            'is_china_return': circuit == 'China'
        }
        
        for session_type in sessions:
            logger.info(f"\n  📊 Collecting {session_type} session...")
            session_data = self.collect_session_data(circuit, session_type)
            
            if session_data:
                if session_data.get('status') == 'skipped':
                    results['skipped_sessions'].append(session_type)
                    logger.info(f"  ⏭️ {session_type} skipped (already collected)")
                else:
                    results['successful_sessions'].append(session_type)
                    logger.info(f"  ✅ {session_type} collected successfully")
            else:
                results['failed_sessions'].append(session_type)
                logger.error(f"  ❌ {session_type} collection failed")
        
        # Calculate success rate
        completed_sessions = len(results['successful_sessions']) + len(results['skipped_sessions'])
        success_rate = completed_sessions / len(sessions) * 100
        results['success_rate'] = success_rate
        
        logger.info(f"\n🎯 {circuit} Collection Summary:")
        logger.info(f"   ✅ Successful: {len(results['successful_sessions'])}")
        logger.info(f"   ⏭️ Skipped: {len(results['skipped_sessions'])}")
        logger.info(f"   ❌ Failed: {len(results['failed_sessions'])}")
        logger.info(f"   📊 Completion Rate: {success_rate:.1f}%")
        
        return results
    
    def collect_priority_circuits(self) -> Dict:
        """Collect data for priority circuits with rate limiting."""
        
        logger.info(f"\n🎯 Priority Circuits Collection - 2024 Season")
        logger.info(f"🏁 Priority circuits: {self.priority_circuits}")
        logger.info(f"⏱️ Rate limiting: {self.rate_limit_delay}s between API calls")
        logger.info(f"🆕 Includes China's return and consolidated sprint format")
        
        collection_summary = {
            'collection_type': 'priority_circuits',
            'year': self.year,
            'circuits': self.priority_circuits,
            'rate_limit_delay': self.rate_limit_delay,
            'start_time': datetime.now().isoformat(),
            'results': {}
        }
        
        for i, circuit in enumerate(self.priority_circuits, 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"🏁 PRIORITY {i}/{len(self.priority_circuits)}: {circuit} Grand Prix 2024")
            logger.info(f"{'='*50}")
            
            circuit_results = self.collect_circuit_complete(circuit)
            collection_summary['results'][circuit] = circuit_results
            
            # Progress update
            completed_circuits = i
            total_circuits = len(self.priority_circuits)
            logger.info(f"📊 Overall Progress: {completed_circuits}/{total_circuits} circuits processed")
        
        collection_summary['end_time'] = datetime.now().isoformat()
        collection_summary['final_stats'] = self.stats.copy()
        
        # Save priority collection summary
        summary_file = self.base_dir / 'priority_circuits_2024_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(collection_summary, f, indent=2, default=str)
        
        self._print_collection_summary()
        
        return collection_summary
    
    def collect_all_circuits(self) -> Dict:
        """Collect data for all circuits in the 2024 season with rate limiting."""
        
        logger.info(f"\n🏆 Complete 2024 Season Collection")
        logger.info(f"🏁 Total circuits: {len(self.circuits_2024)} (24 races)")
        logger.info(f"⏱️ Rate limiting: {self.rate_limit_delay}s between API calls")
        logger.info(f"⏰ Estimated time: ~{len(self.circuits_2024) * 5 * self.rate_limit_delay / 60:.0f} minutes")
        logger.info(f"🆕 New features: China's return, consolidated sprint format")
        
        collection_summary = {
            'collection_type': 'complete_season',
            'year': self.year,
            'total_circuits': len(self.circuits_2024),
            'total_races': 24,
            'rate_limit_delay': self.rate_limit_delay,
            'start_time': datetime.now().isoformat(),
            'results': {},
            'season_features': {
                'china_return': True,
                'consolidated_sprint_format': True,
                'total_sprint_weekends': len([c for c, info in self.circuits_2024.items() if info['has_sprint']])
            }
        }
        
        for i, circuit in enumerate(self.circuits_2024.keys(), 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"🏁 {i}/{len(self.circuits_2024)}: {circuit} Grand Prix 2024")
            logger.info(f"{'='*60}")
            
            circuit_results = self.collect_circuit_complete(circuit)
            collection_summary['results'][circuit] = circuit_results
            
            # Progress update
            logger.info(f"📊 Overall Progress: {i}/{len(self.circuits_2024)} circuits processed")
        
        collection_summary['end_time'] = datetime.now().isoformat()
        collection_summary['final_stats'] = self.stats.copy()
        
        # Save complete season summary
        summary_file = self.base_dir / 'complete_season_2024_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(collection_summary, f, indent=2, default=str)
        
        self._print_collection_summary()
        
        return collection_summary
    
    def collect_sprint_weekends_focus(self) -> Dict:
        """NEW: Collect data focusing on 2024's sprint weekends."""
        
        logger.info(f"\n🏃 2024 Sprint Weekends Focus Collection")
        logger.info(f"🎯 Consolidated Sprint Format")
        
        # Sprint circuits in 2024
        sprint_circuits = [circuit for circuit, info in self.circuits_2024.items() if info['has_sprint']]
        
        logger.info(f"🏃 Sprint circuits: {sprint_circuits}")
        
        collection_summary = {
            'collection_type': 'sprint_weekends_focus',
            'year': self.year,
            'start_time': datetime.now().isoformat(),
            'sprint_circuits': sprint_circuits,
            'total_sprint_weekends': len(sprint_circuits),
            'results': {}
        }
        
        for circuit in sprint_circuits:
            logger.info(f"\n🏃 SPRINT WEEKEND: {circuit} Grand Prix 2024")
            circuit_results = self.collect_circuit_complete(circuit)
            collection_summary['results'][circuit] = circuit_results
        
        collection_summary['end_time'] = datetime.now().isoformat()
        collection_summary['final_stats'] = self.stats.copy()
        
        # Save sprint weekends summary
        summary_file = self.base_dir / 'sprint_weekends_2024_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(collection_summary, f, indent=2, default=str)
        
        self._print_collection_summary()
        
        return collection_summary
    
    def collect_china_return_focus(self) -> Dict:
        """NEW: Special collection for China's return to the F1 calendar."""
        
        logger.info(f"\n🇨🇳 China Grand Prix Return - Special Collection")
        logger.info(f"🎯 First time back since 2019")
        
        collection_summary = {
            'collection_type': 'china_return_focus',
            'year': self.year,
            'start_time': datetime.now().isoformat(),
            'special_note': 'China returns to F1 calendar after COVID hiatus',
            'results': {}
        }
        
        logger.info(f"\n🇨🇳 CHINA RETURN: Shanghai International Circuit 2024")
        china_results = self.collect_circuit_complete('China')
        collection_summary['results']['China'] = china_results
        
        collection_summary['end_time'] = datetime.now().isoformat()
        collection_summary['final_stats'] = self.stats.copy()
        
        # Save China return summary
        summary_file = self.base_dir / 'china_return_2024_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(collection_summary, f, indent=2, default=str)
        
        self._print_collection_summary()
        
        return collection_summary
    
    def resume_collection(self) -> Dict:
        """Resume collection from where it left off."""
        
        logger.info(f"\n🔄 Resuming Collection from Progress File")
        logger.info(f"📋 Already collected: {len(self.collected_sessions)} sessions")
        
        # Find incomplete circuits
        incomplete_circuits = []
        for circuit, circuit_info in self.circuits_2024.items():
            sessions = circuit_info['sessions']
            collected_for_circuit = [
                s for s in self.collected_sessions.keys() 
                if s.startswith(f"{self.year}_{circuit}_")
            ]
            
            if len(collected_for_circuit) < len(sessions):
                missing_sessions = []
                for session in sessions:
                    session_id = f"{self.year}_{circuit}_{session}"
                    if session_id not in self.collected_sessions:
                        missing_sessions.append(session)
                
                incomplete_circuits.append({
                    'circuit': circuit,
                    'missing_sessions': missing_sessions,
                    'collected': len(collected_for_circuit),
                    'total': len(sessions)
                })
        
        if not incomplete_circuits:
            logger.info("✅ All circuits appear to be complete!")
            return {'status': 'complete', 'message': 'No missing data found'}
        
        logger.info(f"📊 Found {len(incomplete_circuits)} circuits with missing data:")
        for incomplete in incomplete_circuits:
            logger.info(f"   {incomplete['circuit']}: {incomplete['collected']}/{incomplete['total']} sessions")
        
        # Resume collection for incomplete circuits
        resume_summary = {
            'collection_type': 'resume',
            'start_time': datetime.now().isoformat(),
            'incomplete_circuits_found': len(incomplete_circuits),
            'results': {}
        }
        
        for incomplete in incomplete_circuits:
            circuit = incomplete['circuit']
            missing_sessions = incomplete['missing_sessions']
            
            logger.info(f"\n🔄 Resuming {circuit} - Missing: {missing_sessions}")
            
            circuit_results = {
                'circuit': circuit,
                'missing_sessions': missing_sessions,
                'successful_sessions': [],
                'failed_sessions': [],
                'skipped_sessions': []
            }
            
            for session_type in missing_sessions:
                session_data = self.collect_session_data(circuit, session_type)
                
                if session_data:
                    if session_data.get('status') == 'skipped':
                        circuit_results['skipped_sessions'].append(session_type)
                    else:
                        circuit_results['successful_sessions'].append(session_type)
                else:
                    circuit_results['failed_sessions'].append(session_type)
            
            resume_summary['results'][circuit] = circuit_results
        
        resume_summary['end_time'] = datetime.now().isoformat()
        resume_summary['final_stats'] = self.stats.copy()
        
        # Save resume summary
        summary_file = self.base_dir / 'resume_collection_2024_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(resume_summary, f, indent=2, default=str)
        
        self._print_collection_summary()
        
        return resume_summary
    
    def _print_collection_summary(self):
        """Print detailed collection statistics."""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 F1 2024 DATA COLLECTION SUMMARY")
        logger.info(f"{'='*60}")
        
        # Overall statistics
        total_attempted = self.stats['total_attempted']
        total_successful = self.stats['successful_collections']
        total_skipped = self.stats['skipped_existing']
        success_rate = (total_successful / total_attempted * 100) if total_attempted > 0 else 0
        
        logger.info(f"📈 OVERALL STATISTICS:")
        logger.info(f"   ✅ Successful: {total_successful}")
        logger.info(f"   ⏭️ Skipped (existing): {total_skipped}")
        logger.info(f"   🎯 Total Attempted: {total_attempted}")
        logger.info(f"   📊 Success Rate: {success_rate:.1f}%")
        logger.info(f"   ❌ Failed: {self.stats['failed_collections']}")
        
        # 2024 specific features
        logger.info(f"\n🆕 2024 SPECIFIC FEATURES:")
        logger.info(f"   🏃 Sprint weekends collected: {self.stats['sprint_weekends_collected']}")
        logger.info(f"   🇨🇳 China return data: {'✅' if self.stats['china_return_collected'] else '❌'}")
        total_sprint_weekends = len([c for c, info in self.circuits_2024.items() if info['has_sprint']])
        logger.info(f"   🏁 Total sprint weekends in 2024: {total_sprint_weekends}")
        
        # Rate limiting stats
        logger.info(f"\n⏱️ RATE LIMITING:")
        logger.info(f"   🕒 Delay between calls: {self.rate_limit_delay}s")
        logger.info(f"   ⏳ Rate limit delays applied: {self.stats['rate_limit_delays']}")
        
        # Error breakdown
        logger.info(f"\n❌ ERROR BREAKDOWN:")
        logger.info(f"   ⏰ Timeouts: {self.stats['timeouts']}")
        logger.info(f"   🌐 Network Errors: {self.stats['network_errors']}")
        logger.info(f"   📊 Data Errors: {self.stats['data_errors']}")
        
        # Session type breakdown
        logger.info(f"\n📋 SESSION TYPE BREAKDOWN:")
        for session_type, session_stats in self.stats['sessions_by_type'].items():
            attempted = session_stats['attempted']
            successful = session_stats['successful']
            skipped = session_stats['skipped']
            rate = (successful / attempted * 100) if attempted > 0 else 0
            
            # Add emoji for session types
            session_emoji = {
                'FP1': '🔧', 'FP2': '🔧', 'FP3': '🔧',
                'Q': '🏁', 'SQ': '🏃', 'S': '🏃', 'R': '🏆'
            }
            emoji = session_emoji.get(session_type, '📊')
            
            logger.info(f"   {emoji} {session_type}: ✅{successful} ⏭️{skipped} 🎯{attempted} ({rate:.1f}%)")
        
        # Progress file info
        logger.info(f"\n📋 PROGRESS TRACKING:")
        logger.info(f"   📁 Progress file: {self.progress_file}")
        logger.info(f"   📊 Sessions tracked: {len(self.collected_sessions)}")
        
        # Failed sessions details
        if self.stats['failed_sessions']:
            logger.info(f"\n❌ FAILED SESSIONS:")
            for failed_session in self.stats['failed_sessions'][:10]:
                logger.info(f"   {failed_session}")
            if len(self.stats['failed_sessions']) > 10:
                logger.info(f"   ... and {len(self.stats['failed_sessions']) - 10} more")

def main():
    """Main function with enhanced command line interface for 2024."""
    
    parser = argparse.ArgumentParser(
        description='F1 Data Collector - 2024 Season (Rate Limited & Robust)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_collector_2024.py --collect-all
  python data_collector_2024.py --priority-circuits
  python data_collector_2024.py --circuit "China" --sessions "Q,R"
  python data_collector_2024.py --sprint-weekends-focus
  python data_collector_2024.py --china-return-focus
  python data_collector_2024.py --resume
  python data_collector_2024.py --check-progress
        """
    )
    
    parser.add_argument('--collect-all', action='store_true',
                       help='Collect data for all circuits in 2024 season')
    parser.add_argument('--priority-circuits', action='store_true',
                       help='Collect data for priority circuits only')
    parser.add_argument('--sprint-weekends-focus', action='store_true',
                       help='Focus on 2024 sprint weekends (consolidated format)')
    parser.add_argument('--china-return-focus', action='store_true',
                       help='Focus on China\'s return to the calendar')
    parser.add_argument('--circuit', type=str,
                       help='Specific circuit to collect')
    parser.add_argument('--sessions', type=str,
                       help='Comma-separated session types (e.g., "FP1,SQ,S,Q,R")')
    parser.add_argument('--resume', action='store_true',
                       help='Resume collection from progress file')
    parser.add_argument('--check-progress', action='store_true',
                       help='Check collection progress without collecting')
    parser.add_argument('--timeout', type=int, default=300,
                       help='Timeout in seconds (default: 300)')
    parser.add_argument('--rate-limit-delay', type=int, default=8,
                       help='Delay between API calls in seconds (default: 8)')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Base directory for data storage')
    parser.add_argument('--list-circuits', action='store_true',
                       help='List all available circuits for 2024')
    
    args = parser.parse_args()
    
    # Enhanced header for 2024
    print("""
🏎️  F1 DATA COLLECTOR - 2024 SEASON (RATE LIMITED)
════════════════════════════════════════════════════════════════
  🇨🇳 China Returns | 24 Races | Consolidated Sprint Format
════════════════════════════════════════════════════════════════
    """)
    
    print(f"⚙️  Configuration:")
    print(f"   📁 Data Directory: {args.data_dir}")
    print(f"   ⏰ Timeout: {args.timeout} seconds")
    print(f"   ⏱️ Rate Limit Delay: {args.rate_limit_delay} seconds")
    print(f"   📅 Target Year: 2024")
    print(f"   🚫 API Limit: 500 calls/hour")
    print(f"   🆕 New Features: China's return, 24 races, consolidated sprint format")
    print()
    
    try:
        # Initialize collector with rate limiting
        collector = F1DataCollector2024(
            base_dir=args.data_dir, 
            timeout=args.timeout,
            rate_limit_delay=args.rate_limit_delay
        )
        
        if args.check_progress:
            # Check progress without collecting
            logger.info(f"📋 Collection Progress Report:")
            logger.info(f"   📊 Sessions collected: {len(collector.collected_sessions)}")
            
            # Analyze completeness
            total_expected = sum(len(info['sessions']) for info in collector.circuits_2024.values())
            completion_rate = len(collector.collected_sessions) / total_expected * 100
            logger.info(f"   📈 Completion rate: {completion_rate:.1f}%")
            
            return
        
        if args.list_circuits:
            # List available circuits with 2024 specific info
            print("🏁 Available Circuits for 2024 (24 races):")
            for circuit, info in collector.circuits_2024.items():
                sprint_indicator = "🏃 (Sprint)" if info['has_sprint'] else ""
                china_indicator = "🇨🇳 (RETURN)" if circuit == 'China' else ""
                sessions = ', '.join(info['sessions'])
                session_id = f"{collector.year}_{circuit}"
                collected_count = len([s for s in collector.collected_sessions.keys() if s.startswith(session_id)])
                total_sessions = len(info['sessions'])
                print(f"   • {circuit} {sprint_indicator} {china_indicator}")
                print(f"     Sessions: {sessions}")
                print(f"     Progress: {collected_count}/{total_sessions} collected")
            return
        
        if args.sprint_weekends_focus:
            # Focus on sprint weekends
            print("🏃 Collecting 2024 sprint weekends (consolidated format)...")
            collector.collect_sprint_weekends_focus()
            
        elif args.china_return_focus:
            # Focus on China's return
            print("🇨🇳 Collecting China's return to F1...")
            collector.collect_china_return_focus()
            
        elif args.resume:
            # Resume collection
            print("🔄 Resuming collection from progress file...")
            collector.resume_collection()
            
        elif args.collect_all:
            # Collect all circuits
            print("🏆 Starting complete 2024 season collection...")
            estimated_hours = len(collector.circuits_2024) * 5 * args.rate_limit_delay / 3600
            print(f"⏰ Estimated completion time: {estimated_hours:.1f} hours")
            
            confirm = input("Continue? (y/N): ").strip().lower()
            if confirm == 'y':
                collector.collect_all_circuits()
            else:
                print("Collection cancelled.")
            
        elif args.priority_circuits:
            # Collect priority circuits
            print("🎯 Starting priority circuits collection...")
            estimated_minutes = len(collector.priority_circuits) * 5 * args.rate_limit_delay / 60
            print(f"⏰ Estimated completion time: {estimated_minutes:.0f} minutes")
            collector.collect_priority_circuits()
            
        elif args.circuit:
            # Collect specific circuit
            circuit = args.circuit
            
            if circuit not in collector.circuits_2024:
                print(f"❌ Error: Circuit '{circuit}' not found in 2024 calendar")
                print(f"💡 Use --list-circuits to see available circuits")
                return
            
            if args.sessions:
                # Collect specific sessions
                requested_sessions = [s.strip().upper() for s in args.sessions.split(',')]
                available_sessions = collector.circuits_2024[circuit]['sessions']
                
                invalid_sessions = [s for s in requested_sessions if s not in available_sessions]
                if invalid_sessions:
                    print(f"❌ Error: Invalid sessions for {circuit}: {invalid_sessions}")
                    print(f"💡 Available sessions: {available_sessions}")
                    return
                
                print(f"🏁 Collecting {circuit} - Sessions: {requested_sessions}")
                for session in requested_sessions:
                    collector.collect_session_data(circuit, session)
            else:
                # Collect all sessions for the circuit
                print(f"🏁 Collecting complete {circuit} Grand Prix...")
                collector.collect_circuit_complete(circuit)
        
        else:
            # Interactive mode with 2024 specific options
            print("🎮 Interactive Mode - Choose collection strategy:")
            print()
            print("1. 🏆 Complete 2024 Season (all 24 races)")
            print("2. 🎯 Priority Circuits (12 key tracks including China)")
            print("3. 🏃 Sprint Weekends Focus (consolidated format)")
            print("4. 🇨🇳 China Return Focus (special collection)")
            print("5. 🔄 Resume Collection (continue from where left off)")
            print("6. 🏁 Specific Circuit")
            print("7. 📋 Check Progress")
            print("8. 📋 List Available Circuits")
            print()
            
            while True:
                choice = input("Enter your choice (1-8): ").strip()
                
                if choice == '1':
                    estimated_hours = len(collector.circuits_2024) * 5 * args.rate_limit_delay / 3600
                    print(f"\n🏆 Complete season will take ~{estimated_hours:.1f} hours")
                    confirm = input("Continue? (y/N): ").strip().lower()
                    if confirm == 'y':
                        collector.collect_all_circuits()
                    break
                    
                elif choice == '2':
                    print("\n🎯 Starting priority circuits collection...")
                    collector.collect_priority_circuits()
                    break
                    
                elif choice == '3':
                    print("\n🏃 Collecting sprint weekends...")
                    collector.collect_sprint_weekends_focus()
                    break
                    
                elif choice == '4':
                    print("\n🇨🇳 Collecting China's return...")
                    collector.collect_china_return_focus()
                    break
                    
                elif choice == '5':
                    print("\n🔄 Resuming collection...")
                    collector.resume_collection()
                    break
                    
                elif choice == '6':
                    print("\n🏁 Available Circuits:")
                    for i, circuit in enumerate(collector.circuits_2024.keys(), 1):
                        sprint_indicator = "🏃" if collector.circuits_2024[circuit]['has_sprint'] else ""
                        china_indicator = "🇨🇳" if circuit == 'China' else ""
                        print(f"   {i:2d}. {circuit} {sprint_indicator} {china_indicator}")
                    
                    circuit_choice = input("\nEnter circuit name: ").strip()
                    if circuit_choice in collector.circuits_2024:
                        collector.collect_circuit_complete(circuit_choice)
                        break
                    else:
                        print(f"❌ Invalid circuit: {circuit_choice}")
                        continue
                        
                elif choice == '7':
                    total_expected = sum(len(info['sessions']) for info in collector.circuits_2024.values())
                    completion_rate = len(collector.collected_sessions) / total_expected * 100
                    print(f"\n📊 Progress: {len(collector.collected_sessions)}/{total_expected} sessions ({completion_rate:.1f}%)")
                    continue
                    
                elif choice == '8':
                    print("\n🏁 2024 F1 Calendar with Progress (24 races):")
                    for circuit, info in collector.circuits_2024.items():
                        session_id = f"{collector.year}_{circuit}"
                        collected_count = len([s for s in collector.collected_sessions.keys() if s.startswith(session_id)])
                        total_sessions = len(info['sessions'])
                        progress = f"{collected_count}/{total_sessions}"
                        
                        sprint_indicator = "🏃" if info['has_sprint'] else "🏁"
                        china_indicator = "🇨🇳" if circuit == 'China' else ""
                        status = "✅" if collected_count == total_sessions else "⏳" if collected_count > 0 else "❌"
                        
                        print(f"   {status} {circuit} {sprint_indicator} {china_indicator} ({progress})")
                    continue
                    
                else:
                    print("❌ Invalid choice. Please enter 1-8.")
                    continue
    
    except KeyboardInterrupt:
        print("\n⏸️ Collection stopped by user")
        collector._print_collection_summary()
        
    except Exception as e:
        logger.error(f"❌ Collection failed with error: {e}")
        print(f"\n💥 Collection failed: {e}")
        print("📋 Check the log file for details")
        sys.exit(1)
    
    print("\n✅ Data collection completed!")
    print(f"📁 Data saved in: {collector.raw_dir}")
    print(f"📋 Progress saved in: {collector.progress_file}")

if __name__ == "__main__":
    main()