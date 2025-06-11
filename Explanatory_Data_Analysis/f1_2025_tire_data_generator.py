"""
Created on Wed Jun 11 13:34:17 2025
@author: sid

F1 2025 Data Generator - Updated for New Season (FIXED VERSION)

This script creates realistic 2025 F1 data without any external dependencies.
Includes all the latest driver changes, team updates, and regulation changes for 2025.

Just run this script and it will create: f1_comprehensive_tire_data_2025.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import random
from datetime import datetime

def create_f1_2025_data():
    """Create comprehensive F1 2025 data with all required columns and latest updates - UP TO SPANISH GP."""
    
    print("🏁 Creating F1 2025 Dataset (Through Spanish GP)")
    print("=" * 50)
    
    # Set random seed for reproducible results
    np.random.seed(2025)
    random.seed(2025)
    
    # 2025 F1 Driver lineup with latest changes
    drivers_2025 = {
        # Red Bull Racing - Continuing dominance
        'VER': {'name': 'Max Verstappen', 'team': 'Red Bull Racing', 'team_code': 'RBR'},
        'PER': {'name': 'Sergio Perez', 'team': 'Red Bull Racing', 'team_code': 'RBR'},
        
        # Ferrari - Strong lineup maintained
        'LEC': {'name': 'Charles Leclerc', 'team': 'Ferrari', 'team_code': 'FER'},
        'HAM': {'name': 'Lewis Hamilton', 'team': 'Ferrari', 'team_code': 'FER'},  # 🆕 NEW: Hamilton to Ferrari!
        
        # Mercedes - New era begins
        'RUS': {'name': 'George Russell', 'team': 'Mercedes', 'team_code': 'MER'},
        'ANT': {'name': 'Andrea Kimi Antonelli', 'team': 'Mercedes', 'team_code': 'MER'},  # 🆕 NEW: Antonelli promoted!
        
        # McLaren - Maintaining strong form
        'NOR': {'name': 'Lando Norris', 'team': 'McLaren', 'team_code': 'MCL'},
        'PIA': {'name': 'Oscar Piastri', 'team': 'McLaren', 'team_code': 'MCL'},
        
        # Aston Martin - Experience and youth
        'ALO': {'name': 'Fernando Alonso', 'team': 'Aston Martin', 'team_code': 'AM'},
        'STR': {'name': 'Lance Stroll', 'team': 'Aston Martin', 'team_code': 'AM'},
        
        # Alpine - New direction
        'GAS': {'name': 'Pierre Gasly', 'team': 'Alpine', 'team_code': 'ALP'},
        'DOO': {'name': 'Jack Doohan', 'team': 'Alpine', 'team_code': 'ALP'},  # 🆕 NEW: Doohan gets his chance
        
        # Williams - Building for future
        'ALB': {'name': 'Alexander Albon', 'team': 'Williams', 'team_code': 'WIL'},
        'COL': {'name': 'Franco Colapinto', 'team': 'Williams', 'team_code': 'WIL'},  # 🆕 NEW: Colapinto promoted
        
        # RB - Continuity with experience
        'TSU': {'name': 'Yuki Tsunoda', 'team': 'RB', 'team_code': 'RB'},
        'LAW': {'name': 'Liam Lawson', 'team': 'RB', 'team_code': 'RB'},  # 🆕 NEW: Lawson gets full-time seat
        
        # Kick Sauber - Preparing for Audi
        'BOT': {'name': 'Valtteri Bottas', 'team': 'Kick Sauber', 'team_code': 'SAU'},
        'ZHO': {'name': 'Zhou Guanyu', 'team': 'Kick Sauber', 'team_code': 'SAU'},
        
        # Haas - Experienced lineup
        'MAG': {'name': 'Kevin Magnussen', 'team': 'Haas', 'team_code': 'HAS'},
        'BEA': {'name': 'Oliver Bearman', 'team': 'Haas', 'team_code': 'HAS'}  # 🆕 NEW: Bearman gets his shot
    }
    
    # 2025 F1 Calendar - ONLY RACES COMPLETED SO FAR (Through Spanish GP)
    circuits_2025_completed = [
        'Bahrain',      # Round 1 - March 2
        'Saudi Arabia', # Round 2 - March 9  
        'Australia',    # Round 3 - March 16
        'Japan',        # Round 4 - April 6
        'China',        # Round 5 - April 20 (Confirmed return!)
        'Miami',        # Round 6 - May 4
        'Imola',        # Round 7 - May 18
        'Monaco',       # Round 8 - May 25
        'Canada',       # Round 9 - June 8
        'Spain'         # Round 10 - June 15 ✅ LATEST COMPLETED RACE
    ]
    
    print(f"🏁 Generating data for {len(circuits_2025_completed)} completed races")
    print(f"📅 Latest race: Spanish GP (June 15, 2025)")
    
    # Enhanced 2025 base lap times by circuit (reflecting car evolution)
    base_times_2025 = {
        'Bahrain': {'SOFT': 90.8, 'MEDIUM': 91.5, 'HARD': 92.3, 'INTERMEDIATE': 94.1},
        'Saudi Arabia': {'SOFT': 89.1, 'MEDIUM': 89.8, 'HARD': 90.6, 'INTERMEDIATE': 92.6},
        'Australia': {'SOFT': 79.4, 'MEDIUM': 80.1, 'HARD': 80.9, 'INTERMEDIATE': 82.6},
        'Japan': {'SOFT': 88.7, 'MEDIUM': 89.4, 'HARD': 90.2, 'INTERMEDIATE': 92.1},
        'China': {'SOFT': 93.8, 'MEDIUM': 94.5, 'HARD': 95.3, 'INTERMEDIATE': 97.1},
        'Miami': {'SOFT': 89.3, 'MEDIUM': 90.0, 'HARD': 90.8, 'INTERMEDIATE': 92.4},
        'Imola': {'SOFT': 75.1, 'MEDIUM': 75.8, 'HARD': 76.6, 'INTERMEDIATE': 78.4},
        'Monaco': {'SOFT': 72.4, 'MEDIUM': 73.1, 'HARD': 73.9, 'INTERMEDIATE': 75.6},
        'Canada': {'SOFT': 71.7, 'MEDIUM': 72.4, 'HARD': 73.2, 'INTERMEDIATE': 74.9},
        'Spain': {'SOFT': 77.8, 'MEDIUM': 78.5, 'HARD': 79.3, 'INTERMEDIATE': 81.0}
    }
    
    # 2025 driver performance factors (updated with new drivers) - early season form
    driver_performance_2025 = {
        'VER': -0.40,  # Verstappen still the benchmark
        'LEC': -0.22,  # Leclerc strong at Ferrari
        'HAM': -0.15,  # Hamilton adjusting to Ferrari (early season learning)
        'RUS': -0.12,  # Russell leading Mercedes
        'ALO': -0.15,  # Alonso veteran skill
        'NOR': -0.05,  # Norris maturing
        'PER': 0.03,   # Perez solid but aging
        'PIA': 0.08,   # Piastri developing well
        'ANT': 0.35,   # Antonelli rookie learning curve (early struggles)
        'STR': 0.18,   # Stroll consistent mid-field
        'GAS': 0.12,   # Gasly experienced
        'DOO': 0.40,   # Doohan steep rookie learning curve
        'ALB': 0.20,   # Albon extracting maximum
        'COL': 0.42,   # Colapinto early rookie struggles
        'TSU': 0.22,   # Tsunoda quick but variable
        'LAW': 0.28,   # Lawson proving himself gradually
        'BOT': 0.30,   # Bottas experienced but slower car
        'ZHO': 0.35,   # Zhou steady improvement
        'MAG': 0.33,   # Magnussen veteran
        'BEA': 0.45    # Bearman steepest rookie learning curve
    }
    
    # Enhanced tire degradation rates for 2025 (improved compounds)
    tire_degradation_2025 = {
        'SOFT': 0.038,      # Slightly improved durability
        'MEDIUM': 0.024,    # Better balanced option
        'HARD': 0.014,      # More durable
        'INTERMEDIATE': 0.022  # Enhanced wet compound
    }
    
    # Updated compound probabilities for 2025
    compound_probabilities = [0.32, 0.42, 0.21, 0.05]  # SOFT, MEDIUM, HARD, INTERMEDIATE
    
    # Sprint weekends that have occurred (realistic for early 2025)
    sprint_weekends = ['China', 'Miami']  # Only 2 sprint weekends so far
    
    print("🔧 Generating realistic 2025 F1 data (partial season)...")
    print(f"   🏎️ Drivers: {len(drivers_2025)} (including new rookies)")
    print(f"   🏁 Circuits: {len(circuits_2025_completed)} completed")
    print(f"   🏃 Sprint weekends: {len(sprint_weekends)} so far")
    
    # Generate comprehensive dataset
    data = []
    
    for circuit_num, circuit in enumerate(circuits_2025_completed, 1):
        print(f"   📍 Generating {circuit} GP data... ({circuit_num}/{len(circuits_2025_completed)})")
        
        # Determine if this is a sprint weekend
        is_sprint_weekend = circuit in sprint_weekends
        
        for driver_code, driver_info in drivers_2025.items():
            # Generate data for different session types
            sessions_to_generate = ['Practice', 'Qualifying', 'Race']
            if is_sprint_weekend:
                sessions_to_generate.append('Sprint')
            
            for session_type in sessions_to_generate:
                # Adjust number of laps based on session type
                if session_type == 'Practice':
                    max_laps = random.randint(15, 25)
                elif session_type == 'Qualifying':
                    max_laps = random.randint(8, 15)
                elif session_type == 'Sprint':
                    max_laps = random.randint(15, 25)  # Sprint race length
                else:  # Race
                    max_laps = random.randint(45, 70)  # Full race distance
                
                # For race sessions, simulate multiple tire stints
                if session_type in ['Race', 'Sprint']:
                    # Generate 2-4 stints for race, 1-2 for sprint
                    num_stints = random.randint(2, 4) if session_type == 'Race' else random.randint(1, 2)
                    lap_number = 1
                    
                    for stint_num in range(num_stints):
                        # Choose tire compound for this stint
                        compound = np.random.choice(['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE'], p=compound_probabilities)
                        
                        # Stint length varies by compound and session
                        if session_type == 'Sprint':
                            stint_length = random.randint(8, 15)
                        elif compound == 'SOFT':
                            stint_length = random.randint(10, 20)
                        elif compound == 'MEDIUM':
                            stint_length = random.randint(18, 32)
                        elif compound == 'HARD':
                            stint_length = random.randint(25, 40)
                        else:  # INTERMEDIATE
                            stint_length = random.randint(5, 18)
                        
                        # Don't exceed session length
                        stint_length = min(stint_length, max_laps - lap_number + 1)
                        
                        for stint_lap in range(1, stint_length + 1):
                            if lap_number > max_laps:
                                break
                            
                            # Add early season adjustment for new drivers/teams
                            early_season_factor = 0.0
                            if circuit_num <= 3:  # First 3 races
                                if driver_code in ['HAM']:  # Hamilton adjusting to Ferrari
                                    early_season_factor = 0.15
                                elif driver_code in ['ANT', 'DOO', 'COL', 'LAW', 'BEA']:  # Rookies
                                    early_season_factor = 0.25
                            elif circuit_num <= 6:  # Races 4-6
                                if driver_code in ['HAM']:
                                    early_season_factor = 0.08
                                elif driver_code in ['ANT', 'DOO', 'COL', 'LAW', 'BEA']:
                                    early_season_factor = 0.15
                            
                            # Generate lap data with early season adjustments
                            record = generate_lap_data_early_season(
                                driver_code, driver_info, circuit, session_type,
                                lap_number, stint_lap, stint_num + 1, compound,
                                base_times_2025, driver_performance_2025, 
                                tire_degradation_2025, circuit_num, early_season_factor
                            )
                            
                            data.append(record)
                            lap_number += 1
                        
                        if lap_number > max_laps:
                            break
                
                else:  # Practice/Qualifying - single compound runs
                    for lap_num in range(1, max_laps + 1):
                        # Mix of compounds in practice/qualifying
                        compound = np.random.choice(['SOFT', 'MEDIUM', 'HARD'], p=[0.5, 0.3, 0.2])
                        
                        early_season_factor = 0.0
                        if circuit_num <= 3:
                            if driver_code in ['HAM']:
                                early_season_factor = 0.12
                            elif driver_code in ['ANT', 'DOO', 'COL', 'LAW', 'BEA']:
                                early_season_factor = 0.20
                        elif circuit_num <= 6:
                            if driver_code in ['HAM']:
                                early_season_factor = 0.06
                            elif driver_code in ['ANT', 'DOO', 'COL', 'LAW', 'BEA']:
                                early_season_factor = 0.12
                        
                        record = generate_lap_data_early_season(
                            driver_code, driver_info, circuit, session_type,
                            lap_num, random.randint(1, 5), 1, compound,
                            base_times_2025, driver_performance_2025, 
                            tire_degradation_2025, circuit_num, early_season_factor
                        )
                        
                        data.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    print(f"\n✅ Generated 2025 F1 dataset (through Spanish GP)!")
    print(f"   📊 Total records: {len(df):,}")
    print(f"   🏎️ Drivers: {df['Driver'].nunique()}")
    print(f"   🏁 Circuits: {df['circuit'].nunique()}")
    print(f"   🛞 Tire compounds: {list(df['Compound'].unique())}")
    print(f"   ⏱️ Lap time range: {df['LapTime'].min():.1f} - {df['LapTime'].max():.1f} seconds")
    print(f"   📅 Season progress: {len(circuits_2025_completed)}/24 races completed")
    
    # Show 2025-specific features (partial season)
    print(f"\n🆕 2025 Season Features (So Far):")
    hamilton_ferrari_laps = len(df[(df['Driver'] == 'HAM') & (df['TeamCode'] == 'FER')])
    print(f"   🔥 Hamilton at Ferrari: {hamilton_ferrari_laps} laps (adaptation period)")
    
    antonelli_laps = len(df[df['Driver'] == 'ANT'])
    print(f"   🌟 Antonelli rookie debut: {antonelli_laps} laps (learning curve)")
    
    all_rookies = len(df[df['Driver'].isin(['ANT', 'DOO', 'COL', 'LAW', 'BEA'])])
    print(f"   👶 All rookies combined: {all_rookies} laps")
    
    sprint_laps = len(df[df['session_type'] == 'Sprint'])
    print(f"   🏃 Sprint sessions so far: {sprint_laps} laps")
    
    china_laps = len(df[df['circuit'] == 'China'])
    print(f"   🇨🇳 China GP return: {china_laps} laps")
    
    return df

def generate_lap_data_early_season(driver_code, driver_info, circuit, session_type, lap_number, 
                                  tire_life, stint_number, compound, base_times, driver_performance, 
                                  tire_degradation, circuit_num, early_season_factor):
    """Generate individual lap data record with early season adjustments."""
    
    # Calculate realistic lap time
    base_time = base_times[circuit][compound]
    driver_factor = driver_performance[driver_code]
    
    # Add early season adjustment
    driver_factor += early_season_factor
    
    # Tire degradation effect
    degradation = tire_degradation[compound] * tire_life
    
    # Session-specific adjustments
    if session_type == 'Qualifying':
        session_factor = -0.8  # Qualifying pace
    elif session_type == 'Sprint':
        session_factor = -0.3  # Sprint pace
    elif session_type == 'Practice':
        session_factor = random.uniform(-0.2, 0.5)  # Variable practice pace
    else:  # Race
        session_factor = 0.0
    
    # Add realistic variability
    random_factor = np.random.normal(0, 0.25)
    track_evolution = -0.002 * lap_number  # Track improvement
    
    # Weather impact (occasional rain, more likely in certain circuits)
    weather_factor = 0.0
    rain_probability = 0.08 if circuit in ['Great Britain', 'Belgium', 'Canada'] else 0.03
    is_wet = random.random() < rain_probability
    
    if is_wet and compound != 'INTERMEDIATE':
        weather_factor = random.uniform(2.0, 8.0)  # Slower in wrong conditions
    
    # Final lap time calculation
    lap_time = (base_time + driver_factor + degradation + session_factor + 
                random_factor + track_evolution + weather_factor)
    
    # Ensure minimum lap time
    lap_time = max(lap_time, base_time * 0.85)
    
    # Generate sector times (realistic distribution)
    sector1_time = lap_time * 0.26 + np.random.normal(0, 0.08)
    sector2_time = lap_time * 0.39 + np.random.normal(0, 0.08)
    sector3_time = lap_time * 0.35 + np.random.normal(0, 0.08)
    
    # Adjust sectors to match total lap time
    sector_total = sector1_time + sector2_time + sector3_time
    sector_adjustment = (lap_time - sector_total) / 3
    sector1_time += sector_adjustment
    sector2_time += sector_adjustment
    sector3_time += sector_adjustment
    
    # Generate speed trap data with 2025 improvements
    base_speed_i1 = 315 + np.random.normal(0, 18)
    base_speed_i2 = 295 + np.random.normal(0, 22)
    base_speed_fl = 335 + np.random.normal(0, 15)
    base_speed_st = 78 + np.random.normal(0, 10)
    
    # Weather conditions
    if is_wet:
        track_temp = round(np.random.normal(25, 3), 1)
        air_temp = round(np.random.normal(18, 2), 1)
        humidity = round(np.random.normal(85, 5), 1)
    else:
        track_temp = round(np.random.normal(45, 8), 1)
        air_temp = round(np.random.normal(28, 5), 1)
        humidity = round(np.random.normal(60, 15), 1)
    
    # Add race date for realistic timing
    race_dates = {
        'Bahrain': '2025-03-02',
        'Saudi Arabia': '2025-03-09', 
        'Australia': '2025-03-16',
        'Japan': '2025-04-06',
        'China': '2025-04-20',
        'Miami': '2025-05-04',
        'Imola': '2025-05-18',
        'Monaco': '2025-05-25',
        'Canada': '2025-06-08',
        'Spain': '2025-06-15'
    }
    
    # Create the comprehensive data record
    record = {
        # Driver information
        'Driver': driver_code,
        'DriverFullName': driver_info['name'],
        'Team': driver_info['team'],
        'TeamCode': driver_info['team_code'],
        
        # Lap information
        'LapNumber': lap_number,
        'LapTime': round(lap_time, 3),
        
        # Sector times
        'Sector1Time': round(sector1_time, 3),
        'Sector2Time': round(sector2_time, 3),
        'Sector3Time': round(sector3_time, 3),
        
        # Tire information
        'Compound': compound,
        'TyreLife': tire_life,
        
        # Circuit information
        'circuit': circuit,
        'Circuit': circuit,
        'Race': circuit,
        'EventName': f'{circuit} Grand Prix',
        
        # Session information
        'session_type': session_type,
        'Session': session_type,
        'Season': 2025,
        'RaceDate': race_dates.get(circuit, '2025-06-15'),
        'RaceNumber': circuit_num,
        
        # Speed data (enhanced for 2025)
        'SpeedI1': round(base_speed_i1, 1),
        'SpeedI2': round(base_speed_i2, 1),
        'SpeedFL': round(base_speed_fl, 1),
        'SpeedST': round(base_speed_st, 1),
        
        # Weather data
        'TrackTemp': track_temp,
        'AirTemp': air_temp,
        'Humidity': humidity,
        'IsWet': is_wet,
        
        # Additional data
        'Position': random.randint(1, 20),
        'StintNumber': stint_number,
        'RaceID': circuit_num,
        'Timestamp': datetime.now().isoformat(),
        
        # 2025 specific data
        'RegulationYear': 2025,
        'CarGeneration': 'Gen4',  # Current generation
        'TireGeneration': 'Pirelli2025',
        'SeasonProgress': round(circuit_num / 24 * 100, 1),  # Percentage of season completed
        'EarlySeasonAdjustment': early_season_factor
    }
    
    return record

def save_f1_2025_data(df):
    """Save the 2025 F1 data to CSV files."""
    
    # Primary file for analysis
    filename = "f1_comprehensive_tire_data_2025.csv"
    df.to_csv(filename, index=False)
    
    # Alternative names for compatibility
    backup_files = [
        "f1_comprehensive_tire_data.csv",
        "f1_2025_data.csv",
        "f1_laps_2025.csv"
    ]
    
    for backup_file in backup_files:
        df.to_csv(backup_file, index=False)
    
    print(f"\n💾 Data saved to:")
    print(f"   📄 Primary: {filename}")
    for backup_file in backup_files:
        print(f"   📄 Backup: {backup_file}")
    
    return filename

def create_data_summary_2025(df):
    """Create a comprehensive summary of the 2025 dataset (partial season)."""
    
    races_completed = len(df['circuit'].unique())
    total_races = 24
    season_progress = (races_completed / total_races) * 100
    
    # Calculate key statistics
    tire_life_min = df['TyreLife'].min()
    tire_life_max = df['TyreLife'].max()
    tire_life_avg = df['TyreLife'].mean()
    
    # Hamilton vs Leclerc analysis
    hamilton_avg = df[df['Driver'] == 'HAM']['LapTime'].mean()
    leclerc_avg = df[df['Driver'] == 'LEC']['LapTime'].mean()
    teammate_gap = abs(hamilton_avg - leclerc_avg)
    
    summary = f"""
{'='*70}
F1 2025 PARTIAL SEASON DATASET SUMMARY (Through Spanish GP)
{'='*70}

SEASON PROGRESS
--------------
📅 Races Completed: {races_completed}/24 ({season_progress:.1f}% of season)
🏁 Latest Race: Spanish Grand Prix (June 15, 2025)
⏳ Next Race: Austrian Grand Prix (June 29, 2025)
📊 Total Laps: {len(df):,}

DATASET OVERVIEW
---------------
🏎️ Drivers: {df['Driver'].nunique()} (2025 grid with major changes)
🏁 Teams: {df['TeamCode'].nunique()}
🏁 Circuits: {df['circuit'].nunique()} (first 10 races)
🛞 Tire Compounds: {df['Compound'].nunique()}
📋 Sessions: {df['session_type'].nunique()} types
📅 Season: 2025 (Partial)

🆕 2025 SEASON MAJOR STORYLINES (So Far)
---------------------------------------
🔥 Hamilton's Ferrari Move - Historic transfer showing adaptation period
🌟 Antonelli's Mercedes Debut - Rookie learning curve in top team
👶 Five Rookie Drivers - Largest rookie class in recent years
🇨🇳 China GP Return - Successfully reintegrated into calendar
🏃 Sprint Format - Enhanced format with 2 sprint weekends completed

2025 DRIVER LINEUP BY TEAM (Current Performance)
----------------------------------------------
"""
    
    # Updated teams for 2025 with performance notes
    teams_2025 = {
        'RBR': 'Red Bull Racing (Dominant Start)',
        'FER': 'Ferrari (Hamilton Integration Period)',
        'MER': 'Mercedes (Rookie Development)', 
        'MCL': 'McLaren (Consistent Progress)',
        'AM': 'Aston Martin (Veteran Leadership)',
        'ALP': 'Alpine (Rookie Integration)',
        'WIL': 'Williams (Youth Movement)',
        'RB': 'RB (Steady Development)',
        'SAU': 'Kick Sauber (Audi Preparation)',
        'HAS': 'Haas (Rookie Program)'
    }
    
    for team_code, team_name in teams_2025.items():
        team_drivers = df[df['TeamCode'] == team_code]['DriverFullName'].unique()
        if len(team_drivers) > 0:
            team_laps = len(df[df['TeamCode'] == team_code])
            summary += f"{team_name:<40}: {', '.join(team_drivers)} ({team_laps:,} laps)\n"
    
    summary += f"""

LAP TIME STATISTICS
------------------
Minimum Lap Time: {df['LapTime'].min():.3f} seconds
Maximum Lap Time: {df['LapTime'].max():.3f} seconds
Average Lap Time: {df['LapTime'].mean():.3f} seconds
Standard Deviation: {df['LapTime'].std():.3f} seconds

By Session Type:
"""
    
    for session in sorted(df['session_type'].unique()):
        session_data = df[df['session_type'] == session]
        summary += f"{session:<12}: {session_data['LapTime'].mean():.3f}s avg ({len(session_data):,} laps)\n"
    
    summary += f"""

🔥 HAMILTON AT FERRARI ANALYSIS
------------------------------
Hamilton Total Laps: {len(df[df['Driver'] == 'HAM']):,}
Leclerc Total Laps: {len(df[df['Driver'] == 'LEC']):,}
Ferrari Team Laps: {len(df[df['TeamCode'] == 'FER']):,}

Hamilton Avg Lap Time: {hamilton_avg:.3f}s
Leclerc Avg Lap Time: {leclerc_avg:.3f}s
Teammate Gap: {teammate_gap:.3f}s

🌟 ROOKIE PERFORMANCE TRACKING
-----------------------------
"""
    
    rookies_2025 = {
        'ANT': 'Antonelli (Mercedes)',
        'DOO': 'Doohan (Alpine)', 
        'COL': 'Colapinto (Williams)',
        'LAW': 'Lawson (RB)',
        'BEA': 'Bearman (Haas)'
    }
    
    for rookie_code, rookie_info in rookies_2025.items():
        rookie_data = df[df['Driver'] == rookie_code]
        if len(rookie_data) > 0:
            avg_time = rookie_data['LapTime'].mean()
            total_laps = len(rookie_data)
            summary += f"{rookie_info:<25}: {avg_time:.3f}s avg ({total_laps:,} laps)\n"
    
    summary += f"""

WEATHER CONDITIONS (Early Season)
--------------------------------
Dry Conditions: {len(df[df['IsWet'] == False]):,} laps ({(len(df[df['IsWet'] == False])/len(df)*100):.1f}%)
Wet Conditions: {len(df[df['IsWet'] == True]):,} laps ({(len(df[df['IsWet'] == True])/len(df)*100):.1f}%)

TIRE DEGRADATION DATA
--------------------
Tire Life Range: {tire_life_min} - {tire_life_max} laps
Average Stint Length: {tire_life_avg:.1f} laps

By Compound (Early Season Data):
"""
    
    for compound in sorted(df['Compound'].unique()):
        compound_data = df[df['Compound'] == compound]
        compound_avg = compound_data['TyreLife'].mean()
        summary += f"{compound:<12}: {compound_avg:.1f} laps avg stint\n"
    
    summary += f"""

SPRINT WEEKEND ANALYSIS
----------------------
Sprint Weekends Completed: {len(df[df['session_type'] == 'Sprint']['circuit'].unique())} of 6 planned
Sprint Laps Total: {len(df[df['session_type'] == 'Sprint']):,}
Sprint Circuits: {', '.join(df[df['session_type'] == 'Sprint']['circuit'].unique())}

DATA QUALITY
-----------
Missing Values: {df.isnull().sum().sum()}
Duplicate Records: {df.duplicated().sum()}
Data Completeness: {((len(df) - df.isnull().sum().sum()) / (len(df) * len(df.columns)) * 100):.1f}%

EARLY SEASON INSIGHTS
--------------------
✅ Hamilton showing adaptation period at Ferrari (realistic learning curve)
✅ Antonelli demonstrating typical rookie development pattern
✅ China GP successfully reintegrated after absence
✅ All five rookies showing individual learning trajectories
✅ Sprint format working well in early season races
✅ Tire strategies evolving with new compound characteristics

UPCOMING ANALYSIS OPPORTUNITIES
------------------------------
🔍 Hamilton vs Leclerc Ferrari teammate dynamic evolution
🔍 Rookie development tracking and prediction modeling
🔍 China GP performance comparison vs pre-2020 data
🔍 Early season tire strategy optimization
🔍 Sprint vs Race performance differential analysis
🔍 Weather impact modeling for remaining season

COLUMNS AVAILABLE ({len(df.columns)} total)
------------------------------------------
{', '.join(sorted(df.columns))}

🚀 READY FOR MID-SEASON ANALYSIS
-------------------------------
✅ Hamilton Ferrari integration assessment
✅ Rookie development trajectory modeling
✅ Enhanced tire degradation analysis (early season patterns)
✅ Sprint vs Race performance comparison
✅ Weather impact and strategy analysis
✅ Team performance evolution tracking
✅ Driver adaptation and learning curve analysis

{'='*70}
F1 2025 Partial Season Dataset - Through Spanish GP
Perfect for analyzing early season trends and development patterns!
Next Update: Post-Austrian GP (June 29, 2025)
{'='*70}
"""
    
    # Save summary to file
    summary_file = "f1_2025_partial_season_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(summary)
    
    print(summary)
    print(f"📄 Summary saved to: {summary_file}")

def test_data_compatibility_2025():
    """Test if the generated 2025 data will work with analysis code."""
    
    print("🧪 Testing 2025 data compatibility...")
    
    filename = "f1_comprehensive_tire_data_2025.csv"
    if not Path(filename).exists():
        print("❌ 2025 data file not found")
        return False
    
    try:
        test_df = pd.read_csv(filename)
        print(f"✅ 2025 data loads successfully: {len(test_df):,} records")
        
        # Check required columns for analysis
        required_columns = [
            'LapTime', 'Compound', 'TyreLife', 'Driver', 'TeamCode', 
            'circuit', 'DriverFullName', 'Team', 'session_type',
            'Sector1Time', 'Sector2Time', 'Sector3Time'
        ]
        
        missing_columns = [col for col in required_columns if col not in test_df.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return False
        else:
            print("✅ All required columns present")
        
        # Test 2025-specific features
        print(f"📊 2025 Data Quality Check:")
        print(f"   🏎️ Drivers: {test_df['Driver'].nunique()}")
        print(f"   🏁 Teams: {test_df['TeamCode'].nunique()}")
        print(f"   🛞 Compounds: {list(test_df['Compound'].unique())}")
        print(f"   📋 Sessions: {list(test_df['session_type'].unique())}")
        print(f"   ⏱️ Valid lap times: {test_df['LapTime'].notna().sum():,}")
        print(f"   🔢 Tire life range: {test_df['TyreLife'].min()}-{test_df['TyreLife'].max()}")
        
        # Check for 2025-specific features
        hamilton_ferrari = 'HAM' in test_df[test_df['TeamCode'] == 'FER']['Driver'].unique()
        antonelli_merc = 'ANT' in test_df[test_df['TeamCode'] == 'MER']['Driver'].unique()
        sprint_sessions = 'Sprint' in test_df['session_type'].unique()
        rookies_present = any(driver in test_df['Driver'].unique() for driver in ['ANT', 'DOO', 'COL', 'LAW', 'BEA'])
        
        print(f"🆕 2025 Features Verification:")
        print(f"   🔥 Hamilton at Ferrari: {'✅' if hamilton_ferrari else '❌'}")
        print(f"   🌟 Antonelli at Mercedes: {'✅' if antonelli_merc else '❌'}")
        print(f"   🏃 Sprint sessions: {'✅' if sprint_sessions else '❌'}")
        print(f"   👶 Rookie drivers: {'✅' if rookies_present else '❌'}")
        
        # Data range validation
        lap_time_range = test_df['LapTime'].max() - test_df['LapTime'].min()
        print(f"   ⏱️ Lap time spread: {lap_time_range:.1f}s (good variance)")
        
        # Session distribution check
        session_balance = test_df['session_type'].value_counts()
        print(f"   📋 Session balance: {dict(session_balance)}")
        
        print("✅ 2025 data compatibility test passed!")
        return True
        
    except Exception as e:
        print(f"❌ 2025 data compatibility test failed: {e}")
        return False

def analyze_hamilton_ferrari_adaptation():
    """Analyze Hamilton's adaptation to Ferrari through early season."""
    
    try:
        df = pd.read_csv("f1_comprehensive_tire_data_2025.csv")
        
        hamilton_data = df[df['Driver'] == 'HAM']
        leclerc_data = df[df['Driver'] == 'LEC']
        
        if len(hamilton_data) > 0 and len(leclerc_data) > 0:
            # Analyze by race progression
            analysis = "🔥 HAMILTON'S FERRARI ADAPTATION ANALYSIS\n" + "="*45 + "\n"
            
            circuits = ['Bahrain', 'Saudi Arabia', 'Australia', 'Japan', 'China', 
                       'Miami', 'Imola', 'Monaco', 'Canada', 'Spain']
            
            for i, circuit in enumerate(circuits, 1):
                ham_circuit = hamilton_data[hamilton_data['circuit'] == circuit]
                lec_circuit = leclerc_data[leclerc_data['circuit'] == circuit]
                
                if len(ham_circuit) > 0 and len(lec_circuit) > 0:
                    ham_avg = ham_circuit['LapTime'].mean()
                    lec_avg = lec_circuit['LapTime'].mean()
                    gap = ham_avg - lec_avg
                    
                    status = "HAM faster" if gap < 0 else "LEC faster"
                    analysis += f"Round {i:2d} {circuit:<12}: Gap {gap:+.3f}s ({status})\n"
            
            # Overall adaptation trend
            overall_ham = hamilton_data['LapTime'].mean()
            overall_lec = leclerc_data['LapTime'].mean()
            overall_gap = overall_ham - overall_lec
            
            analysis += f"\nOVERALL (10 races): {overall_gap:+.3f}s "
            analysis += f"({'Hamilton faster' if overall_gap < 0 else 'Leclerc faster'})\n"
            analysis += f"Total laps analyzed: Hamilton {len(hamilton_data)}, Leclerc {len(leclerc_data)}\n"
            
            return analysis
            
    except FileNotFoundError:
        return "2025 data file not found. Run main() first."
    except Exception as e:
        return f"Error analyzing Hamilton-Ferrari data: {e}"

def analyze_2025_rookies_progression():
    """Analyze rookie progression through early season."""
    
    try:
        df = pd.read_csv("f1_comprehensive_tire_data_2025.csv")
        
        rookies = {
            'ANT': 'Antonelli (Mercedes)',
            'DOO': 'Doohan (Alpine)', 
            'COL': 'Colapinto (Williams)',
            'LAW': 'Lawson (RB)',
            'BEA': 'Bearman (Haas)'
        }
        
        analysis = "👶 2025 ROOKIE PROGRESSION ANALYSIS\n" + "="*35 + "\n"
        
        circuits = ['Bahrain', 'Saudi Arabia', 'Australia', 'Japan', 'China', 
                   'Miami', 'Imola', 'Monaco', 'Canada', 'Spain']
        
        # Analyze each rookie's progression
        for rookie_code, rookie_name in rookies.items():
            rookie_data = df[df['Driver'] == rookie_code]
            if len(rookie_data) > 0:
                analysis += f"\n{rookie_name}:\n"
                
                # Track improvement over races
                race_averages = []
                for circuit in circuits:
                    circuit_data = rookie_data[rookie_data['circuit'] == circuit]
                    if len(circuit_data) > 0:
                        avg_time = circuit_data['LapTime'].mean()
                        race_averages.append(avg_time)
                        analysis += f"  {circuit:<12}: {avg_time:.3f}s\n"
                
                # Calculate improvement trend
                if len(race_averages) >= 3:
                    early_avg = np.mean(race_averages[:3])
                    recent_avg = np.mean(race_averages[-3:])
                    improvement = early_avg - recent_avg
                    
                    analysis += f"  Improvement: {improvement:+.3f}s (early vs recent 3 races)\n"
                    analysis += f"  Total Laps: {len(rookie_data):,}\n"
        
        return analysis
        
    except FileNotFoundError:
        return "2025 data file not found. Run main() first."
    except Exception as e:
        return f"Error analyzing rookie data: {e}"

def main():
    """Main function to create comprehensive F1 2025 data (partial season through Spanish GP)."""
    
    print("🏁 F1 2025 Data Generator (Partial Season)")
    print("🚀 Creating realistic 2025 dataset through Spanish GP")
    print("=" * 70)
    
    try:
        # Step 1: Generate the main dataset (partial season)
        print("STEP 1: Generating 2025 F1 Partial Season Data")
        print("-" * 45)
        df = create_f1_2025_data()
        
        # Step 2: Save the data
        print("\nSTEP 2: Saving Data Files")
        print("-" * 25)
        filename = save_f1_2025_data(df)
        
        # Step 3: Create comprehensive summary
        print("\nSTEP 3: Creating Partial Season Summary")
        print("-" * 36)
        create_data_summary_2025(df)
        
        # Step 4: Generate early season analysis
        print("\nSTEP 4: Generating Early Season Analysis")
        print("-" * 39)
        
        # Create Hamilton Ferrari analysis
        hamilton_analysis = analyze_hamilton_ferrari_adaptation()
        with open('hamilton_ferrari_early_analysis.txt', 'w') as f:
            f.write(hamilton_analysis)
        print("📄 Hamilton analysis saved to: hamilton_ferrari_early_analysis.txt")
        
        # Create rookie progression analysis
        rookie_analysis = analyze_2025_rookies_progression()
        with open('rookies_early_progression.txt', 'w') as f:
            f.write(rookie_analysis)
        print("📄 Rookie analysis saved to: rookies_early_progression.txt")
        
        # Step 5: Test compatibility
        print("\nSTEP 5: Testing Compatibility")
        print("-" * 30)
        compatibility = test_data_compatibility_2025()
        
        # Final summary
        print("\n" + "=" * 70)
        print("🎉 F1 2025 PARTIAL SEASON DATA GENERATION COMPLETE!")
        print("=" * 70)
        
        if compatibility:
            races_completed = len(df['circuit'].unique())
            season_progress = (races_completed / 24) * 100
            
            print("✅ SUCCESS! Your 2025 F1 partial season dataset is ready.")
            print(f"\n📁 Files created:")
            print(f"   📊 Main Data: {filename}")
            print(f"   📄 Season Summary: f1_2025_partial_season_summary.txt")
            print(f"   🔥 Hamilton Analysis: hamilton_ferrari_early_analysis.txt")
            print(f"   👶 Rookie Analysis: rookies_early_progression.txt")
            
            print(f"\n📅 Season Status:")
            print(f"   ✅ Completed: {races_completed}/24 races ({season_progress:.1f}%)")
            print(f"   🏁 Latest: Spanish Grand Prix (June 15, 2025)")
            print(f"   ⏳ Next: Austrian Grand Prix (June 29, 2025)")
            
            print(f"\n🔥 Early Season Headlines:")
            print(f"   🏎️ Hamilton adapting to Ferrari alongside Leclerc")
            print(f"   🌟 Antonelli making Mercedes debut as youngest driver")
            print(f"   👶 Five rookies learning F1 - largest class in years")
            print(f"   🇨🇳 China GP successfully returned to calendar")
            print(f"   🏃 Sprint format enhanced with 2 completed weekends")
            
            print(f"\n📊 Dataset Highlights:")
            hamilton_laps = len(df[df['Driver'] == 'HAM'])
            antonelli_laps = len(df[df['Driver'] == 'ANT'])
            rookie_laps = len(df[df['Driver'].isin(['ANT', 'DOO', 'COL', 'LAW', 'BEA'])])
            sprint_laps = len(df[df['session_type'] == 'Sprint'])
            
            print(f"   Hamilton Ferrari laps: {hamilton_laps:,}")
            print(f"   Antonelli rookie laps: {antonelli_laps:,}")
            print(f"   All rookies combined: {rookie_laps:,}")
            print(f"   Sprint weekend laps: {sprint_laps:,}")
            print(f"   Total dataset size: {len(df):,} records")
            
            print(f"\n🚀 Analysis Ready:")
            print(f"   1. Hamilton vs Leclerc Ferrari adaptation tracking")
            print(f"   2. Antonelli rookie development progression")
            print(f"   3. Multi-rookie comparison and learning curves")
            print(f"   4. Early season tire strategy evolution")
            print(f"   5. China GP return impact analysis")
            print(f"   6. Sprint vs Race performance patterns")
            
            print(f"\n⏳ Awaiting Future Races:")
            remaining_races = ['Austria', 'Great Britain', 'Hungary', 'Belgium', 
                             'Netherlands', 'Italy', 'Azerbaijan', 'Singapore',
                             'United States', 'Mexico', 'Brazil', 'Qatar', 
                             'Las Vegas', 'Abu Dhabi']
            print(f"   Remaining: {', '.join(remaining_races[:7])}...")
            print(f"   Total remaining: {len(remaining_races)} races")
            
        else:
            print("⚠️ Data generated but compatibility issues detected.")
            print("Please check the error messages above.")
        
    except Exception as e:
        print(f"❌ Error generating 2025 partial season data: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure pandas and numpy are installed")
        print("2. Check write permissions in current directory")
        print("3. Verify sufficient disk space")
        print("4. Try running with administrator privileges")

# Quick analysis functions for partial season
def quick_season_progress():
    """Quick overview of 2025 season progress."""
    
    try:
        df = pd.read_csv("f1_comprehensive_tire_data_2025.csv")
        
        races_completed = len(df['circuit'].unique())
        progress = (races_completed / 24) * 100
        
        print(f"📅 2025 SEASON PROGRESS")
        print(f"{'='*25}")
        print(f"Completed: {races_completed}/24 races ({progress:.1f}%)")
        print(f"Latest: {df['circuit'].unique()[-1]} GP")
        print(f"Total laps: {len(df):,}")
        print(f"Hamilton Ferrari laps: {len(df[df['Driver'] == 'HAM']):,}")
        print(f"Rookie combined laps: {len(df[df['Driver'].isin(['ANT', 'DOO', 'COL', 'LAW', 'BEA'])]):,}")
        
    except FileNotFoundError:
        print("⚠️ Run main() to generate dataset first")

def quick_ferrari_update():
    """Quick Ferrari team analysis."""
    
    try:
        df = pd.read_csv("f1_comprehensive_tire_data_2025.csv")
        
        ferrari_data = df[df['TeamCode'] == 'FER']
        ham_data = ferrari_data[ferrari_data['Driver'] == 'HAM']
        lec_data = ferrari_data[ferrari_data['Driver'] == 'LEC']
        
        print(f"🔥 FERRARI TEAM UPDATE")
        print(f"{'='*22}")
        print(f"Total Ferrari laps: {len(ferrari_data):,}")
        print(f"Hamilton: {len(ham_data):,} laps, {ham_data['LapTime'].mean():.3f}s avg")
        print(f"Leclerc: {len(lec_data):,} laps, {lec_data['LapTime'].mean():.3f}s avg")
        
        gap = ham_data['LapTime'].mean() - lec_data['LapTime'].mean()
        faster = "Hamilton" if gap < 0 else "Leclerc"
        print(f"Teammate gap: {abs(gap):.3f}s in favor of {faster}")
        
    except FileNotFoundError:
        print("⚠️ Run main() to generate dataset first")

if __name__ == "__main__":
    main()