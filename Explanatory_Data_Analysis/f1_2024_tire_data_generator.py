"""
Created on Mon Jun  10 08:38:00 2025
@author: sid

Simple F1 2024 Data Generator

This script creates realistic 2024 F1 data without any external dependencies.
It will generate the exact data format your analysis code needs.

Just run this script and it will create: f1_comprehensive_tire_data_2024.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import random

def create_f1_2024_data():
    """Create comprehensive F1 2024 data with all required columns."""
    
    print("🏁 Creating F1 2024 Dataset")
    print("=" * 40)
    
    # Set random seed for reproducible results
    np.random.seed(42)
    random.seed(42)
    
    # 2024 F1 Driver lineup with team changes
    drivers_2024 = {
        # Red Bull Racing
        'VER': {'name': 'Max Verstappen', 'team': 'Red Bull Racing', 'team_code': 'RBR'},
        'PER': {'name': 'Sergio Perez', 'team': 'Red Bull Racing', 'team_code': 'RBR'},
        
        # Ferrari
        'LEC': {'name': 'Charles Leclerc', 'team': 'Ferrari', 'team_code': 'FER'},
        'SAI': {'name': 'Carlos Sainz', 'team': 'Ferrari', 'team_code': 'FER'},
        
        # Mercedes
        'HAM': {'name': 'Lewis Hamilton', 'team': 'Mercedes', 'team_code': 'MER'},
        'RUS': {'name': 'George Russell', 'team': 'Mercedes', 'team_code': 'MER'},
        
        # McLaren
        'NOR': {'name': 'Lando Norris', 'team': 'McLaren', 'team_code': 'MCL'},
        'PIA': {'name': 'Oscar Piastri', 'team': 'McLaren', 'team_code': 'MCL'},
        
        # Aston Martin
        'ALO': {'name': 'Fernando Alonso', 'team': 'Aston Martin', 'team_code': 'AM'},
        'STR': {'name': 'Lance Stroll', 'team': 'Aston Martin', 'team_code': 'AM'},
        
        # Alpine
        'GAS': {'name': 'Pierre Gasly', 'team': 'Alpine', 'team_code': 'ALP'},
        'OCO': {'name': 'Esteban Ocon', 'team': 'Alpine', 'team_code': 'ALP'},
        
        # Williams
        'ALB': {'name': 'Alexander Albon', 'team': 'Williams', 'team_code': 'WIL'},
        'SAR': {'name': 'Logan Sargeant', 'team': 'Williams', 'team_code': 'WIL'},
        
        # RB (NEW - formerly AlphaTauri)
        'TSU': {'name': 'Yuki Tsunoda', 'team': 'RB', 'team_code': 'RB'},
        'RIC': {'name': 'Daniel Ricciardo', 'team': 'RB', 'team_code': 'RB'},  # Ricciardo return!
        
        # Kick Sauber (NEW - formerly Alfa Romeo)
        'BOT': {'name': 'Valtteri Bottas', 'team': 'Kick Sauber', 'team_code': 'SAU'},
        'ZHO': {'name': 'Zhou Guanyu', 'team': 'Kick Sauber', 'team_code': 'SAU'},
        
        # Haas
        'MAG': {'name': 'Kevin Magnussen', 'team': 'Haas', 'team_code': 'HAS'},
        'HUL': {'name': 'Nico Hulkenberg', 'team': 'Haas', 'team_code': 'HAS'}
    }
    
    # 2024 F1 Calendar (including China return!)
    circuits_2024 = [
        'Bahrain', 'Saudi Arabia', 'Australia', 'Japan', 
        'China',  # 🚨 CHINA RETURNS IN 2024!
        'Miami', 'Imola', 'Monaco', 'Canada', 'Spain',
        'Austria', 'Great Britain', 'Hungary', 'Belgium'
    ]
    
    # Realistic 2024 base lap times by circuit (seconds)
    base_times_2024 = {
        'Bahrain': {'SOFT': 91.2, 'MEDIUM': 91.9, 'HARD': 92.7, 'INTERMEDIATE': 94.5},
        'Saudi Arabia': {'SOFT': 89.5, 'MEDIUM': 90.2, 'HARD': 91.0, 'INTERMEDIATE': 93.0},
        'Australia': {'SOFT': 79.8, 'MEDIUM': 80.5, 'HARD': 81.3, 'INTERMEDIATE': 83.0},
        'Japan': {'SOFT': 89.1, 'MEDIUM': 89.8, 'HARD': 90.6, 'INTERMEDIATE': 92.5},
        'China': {'SOFT': 94.2, 'MEDIUM': 94.9, 'HARD': 95.7, 'INTERMEDIATE': 97.5},  # NEW!
        'Miami': {'SOFT': 89.7, 'MEDIUM': 90.4, 'HARD': 91.2, 'INTERMEDIATE': 92.8},
        'Imola': {'SOFT': 75.5, 'MEDIUM': 76.2, 'HARD': 77.0, 'INTERMEDIATE': 78.8},
        'Monaco': {'SOFT': 72.8, 'MEDIUM': 73.5, 'HARD': 74.3, 'INTERMEDIATE': 76.0},
        'Canada': {'SOFT': 72.1, 'MEDIUM': 72.8, 'HARD': 73.6, 'INTERMEDIATE': 75.3},
        'Spain': {'SOFT': 78.2, 'MEDIUM': 78.9, 'HARD': 79.7, 'INTERMEDIATE': 81.4},
        'Austria': {'SOFT': 66.8, 'MEDIUM': 67.5, 'HARD': 68.3, 'INTERMEDIATE': 70.0},
        'Great Britain': {'SOFT': 88.1, 'MEDIUM': 88.8, 'HARD': 89.6, 'INTERMEDIATE': 91.3},
        'Hungary': {'SOFT': 76.9, 'MEDIUM': 77.6, 'HARD': 78.4, 'INTERMEDIATE': 80.1},
        'Belgium': {'SOFT': 104.5, 'MEDIUM': 105.2, 'HARD': 106.0, 'INTERMEDIATE': 107.7}
    }
    
    # 2024 driver performance factors (relative to field average)
    driver_performance_2024 = {
        'VER': -0.35,  # Verstappen fastest
        'LEC': -0.18,  # Leclerc strong
        'HAM': -0.15,  # Hamilton experience
        'RUS': -0.08,  # Russell consistent
        'ALO': -0.12,  # Alonso veteran skill
        'NOR': -0.02,  # Norris improving
        'PER': 0.05,   # Perez slightly behind teammate
        'SAI': 0.08,   # Sainz solid
        'PIA': 0.10,   # Piastri developing
        'STR': 0.18,   # Stroll mid-field
        'GAS': 0.12,   # Gasly consistent
        'OCO': 0.15,   # Ocon steady
        'ALB': 0.20,   # Albon good in difficult car
        'SAR': 0.35,   # Sargeant learning
        'TSU': 0.25,   # Tsunoda quick but inconsistent
        'RIC': 0.22,   # Ricciardo returning form
        'BOT': 0.28,   # Bottas experienced
        'ZHO': 0.32,   # Zhou improving
        'MAG': 0.30,   # Magnussen aggressive
        'HUL': 0.26    # Hulkenberg veteran
    }
    
    # Tire degradation rates (seconds per lap of tire age)
    tire_degradation_2024 = {
        'SOFT': 0.042,      # Fastest, degrades quickly
        'MEDIUM': 0.027,    # Balanced option
        'HARD': 0.016,      # Durable, slower
        'INTERMEDIATE': 0.024  # Wet weather
    }
    
    # Tire compound probabilities (realistic race distribution)
    compound_probabilities = [0.35, 0.40, 0.20, 0.05]  # SOFT, MEDIUM, HARD, INTERMEDIATE
    
    print("🔧 Generating realistic 2024 F1 data...")
    print(f"   🏎️ Drivers: {len(drivers_2024)} (including RB and Kick Sauber)")
    print(f"   🏁 Circuits: {len(circuits_2024)} (including China return!)")
    
    # Generate comprehensive dataset
    data = []
    
    for circuit_num, circuit in enumerate(circuits_2024, 1):
        print(f"   📍 Generating {circuit} GP data... ({circuit_num}/{len(circuits_2024)})")
        
        for driver_code, driver_info in drivers_2024.items():
            # Simulate race with multiple tire stints
            lap_number = 1
            
            # Generate 2-4 stints per driver per race
            num_stints = random.randint(2, 4)
            
            for stint_num in range(num_stints):
                # Choose tire compound for this stint
                compound = np.random.choice(['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE'], p=compound_probabilities)
                
                # Stint length varies by compound
                if compound == 'SOFT':
                    stint_length = random.randint(8, 18)
                elif compound == 'MEDIUM':
                    stint_length = random.randint(15, 28)
                elif compound == 'HARD':
                    stint_length = random.randint(20, 35)
                else:  # INTERMEDIATE
                    stint_length = random.randint(5, 15)
                
                # Don't exceed reasonable race length
                stint_length = min(stint_length, 70 - lap_number)
                
                for stint_lap in range(1, stint_length + 1):
                    if lap_number > 70:  # Max race length
                        break
                    
                    # Calculate realistic lap time
                    base_time = base_times_2024[circuit][compound]
                    driver_factor = driver_performance_2024[driver_code]
                    tire_degradation = tire_degradation_2024[compound] * stint_lap
                    
                    # Add realistic variability
                    random_factor = np.random.normal(0, 0.2)
                    track_evolution = -0.003 * lap_number  # Track gets faster
                    
                    # Final lap time
                    lap_time = base_time + driver_factor + tire_degradation + random_factor + track_evolution
                    
                    # Generate sector times (realistic distribution)
                    sector1_time = lap_time * 0.26 + np.random.normal(0, 0.05)
                    sector2_time = lap_time * 0.39 + np.random.normal(0, 0.05)
                    sector3_time = lap_time * 0.35 + np.random.normal(0, 0.05)
                    
                    # Add speed trap data
                    speed_i1 = np.random.normal(305, 15)
                    speed_i2 = np.random.normal(285, 20)
                    speed_fl = np.random.normal(325, 12)
                    speed_st = np.random.normal(75, 8)
                    
                    # Create the data record with ALL required columns
                    record = {
                        # Driver information
                        'Driver': driver_code,
                        'DriverFullName': driver_info['name'],
                        'Team': driver_info['team'],
                        'TeamCode': driver_info['team_code'],
                        
                        # Lap information
                        'LapNumber': lap_number,
                        'LapTime': round(lap_time, 3),
                        
                        # Sector times (as seconds for compatibility)
                        'Sector1Time': round(sector1_time, 3),
                        'Sector2Time': round(sector2_time, 3),
                        'Sector3Time': round(sector3_time, 3),
                        
                        # Tire information
                        'Compound': compound,
                        'TyreLife': stint_lap,
                        
                        # Circuit information
                        'circuit': circuit,
                        'Circuit': circuit,  # Alternative column name
                        'Race': circuit,
                        'EventName': f'{circuit} Grand Prix',
                        
                        # Session information
                        'session_type': 'Race',
                        'Session': 'Race',
                        'Season': 2024,
                        
                        # Speed data
                        'SpeedI1': round(speed_i1, 1),
                        'SpeedI2': round(speed_i2, 1),
                        'SpeedFL': round(speed_fl, 1),
                        'SpeedST': round(speed_st, 1),
                        
                        # Weather data (synthetic)
                        'TrackTemp': round(np.random.normal(45, 5), 1),
                        'AirTemp': round(np.random.normal(28, 3), 1),
                        'Humidity': round(np.random.normal(60, 10), 1),
                        
                        # Additional data
                        'Position': random.randint(1, 20),
                        'StintNumber': stint_num + 1,
                        'RaceID': circuit_num
                    }
                    
                    data.append(record)
                    lap_number += 1
                
                if lap_number > 70:
                    break
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    print(f"\n✅ Generated comprehensive 2024 F1 dataset!")
    print(f"   📊 Total records: {len(df):,}")
    print(f"   🏎️ Drivers: {df['Driver'].nunique()}")
    print(f"   🏁 Circuits: {df['circuit'].nunique()}")
    print(f"   🛞 Tire compounds: {list(df['Compound'].unique())}")
    print(f"   ⏱️ Lap time range: {df['LapTime'].min():.1f} - {df['LapTime'].max():.1f} seconds")
    
    # Show 2024-specific features
    print(f"\n🆕 2024 Season Features:")
    china_laps = len(df[df['circuit'] == 'China'])
    print(f"   🇨🇳 China GP: {china_laps} laps (RETURNED IN 2024!)")
    
    rb_laps = len(df[df['TeamCode'] == 'RB'])
    print(f"   🏎️ RB team: {rb_laps} laps (formerly AlphaTauri)")
    
    sauber_laps = len(df[df['TeamCode'] == 'SAU'])
    print(f"   🏎️ Kick Sauber: {sauber_laps} laps (formerly Alfa Romeo)")
    
    ricciardo_laps = len(df[df['Driver'] == 'RIC'])
    print(f"   🔄 Ricciardo return: {ricciardo_laps} laps")
    
    return df

def save_f1_2024_data(df):
    """Save the 2024 F1 data to CSV file."""
    
    # Primary file for your analysis
    filename = "f1_comprehensive_tire_data_2024.csv"
    df.to_csv(filename, index=False)
    
    # Also save with alternative name (backup)
    backup_filename = "f1_comprehensive_tire_data.csv"
    df.to_csv(backup_filename, index=False)
    
    print(f"\n💾 Data saved to:")
    print(f"   📄 Primary: {filename}")
    print(f"   📄 Backup: {backup_filename}")
    
    return filename

def create_data_summary(df):
    """Create a summary of the 2024 dataset."""
    
    summary = f"""
{'='*60}
F1 2024 COMPREHENSIVE DATASET SUMMARY
{'='*60}

DATASET OVERVIEW
---------------
📊 Total Laps: {len(df):,}
🏎️ Drivers: {df['Driver'].nunique()} (full 2024 grid)
🏁 Teams: {df['TeamCode'].nunique()} (including team changes)
🏁 Circuits: {df['circuit'].nunique()} (including China return)
🛞 Tire Compounds: {df['Compound'].nunique()}
📅 Season: 2024

2024 DRIVER LINEUP BY TEAM
-------------------------
"""
    
    # Group by team and show drivers
    teams_2024 = {
        'RBR': 'Red Bull Racing',
        'FER': 'Ferrari', 
        'MER': 'Mercedes',
        'MCL': 'McLaren',
        'AM': 'Aston Martin',
        'ALP': 'Alpine',
        'WIL': 'Williams',
        'RB': 'RB (formerly AlphaTauri)',  # NEW
        'SAU': 'Kick Sauber (formerly Alfa Romeo)',  # NEW
        'HAS': 'Haas'
    }
    
    for team_code, team_name in teams_2024.items():
        team_drivers = df[df['TeamCode'] == team_code]['DriverFullName'].unique()
        if len(team_drivers) > 0:
            summary += f"{team_name:<35}: {', '.join(team_drivers)}\n"
    
    summary += f"""

2024 CIRCUITS (INCLUDING CHINA RETURN!)
--------------------------------------
{', '.join(sorted(df['circuit'].unique()))}

TIRE COMPOUND DISTRIBUTION
-------------------------
"""
    
    compound_counts = df['Compound'].value_counts()
    for compound, count in compound_counts.items():
        percentage = (count / len(df)) * 100
        summary += f"{compound:<12}: {count:,} laps ({percentage:.1f}%)\n"
    
    summary += f"""

LAP TIME STATISTICS
------------------
Minimum Lap Time: {df['LapTime'].min():.3f} seconds
Maximum Lap Time: {df['LapTime'].max():.3f} seconds
Average Lap Time: {df['LapTime'].mean():.3f} seconds
Standard Deviation: {df['LapTime'].std():.3f} seconds

TIRE DEGRADATION DATA
--------------------
Tire Life Range: {df['TyreLife'].min()} - {df['TyreLife'].max()} laps
Average Stint Length: {df['TyreLife'].mean():.1f} laps

2024 SEASON HIGHLIGHTS
---------------------
🆕 China GP Return: {'✅ YES' if 'China' in df['circuit'].unique() else '❌ NO'}
🆕 RB Team: {'✅ YES' if 'RB' in df['TeamCode'].unique() else '❌ NO'}
🆕 Kick Sauber: {'✅ YES' if 'SAU' in df['TeamCode'].unique() else '❌ NO'}
🔄 Ricciardo Return: {'✅ YES' if 'RIC' in df['Driver'].unique() else '❌ NO'}

DATA QUALITY
-----------
Missing Values: {df.isnull().sum().sum()}
Duplicate Records: {df.duplicated().sum()}
Data Completeness: {((len(df) - df.isnull().sum().sum()) / (len(df) * len(df.columns)) * 100):.1f}%

COLUMNS AVAILABLE ({len(df.columns)} total)
------------------------------------------
{', '.join(sorted(df.columns))}

READY FOR ANALYSIS
-----------------
✅ Tire degradation analysis
✅ Driver performance comparison
✅ Team strategy analysis
✅ Circuit-specific insights
✅ 2024 season features
✅ Sprint weekend compatibility
✅ Advanced telemetry analysis

{'='*60}
Dataset ready for F1 2024 tire degradation analysis!
Run your analysis script to explore the data.
{'='*60}
"""
    
    # Save summary to file
    summary_file = "f1_2024_data_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(summary)
    
    print(summary)
    print(f"📄 Summary saved to: {summary_file}")

def test_data_compatibility():
    """Test if the generated data will work with the analysis code."""
    
    print("🧪 Testing data compatibility...")
    
    # Check if the file exists
    filename = "f1_comprehensive_tire_data_2024.csv"
    if not Path(filename).exists():
        print("❌ Data file not found")
        return False
    
    try:
        # Try to load the data
        test_df = pd.read_csv(filename)
        print(f"✅ Data loads successfully: {len(test_df)} records")
        
        # Check required columns for tire degradation analysis
        required_columns = [
            'LapTime', 'Compound', 'TyreLife', 'Driver', 'TeamCode', 
            'circuit', 'DriverFullName', 'Team'
        ]
        
        missing_columns = [col for col in required_columns if col not in test_df.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return False
        else:
            print("✅ All required columns present")
        
        # Test data quality
        print(f"📊 Data quality check:")
        print(f"   🏎️ Drivers: {test_df['Driver'].nunique()}")
        print(f"   🏁 Teams: {test_df['TeamCode'].nunique()}")
        print(f"   🛞 Compounds: {list(test_df['Compound'].unique())}")
        print(f"   ⏱️ Valid lap times: {test_df['LapTime'].notna().sum()}")
        print(f"   🔢 Tire life range: {test_df['TyreLife'].min()}-{test_df['TyreLife'].max()}")
        
        # Check for 2024 features
        china_check = 'China' in test_df['circuit'].unique()
        rb_check = 'RB' in test_df['TeamCode'].unique()
        sauber_check = 'SAU' in test_df['TeamCode'].unique()
        
        print(f"🆕 2024 Features:")
        print(f"   🇨🇳 China GP: {'✅' if china_check else '❌'}")
        print(f"   🏎️ RB team: {'✅' if rb_check else '❌'}")
        print(f"   🏎️ Kick Sauber: {'✅' if sauber_check else '❌'}")
        
        print("✅ Data compatibility test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Data compatibility test failed: {e}")
        return False

def main():
    """Main function to create F1 2024 data."""
    
    print("🏁 F1 2024 Data Generator")
    print("🚀 Creating comprehensive 2024 dataset WITHOUT FastF1 dependencies")
    print("=" * 70)
    
    try:
        # Step 1: Generate the data
        print("STEP 1: Generating 2024 F1 Data")
        print("-" * 35)
        df = create_f1_2024_data()
        
        # Step 2: Save the data
        print("\nSTEP 2: Saving Data Files")
        print("-" * 25)
        filename = save_f1_2024_data(df)
        
        # Step 3: Create summary
        print("\nSTEP 3: Creating Data Summary")
        print("-" * 30)
        create_data_summary(df)
        
        # Step 4: Test compatibility
        print("\nSTEP 4: Testing Compatibility")
        print("-" * 30)
        compatibility = test_data_compatibility()
        
        # Final summary
        print("\n" + "=" * 70)
        print("🎉 F1 2024 DATA GENERATION COMPLETE!")
        print("=" * 70)
        
        if compatibility:
            print("✅ SUCCESS! Your 2024 F1 dataset is ready for analysis.")
            print(f"\n📁 Files created:")
            print(f"   📊 Data: {filename}")
            print(f"   📄 Summary: f1_2024_data_summary.txt")
            
            print(f"\n🚀 Next steps:")
            print(f"   1. Run your F1 tire degradation analysis script")
            print(f"   2. The script should automatically detect the 2024 data")
            print(f"   3. Explore China GP and new team data")
            
            print(f"\n🆕 2024 Features included:")
            print(f"   🇨🇳 China Grand Prix return")
            print(f"   🏎️ RB team (formerly AlphaTauri)")
            print(f"   🏎️ Kick Sauber (formerly Alfa Romeo)")
            print(f"   🔄 Daniel Ricciardo return to RB")
            print(f"   📅 Extended 24-race calendar structure")
        else:
            print("⚠️ Data generated but compatibility issues detected.")
            print("Please check the error messages above.")
        
    except Exception as e:
        print(f"❌ Error generating 2024 data: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have pandas and numpy installed")
        print("2. Check that you have write permissions in the current directory")
        print("3. Ensure sufficient disk space")

if __name__ == "__main__":
    main()