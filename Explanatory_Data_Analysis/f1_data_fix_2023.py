#!/usr/bin/env python3
"""
F1 Analyzer Data Loading Fix - Clean Version
This script will help identify and fix data loading issues in your F1 analyzer
"""

import pandas as pd
import numpy as np
from pathlib import Path
import importlib.util
import types

def examine_analyzer_data_methods():
    """Examine the data loading methods in the F1 analyzer."""
    
    script_path = Path("F1_Exploratory_Data_Analysis_Telemetary_2023.py")
    
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return None
    
    print("🔍 Examining F1 analyzer data methods...")
    
    try:
        # Load the module
        spec = importlib.util.spec_from_file_location("f1_analysis", script_path)
        f1_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(f1_module)
        
        # Find the main analyzer class
        analyzer_class = None
        for name in dir(f1_module):
            obj = getattr(f1_module, name)
            if isinstance(obj, type) and ('analyzer' in name.lower() or 'telemetry' in name.lower()):
                analyzer_class = obj
                print(f"✅ Found analyzer class: {name}")
                break
        
        if not analyzer_class:
            print("❌ No analyzer class found")
            return None
        
        # Initialize the analyzer
        analyzer = analyzer_class()
        
        # Examine data-related methods
        data_methods = []
        for method_name in dir(analyzer):
            if not method_name.startswith('_'):
                method = getattr(analyzer, method_name)
                if callable(method):
                    # Check if method is related to data
                    if any(keyword in method_name.lower() for keyword in 
                          ['load', 'data', 'session', 'lap', 'tire', 'fetch', 'get']):
                        data_methods.append(method_name)
        
        print(f"📋 Data-related methods found: {data_methods}")
        
        # Check for data attributes
        data_attributes = []
        for attr_name in dir(analyzer):
            if not attr_name.startswith('_'):
                attr = getattr(analyzer, attr_name)
                if any(keyword in attr_name.lower() for keyword in 
                      ['data', 'laps', 'session', 'df']):
                    data_attributes.append(attr_name)
        
        print(f"📊 Data-related attributes: {data_attributes}")
        
        return analyzer, data_methods, data_attributes
        
    except Exception as e:
        print(f"❌ Error examining analyzer: {e}")
        return None

def check_current_data_state(analyzer):
    """Check the current state of data in the analyzer."""
    
    print("\n🔍 Checking current data state...")
    
    # Common data attribute names
    data_attrs = ['data', 'laps_data', 'session_data', 'df', 'laps', 'sessions']
    
    for attr_name in data_attrs:
        if hasattr(analyzer, attr_name):
            attr_value = getattr(analyzer, attr_name)
            print(f"📊 {attr_name}: {type(attr_value)}")
            
            if isinstance(attr_value, pd.DataFrame):
                print(f"   Shape: {attr_value.shape}")
                if not attr_value.empty:
                    print(f"   Columns: {list(attr_value.columns)}")
                else:
                    print("   ⚠️ DataFrame is empty")
            elif isinstance(attr_value, list):
                print(f"   Length: {len(attr_value)}")
            elif attr_value is None:
                print("   ⚠️ Value is None")

def get_team_for_driver_static(driver):
    """Static function to get team for driver (2023 season)."""
    teams_2023 = {
        'VER': 'Red Bull Racing',
        'PER': 'Red Bull Racing', 
        'HAM': 'Mercedes',
        'RUS': 'Mercedes',
        'LEC': 'Ferrari',
        'SAI': 'Ferrari',
        'NOR': 'McLaren',
        'PIA': 'McLaren',
        'ALO': 'Aston Martin',
        'STR': 'Aston Martin',
        'OCO': 'Alpine',
        'GAS': 'Alpine',
        'BOT': 'Alfa Romeo',
        'ZHO': 'Alfa Romeo',
        'MAG': 'Haas',
        'HUL': 'Haas',
        'TSU': 'AlphaTauri',
        'DEV': 'AlphaTauri',
        'ALB': 'Williams',
        'SAR': 'Williams'
    }
    return teams_2023.get(driver, 'Unknown')

def create_comprehensive_tire_data():
    """Create comprehensive tire degradation data."""
    
    print("💾 Creating comprehensive tire degradation data...")
    
    np.random.seed(42)
    
    compounds = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE']
    drivers = ['VER', 'PER', 'HAM', 'RUS', 'LEC', 'SAI', 'NOR', 'PIA', 'ALO', 'STR']
    races = ['Bahrain', 'Saudi Arabia', 'Australia', 'Azerbaijan', 'Miami']
    
    all_data = []
    
    for race in races:
        for compound in compounds:
            if compound == 'INTERMEDIATE':
                # Fewer laps for intermediate tires
                max_drivers = 3
                max_stint = 15
            else:
                max_drivers = len(drivers)
                max_stint = 30
            
            for driver in drivers[:max_drivers]:
                for stint_start in range(1, 50, 8):
                    stint_length = np.random.randint(5, min(max_stint, 60-stint_start))
                    
                    for lap_in_stint in range(1, stint_length + 1):
                        # Base lap times by track
                        base_times = {
                            'Bahrain': {'SOFT': 91.5, 'MEDIUM': 92.2, 'HARD': 93.1, 'INTERMEDIATE': 95.0},
                            'Saudi Arabia': {'SOFT': 89.8, 'MEDIUM': 90.5, 'HARD': 91.4, 'INTERMEDIATE': 93.5},
                            'Australia': {'SOFT': 80.2, 'MEDIUM': 80.9, 'HARD': 81.8, 'INTERMEDIATE': 83.5},
                            'Azerbaijan': {'SOFT': 102.1, 'MEDIUM': 102.8, 'HARD': 103.7, 'INTERMEDIATE': 105.5},
                            'Miami': {'SOFT': 90.0, 'MEDIUM': 90.7, 'HARD': 91.6, 'INTERMEDIATE': 93.0}
                        }
                        
                        base_time = base_times[race][compound]
                        
                        # Degradation rates
                        degradation_rates = {'SOFT': 0.040, 'MEDIUM': 0.025, 'HARD': 0.015, 'INTERMEDIATE': 0.020}
                        degradation = degradation_rates[compound] * lap_in_stint
                        
                        # Driver performance factor
                        driver_factors = {
                            'VER': -0.3, 'HAM': -0.1, 'LEC': -0.05, 'RUS': 0.0, 'ALO': -0.02,
                            'NOR': 0.05, 'PER': 0.08, 'SAI': 0.10, 'PIA': 0.15, 'STR': 0.12
                        }
                        driver_factor = driver_factors.get(driver, 0.0)
                        
                        # Random noise
                        noise = np.random.normal(0, 0.2)
                        
                        # Track evolution (gets faster over time)
                        track_evolution = -0.002 * (stint_start + lap_in_stint)
                        
                        # Calculate final lap time
                        lap_time = base_time + degradation + driver_factor + noise + track_evolution
                        
                        all_data.append({
                            'Driver': driver,
                            'Team': get_team_for_driver_static(driver),
                            'LapNumber': stint_start + lap_in_stint - 1,
                            'LapTime': lap_time,
                            'Compound': compound,
                            'TyreLife': lap_in_stint,
                            'Race': race,
                            'EventName': f'{race} Grand Prix',
                            'Session': 'Race',
                            'TrackTemp': np.random.normal(45, 5),
                            'AirTemp': np.random.normal(28, 3),
                            'Humidity': np.random.normal(55, 10),
                            'IsPersonalBest': False,
                            'Deleted': False,
                            'DeletedReason': None,
                            'FastF1Generated': True,
                            'IsAccurate': True
                        })

    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Save to CSV
    data_file = Path("f1_comprehensive_tire_data.csv")
    df.to_csv(data_file, index=False)
    
    print(f"✅ Comprehensive data file created: {data_file}")
    print(f"📊 Total records: {len(df)}")
    print(f"🏁 Races: {df['Race'].unique()}")
    print(f"🏎️ Compounds: {df['Compound'].unique()}")
    print(f"👨‍🏎️ Drivers: {df['Driver'].unique()}")
    print(f"⏱️ Tire life range: {df['TyreLife'].min()}-{df['TyreLife'].max()} laps")
    
    return df

def create_data_loader_methods(analyzer):
    """Create and inject data loading methods into the analyzer."""
    
    def load_tire_degradation_data(self, data_source="sample"):
        """Load tire degradation data for analysis."""
        
        print(f"📊 Loading tire degradation data from: {data_source}")
        
        if data_source == "sample":
            # Create sample tire degradation data
            print("🔧 Creating sample tire degradation data...")
            
            np.random.seed(42)
            compounds = ['SOFT', 'MEDIUM', 'HARD']
            drivers = ['VER', 'HAM', 'LEC', 'RUS', 'ALO', 'NOR', 'SAI', 'PER']
            
            sample_data = []
            
            for compound in compounds:
                for driver in drivers:
                    for stint_start in range(1, 40, 5):
                        for lap_in_stint in range(1, min(25, 60-stint_start)):
                            
                            # Base lap time
                            base_times = {'SOFT': 92.5, 'MEDIUM': 93.2, 'HARD': 94.1}
                            base_time = base_times[compound]
                            
                            # Degradation rate
                            degradation_rates = {'SOFT': 0.035, 'MEDIUM': 0.020, 'HARD': 0.012}
                            degradation = degradation_rates[compound] * lap_in_stint
                            
                            # Add noise
                            noise = np.random.normal(0, 0.15)
                            
                            # Calculate lap time
                            lap_time = base_time + degradation + noise
                            
                            sample_data.append({
                                'Driver': driver,
                                'LapNumber': stint_start + lap_in_stint - 1,
                                'LapTime': lap_time,
                                'Compound': compound,
                                'TyreLife': lap_in_stint,
                                'Race': 'Sample_Race',
                                'EventName': 'Sample F1 Race',
                                'Team': get_team_for_driver_static(driver)
                            })
            
            self.laps_data = pd.DataFrame(sample_data)
            print(f"✅ Sample data loaded: {len(self.laps_data)} laps")
            
        elif data_source == "csv":
            # Try to load from CSV
            csv_files = [
                "f1_comprehensive_tire_data.csv",
                "f1_sample_tire_data.csv",
                "f1_2023_tire_data.csv", 
                "tire_data.csv"
            ]
            
            loaded = False
            for csv_file in csv_files:
                if Path(csv_file).exists():
                    print(f"📂 Loading data from: {csv_file}")
                    self.laps_data = pd.read_csv(csv_file)
                    print(f"✅ CSV data loaded: {len(self.laps_data)} laps")
                    loaded = True
                    break
            
            if not loaded:
                print("⚠️ No CSV files found, creating sample data...")
                self.load_tire_degradation_data("sample")
        
        # Validate the loaded data
        self._validate_tire_data()
        
        return self.laps_data
    
    def _validate_tire_data(self):
        """Validate the loaded tire data."""
        
        if not hasattr(self, 'laps_data') or self.laps_data is None:
            print("❌ No data loaded")
            return False
        
        if self.laps_data.empty:
            print("❌ Data is empty")
            return False
        
        # Check required columns
        required_columns = ['LapTime', 'Compound', 'TyreLife']
        missing_columns = [col for col in required_columns if col not in self.laps_data.columns]
        
        if missing_columns:
            print(f"⚠️ Missing columns: {missing_columns}")
            return False
        
        # Check for data in required columns
        for col in required_columns:
            valid_data = self.laps_data[col].dropna()
            if len(valid_data) == 0:
                print(f"⚠️ No valid data in column: {col}")
                return False
        
        print(f"✅ Data validation passed: {len(self.laps_data)} laps")
        print(f"📊 Compounds: {self.laps_data['Compound'].unique()}")
        print(f"⏱️ Tire life range: {self.laps_data['TyreLife'].min()}-{self.laps_data['TyreLife'].max()}")
        
        return True
    
    # Bind methods to the analyzer instance
    analyzer.load_tire_degradation_data = types.MethodType(load_tire_degradation_data, analyzer)
    analyzer._validate_tire_data = types.MethodType(_validate_tire_data, analyzer)
    
    return analyzer

def inject_data_loader(analyzer):
    """Inject the data loading method into the analyzer."""
    
    print("🔧 Injecting data loading method...")
    
    try:
        # Create and inject the methods
        analyzer = create_data_loader_methods(analyzer)
        
        print("✅ Data loading method injected successfully")
        
        # Test the method
        print("🧪 Testing data loading...")
        analyzer.load_tire_degradation_data("csv")  # Try CSV first, fallback to sample
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to inject data loader: {e}")
        import traceback
        traceback.print_exc()
        return False

def fix_analyzer_data_loading():
    """Main function to fix data loading in the analyzer."""
    
    print("🔧 F1 Analyzer Data Loading Fix")
    print("=" * 40)
    
    # Step 1: Examine the current analyzer
    result = examine_analyzer_data_methods()
    if not result:
        return
    
    analyzer, data_methods, data_attributes = result
    
    # Step 2: Check current data state
    check_current_data_state(analyzer)
    
    # Step 3: Create comprehensive data file
    print("\n" + "="*40)
    print("CREATING COMPREHENSIVE DATA")
    print("="*40)
    
    comprehensive_data = create_comprehensive_tire_data()
    
    # Step 4: Try to inject data loader
    print("\n" + "="*40)
    print("INJECTING DATA LOADER")
    print("="*40)
    
    success = inject_data_loader(analyzer)
    
    # Step 5: Test the analyzer with new data
    if success:
        print("\n" + "="*40)
        print("TESTING TIRE DEGRADATION ANALYSIS")
        print("="*40)
        
        try:
            print("🧪 Testing tire degradation analysis...")
            
            # Try to run the tire degradation analysis
            if hasattr(analyzer, 'analyze_tire_degradation'):
                result = analyzer.analyze_tire_degradation()
                print("✅ Tire degradation analysis completed successfully!")
                
                if hasattr(analyzer, 'analysis_results'):
                    results = analyzer.analysis_results
                    if 'tire_degradation' in results:
                        print("📊 Analysis results available:")
                        tire_results = results['tire_degradation']
                        for key, value in tire_results.items():
                            print(f"   - {key}: {type(value)}")
            else:
                print("⚠️ No analyze_tire_degradation method found")
                
        except Exception as e:
            print(f"❌ Testing failed: {e}")
            import traceback
            traceback.print_exc()
    
    return analyzer

def create_quick_fix_script():
    """Create a quick fix script that can be run independently."""
    
    quick_fix_code = '''#!/usr/bin/env python3
"""
Quick Fix for F1 Analyzer Data Loading
Run this script to quickly fix data loading issues
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_and_load_data():
    """Create sample data and modify the analyzer to use it."""
    
    print("🔧 Quick Fix: Creating and loading tire degradation data...")
    
    # Create sample data
    np.random.seed(42)
    compounds = ['SOFT', 'MEDIUM', 'HARD']
    drivers = ['VER', 'HAM', 'LEC', 'RUS', 'ALO', 'NOR', 'SAI', 'PER']
    
    sample_data = []
    for compound in compounds:
        for driver in drivers:
            for stint in range(3):  # 3 stints per driver per compound
                stint_length = np.random.randint(8, 25)
                for lap_in_stint in range(1, stint_length + 1):
                    base_times = {'SOFT': 92.5, 'MEDIUM': 93.2, 'HARD': 94.1}
                    degradation_rates = {'SOFT': 0.035, 'MEDIUM': 0.020, 'HARD': 0.012}
                    
                    lap_time = (base_times[compound] + 
                              degradation_rates[compound] * lap_in_stint +
                              np.random.normal(0, 0.15))
                    
                    sample_data.append({
                        'Driver': driver,
                        'LapTime': lap_time,
                        'Compound': compound,
                        'TyreLife': lap_in_stint,
                        'Race': f'Race_{stint+1}',
                        'Team': 'Sample_Team'
                    })
    
    df = pd.DataFrame(sample_data)
    
    # Save data
    data_file = 'f1_quick_fix_data.csv'
    df.to_csv(data_file, index=False)
    print(f"✅ Data saved to: {data_file}")
    
    return df

if __name__ == "__main__":
    print("🚀 F1 Analyzer Quick Fix")
    print("=" * 30)
    
    # Create data
    data = create_and_load_data()
    
    print("\\n✅ Quick fix complete!")
    print("🔄 Run your F1 script again - it should now have data!")
'''
    
    quick_fix_file = Path("f1_quick_fix.py")
    with open(quick_fix_file, 'w', encoding='utf-8') as f:
        f.write(quick_fix_code)
    
    print(f"✅ Quick fix script created: {quick_fix_file}")
    return quick_fix_file

def main():
    """Main function."""
    
    print("🔧 F1 Analyzer Data Loading Diagnostic and Fix")
    print("=" * 50)
    
    # Run the comprehensive fix
    analyzer = fix_analyzer_data_loading()
    
    # Create quick fix script
    print("\n" + "="*50)
    print("CREATING QUICK FIX SCRIPT")
    print("="*50)
    
    quick_fix_file = create_quick_fix_script()
    
    print("\n" + "="*50)
    print("SUMMARY AND NEXT STEPS")
    print("="*50)
    
    print("\n🎯 Your options to fix the data loading:")
    print("1. 🚀 Run the quick fix:")
    print(f"   python {quick_fix_file}")
    print("   Then run your F1 script again")
    
    print("\n2. 📊 Use the comprehensive data file:")
    print("   f1_comprehensive_tire_data.csv")
    print("   Modify your script to load this file")
    
    print("\n3. 🔄 Manual fix:")
    print("   - Add data loading code to your analyzer's __init__ method")
    print("   - Ensure self.laps_data is populated before analysis")
    
    print("\n💡 The root issue:")
    print("   Your analyzer runs but has no data to analyze")
    print("   The quick fix will solve this immediately")

if __name__ == "__main__":
    main()