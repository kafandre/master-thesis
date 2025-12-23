import rdata
import pandas as pd
import requests
import tarfile
import io
import os

VERSION = "0.1-10"
URL = f"https://cran.r-project.org/src/contrib/hdi_{VERSION}.tar.gz"
DEST_FOLDER = "data/riboflavin"
CSV_PATH = os.path.join(DEST_FOLDER, "riboflavin.csv")

os.makedirs(DEST_FOLDER, exist_ok=True)

print(f"1. Downloading source package from CRAN (v{VERSION})...")
response = requests.get(URL)
response.raise_for_status()

print("2. Processing archive...")
with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
    # Find the right file (case insensitive search)
    member = next(
        (m for m in tar.getmembers() if "riboflavin" in m.name.lower() and (m.name.endswith(".rda") or m.name.endswith(".RData"))), 
        None
    )
    
    if not member:
        raise ValueError("riboflavin dataset not found in package!")

    print(f"   -> Found file: {member.name}")
    
    # Extract
    f = tar.extractfile(member)
    parsed = rdata.parser.parse_file(f)
    
    # --- 3. Configure Converter Safely ---
    # 1. Get default configuration
    defaults = rdata.conversion.SimpleConverter().constructor_dict
    # 2. Create a mutable copy
    new_map = dict(defaults)
    # 3. Remove the 'data.frame' handler to prevent the crash
    if "data.frame" in new_map:
        del new_map["data.frame"]
        
    # 4. Create a NEW converter with our safe configuration
    print("3. Converting (safe mode)...")
    converter = rdata.conversion.SimpleConverter(constructor_dict=new_map)
    converted = converter.convert(parsed)

# --- 4. Reconstruct DataFrame Manually ---
print("4. Reconstructing DataFrame...")

# The result is now a nested dictionary (since we disabled DataFrame auto-conversion)
keys = list(converted.keys())
# Grab the first object found (usually 'riboflavin')
ribo_data = converted[keys[0]]

# Extract components
y_values = ribo_data['y']
x_matrix = ribo_data['x']

# Create Pandas DataFrame
print("   Building feature matrix...")
X_df = pd.DataFrame(x_matrix)
X_df.columns = [f"gene_{i}" for i in range(X_df.shape[1])] 

y_series = pd.Series(y_values, name="target_y")

# Combine
df_final = pd.concat([y_series, X_df], axis=1)

# --- 5. Save ---
df_final.to_csv(CSV_PATH, index=False)
print(f"Success! Data shape: {df_final.shape}")
print(f"Saved to: {CSV_PATH}")