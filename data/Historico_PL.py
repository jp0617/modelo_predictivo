# %pip install kagglehub

import kagglehub
import shutil
import os

# Download latest version
path = kagglehub.dataset_download("marcohuiii/english-premier-league-epl-match-data-2000-2025")
print("Path to dataset files:", path)

# Find CSV file in downloaded path
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
csv_path = os.path.join(path, csv_files[0])

# Copy to Unity Catalog Volume (ajusta catalog/schema/volume según tu configuración)
volume_path = "/Volumes/workspace/bronze/premier_league/epl_final.csv"
os.makedirs(os.path.dirname(volume_path), exist_ok=True)
shutil.copy(csv_path, volume_path)
print(f"File copied to: {volume_path}")

# Now read from Volume and write to table
df = spark.read.option("header", "true").csv(volume_path)
df.write.mode("overwrite").saveAsTable("bronze.historico_premier")

display(spark.table("bronze.historico_premier"))