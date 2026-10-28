import pandas as pd
import kaggle

# Authenticate & search public datasets on Kaggle
kaggle.api.dataset_download_files(
    "senapatirajesh/netflix-tv-shows-and-movies", path="./data", unzip=True
)

# Load into DataFrame
df = pd.read_csv("./data/NetFlix.csv")
print(df.head())