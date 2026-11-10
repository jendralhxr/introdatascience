import pandas as pd

# 1. Read the raw dataset
df = pd.read_csv('eai-databmi - Form Responses 1.csv')

# 2. Check how many duplicates exist based strictly on 'Email Address'
duplicate_count = df.duplicated(subset=['Email Address']).sum()
print(f"Number of duplicate email entries found: {duplicate_count}")

# 3. View the duplicated entries side-by-side before dropping
duplicate_rows = df[df.duplicated(subset=['Email Address'], keep=False)]
print("\n--- Duplicate Submissions ---")
print(duplicate_rows[['Email Address', 'Timestamp', 'Height (cm)', 'Weight (kg)']])

# 4. Remove duplicates based ONLY on 'Email Address'
# keep='last' preserves the latest submission, keep='first' preserves the earliest
df_cleaned = df.drop_duplicates(subset=['Email Address'], keep='last').reset_index(drop=True)

print(f"\nOriginal row count: {len(df)}")
print(f"Cleaned row count:  {len(df_cleaned)}")

#------

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Load & Deduplicate Data
df = pd.read_csv("eai-databmi - Form Responses 1.csv")
df_clean = df.drop_duplicates(subset=["Email Address"], keep="last").reset_index(
    drop=True
)
df_clean.rename(
    columns={"Height (cm)": "Height", "Weight (kg)": "Weight"}, inplace=True
)

df_clean= df_cleaned

# Set visual style
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))


import seaborn as sns

# --- Height Histogram ---
mean_h = df_clean["Height"].mean()
median_h = df_clean["Height"].median()

sns.histplot(df_clean["Height"], kde=True, ax=axes[0], color="skyblue", bins=8)
axes[0].axvline(
    mean_h,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Mean: {mean_h:.1f} cm",
)
axes[0].axvline(
    median_h,
    color="green",
    linestyle="-",
    linewidth=2,
    label=f"Median: {median_h:.1f} cm",
)
axes[0].set_title("Height Distribution", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Height (cm)")
axes[0].set_ylabel("Student Count")
axes[0].legend()

# --- Weight Histogram ---
mean_w = df_clean["Weight"].mean()
median_w = df_clean["Weight"].median()

sns.histplot(df_clean["Weight"], kde=True, ax=axes[1], color="salmon", bins=8)
axes[1].axvline(
    mean_w,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Mean: {mean_w:.1f} kg",
)
axes[1].axvline(
    median_w,
    color="green",
    linestyle="-",
    linewidth=2,
    label=f"Median: {median_w:.1f} kg",
)
axes[1].set_title("Weight Distribution", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Weight (kg)")
axes[1].set_ylabel("Student Count")
axes[1].legend()

plt.tight_layout()
plt.show()


#------

# BMI formula: Weight (kg) / (Height (m) ^ 2)
df_clean['BMI'] = df_clean['Weight'] / (df_clean['Height'] ** 2)

# Categorize BMI according to WHO standard categories
def get_bmi_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 25.0:
        return 'Normal weight'
    elif 25.0 <= bmi < 30.0:
        return 'Overweight'
    else:
        return 'Obese'

df_clean['Category'] = df_clean['BMI'].apply(get_bmi_category)

print("BMI Summary Statistics:")
print(df_clean['BMI'].describe())

print("\nCategory Distribution:")
print(df_clean['Category'].value_counts())

print("\nFirst 10 records with BMI:")
print(df_clean[['Email Address', 'Height', 'Weight', 'BMI', 'Category']].head(10))


#--

# Plotting BMI Histogram
sns.set_theme(style="whitegrid")
plt.figure(figsize=(9, 5))

mean_bmi = df_clean['BMI'].mean()
median_bmi = df_clean['BMI'].median()

# Histogram plot with KDE line
ax = sns.histplot(df_clean['BMI'], kde=True, color='purple', bins=10)

# Overlay Mean and Median lines
plt.axvline(mean_bmi, color='red', linestyle='--', linewidth=2, label=f'Mean BMI ({mean_bmi:.2f})')
plt.axvline(median_bmi, color='green', linestyle='-', linewidth=2, label=f'Median BMI ({median_bmi:.2f})')

# Annotate WHO standard thresholds
plt.axvline(18.5, color='orange', linestyle=':', label='Underweight Threshold (< 18.5)')
plt.axvline(25.0, color='darkred', linestyle=':', label='Overweight Threshold (≥ 25.0)')

plt.title('Class BMI Distribution', fontsize=14, fontweight='bold')
plt.xlabel('BMI (kg/m²)', fontsize=12)
plt.ylabel('Student Count', fontsize=12)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()

#---


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Setup Data & Compute Metrics
mean_bmi = df_clean["BMI"].mean()
median_bmi = df_clean["BMI"].median()

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

# 2. Plot Histogram & KDE Curve
ax = sns.histplot(
    df_clean["BMI"],
    kde=True,
    color="#6C5CE7",
    bins=10,
    edgecolor="white",
    alpha=0.6,
)

# 3. Add Threshold Lines
plt.axvline(
    18.5,
    color="#E67E22",
    linestyle=":",
    linewidth=2,
    label="Underweight (< 18.5)",
)
plt.axvline(
    25.0, color="#C0392B", linestyle=":", linewidth=2, label="Overweight (≥ 25.0)"
)

# 4. Add Mean & Median Lines
plt.axvline(
    mean_bmi,
    color="#D63031",
    linestyle="--",
    linewidth=2.5,
    label=f"Mean ({mean_bmi:.2f})",
)
plt.axvline(
    median_bmi,
    color="#2ECC71",
    linestyle="-",
    linewidth=2.5,
    label=f"Median ({median_bmi:.2f})",
)

# 5. Fix Layout & Prevent Legend Collision
plt.ylim(0, 14)  # Extends top range to give legend breathing room
plt.title("Class BMI Distribution", fontsize=15, fontweight="bold", pad=15)
plt.xlabel("BMI (kg/m²)", fontsize=12)
plt.ylabel("Student Count", fontsize=12)

# Place legend in cleanThe main reason this snippet "doesn't play nice" is a visual overlap issue: when you draw vertical threshold lines at `18.5` and `25.0` inside a tightly packed plot, the **legend items and vertical lines clutter the plot area**, and `sns.histplot()` with default `kde=True` can sometimes clip or look misaligned depending on bin alignment.

Here is a refined version that fixes the layout, adds clear background color zones for WHO categories, and formats the plot so all text and lines are clean and readable:

```python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Ensure BMI is calculated
df_clean["BMI"] = df_clean["Weight"] / ((df_clean["Height"] / 100) ** 2)

mean_bmi = df_clean["BMI"].mean()
median_bmi = df_clean["BMI"].median()

sns.set_theme(style="white")
fig, ax = plt.subplots(figsize=(10, 6))

# 1. Shaded WHO Background Zones for clear context
ax.axvspan(15, 18.5, color="#ffe6cc", alpha=0.5, label="Underweight (< 18.5)")
ax.axvspan(
    18.5, 25.0, color="#d9f2d9", alpha=0.5, label="Normal (18.5 – 24.9)"
)
ax.axvspan(
    25.0, 30.0, color="#fff0b3", alpha=0.5, label="Overweight (25.0 – 29.9)"
)
ax.axvspan(30.0, 35.0, color="#ffcccc", alpha=0.5, label="Obese (≥ 30.0)")

# 2. Histogram with KDE line
sns.histplot(
    data=df_clean,
    x="BMI",
    kde=True,
    color="#4b0082",
    bins=12,
    ax=ax,
    edgecolor="white",
)

# 3. Mean & Median Lines (Higher zorder so they sit on top of bars)
ax.axvline(
    mean_bmi,
    color="red",
    linestyle="--",
    linewidth=2.5,
    zorder=5,
    label=f"Mean: {mean_bmi:.2f}",
)
ax.axvline(
    median_bmi,
    color="darkgreen",
    linestyle="-",
    linewidth=2.5,
    zorder=5,
    label=f"Median: {median_bmi:.2f}",
)

# 4. Clean labels and bounds
ax.set_xlim(15, 33)
ax.set_title("Class BMI Distribution & WHO Categories", fontsize=14, pad=15)
ax.set_xlabel("BMI (kg/m²)", fontsize=12)
ax.set_ylabel("Student Count", fontsize=12)

# Place legend outside or neatly at top right
ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

sns.despine()
plt.tight_layout()
plt.show()



#-----

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

height = df_clean['Height'].values
weight = df_clean['Weight'].values

# Combine features into matrix X
X = np.column_stack((height, weight))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Fit K-Means with K=3 (Small, Medium, Large)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
centroids_scaled = kmeans.cluster_centers_
centroids_unscaled = scaler.inverse_transform(centroids_scaled)

# Assign cluster labels based on height/weight centroids (S: lowest height/weight, L: highest)
cluster_order = np.argsort(centroids_unscaled[:, 0])  # Sort by height centroid
size_names = ['Small (S)', 'Medium (M)', 'Large (L)']
cluster_mapping = {original_idx: size_names[new_rank] for new_rank, original_idx in enumerate(cluster_order)}

df_clean['Cluster_ID'] = labels
df_clean['Shirt_Size'] = df_clean['Cluster_ID'].map(cluster_mapping)

# Print Summary Breakdown
size_counts = df_clean['Shirt_Size'].value_counts().reindex(['Small (S)', 'Medium (M)', 'Large (L)'])
print("--- Shirt Size Inventory Order ---")
print(size_counts)

print("\n--- Cluster Centroids (Average Height & Weight) ---")
for original_idx in cluster_order:
    c_h, c_w = centroids_unscaled[original_idx]
    s_label = cluster_mapping[original_idx]
    count = (df_clean['Shirt_Size'] == s_label).sum()
    print(f"{s_label}: Centroid = ({c_h:.1f} cm, {c_w:.1f} kg) | Count = {count} students")

# 3. Visualization of K-Means Clusters
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

palette = {'Small (S)': '#2ECC71', 'Medium (M)': '#3498DB', 'Large (L)': '#E74C3C'}

# Plot clustered data points
sns.scatterplot(
    data=df_clean, x='Height', y='Weight', hue='Shirt_Size',
    hue_order=['Small (S)', 'Medium (M)', 'Large (L)'],
    palette=palette, s=90, alpha=0.9, zorder=3
)

# Plot Centroids
for original_idx in cluster_order:
    c_h, c_w = centroids_unscaled[original_idx]
    s_label = cluster_mapping[original_idx]
    plt.scatter(c_h, c_w, color='black', s=220, marker='X', zorder=5)
    plt.annotate(f"Centroid {s_label.split()[0]}\n({c_h:.1f}cm, {c_w:.1f}kg)", 
                 (c_h, c_w), textcoords="offset points", xytext=(0,10), 
                 ha='center', fontsize=9, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1, alpha=0.8))

plt.title('K-Means Clustering (K=3) for Class T-Shirt Sizing', fontsize=15, fontweight='bold', pad=15)
plt.xlabel('Height (cm)', fontsize=12)
plt.ylabel('Weight (kg)', fontsize=12)
plt.legend(title='Assigned Shirt Size', frameon=True, facecolor='white', framealpha=0.9, fontsize=10, loc='upper left')

plt.tight_layout()

