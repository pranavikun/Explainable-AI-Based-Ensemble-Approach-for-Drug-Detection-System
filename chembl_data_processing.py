#1
import pandas as pd
from chembl_webresource_client.new_client import new_client

target = new_client.target
target_query = target.search('coronavirus')
targets = pd.DataFrame.from_dict(target_query)

print("Available columns in target data:", targets.columns)

columns_to_display = ['target_chembl_id', 'pref_name', 'organism', 'target_type']
if 'description' in targets.columns:
    columns_to_display.append('description')
print("\nDrug Target Information:")
print(targets[columns_to_display].head(10))
selected_protein_target = targets['target_chembl_id'][6]  

bioactivity_data = new_client.activity
filtered_data = bioactivity_data.filter(target_chembl_id=selected_protein_target).filter(standard_type="IC50")
bioactivity_DF = pd.DataFrame.from_dict(filtered_data)
bioactivity_DF = bioactivity_DF[bioactivity_DF.standard_value.notna()]

print(f"\nBioactivity Data for Target: {selected_protein_target}")
print(bioactivity_DF[['molecule_chembl_id', 'canonical_smiles', 'standard_value']].head(10))

bioactivity_DF.to_csv('raw_bioactivity_data.csv', index=False)

activity_classes = []
for value in bioactivity_DF.standard_value:
    if float(value) >= 10000:
        activity_classes.append("inactive")
    elif float(value) <= 1000:
        activity_classes.append("active")
    else:
        activity_classes.append("moderate")

bioactivity_DF['activity_class'] = activity_classes
activity_summary = bioactivity_DF['activity_class'].value_counts()
print("\nBioactivity Classification Summary:")
print(activity_summary)
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
sns.countplot(data=bioactivity_DF, x='activity_class', hue='activity_class', palette="viridis", order=["active", "moderate", "inactive"], legend=False)
plt.title(f"Bioactivity Distribution for Target {selected_protein_target}")
plt.xlabel("Activity Class")
plt.ylabel("Frequency")
plt.show()
