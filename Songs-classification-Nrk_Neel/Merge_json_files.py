import json

# Load data from the two JSON files
with open('Hindi_Songs_Classification/combined_data_nrk.json', 'r') as f1:
    data1 = json.load(f1)

with open('Hindi_Songs_Classification/combined_data_neel.json', 'r') as f2:
    data2 = json.load(f2)

# Merge the two dictionaries
merged_data = {**data1, **data2}

print(len(merged_data))

# Save the merged data into a new JSON file
with open('Hindi_Merged_Songs_Features.json', 'w') as f:
    json.dump(merged_data, f, indent=4)

print("JSON files merged successfully into 'merged_file.json'")
