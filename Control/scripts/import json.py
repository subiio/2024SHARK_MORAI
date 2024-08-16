import json

# Load the original data from the optimal_map.json file
with open('/home/gigacha/catkin_ws/src/beginner_tutorials_blanks/scripts/optimal_map.json', 'r') as file:
    data = json.load(file)

# Create a new dictionary with the keys adjusted by subtracting 1500
new_data = {str(int(k) - 1500): v for k, v in data.items()}

# Save the new data back to a JSON file
with open('optimal_map.json', 'w') as file:
    json.dump(new_data, file, indent=4)

print("Updated optimal_map.json has been created as optimal_map_updated.json")
