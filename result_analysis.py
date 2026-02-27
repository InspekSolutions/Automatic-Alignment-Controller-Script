import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file (make sure it is in the same directory as the script)
df = pd.read_csv("test_16-04-20_4-16.csv")
# Compute global axis limits from all the data
global_x_min = df['x'].min()
global_x_max = df['x'].max()
global_y_min = df['y'].min()
global_y_max = df['y'].max()

# Group the DataFrame by the parameter columns: Range, Step, Speed
grouped = df.groupby(['Range', 'Step', 'Speed'])

# Function to compute Euclidean average deviation from the centroid.
def compute_deviation(group):
    mean_x = group['x'].mean()
    mean_y = group['y'].mean()
    # Euclidean distances from the centroid for all points in this group
    distances = np.sqrt((group['x'] - mean_x)**2 + (group['y'] - mean_y)**2)
    # Average deviation (mean Euclidean distance)
    avg_deviation = distances.mean()
    return mean_x, mean_y, group['x'].std(), group['y'].std(), avg_deviation

# Loop through each parameter set (group) and plot
for (rng, step, speed), group in grouped:
    # Compute deviation statistics
    mean_x, mean_y, std_x, std_y, avg_dev = compute_deviation(group)
    
    # Create a new figure for each group
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Scatter plot the x and y values
    scatter = ax.scatter(group['x'], group['y'], s=100, c='blue')
    
    # Annotate each point with its order number
    for idx, row in group.iterrows():
        ax.text(row['x'], row['y'], str(int(row['Order'])), fontsize=9,
                color='red', ha='right', va='bottom')
    
    # Set the same axis limits for every plot
    ax.set_xlim(global_x_min, global_x_max)
    ax.set_ylim(global_y_min, global_y_max)
    
    # Set labels and title with the parameter info and computed deviations
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    title = (f"Range={rng}, Step={step}, Speed={speed}\n"
             f"Std_x={std_x:.2f}, Std_y={std_y:.2f}, AvgDev={avg_dev:.2f}")
    ax.set_title(title)
    
    # Optionally, plot the centroid
    ax.plot(mean_x, mean_y, marker='x', markersize=10, color='black')
    
    plt.tight_layout()
    plt.show()
    
    # Also print the deviation stats for this parameter set in the console
    print(f"Parameters: Range={rng}, Step={step}, Speed={speed}")
    print(f"  Mean x: {mean_x:.2f}, Mean y: {mean_y:.2f}")
    print(f"  Std Dev x: {std_x:.2f}, Std Dev y: {std_y:.2f}")
    print(f"  Average Euclidean Deviation: {avg_dev:.2f}\n")
