import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Define transformation formula for waypoint (assuming real-world coordinates need to be transformed)
def transform_coordinates(x, y):
    new_x = 2000 + x * 20
    new_y = 2000 - y * 20
    return new_x, new_y

# Load the image
img = mpimg.imread('/home/daniel/maps/my_map0924.png')

# Create a plot
fig, ax = plt.subplots()
ax.imshow(img)

# Read data from the CSV file
csv_file = '/home/daniel/maps/wall_data.csv'  # Replace with your CSV file name

with open(csv_file, newline='') as file:
    reader = csv.DictReader(file)
    
    # Loop over each row in the CSV
    for idx, row in enumerate(reader):
        # Parse the data from the CSV
        wp_x = float(row['Waypoint X'])
        wp_y = float(row['Waypoint Y'])
        lw_x = float(row['Left Wall X'])
        lw_y = float(row['Left Wall Y'])
        rw_x = float(row['Right Wall X'])
        rw_y = float(row['Right Wall Y'])
        center_x = float(row['Center X'])
        center_y = float(row['Center Y'])
        
        # Transform waypoint coordinates
        waypoint = transform_coordinates(wp_x, wp_y)
        
        # Plot the waypoint (orange)
        ax.plot(waypoint[0], waypoint[1], 'o', color='orange', label=f'WP{idx+1}' if idx == 0 else "")
        ax.text(waypoint[0] - 10, waypoint[1] - 10, f'WP{idx+1}', color='orange', fontsize=12)

        # Plot left and right wall points
        ax.plot(lw_x, lw_y, 'o', color='red', label=f'Left Wall{idx+1}' if idx == 0 else "")

        
        ax.plot(rw_x, rw_y, 'o', color='blue', label=f'Right Wall{idx+1}' if idx == 0 else "")

        
        # Plot center point (green)
        ax.plot(center_x, center_y, 'o', color='green', label=f'Center{idx+1}' if idx == 0 else "")


# Add a legend
ax.legend(loc='upper right')

# Show the image with annotated points
plt.show()
