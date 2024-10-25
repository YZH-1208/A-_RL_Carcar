import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def transform_coordinates(x, y):
    new_x = 2000 + x * 20
    new_y = 2000 - y * 20
    return new_x, new_y

# Load the image
img_path = '/home/daniel/maps/my_map0924_2.png'  # Example image path; replace as needed
img = mpimg.imread(img_path)

# Define a variable for A* path points (these are already transformed)
astar_points = [(1.25, 0.1), (1.25, 0.1), (2.3, 0.15), (3.4, 0.2), (4.4, 0.15), (5.5, 0.2), (6.5, 0.25), (7.55, 0.25), (8.65, 0.35), (9.65, 0.4), (10.65, 0.45), (11.7, 0.5), (12.75, 0.55), (13.85, 0.55), (14.95, 0.65), (15.95, 0.75), (17.0, 0.8), (18.05, 0.85), (19.05, 0.9), (20.1, 0.95), (21.15, 1.0), (22.2, 1.05), (24.4, 1.2), (24.4, 1.2), (25.45, 1.25), (26.55, 1.35), (28.55, 1.45), (28.65, 1.45), (29.65, 1.5), (30.7, 1.6), (31.75, 1.65), (32.8, 1.75), (33.85, 1.85), (35.45, 2.0), (35.95, 2.0), (37.3, 2.0), (38.6, 1.9), (39.55, 1.9), (40.75, 2.25), (41.15, 2.25), (42.2, 2.3), (43.3, 2.35), (44.4, 2.4), (45.5, 2.45), (47.65, 2.5), (47.65, 2.5), (48.65, 2.65), (49.75, 2.65), (51.05, 2.65), (52.5, 2.7), (53.3, 2.75), (54.15, 2.75), (55.5, 2.5), (56.85, 2.55), (58.25, 3.6), (58.25, 3.6), (59.1, 4.15), (59.6, 6.25), (59.4, 7.25), (59.35, 8.3), (59.35, 8.3), (59.35, 10.35), (59.35, 11.35), (59.3, 12.4), (59.3, 12.4), (59.3, 14.5), (59.3, 15.5), (59.1, 16.5), (58.7, 17.5), (58.7, 17.5), (57.6, 17.85), (56.15, 17.6), (55.0, 17.55), (53.95, 17.5), (52.6, 17.4), (52.0, 17.25), (50.55, 17.05), (49.5, 16.95), (48.45, 16.8), (48.45, 16.8), (48.45, 16.65), (47.5, 16.45), (44.05, 17.75), (45.4, 17.5), (41.95, 17.4), (40.9, 17.35), (40.15, 16.45), (39.75, 16.45), (39.75, 16.45), (36.95, 16.2), (36.9, 16.15), (36.2, 15.0), (35.55, 14.0), (35.55, 14.0), (34.0, 12.65), (32.95, 12.65), (31.9, 12.65), (30.9, 12.65), (29.85, 12.6), (28.75, 12.55), (27.75, 12.5), (26.75, 12.45), (25.75, 12.4), (24.65, 12.35), (23.7, 12.3), (22.45, 12.25), (21.35, 12.2), (20.6, 12.05), (19.25, 11.9), (18.35, 11.9), (17.3, 11.85), (16.25, 11.8), (14.95, 11.75), (13.95, 11.7), (12.65, 11.65), (11.4, 11.6), (11.1, 11.6), (9.75, 13.05), (9.75, 13.05), (8.35, 14.4), (7.85, 14.85), (7.65, 14.85), (5.85, 14.25), (5.75, 14.05), (5.55, 14.05), (4.5, 13.85), (1.0, 13.65), (0.3, 13.65), (-1.1, 13.55), (-2.25, 13.5), (-3.1, 13.35), (-4.05, 13.55), (-5.1, 13.85), (-6.1, 13.95), (-7.15, 13.9), (-8.2, 13.85), (-9.2, 13.75), (-10.3, 13.65), (-11.25, 13.35), (-12.0, 12.65), (-12.35, 11.7), (-12.45, 10.65), (-12.45, 10.65), (-12.35, 9.6), (-12.25, 8.6), (-12.15, 7.5), (-12.1, 6.45), (-12.1, 5.4), (-12.05, 4.3), (-12.0, 3.25), (-11.95, 2.25), (-11.8, 1.2), (-10.55, -0.55), (-10.55, -0.55), (-9.45, -0.8), (-8.4, -0.8), (-7.4, -0.6), (-6.05, -0.55), (-5.3334, -0.3768)]



fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(img)

# Read and plot data from the CSV file
csv_file = '/home/daniel/maps/wall_data.csv'  # Replace with the actual CSV path

# Attempt to read the CSV file
try:
    with open(csv_file, newline='') as file:
        reader = csv.DictReader(file)
        
        for idx, row in enumerate(reader):
            # Parse the coordinates directly, as they are already transformed
            wp_x = float(row['Waypoint X'])
            wp_y = float(row['Waypoint Y'])
            lw_x = float(row['Left Wall X'])
            lw_y = float(row['Left Wall Y'])
            rw_x = float(row['Right Wall X'])
            rw_y = float(row['Right Wall Y'])
            center_x = float(row['Center X'])
            center_y = float(row['Center Y'])

            waypoint = transform_coordinates(wp_x,wp_y)
            # waypoint = astar_points

            # Plot each point with distinct markers and colors
            ax.plot(waypoint[0], waypoint[1], 'o', color='orange', markersize=8, label='Waypoint' if idx == 0 else "")
            ax.plot(lw_x, lw_y, 'o', color='red', markersize=8, label='Left Wall' if idx == 0 else "")
            ax.plot(rw_x, rw_y, 'o', color='blue', markersize=8, label='Right Wall' if idx == 0 else "")
            ax.plot(center_x, center_y, 'o', color='green', markersize=8, label='Center' if idx == 0 else "")

except FileNotFoundError:
    print(f"The file at {csv_file} was not found. Please check the file path.")

# Plot the A* path points
x_astar, y_astar = zip(*astar_points)
# 將 astar_points 的 x 和 y 分別轉換
x_astar_transformed = [2000 + 20 * x for x in x_astar]
y_astar_transformed = [2000 - 20 * y for y in y_astar]

ax.plot(x_astar_transformed,y_astar_transformed, 'o-', color='purple', markersize=10, linewidth=2, label="A* Path")

# Add legend
ax.legend(loc='upper right')

# Remove coordinate display on plot
ax.set_xticks([])
ax.set_yticks([])

# Show the image with annotated points
plt.title("Path Visualization with A*")
plt.show()