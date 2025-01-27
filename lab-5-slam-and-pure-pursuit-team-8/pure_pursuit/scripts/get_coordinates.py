from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# def create_obstacles():# Open the image file
#     filename = "/home/yihsuan/sim_ws/src/f1tenth_gym_ros/maps/classroom.pgm"
#     img = Image.open(filename)

#     # Convert the image to a numpy array
#     img_array = np.array(img)
#     x_shape, y_shape = img_array.shape
#     print("x_shape", x_shape, y_shape)
#     # Define a threshold
#     white_threshold = 200


#     y_indices, x_indices = np.where(img_array < white_threshold)
#     x_origin, y_origin = -6.13, -4.49
#     x_coord, y_coord = x_origin + 0.05 * x_indices, y_origin + 0.05 * (x_shape-y_indices)
#     return np.vstack([x_coord, y_coord]).T
# h = create_obstacles()
# print(len(h))
# Open the image file
filename = "/home/yihsuan/sim_ws/src/f1tenth_gym_ros/maps/levine_all.pgm"
img = Image.open(filename)

# Convert the image to a numpy array
img_array = np.array(img)
x_shape, y_shape = img_array.shape
#print("x_shape", x_shape, y_shape)
# Define a threshold
white_threshold = 249



# Get the indices where the condition is met
y_indices, x_indices = np.where(img_array < white_threshold)
# x_indices, y_indices = 0, 0
# print(img_array[0, 0])
# #x_origin, y_origin = 2.2, 5.66
# x_origin, y_origin = -11.1, -0.681
# #x_indices, y_indices = y_shape, x_shape  # X coordinates
# x_coord, y_coord = x_origin + 0.05 * x_indices, y_origin + 0.05 * (x_shape-y_indices)
# # Print or process the coordinates
# # print("Coordinates with values above", white_threshold, ":")
# # for x, y in zip(x_indices, y_indices):
# #     print(f"({x}, {y})")

# y_indices, x_indices = np.where(img_array < white_threshold)
# x_origin, y_origin = -6.13, -4.49
# x_indices, y_indices = y_shape, x_shape  # X coordinates
# x_coord, y_coord = x_origin + 0.05 * x_indices, y_origin + 0.05 * (x_shape-y_indices)

# print(x_coord, y_coord)
# print("Coordinates with values above", white_threshold, ":")
# #for x, y in zip(x_indices, y_indices):
#     #print(f"({x}, {y})")
#print(max(x_indices))
# Optional: Show the image with matplotlib to visually confirm the points
plt.imshow(img_array, cmap='gray')
plt.colorbar()
plt.scatter(x_indices, y_indices, color='red', s=1)  # Mark the points in red
plt.show()
