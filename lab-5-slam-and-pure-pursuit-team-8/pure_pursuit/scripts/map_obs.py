from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#filename = "/home/yihsuan/sim_ws/src/lab-5-slam-and-pure-pursuit-team-8/pure_pursuit/maps/levine_3rd.pgm"
filename = "/home/yihsuan/sim_ws/src/f1tenth_gym_ros/maps/final_race.pgm"
img = Image.open(filename)

img_array = np.array(img)
# x_start, x_end = 295, 305  # X coordinates
# y_start, y_end = 390, 400  # Y coordinates
# white_threshold = 200
# img_array[x_start:x_end, y_start:y_end][img_array[x_start:x_end, y_start:y_end] > white_threshold] = 0

# x_start, x_end = 285, 300
# y_start, y_end = 580, 590
# img_array[x_start:x_end, y_start:y_end][img_array[x_start:x_end, y_start:y_end] > white_threshold] = 0

# x_start, x_end = 370, 380
# y_start, y_end = 565, 575
# img_array[x_start:x_end, y_start:y_end][img_array[x_start:x_end, y_start:y_end] > white_threshold] = 0

# x_start, x_end = 240, 250
# y_start, y_end = 568, 578
# img_array[x_start:x_end, y_start:y_end][img_array[x_start:x_end, y_start:y_end] > white_threshold] = 0

# x_start, x_end = 410, 420
# y_start, y_end = 580, 592
# img_array[x_start:x_end, y_start:y_end][img_array[x_start:x_end, y_start:y_end] > white_threshold] = 0

# modified_img = Image.fromarray(img_array)
# #for i in    

# # Save the modified image
# modified_image_path = '/home/yihsuan/sim_ws/src/lab-5-slam-and-pure-pursuit-team-8/pure_pursuit/maps/levine_3rd_obs.pgm'
# modified_img.save(modified_image_path)

# #for i in    
# #img_array[300, 400] = 0
# #print(img_array)

plt.imshow(img_array, cmap='gray')  # Use the grayscale color map
plt.colorbar()  # Optionally add a colorbar to see the value mapping
plt.show()
