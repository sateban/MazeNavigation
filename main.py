	# %reset
import ctypes
import numpy as np
import cv2
import time

def get_screen_size():
	user32 = ctypes.windll.user32
	screensize = (user32.GetSystemMetrics(0), user32.GetSystemMetrics(1))
	return screensize

# def combineImage(baseImage, overlayImage):
	# Load the JPG image
	# jpg_image = Image.open(baseImage)

	# # Load the PNG image with transparency
	# png_image = Image.open(overlayImage)

	# # Ensure both images have the same mode and alpha channel
	# jpg_image = jpg_image.convert('RGBA')
	# png_image = png_image.convert('RGBA')

	# # Resize the PNG image to fit onto the JPG image
	# png_image = png_image.resize(jpg_image.size, Image.LANCZOS)

	# # Composite the images, using the alpha channel of the PNG image as a mask
	# result_image = Image.alpha_composite(jpg_image, png_image)

	# # Save or display the result
	# result_image.show() 
	# result_image.save('./samples/overlayed_image.jpg')

def removeBlack(image):	
	# filename = './samples/maze.png'
	# image = cv2.imread(filename)
	# Convert the image to the HSV color space for better color manipulation
	# hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	# image = image.reshape(image.shape + (1,))
	hsv_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

	# Define the lower and upper bounds for the black color in HSV
	# lower_black = np.array([0, 0, 0], dtype=np.uint8)
	# upper_black = np.array([180, 255, 30], dtype=np.uint8)

	# # Create a mask for the black color in the image
	# black_mask = cv2.inRange(hsv_image, lower_black, upper_black)

	# # Invert the mask so that black becomes white and vice versa
	# inverse_mask = cv2.bitwise_not(black_mask)

	# Create a 4-channel image (BGRA) with an alpha channel
	h, w = image.shape[:2]
	bgra_image = np.zeros((h, w, 4), dtype=np.uint8)

	# Copy the RGB channels from the original image to the BGRA image
	# bgra_image[:, :, :3] = image
	# bgra_image[:, :, :3] = [255, 0, 0] # Change the color of White to other 
	bgra_image[:, :, :3][image != 0] = [255, 0, 0]
	
	bgra_image[:, :, 3] = image

	# cv2.imshow("Corners", bgra_image)

	# Set the alpha channel based on the inverse mask
	# bgra_image[:, :, 3] = inverse_mask

	return bgra_image

def detect_leading_lines(leadFrame):
	# Convert the frame to grayscale
	# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	# Apply GaussianBlur to reduce noise and help Canny edge detection
	blurred = cv2.GaussianBlur(leadFrame, (5, 5), 0)
	
	# Apply Canny edge detection
	edges = cv2.Canny(blurred, 50, 150)
	
	# Apply Hough Line Transform to detect lines in the image
	lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=30)
	
	# Check if there are lines and analyze them
	# Detects any line positionss
	# if lines is not None:
	#     for line in lines:
	#         x1, y1, x2, y2 = line[0]
	#         # You can perform further analysis or filtering based on the slope, position, etc.
	#         # For simplicity, let's just draw the lines on the original leadFrame
	#         cv2.line(leadFrame, (x1, y1), (x2, y2), (0, 255, 0), 2)

	# lower_line_y_coordinates = []
	# frame_height = leadFrame.shape[0]
	# max_avg = 0

	# if lines is not None:
	# 	for line in lines:
	# 		x1, y1, x2, y2 = line[0]
	# 		# print(f'max(y1, y2): {max(y1, y2)}')
	# 		max_avg += max(y1, y2)
	
	# max_avg = max_avg / len(lines)
	# # print(f'max_avg: ', max_avg)
	# # print(f'max_avg: ', max_avg)

	# # Detects horizontal lines only
	# if lines is not None:
	# 	# print("Lines", lines)
	# 	for line in lines:
	# 		x1, y1, x2, y2 = line[0]
			
	# 		# Calculate the slope of the line
	# 		slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Adding a small value to avoid division by zero
			
	# 		# Set a threshold to consider a line as horizontal
	# 		slope_threshold = 0.8

	# 		# Check if the line is approximately horizontal
	# 		if (abs(slope) < slope_threshold 
	# 			and max(y1, y2) > max_avg - 10 # Lowest part of the frame with line 
	# 			and max(y1, y2) > frame_height / 2): # Detect half portion of the frame only.
	# 			# Draw the line on the original leadFrame
	# 			cv2.line(leadFrame, (x1, y1), (x2, y2), (0, 255, 0), 2)
			# cv2.line(leadFrame, (100, 20), (50, 50), (0, 255, 0), 2)

			
	return lines

def harrisCorner(harrisImage):
	
	gray = cv2.cvtColor(harrisImage, cv2.COLOR_BGR2GRAY)
# # Find Harris corners
	gray = np.float32(gray)
	dst = cv2.cornerHarris(gray, 2, 3, 0.04)
	dst = cv2.dilate(dst, None)
	ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
	dst = np.uint8(dst)

# # Find centroids
	ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

# # Define the criteria to stop and refine the corners
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
	corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

# # Now draw them
	for i in range(len(corners)):
		x, y = corners[i]
		cv2.circle(harrisImage, (int(x), int(y)), 3, (0, 150, 255), -1)

	return harrisImage

lk_params = dict(winSize=(15, 15),
				 maxLevel=2,
				 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=20,
					  qualityLevel=0.3,
					  minDistance=10,
					  blockSize=7)

trajectory_len = 40
detect_interval = 5
trajectories = []
roi_trajectories = []
object_max_search = 0
frame_idx = 0

# vid_path = './maze/VID_20231226_124638.mp4'
# vid_path = './maze/VID_20231226_124740.mp4'
# vid_path = './maze/VID_20231226_124828.mp4'
# vid_path = './maze/VID_20231226_124843.mp4'
# vid_path = './maze/VID_20231226_124851.mp4'

# vid_path = './maze/from_start1_rotate.mp4'
# vid_path = './maze/start1.mp4'
# vid_path = './videos/boy-walking.mp4'
# vid_path = './maze/start1_steady.mp4'
vid_path = './maze/with_obstacle.mp4'
# vid_path = './maze/with_obstacle_steady.mp4'

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(vid_path)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# hog.setSVMDetector(cv2.HOGDescriptor_setSVMDetector())
# cv2.HOGDescriptor_setSVMDetector

# Create a term criteria for CAMShift
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
capCounter = 0

while True:
	suc, frame = cap.read()
	
	if not suc:
		print("There is No Frame ")
		break

	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame_graye = cv2.Canny(frame_gray, 100, 200)
	frame_canny_merge = frame_gray
	frame_hog = frame_gray.copy()
	img = frame #.copy()

	boxes, weights = hog.detectMultiScale(frame_hog, winStride=(8, 8), padding=(8, 8), scale=1.05)

	# Use the first detected person to initialize CAMShift tracking
	# if len(boxes) > 0:
	# 	x, y, w, h = boxes[0]
	# 	track_window = (x, y, w, h)

	for (x, y, w, h) in boxes:
		cv2.rectangle(frame_hog, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# Calculate optical flow for a sparse feature set using the iterative Lucas-Kanade Method
	if len(trajectories) > 0:
		# print("trajectories", len(trajectories[0]))

		img0, img1 = prev_gray, frame_graye
		p0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)
		
		# calcOpticalFlowPyrLK -> optical flow-based feature tracking algorithm
		p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
		p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
		d = abs(p0 - p0r).reshape(-1, 2).max(-1)
		good = d < 1

		new_trajectories = []
		object_trajectories = []
		isObjectDetected = False

		# Get all the trajectories
		for trajectory, (x, y), good_flag in zip(trajectories, p1.reshape(-1, 2), good):
			if not good_flag:
				continue
			trajectory.append((x, y))
			if len(trajectory) > trajectory_len:
				del trajectory[0]
			new_trajectories.append(trajectory)

			# Newest detected point
			# if len(object_trajectories) == 0:
			# 	object_trajectories.append(trajectory)

			if object_max_search < y and len(new_trajectories) > 200:
				object_trajectories.append(trajectory)
				print("Object Detected")
				cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)
				isObjectDetected = True
			else:
				isObjectDetected = False

			# print(len(trajectory))
				# print("Object Detected")

			# roi_trajectories

		trajectories = new_trajectories
		# trajectories = new_trajectories

		# Draw all the trajectories
		cv2.polylines(img, [np.int32(trajectory) for trajectory in trajectories], False, (0, 255, 0))
		cv2.putText(img, 'track count: %d' % len(trajectories), (20, 80), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
		if isObjectDetected:
			cv2.putText(img, 'Object Detected', (20, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
		else: 
			cv2.putText(img, '', (20, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

	# Update interval - When to update and detect new features
	if frame_idx % detect_interval == 0:
		mask = np.zeros_like(frame_graye)
		mask[:] = 255

		# Lastest point in latest trajectory
		for x, y in [np.int32(trajectory[-1]) for trajectory in trajectories]:
			# if object_max_search < y:
			cv2.circle(mask, (x, y), 3, 0, -1) # Desc: Dots in the Screen

		# Detect the good features to track -> Lukas-Kanade
		p = cv2.goodFeaturesToTrack(frame_graye, mask=mask, **feature_params)
		if p is not None:
			# If good features can be tracked - add that to the trajectories
			for x, y in np.float32(p).reshape(-1, 2):
				trajectories.append([(x, y)])

	frame_idx += 1
	prev_gray = frame_graye
	# print(img.shape)
	# print(mask.shape)
	# print(frame_graye.shape)

	# if img.shape[2] == 3:
	# 	img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

	# █▀ █ █ █▀█ █ █ █   █▀█ █▄ █ █   █▄█   █   █▀█ █ █ █ █▀▀ █▀█   █   █▀▀ ▄▀█ █▀▄ █ █▄ █ █▀▀   █   █ █▄ █ █▀▀ █▀
	# ▄█ █▀█ █▄█ ▀▄▀▄▀   █▄█ █ ▀█ █▄▄  █    █▄▄ █▄█ ▀▄▀▄▀ ██▄ █▀▄   █▄▄ ██▄ █▀█ █▄▀ █ █ ▀█ █▄█   █▄▄ █ █ ▀█ ██▄ ▄█
	lines = detect_leading_lines(frame_canny_merge)

	lower_line_y_coordinates = []
	frame_height = frame_canny_merge.shape[0]
	max_avg = 0

	if lines is not None:
		for line in lines:
			x1, y1, x2, y2 = line[0]
			# print(f'max(y1, y2): {max(y1, y2)}')
			max_avg += max(y1, y2)
	
	max_avg = max_avg / len(lines)
	# print(f'max_avg: ', max_avg)

	# Detects horizontal lines only
	if lines is not None:
		for line in lines:
			x1, y1, x2, y2 = line[0]
			
			# Calculate the slope of the line
			slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Adding a small value to avoid division by zero
			
			# Set a threshold to consider a line as horizontal
			slope_threshold = 0.8 # 0.2 and below is strictly horizontal, added 0.8 to make it read lines that are changing rotation during turning of camera

			# Check if the line is approximately horizontal
			if (abs(slope) < slope_threshold 
				and max(y1, y2) > max_avg - 10 # Lowest part of the frame with line 
				and max(y1, y2) > frame_height / 2): # Detect half portion of the frame only.
				# Draw the line on the original leadFrame
				cv2.line(frame_canny_merge, (x1, y1), (x2, y2), (0, 255, 0), 2)
				object_max_search = frame_height / 2

	# harrisCornerFrame = harrisCorner(frame)
	# cannyImgInv = removeBlack(frame_graye)

	if img.shape[2] == 3:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

	
	# █▀▀ █▀█ █▄ █ █ █ █▀▀ █▀█ ▀█▀   █ █▀▄▀█ █▀▀   ▀█▀ █▀█   █▄▄ █▀▀ █▀█ ▄▀█
	# █▄▄ █▄█ █ ▀█ ▀▄▀ ██▄ █▀▄  █    █ █ ▀ █ █▄█    █  █▄█   █▄█ █▄█ █▀▄ █▀█ to make the merging of windows possible
	img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA) # Lukas-Kanade
	mask = cv2.cvtColor(mask, cv2.COLOR_BGR2BGRA) # Lukas-Kanade
	frame_graye = cv2.cvtColor(frame_graye, cv2.COLOR_BGR2BGRA) # Canny
	frame_canny_merge = cv2.cvtColor(frame_canny_merge, cv2.COLOR_BGR2BGRA) # Canny
	frame_hog = cv2.cvtColor(frame_hog, cv2.COLOR_BGR2BGRA) # Canny

	# For No Video Frame Only
	m = np.zeros_like(frame_graye)
	m[:] = 100
	print(frame_graye.shape)
	hh = int(frame_graye.shape[0] / 2)
	ww = int(frame_graye.shape[1] / 2) - (5 * 35)
	
	# █▀█ █   ▄▀█ █▀▀ █▀▀   ▀█▀ █▀▀ ▀▄▀ ▀█▀   █▀▀ █▀█ █▀█   █▀▄ █▀▀ █▀ █▀▀ █▀█ █ █▀█ ▀█▀ █ █▀█ █▄ █
	# █▀▀ █▄▄ █▀█ █▄▄ ██▄    █  ██▄ █ █  █    █▀  █▄█ █▀▄   █▄▀ ██▄ ▄█ █▄▄ █▀▄ █ █▀▀  █  █ █▄█ █ ▀█
	cv2.putText(img, 'Using: Lukas-Kanade (flow-based feature tracking)', (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
	cv2.putText(mask, 'Using: Lukas-Kanade (flow estimation & good features tracking)', (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
	cv2.putText(frame_graye, 'Using: Canny (Edge Detection)', (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
	cv2.putText(frame_canny_merge, 'Using: Hough Lines (Leading Lines Detection)', (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 100, 0), 2)
	cv2.putText(frame_hog, 'Using: HOG (People Detector)', (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 100, 0), 2)
	cv2.putText(m, 'NO VIDEO', (ww, hh), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 100), 2)
	# print(h, w)
	# Combine Optical Flow, , and Canny (Edge Detection)
	# result = np.concatenate((img, mask, frame_graye), axis=1)  # Concatenate vertically
	# h = int(result.shape[1] / 1) # if using hd camera, divide by 3 instead else 2
	# w = int(result.shape[0] / 1)
	# resized_image = cv2.resize(result, (h, w))

	
	# █▀▀ █▀█ █▀▄▀█ █▄▄ █ █▄ █ █▀▀   █▀▀ █▀█ ▄▀█ █▀▄▀█ █▀▀   ▀█▀ █▀█   █▀█ █▄ █ █▀▀   █ █ █ █ █▄ █ █▀▄ █▀█ █ █ █
	# █▄▄ █▄█ █ ▀ █ █▄█ █ █ ▀█ ██▄   █▀  █▀▄ █▀█ █ ▀ █ ██▄    █  █▄█   █▄█ █ ▀█ ██▄   ▀▄▀▄▀ █ █ ▀█ █▄▀ █▄█ ▀▄▀▄▀
	top_row = np.concatenate((img, mask), axis=1)
	middle_row = np.concatenate((frame_graye, frame_canny_merge), axis=1)
	bottom_row = np.concatenate((frame_hog, m), axis=1)

	# Set the dimensions of the window you want to create
	window_width = get_screen_size()[0]
	window_height = get_screen_size()[1]

	combined_frames = np.concatenate((top_row, middle_row, bottom_row), axis=0)
	
	scaling_factor = min(window_height / combined_frames.shape[1], window_width / combined_frames.shape[0])
	h = window_width #int(combined_frames.shape[1] / 1.5) # if using hd camera, divide by 3 instead else 2
	w = window_height #int(combined_frames.shape[0] / 1.5)
	# resized_image = cv2.resize(combined_frames, (h, w))
	resized_image = cv2.resize(combined_frames, (int(combined_frames.shape[1] * scaling_factor), int(combined_frames.shape[0] * scaling_factor)))

	height, width = resized_image.shape[:2]
	
	x_position = int(abs(width - window_width) / 2)
	y_position = int(abs(height - window_height) / 2) - 30 #int(abs(height - window_height))

	
	# █▀ █ █ █▀█ █ █ █   █ █ █ █ █▄ █ █▀▄ █▀█ █ █ █
	# ▄█ █▀█ █▄█ ▀▄▀▄▀   ▀▄▀▄▀ █ █ ▀█ █▄▀ █▄█ ▀▄▀▄▀
	# Show Results, 
	cv2.imshow('Merged Frames', resized_image)
	cv2.moveWindow('Merged Frames', x_position, y_position)
	# cv2.imshow('Mask', mask)

	if cv2.waitKey(10) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()