import cv2 as cv
import numpy as np
import glob
import math
import re
import scipy.ndimage

bins = np.arange(256).reshape(256,1)

messageStrings = {
	1: ('Invalid'),
	2: ('Conflict'),
	3: ('Incomplete'),
}

components = {
	0: ('Blue'),
	1: ('Green'),
	2: ('Red'),
	3: ('Grey'),
}

palettes = {
	'Autumn': 0,
	'Bone': 1,
	'Jet': 2,
	'Winter': 3,
	'Rainbow': 4,
	'Ocean': 5,
	'Summer': 6,
	'Spring': 7,
	'Cool': 8,
	'HSV': 9,
	'Pink': 10,
	'Hot': 11,
}

def throwError(messageIndex, reason):

	print 'Argument ' + messageStrings[messageIndex] + ': ' + reason
	exit()

def padZero(frame, count, intensity):

	return cv.copyMakeBorder(frame, count, count, count, count, cv.BORDER_CONSTANT, value=[intensity, intensity, intensity])

def saveHistogram(frame3, frame1):

	histogramImage = np.ones((300, 256, 3)) * 255

	for x in range(0, 4):

		if x == 3:
			histogram1Array = cv.calcHist([frame1], [0], None, [256], [0, 256])
		else:
			histogram1Array = cv.calcHist([frame3[x]], [0], None, [256], [0, 256])

		cv.normalize(histogram1Array, histogram1Array, 0, 255, cv.NORM_MINMAX)
		histogram1Array = np.int32(np.around(histogram1Array))
		cv.polylines(histogramImage, [np.int32(np.column_stack((bins, histogram1Array)))], False, (255 * (x == 0), 255 * (x == 1), 255 * (x == 2)))

		print components[x] + ' ' + str(histogram1Array.argmax())

	return padZero(np.flipud(histogramImage), 10, 255)

def doSegment(method, frame, override, threshold, sigma):

	if method == 'Canny':

		if override is True:
			return cv.Canny(frame, threshold[0], threshold[1])
		else:
			medianIntensity = np.median(frame)
			return cv.Canny(frame, int(max(0, (1.0 - sigma) * medianIntensity)), int(min(255, (1.0 + sigma) * medianIntensity)))

	elif method == 'Binarize':

		if override is True:
			_, binary = cv.threshold(frame, threshold, 255, 0)
		else:
			_,binary = cv.threshold(frame, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
		return binary

	else:

		return np.zeros((10, 10, 1), np.uint8)

def doAffine(frame, method, scale, angle):

	height, width, matrices = frame.shape

	if method == 'Rotate':

		sineFactor = math.sin(angle * np.pi / 180)
		cosineFactor = math.cos(angle * np.pi / 180)

		heightTarget = int(width * sineFactor + height * cosineFactor)
		widthTarget = int(width * cosineFactor + height * sineFactor)

		warpMatrix = np.float32([[1, 0, (widthTarget - width) / 2], [0, 1, (heightTarget - height) / 2]])
		rotateMatrix = cv.getRotationMatrix2D((widthTarget / 2, heightTarget / 2), angle, 1)

		return cv.warpAffine(cv.warpAffine(frame, warpMatrix, tuple([heightTarget, widthTarget])), rotateMatrix, tuple([heightTarget, widthTarget]))

	elif method == 'Shear':

		factor = math.sin(angle * np.pi / 180)

		sourceTriangle = np.array([(0, 0), (width - 1, 0), (0, height - 1)], np.float32)
		destinationTriangle = np.array([(0, 0), (width - 1, 0), (width * factor, height - 1)], np.float32)

		warpMatrix = cv.getAffineTransform(sourceTriangle, destinationTriangle)

		return cv.warpAffine(frame, warpMatrix, (int(width * factor) + width, height))

	elif method == 'Stretch':

		sourceTriangle = np.array([(0, 0), (width - 1, 0), (0, height - 1)], np.float32)
		destinationTriangle = np.array([(0, 0), (int(width * scale[0]) + width - 1, 0), (0, int(height * scale[1]) + height - 1)], np.float32)

		warpMatrix = cv.getAffineTransform(sourceTriangle, destinationTriangle)

		return cv.warpAffine(frame, warpMatrix, (int(width * scale[0]) + width, int(height * scale[1]) + height))

def swapChannels(frame_Channels, index0, index1):

	c = frame_Channels[index0]
	frame_Channels[index0] = frame_Channels[index1]
	frame_Channels[index1] = c

	return cv.merge(frame_Channels)

def dropChannels(frame_Channels, index):

	height, width = frame_Channels[0].shape

	frame_Channels_copy = list(frame_Channels)
	frame_Channels_copy[index] = np.ones([height, width], np.uint8) * 128

	return cv.merge(frame_Channels_copy)

def enlarge(frame_Channels, scale):

	for x in range(0, 3):

		frame_Channels[x] = scipy.ndimage.zoom(frame_Channels[x], scale, order=3)

	return cv.merge(frame_Channels)

def trim(frame):

	pixels = np.array(np.where(frame != 255))

	return frame[pixels[0, 0]:pixels[0, -1], min(pixels[1, :]):max(pixels[1, :])]

def getLUT(fileName):

	paletteLUT = np.zeros((256, 1, 3), dtype=np.uint8)

	paletteFileHandle = open(fileName, 'r')

	position = 0
	for entry in paletteFileHandle.readlines():
		intensitites = entry.strip('\n').split(' ')
		for channels in range(0, 3):
			paletteLUT[position, 0, channels] = intensitites[2 - channels]
		position += 1

	paletteFileHandle.close()

	return paletteLUT

def processFrame(arguments):

# Argument Handling: Angle

	if (arguments.angle != None) & (arguments.method in ['Rotate', 'Shear']):

		try:

			angle = float(arguments.angle)

			if angle < 0:

				throwError(1, 'Negative angles are not supported')

			if angle > 180:

				throwError(1, 'Value out of domain')

		except ValueError:

			throwError(1, 'Unable to parse')

	elif (arguments.angle != None):

		throwError(2, arguments.method + ' cannot have an angle')

	else:

		angle = 2

# Argument Handling: Direction

	if (arguments.direction != None) & (arguments.method == 'Flip'):

		try:

			direction = int(arguments.direction)

			if direction not in [-1, 0, 1]:

				throwError(1, 'Value out of domain')

		except ValueError:

			throwError(1, 'Unable to parse')

	elif (arguments.direction != None) & (arguments.method == 'Stitch'):

		try:

			direction = arguments.direction

			if direction not in ['H', 'V']:

				throwError(1, 'Value out of domain')

		except ValueError:

			throwError(1, 'Unable to parse')

	elif arguments.direction:

		throwError(2, arguments.method + ' cannot have a direction')

	elif (arguments.direction == None) & (arguments.method == 'Stitch'):

		throwError(3, arguments.method + ' requires a direction')

	else:

		direction = -1

# Argument Handling: Operand

	if (arguments.operand == None) & (arguments.method == 'Stitch'):

		throwError(3, arguments.method + ' requires another operand')

	if (arguments.operand != None) & (arguments.method != 'Stitch'):

		throwError(2, arguments.method + ' cannot have another operand')

# Argument Handling: Palette

	if (arguments.palette == None) & (arguments.method == 'Colorize'):

		throwError(3, arguments.method + ' requires a choice of palette')

	elif (arguments.palette != None) & (arguments.method != 'Colorize'):

		throwError(2, arguments.method + ' cannot have a choice of palette')

# Argument Handling: Scale

	if (arguments.scale != None) & (arguments.method == 'Enlarge'):

		try:

			scale = float(arguments.scale)

			if (scale <= 1):

				throwError(1, 'Value out of domain')

		except ValueError:

			throwError(1, 'Unable to parse')

	elif (arguments.scale != None) & (arguments.method == 'Stretch'):

		try:

			scale = [float(i) for i in arguments.scale.split('x')]

			if (len(scale) != 2):

				throwError(1, 'Unable to parse')

			if (scale[0] <= 0.0) | (scale[1] <= 0.0):

				throwError(1, 'Value out of domain')

			if (scale[0] == 1.0) | (scale[1] == 1.0):

				throwError(1, 'Value out of domain')

		except ValueError:

			throwError(1, 'Unable to parse')

	elif (arguments.scale == None) & (arguments.method in ['Stretch']):

		scale = [0.2, 0.2]

	elif arguments.scale:

		throwError(2, arguments.method + ' cannot have a scale factor')

	else:

		scale = 2.0

# Argument Handling: Sigma

	if (arguments.sigma != None) & (arguments.method == 'Canny'):

		try:

			sigma = float(arguments.sigma)

			if (sigma > 1) | (sigma < 0):

				throwError(1, 'Value out of domain')

		except ValueError:

			throwError(1, 'Unable to parse')

	elif arguments.sigma:

		throwError(2, arguments.method + ' cannot have a sigma')

	else:

		sigma = 0.33

# Argument Handling: Dimensions

	if (arguments.dimension != None) & (arguments.method == 'Crop'):

		try:

			dimension = [int(i) for i in arguments.dimension.split('x')]

			if (len(dimension) != 4):

				throwError(1, 'Unable to parse')

		except ValueError:

			throwError(1, 'Unable to parse')

	elif arguments.dimension:

		throwError(2, arguments.method + ' cannot have a dimension')

	elif (arguments.dimension == None) & (arguments.method == 'Crop'):

		throwError(2, arguments.method + ' must have a dimension')

	else:

		dimension = -1

# Argument Handling: Threshold

	if (arguments.threshold != None) & (arguments.method == 'Canny'):

		try:

			threshold = [int(x) for x in arguments.threshold.split(',')]

		except ValueError:

			throwError(1, 'Unable to parse')

		if len(threshold) != 2:

			throwError(2, 'Canny\'s threshold must have two figures')

		for x in [0, 1]:

			if (threshold[x] > 255) | (threshold[x] < 0):

				throwError(1, 'Value out of domain')

		override = True

	elif (arguments.threshold != None) & (arguments.method == 'Binarize'):

		try:

			threshold = int(arguments.threshold)

		except ValueError:

			throwError(1, 'Unable to parse')

		if threshold > 0:

			override = True

		else:

			threshold = None
			override = False

	elif arguments.threshold:

		throwError(2, arguments.method + ' cannot have a threshold')

	else:

		threshold = None
		override = False

# I/O Transactions

	frame = cv.imread(arguments.frameIn)

	if frame is None:

		throwError(1, 'Unable to find / open ' + arguments.frameIn)

	height, width, _ = frame.shape

	frame_Channels = cv.split(frame)
	frame_Greyscale = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

	if arguments.dump:

		if arguments.dump == 'BGR':

			for x in range(0, 3):

				cv.imwrite(arguments.frameOut + '_BGR_Dump' + str(x) + '.png', frame_Channels[x])

		if arguments.dump == 'Grey':

			cv.imwrite(arguments.frameOut + '_Grey_Dump' + 'X.png', frame_Greyscale)

		if arguments.dump == 'YVU':

			frame_Components = cv.split(cv.cvtColor(frame, cv.COLOR_BGR2YCrCb))

			for x in range(0, 3):

				cv.imwrite(arguments.frameOut + '_YVU_Dump' + str(x) + '.png', frame_Components[x])

		if arguments.dump == 'HSV':

			frame_Components = cv.split(cv.cvtColor(frame, cv.COLOR_BGR2HSV))

			for x in range(0, 3):

				cv.imwrite(arguments.frameOut + '_HSV_Dump' + str(x) + '.png', frame_Components[x])

	if arguments.search:

		for (i, functionName) in enumerate(filter(lambda x: re.search(".*{}.*".format(arguments.search), x, re.IGNORECASE), dir(cv))):
			print("{}. {}".format(i + 1, functionName))

	if arguments.method == 'None':

		pass

	elif arguments.method == 'Crop':

		if (dimension[0] < 0) | (dimension[0] > width):

			throwError(1, 'Value out of domain')

		elif (dimension[1] < 0) | (dimension[1] > height):

			throwError(1, 'Value out of domain')

		elif (dimension[2] < 0) | (dimension[2] > width - dimension[0]):

			throwError(1, 'Value out of domain')

		elif (dimension[3] < 0) | (dimension[3] > height - dimension[1]):

			throwError(1, 'Value out of domain')

		cv.imwrite(arguments.frameOut + arguments.method + '.png', frame[dimension[1]:dimension[1] + dimension[3], dimension[0]:dimension[0] + dimension[2]])

	elif arguments.method == 'Histogram':

		cv.imwrite(arguments.frameOut + arguments.method + '.png', saveHistogram(frame_Channels, frame_Greyscale))

	elif arguments.method == 'Enlarge':

		cv.imwrite(arguments.frameOut + arguments.method + '.png', enlarge(frame_Channels, scale))

	elif arguments.method == 'Negate':

		cv.imwrite(arguments.frameOut + arguments.method + '.png', cv.bitwise_not(frame))

	elif arguments.method == 'Trim':

		cv.imwrite(arguments.frameOut + arguments.method + '.png', trim(frame))

	elif arguments.method == 'Flip':

		cv.imwrite(arguments.frameOut + arguments.method + str(direction + 1) + '.png', cv.flip(frame, direction))

	elif arguments.method == 'ChannelSwap':

		for x in range(0, 3):

			cv.imwrite(arguments.frameOut + arguments.method + str(x) + '.png', swapChannels(frame_Channels, x, (x + 1) % 3))

	elif arguments.method == 'ChannelDrop':

		for x in range(0, 3):

			cv.imwrite(arguments.frameOut + arguments.method + str(x) + '.png', dropChannels(frame_Channels, x))

	elif arguments.method == 'Stitch':

		frame_Stitch = cv.imread(arguments.operand)

		if frame_Stitch is None:

			throwError(1, 'Unable to find / open ' + arguments.operand)

		if direction is 'H':

			try:

				frameOut = np.hstack((frame, frame_Stitch))

			except ValueError:

				throwError(1, 'Inappropriate frame dimension')

		elif direction is 'V':

			try:

				frameOut = np.vstack((frame, frame_Stitch))

			except ValueError:

				throwError(1, 'Inappropriate frame dimension')

		cv.imwrite(arguments.frameOut + arguments.method + direction + '.png', frameOut)

	elif arguments.method in ['Rotate', 'Shear', 'Stretch']:

		cv.imwrite(arguments.frameOut + arguments.method + '.png', doAffine(frame, arguments.method, scale, angle))

	elif arguments.method in ['Binarize', 'Canny']:

		for x in range(0, 3):

			cv.imwrite(arguments.frameOut + arguments.method + str(x) + '.png', doSegment(arguments.method, frame_Channels[x], override, threshold, sigma))

		cv.imwrite(arguments.frameOut + arguments.method + 'X.png', doSegment(arguments.method, frame_Greyscale, override, threshold, sigma))

	elif arguments.method == 'Colorize':

		if (arguments.palette == '?'):

			for name, number in palettes.items():

				print name

		elif (arguments.palette == '*'):

			for name, number in palettes.items():

				cv.imwrite(arguments.frameOut + arguments.method + name + '.png', cv.applyColorMap(frame_Greyscale, number))

		elif (len(arguments.palette.split('.plt')) == 2):

			paletteName = arguments.palette.strip('.plt')

			if (paletteName == '*'):

				for paletteFileName in glob.glob('*.plt'):

					cv.imwrite(arguments.frameOut + arguments.method + paletteFileName.strip('.plt') + '.png', cv.LUT(frame, getLUT(paletteFileName)))

			else:

				cv.imwrite(arguments.frameOut + arguments.method + paletteName + '.png', cv.LUT(frame, getLUT(arguments.palette)))

		elif (arguments.palette not in palettes):

			throwError(1, 'Unknown palette')

		else:

			cv.imwrite(arguments.frameOut + arguments.method + arguments.palette + '.png', cv.applyColorMap(frame_Greyscale, palettes[arguments.palette]))

	else:

		throwError(1, 'Unknown method')

# Add Blend

# Add Skeletonize
# Add Timestamp

# Add Function-Search