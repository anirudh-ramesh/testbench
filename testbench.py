import cv2 as cv
import numpy as np
import argparse
import math
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

	else:

		return np.zeros((10, 10, 1), np.uint8)

def swapChannels(frame_Channels, index0, index1):

	c = frame_Channels[index0]
	frame_Channels[index0] = frame_Channels[index1]
	frame_Channels[index1] = c

	return cv.merge(frame_Channels)

def enlarge(frame_Channels, scale):

	for x in range(0, 3):

		frame_Channels[x] = scipy.ndimage.zoom(frame_Channels[x], scale, order=3)

	return cv.merge(frame_Channels)

def processFrame(args):

# Argument Handling: Angle

	if (args.angle != None) & (args.method in ['Rotate', 'Shear']):

		try:

			angle = float(args.angle)

			if angle < 0:

				throwError(1, 'Negative angles are not supported')

			if angle > 180:

				throwError(1, 'Value out of domain')

		except ValueError:

			throwError(1, 'Unable to parse')

	elif (args.angle != None):

		throwError(2, args.method + ' cannot have an angle')

	else:

		angle = 2

# Argument Handling: Direction

	if (args.direction != None) & (args.method == 'Flip'):

		try:

			direction = int(args.direction)

			if direction not in [-1, 0, 1]:

				throwError(1, 'Value out of domain')

		except ValueError:

			throwError(1, 'Unable to parse')

	if (args.direction != None) & (args.method == 'Stitch'):

		try:

			direction = args.direction

			if direction not in ['H', 'V']:

				throwError(1, 'Value out of domain')

		except ValueError:

			throwError(1, 'Unable to parse')

	elif args.direction:

		throwError(2, args.method + ' cannot have a direction')

	elif (args.direction == None) & (args.method == 'Stitch'):

		throwError(3, args.method + ' requires a direction')

	else:

		direction = -1

# Argument Handling: Operand

	if (args.operand == None) & (args.method == 'Stitch'):

		throwError(3, args.method + ' requires another operand')

	if (args.operand != None) & (args.method != 'Stitch'):

		throwError(2, args.method + ' cannot have another operand')

# Argument Handling: Palette

	if (args.palette == None) & (args.method == 'Palette'):

		throwError(3, args.method + ' requires a choice of palette')

	elif (args.palette != None) & (args.method != 'Palette'):

		throwError(2, args.method + ' cannot have a choice of palette')

# Argument Handling: Scale

	if (args.scale != None) & (args.method == 'Enlarge'):

		try:

			scale = float(args.scale)

			if (scale <= 1):

				throwError(1, 'Value out of domain')

		except ValueError:

			throwError(1, 'Unable to parse')

	elif (args.scale != None) & (args.method == 'Stretch'):

		try:

			scale = [float(i) for i in args.scale.split('x')]

			if (len(scale) != 2):

				throwError(1, 'Unable to parse')

			if (scale[0] <= 0.0) | (scale[1] <= 0.0):

				throwError(1, 'Value out of domain')

			if (scale[0] == 1.0) | (scale[1] == 1.0):

				throwError(1, 'Value out of domain')

		except ValueError:

			throwError(1, 'Unable to parse')

	elif (args.scale == None) & (args.method in ['Stretch']):

		scale = [0.2, 0.2]

	elif args.scale:

		throwError(2, args.method + ' cannot have a scale factor')

	else:

		scale = 2.0

# Argument Handling: Sigma

	if (args.sigma != None) & (args.method == 'Canny'):

		try:

			sigma = float(args.sigma)

			if (sigma > 1) | (sigma < 0):

				throwError(1, 'Value out of domain')

		except ValueError:

			throwError(1, 'Unable to parse')

	elif args.sigma:

		throwError(2, args.method + ' cannot have a sigma')

	else:

		sigma = 0.33

# Argument Handling: Threshold

	if (args.threshold != None) & (args.method == 'Canny'):

		try:

			threshold = [int(x) for x in args.threshold.split(',')]

		except ValueError:

			throwError(1, 'Unable to parse')

		if len(threshold) != 2:

			throwError(2, 'Canny\'s threshold must have two figures')

		for x in [0, 1]:

			if (threshold[x] > 255) | (threshold[x] < 0):

				throwError(1, 'Value out of domain')

		override = True

	elif (args.threshold != None) & (args.method == 'Binarize'):

		try:

			threshold = int(args.threshold)

		except ValueError:

			throwError(1, 'Unable to parse')

		if threshold > 0:

			override = True

		else:

			threshold = None
			override = False

	elif args.threshold:

		throwError(2, args.method + ' cannot have a threshold')

	else:

		threshold = None
		override = False

# I/O Transactions

	frame = cv.imread(args.frameIn)

	if frame is None:

		throwError(1, 'Unable to find / open ' + args.frameIn)

	frame_Channels = cv.split(frame)
	frame_Greyscale = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

	if args.dump:

		if args.dump == 'BGR':

			for x in range(0, 3):

				cv.imwrite(args.frameOut + '_BGR_Dump' + str(x) + '.png', frame_Channels[x])

		if args.dump == 'Grey':

			cv.imwrite(args.frameOut + '_Grey_Dump' + 'X.png', frame_Greyscale)

		if args.dump == 'YVU':

			frame_Components = cv.split(cv.cvtColor(frame, cv.COLOR_BGR2YCrCb))

			for x in range(0, 3):

				cv.imwrite(args.frameOut + '_YVU_Dump' + str(x) + '.png', frame_Components[x])

		if args.dump == 'HSV':

			frame_Components = cv.split(cv.cvtColor(frame, cv.COLOR_BGR2HSV))

			for x in range(0, 3):

				cv.imwrite(args.frameOut + '_HSV_Dump' + str(x) + '.png', frame_Components[x])

	if args.method == 'None':

		pass

	elif args.method == 'Histogram':

		cv.imwrite(args.frameOut + args.method + '.png', saveHistogram(frame_Channels, frame_Greyscale))

	elif args.method == 'Enlarge':

		cv.imwrite(args.frameOut + args.method + '.png', enlarge(frame_Channels, scale))

	elif args.method == 'Negate':

		cv.imwrite(args.frameOut + args.method + '.png', cv.bitwise_not(frame))

	elif args.method == 'Flip':

		cv.imwrite(args.frameOut + args.method + str(direction + 1) + '.png', cv.flip(frame, direction))

	elif args.method == 'ChannelSwap':

		for x in range(0, 3):

			cv.imwrite(args.frameOut + args.method + str(x) + '.png', swapChannels(frame_Channels, x, (x + 1) % 3))

	elif args.method == 'Stitch':

		frame_Stitch = cv.imread(args.operand)

		if frame_Stitch is None:

			throwError(1, 'Unable to find / open ' + args.operand)

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

		cv.imwrite(args.frameOut + args.method + direction + '.png', frameOut)

	elif args.method in ['Rotate', 'Shear', 'Stretch']:

		cv.imwrite(args.frameOut + args.method + '.png', doAffine(frame, args.method, scale, angle))

	elif args.method in ['Binarize', 'Canny']:

		for x in range(0, 3):

			cv.imwrite(args.frameOut + args.method + str(x) + '.png', doSegment(args.method, frame_Channels[x], override, threshold, sigma))

		cv.imwrite(args.frameOut + args.method + 'X.png', doSegment(args.method, frame_Greyscale, override, threshold, sigma))

	elif args.method == 'Palette':

		if (args.palette == '?'):

			for name, number in palettes.items():

				print name

		elif (args.palette == '*'):

			for name, number in palettes.items():

				cv.imwrite(args.frameOut + args.method + name + '.png', cv.applyColorMap(frame_Greyscale, number))

		else:

			cv.imwrite(args.frameOut + args.method + args.palette + '.png', cv.applyColorMap(frame_Greyscale, palettes[args.palette]))

	else:

		throwError(1, 'Unknown method')

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='')

	parser.add_argument('frameIn', help='<frameIn>')
	parser.add_argument('frameOut', help='<frameOut>')
	parser.add_argument('method', help='<method>')
	parser.add_argument('-angle', help='<angle>')
	parser.add_argument('-direction', help='<direction>')
	parser.add_argument('-dump', help='<dump>')
	parser.add_argument('-operand', help='<operand>')
	parser.add_argument('-palette', help='<palette>')
	parser.add_argument('-scale', help='<scale>')
	parser.add_argument('-sigma', help='<sigma>')
	parser.add_argument('-threshold', help='<threshold>')

	args = parser.parse_args()

	processFrame(args)

# Add Blend

# Add False-Color (Custom)
# Add Skeletonize
# Add Timestamp

# Add Function-Search