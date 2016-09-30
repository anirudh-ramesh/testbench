import cv2 as cv
import numpy as np
import argparse

bins = np.arange(256).reshape(256,1)

messageStrings = {
	1: ('Invalid'),
	2: ('Conflict'),
}

components = {
	0: ('Blue'),
	1: ('Green'),
	2: ('Red'),
	3: ('Grey'),
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
			histogram1Array = cv.calcHist([frame1], [0], None, [256], [0,256])
		else:
			histogram1Array = cv.calcHist([frame3[x]], [0], None, [256], [0,256])

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

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='')

	parser.add_argument('frameIn', help='<frameIn>')
	parser.add_argument('frameOut', help='<frameOut>')
	parser.add_argument('method', help='<method>')
	parser.add_argument('-dump', help='<dump>')
	parser.add_argument('-sigma', help='<sigma>')
	parser.add_argument('-threshold', help='<threshold>')

	args = parser.parse_args()

	if (args.sigma != None) & (args.method == 'Canny'):

		try:

			sigma = float(args.sigma)

			if (sigma > 1) | (sigma < 0):

				throwError(1, 'Value out of bounds')

		except ValueError:

			throwError(1, 'Unable to parse')

	elif args.sigma:

		throwError(2, args.method + ' cannot have a sigma')

	else:

		sigma = 0.33

	if (args.threshold != None) & (args.method == 'Canny'):

		try:

			threshold = [int(x) for x in args.threshold.split(',')]

		except ValueError:

			throwError(1, 'Unable to parse')

		if len(threshold) != 2:

			throwError(2, 'Canny\'s threshold must have two figures')

		for x in [0, 1]:

			if (threshold[x] > 255) | (threshold[x] < 0):

				throwError(1, 'Value out of bounds')

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

	frame = cv.imread(args.frameIn)
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

	if args.method == 'None':

		exit()

	elif args.method == 'Histogram':

		cv.imwrite(args.frameOut + 'Histogram.png', saveHistogram(frame_Channels, frame_Greyscale))

		exit()

	else:

		for x in range(0, 3):

			cv.imwrite(args.frameOut + args.method + str(x) + '.png', doSegment(args.method, frame_Channels[x], override, threshold, sigma))

		cv.imwrite(args.frameOut + args.method + 'X.png', doSegment(args.method, frame_Greyscale, override, threshold, sigma))