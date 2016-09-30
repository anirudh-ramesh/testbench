import cv2 as cv
import numpy as np
import argparse

messageStrings = {
	1: ('Invalid'),
	2: ('Conflict'),
}

def throwError(messageIndex, reason):

	print 'Argument ' + messageStrings[messageIndex] + ': ' + reason
	exit()

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

	else:

		threshold = None
		override = False

	frame = cv.imread(args.frameIn)
	frame_Channels = cv.split(frame)
	frame_Greyscale = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

	cv.imwrite(args.frameOut + args.method + '0.png', doSegment(args.method, frame_Channels[0], override, threshold, sigma))
	cv.imwrite(args.frameOut + args.method + '1.png', doSegment(args.method, frame_Channels[1], override, threshold, sigma))
	cv.imwrite(args.frameOut + args.method + '2.png', doSegment(args.method, frame_Channels[2], override, threshold, sigma))
	cv.imwrite(args.frameOut + args.method + 'X.png', doSegment(args.method, frame_Greyscale, override, threshold, sigma))