import cv2 as cv
import numpy as np
import argparse

def doCanny(frame, override, threshold, sigma):

	if override is True:
		try:
			return cv.Canny(frame, int(threshold.split(',')[0]), int(threshold.split(',')[1]))
		except IndexError:
			pass

	medianIntensity = np.median(frame)
	return cv.Canny(frame, int(max(0, (1.0 - sigma) * medianIntensity)), int(min(255, (1.0 + sigma) * medianIntensity)))

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='')

	parser.add_argument('frameIn', help='<frameIn>')
	parser.add_argument('frameOut', help='<frameOut>')
	parser.add_argument('-sigma', help='<sigma>')
	parser.add_argument('-threshold', help='<threshold>')

	args = parser.parse_args()

	if args.sigma:
		sigma = float(args.sigma)
	else:
		sigma = 0.33

	if args.threshold:
		threshold = args.threshold
		override = True
	else:
		threshold = None
		override = False

	frame = cv.imread(args.frameIn)
	frame_Channels = cv.split(frame)
	frame_Greyscale = 255 - cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

	cv.imwrite(args.frameOut + '0.png', doCanny(frame_Channels[0], override, threshold, sigma))
	cv.imwrite(args.frameOut + '1.png', doCanny(frame_Channels[1], override, threshold, sigma))
	cv.imwrite(args.frameOut + '2.png', doCanny(frame_Channels[2], override, threshold, sigma))
	cv.imwrite(args.frameOut + 'X.png', doCanny(frame_Greyscale, override, threshold, sigma))