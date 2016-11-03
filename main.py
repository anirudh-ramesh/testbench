from testbench import processFrame
import argparse

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='')

	parser.add_argument('frameIn', help='<frameIn>')
	parser.add_argument('frameOut', help='<frameOut>')
	parser.add_argument('method', help='<method>')
	parser.add_argument('-angle', help='<angle>')
	parser.add_argument('-colour', help='<colour>')
	parser.add_argument('-direction', help='<direction>')
	parser.add_argument('-dump', help='<dump>')
	parser.add_argument('-operand', help='<operand>')
	parser.add_argument('-palette', help='<palette>')
	parser.add_argument('-scale', help='<scale>')
	parser.add_argument('-sigma', help='<sigma>')
	parser.add_argument('-search', help='<search>')
	parser.add_argument('-start', help='<start>')
	parser.add_argument('-thickness', help='<thickness>')
	parser.add_argument('-threshold', help='<threshold>')

	args = parser.parse_args()

	processFrame(args)