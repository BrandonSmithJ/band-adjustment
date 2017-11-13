from train import train_network
import argparse 

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-s", "--source", help="Source sensor [VI, MSI, OLI, OLCI, AER]")
	parser.add_argument("-t", "--target", help="Target sensor [VI, MSI, OLI, OLCI, AER]")
	parser.add_argument("--filename", help="Name of file to convert")
	parser.add_argument("--datadir", help="Directory data is located in", default="Data")
	parser.add_argument("--preddir", help="Directory predictions should go in", default="Predictions")
	parser.add_argument("--builddir", help="Directory DNN build is located in", default="Build")
	parser.add_argument("--gridsearch", help="Flag to turn on hyperparameter gridsearch", default=False)

	args = parser.parse_args()
	train_network(
		sensor_source = args.source,
		sensor_target = args.target,
		data_path = args.datadir,
		save_path = args.preddir,
		build_path= args.builddir,
		filename  = args.filename,
		gridsearch= args.gridsearch,
	)