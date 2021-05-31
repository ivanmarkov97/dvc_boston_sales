import yaml
import argparse

import numpy as np
import pandas as pd


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Feature engineering parser')
	parser.add_argument('path', type=str, help='Path for dataset')
	parser.add_argument('config', type=str, help='Config yaml path')
	parser.add_argument('output', type=str, help='Output dataset path')
	parser.add_argument('--separator', default=';', type=str, help='Separator for dataset')

	args = parser.parse_args()

	data = pd.read_csv(args.path, sep=args.separator)

	with open(args.config, 'r') as f:
		config = yaml.safe_load(f)['engineering']
		log = config['log']
		if log:
			log_features = config['log_features']
		else:
			log_features = []

	for feature in log_features:
		data[feature] = data[feature].map(np.log1p)

	data.to_csv(args.output, sep=args.separator, index=False)
