import argparse

import pandas as pd

from sklearn.datasets import load_boston


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Data loader parser')
	parser.add_argument('path', type=str, help='Path to save data')
	parser.add_argument('--target', default='target', type=str, help='Name for target variable')
	parser.add_argument('--separator', default=';', type=str, help='Separator for data storage')

	args = parser.parse_args()

	path = args.path
	target_name = args.target
	sep = args.separator

	data = load_boston()
	X = data['data']
	target = data['target']
	features = data['feature_names']

	X = pd.DataFrame(X, columns=features)
	X[target_name] = target

	X.to_csv(path, sep=sep, index=False)
