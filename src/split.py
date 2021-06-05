import yaml
import argparse

import pandas as pd

from sklearn.model_selection import train_test_split


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Feature engineering parser')
	parser.add_argument('path', type=str, help='Path for dataset')
	parser.add_argument('config', type=str, help='Config yaml path')
	parser.add_argument('train_output', type=str, help='Output train dataset path')
	parser.add_argument('test_output', type=str, help='Output test dataset path')
	parser.add_argument('--separator', default=';', type=str, help='Separator for dataset')

	args = parser.parse_args()

	data = pd.read_csv(args.path, sep=args.separator)

	with open(args.config, 'r') as f:
		config = yaml.safe_load(f)['split']

	X_train, X_test = train_test_split(data, test_size=config['test_size'], shuffle=config['shuffle'], random_state=config['random_state'])

	X_train.to_csv(args.train_output, sep=args.separator, index=False)
	X_test.to_csv(args.test_output, sep=args.separator, index=False)
