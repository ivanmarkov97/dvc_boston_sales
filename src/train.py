import yaml
import joblib
import argparse

import pandas as pd

from sklearn.ensemble import RandomForestRegressor


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Feature engineering parser')
	parser.add_argument('train_path', type=str, help='Path for dataset')
	parser.add_argument('config', type=str, help='Config yaml path')
	parser.add_argument('model_path', type=str, help='path to store model')
	parser.add_argument('--separator', default=';', type=str, help='Separator for dataset')

	args = parser.parse_args()

	with open(args.config, 'r') as f:
		config = yaml.safe_load(f)['train']

	train = pd.read_csv(args.train_path, sep=args.separator)

	target = 'target'
	features = train.columns.tolist()
	features.remove(target)

	clf = RandomForestRegressor(n_estimators=config['n_estimators'], random_state=config['random_state'])
	clf.fit(train[features], train[target])

	joblib.dump(clf, open(args.model_path, 'wb'))
