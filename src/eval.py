import json
import joblib
import argparse

import pandas as pd

from sklearn import metrics


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Feature engineering parser')
	parser.add_argument('test_path', type=str, help='Path for dataset')
	parser.add_argument('model_path', type=str, help='path to store model')
	parser.add_argument('scores', type=str, help='scores path')
	parser.add_argument('--separator', default=';', type=str, help='Separator for dataset')

	args = parser.parse_args()

	test = pd.read_csv(args.test_path, sep=args.separator)

	target = 'target'
	features = test.columns.tolist()
	features.remove(target)

	clf = joblib.load(args.model_path)
	y_pred = clf.predict(test[features])

	results = {
		'mae': metrics.mean_absolute_error(test[target], y_pred),
		'mse': metrics.mean_squared_error(test[target], y_pred),
	}

	with open(args.scores, 'w') as f:
		json.dump(results, f)
