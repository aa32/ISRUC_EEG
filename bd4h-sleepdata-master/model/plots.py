import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import *


PLOT_PATH = './output/'


def classification_metrics(Y_pred, Y_true):
	acc = accuracy_score(Y_true, Y_pred)
	precision = precision_score(Y_true, Y_pred, average='weighted')
	recall = recall_score(Y_true, Y_pred, average='weighted')
	f1 = f1_score(Y_true, Y_pred, average='weighted')
	return acc, precision, recall, f1


def print_metrics(model_type, Y_pred, Y_true, save=False):
	acc, precision, recall, f1score = classification_metrics(Y_pred,Y_true)
	print(("Accuracy: {}".format(str(acc))))
	print(("Precision: {}".format(str(precision))))
	print(("Recall: {}".format(str(recall))))
	print(("F1-score: {}".format(str(f1score))))

	if save:
		out_dir = os.path.join(PLOT_PATH, model_type)
		if not os.path.exists(out_dir):
			os.mkdir(out_dir)
		with open(os.path.join(out_dir, 'metrics.txt'), 'w') as f:
			f.write(("Model type: {}\n".format(model_type)))
			f.write(("Accuracy: {}\n".format(str(acc))))
			f.write(("Precision: {}\n".format(str(precision))))
			f.write(("Recall: {}\n".format(str(recall))))
			f.write(("F1-score: {}\n".format(str(f1score))))


def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies, model_type):
	plt.plot(np.arange(len(train_losses)), train_losses, label='Train')
	plt.plot(np.arange(len(valid_losses)), valid_losses, label='Validation')
	plt.ylabel('Loss')
	plt.xlabel('epoch')
	plt.legend(loc="best")

	out_dir = os.path.join(PLOT_PATH, model_type)
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)
	plt.savefig(os.path.join(out_dir, 'loss_curve.png'))

	plt.clf()
	plt.plot(np.arange(len(train_accuracies)), train_accuracies, label='Train')
	plt.plot(np.arange(len(valid_accuracies)), valid_accuracies, label='Validation')
	plt.ylabel('Accuracy')
	plt.xlabel('epoch')
	plt.legend(loc="best")

	plt.savefig(os.path.join(out_dir, 'accuracy_curve.png'))


def plot_confusion_matrix(results, class_names, model_type):
	results = list(zip(*results))
	cm = confusion_matrix(results[0], results[1])
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	print(cm)
	plt.clf()
	plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title('Normalized Confusion Matrix')
	plt.colorbar()
	tick_marks = np.arange(len(class_names))
	plt.xticks(tick_marks, class_names, rotation=45)
	plt.yticks(tick_marks, class_names)

	fmt = '.2f'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
					horizontalalignment="center",
					color="white" if cm[i, j] > thresh else "black")

	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.tight_layout()

	out_dir = os.path.join(PLOT_PATH, model_type)
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)

	plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))