from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt

result = pd.read_csv('result_fedavg.csv')
disp = ConfusionMatrixDisplay.from_predictions(result['label'],result['pred'], display_labels=['Bad','Good'],cmap=plt.cm.Blues)
plt.show()