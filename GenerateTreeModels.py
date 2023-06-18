import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree, ensemble, metrics


class Variable:
    def __init__(self):
        self.value = None
    def set(self, val):
        self.value = val


root = tk.Tk()
root.withdraw()

train_path = filedialog.askopenfilename()
test_path = filedialog.askopenfilename()

root.destroy()

train_xy = pd.read_csv(train_path)
test_xy = pd.read_csv(test_path)

i = 0
label_string = ''
for column in train_xy.columns:
    label_string = label_string + f"\n{i}: {column}"
    i += 1

window = tk.Tk()

number = Variable()

label = tk.Label(window, text=label_string +
                              "\n See the options above, which column should be used to make the prediction " +
                              "(enter a number and press 'submit'): ")
entry = tk.Entry(window, textvariable=number)


def callback():
    number.set(int(entry.get()))
    window.destroy()


button = tk.Button(window, text="Submit", command=callback)

label.pack()
entry.pack()
button.pack()
window.mainloop()

y_column = number.value

window = tk.Tk()
label = tk.Label(text=f"Using {train_xy.columns[y_column]} as the target.")
label.pack()
window.mainloop()

train_y = train_xy.iloc[:, y_column]
test_y = test_xy.iloc[:, y_column]

train_x = train_xy.drop(train_xy.columns[y_column], axis=1, inplace=False)
test_x = test_xy.drop(test_xy.columns[y_column], axis=1, inplace=False)

# I added in the default parameters, so you can adjust them (I put links to the documentation for each type of model)

# Regression tree parameters here:
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
regression_tree = tree.DecisionTreeRegressor(max_depth=5)

# Random forrest parameters here:
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
random_forrest = ensemble.RandomForestRegressor(n_estimators=100, criterion='squared_error', max_depth=None,
                                                min_samples_split=2, min_samples_leaf=1)

# Gradient boosting parameters here (using faster variant of the algorithm):
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html
gradient_boost = ensemble.HistGradientBoostingRegressor(loss='squared_error', quantile=None, learning_rate=0.1,
                                                        max_iter=100, max_leaf_nodes=31, max_depth=None,
                                                        min_samples_leaf=20, l2_regularization=0.0, max_bins=255)

regression_tree.fit(train_x.to_numpy(), train_y.to_numpy())
random_forrest.fit(train_x.to_numpy(), train_y.to_numpy())
gradient_boost.fit(train_x.to_numpy(), train_y.to_numpy())

#rt = tree.export_text(regression_tree, feature_names=list(train_x.columns))
#window = tk.Tk()
#label = tk.Label(text=rt)
#label.pack()
#window.mainloop()

# Using the models to make predictions on both testing and training data
rt_test_preds = regression_tree.predict(test_x.to_numpy())
rt_preds = regression_tree.predict(train_x.to_numpy())

rf_test_preds = random_forrest.predict(test_x.to_numpy())
rf_preds = random_forrest.predict(train_x.to_numpy())

gb_test_preds = gradient_boost.predict(test_x.to_numpy())
gb_preds = gradient_boost.predict(train_x.to_numpy())

window = tk.Tk()
label = tk.Label(text=
                    f"Regression tree train RMSE: " +
                    f"{metrics.mean_squared_error(train_y.to_numpy(), rt_preds) ** .5}" +
                    f"\nRegression tree test RMSE: " +
                    f"{metrics.mean_squared_error(test_y.to_numpy(), rt_test_preds) ** .5}" +
                    f"\nRandom forrest train RMSE: " +
                    f"{metrics.mean_squared_error(train_y.to_numpy(), rf_preds) ** .5}" +
                    f"\nRandom forrest test RMSE: " +
                    f"{metrics.mean_squared_error(test_y.to_numpy(), rf_test_preds) ** .5}" +
                    f"\nGradient boosted train RMSE: " +
                    f"{metrics.mean_squared_error(train_y.to_numpy(), gb_preds) ** .5}" +
                    f"\nGradient boosted test RMSE: " +
                    f"{metrics.mean_squared_error(test_y.to_numpy(), gb_test_preds) ** .5}"
                    )
label.pack()
window.mainloop()

fig, axs = plt.subplots(3, 2, figsize=(20, 17))

axs[0, 0].scatter(rt_preds, train_y.to_numpy())
axs[0, 0].set_title(
    f"Regression Tree Training Set Predictions - MAPE {round(metrics.mean_absolute_percentage_error(train_y.to_numpy(), rt_preds), 2) * 100} %")
axs[0, 0].set_xlabel("Predicted")
axs[0, 0].set_ylabel("Actual")
axs[0, 0].plot([0, 550000], [0, 550000], 'r')

axs[0, 1].scatter(rt_test_preds, test_y.to_numpy())
axs[0, 1].set_title(
    f"Regression Tree Testing Set Predictions - MAPE {round(metrics.mean_absolute_percentage_error(test_y.to_numpy(), rt_test_preds), 2) * 100} %")
axs[0, 1].set_xlabel("Predicted")
axs[0, 1].set_ylabel("Actual")
axs[0, 1].plot([0, 550000], [0, 550000], 'r')

axs[1, 0].scatter(rf_preds, train_y.to_numpy())
axs[1, 0].set_title(
    f"Random Forrest Training Set Predictions - MAPE {round(metrics.mean_absolute_percentage_error(train_y.to_numpy(), rf_preds), 2) * 100} %")
axs[1, 0].set_xlabel("Predicted")
axs[1, 0].set_ylabel("Actual")
axs[1, 0].plot([0, 550000], [0, 550000], 'r')

axs[1, 1].scatter(rf_test_preds, test_y.to_numpy())
axs[1, 1].set_title(
    f"Random Forrest Testing Set Predictions - MAPE {round(metrics.mean_absolute_percentage_error(test_y.to_numpy(), rf_test_preds), 2) * 100} %")
axs[1, 1].set_xlabel("Predicted")
axs[1, 1].set_ylabel("Actual")
axs[1, 1].plot([0, 550000], [0, 550000], 'r')

axs[2, 0].scatter(gb_preds, train_y.to_numpy())
axs[2, 0].set_title(
    f"Gradient Boosting Training Set Predictions - MAPE {round(metrics.mean_absolute_percentage_error(train_y.to_numpy(), gb_preds), 2) * 100} %")
axs[2, 0].set_xlabel("Predicted")
axs[2, 0].set_ylabel("Actual")
axs[2, 0].plot([0, 550000], [0, 550000], 'r')

axs[2, 1].scatter(gb_test_preds, test_y.to_numpy())
axs[2, 1].set_title(
    f"Gradient Boosting Testing Set Predictions - MAPE {round(metrics.mean_absolute_percentage_error(test_y.to_numpy(), gb_test_preds), 2) * 100} %")
axs[2, 1].set_xlabel("Predicted")
axs[2, 1].set_ylabel("Actual")
axs[2, 1].plot([0, 550000], [0, 550000], 'r')

plt.show()
