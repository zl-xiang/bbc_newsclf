#%%%
import utils
import pandas as pd
import fs_n_training as ft
import prepocessing as pp
from sklearn.model_selection import train_test_split


X_ori, y_ori = pp.processing(True)
X_train, X_test, y_train, y_test = train_test_split(
        X_ori, y_ori, test_size=0.2, random_state=42)

svc_grid,softmax_grid = ft.get_grid_instance(X_train,y_train)
# fine-tuning
svc_results = pd.DataFrame.from_records(svc_grid.cv_results_)
softmax_results  = pd.DataFrame.from_records(softmax_grid.cv_results_)

# best score in svc
ft.print_grid_results(svc_grid)
# best score in softmax
ft.print_grid_results(softmax_grid)

# evaluation
ft.print_test_report("SVC Best ",pp.topic_list,svc_grid,X_test,y_test)
ft.print_test_report("Softmax Best ",pp.topic_list,softmax_grid,X_test,y_test)
# %%
