import sys
sys.path.append(r'/Users/haer9000/Documents/codes/python_code/liver_disease/liver_disease/')
import analysis.imbalance.SMOTE as SMOTE


def get_SMOTE_resampled(df):
    data = df.to_numpy()
    balanced_data_arr2 = SMOTE.SMOTE(data)
    return balanced_data_arr2


