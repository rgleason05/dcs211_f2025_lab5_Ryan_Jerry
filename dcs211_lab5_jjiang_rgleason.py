import numpy as np      # numpy is Python's "array" library
import pandas as pd     # Pandas is Python's "data" library ("dataframe" == spreadsheet)
import seaborn as sns   # yay for Seaborn plots!
import matplotlib.pyplot as plt
import random

def drawDigitHeatmap(pixels: np.ndarray, showNumbers: bool = True) -> None:
    ''' Draws a heat map of a given digit based on its 8x8 set of pixel values.
    Parameters:
        pixels: a 2D numpy.ndarray (8x8) of integers of the pixel values for
                the digit
        showNumbers: if True, shows the pixel value inside each square
    Returns:
        None -- just plots into a window
    '''

    (fig, axes) = plt.subplots(figsize = (4.5, 3))  # aspect ratio

    rgb = (0, 0, 0.5)  # each in (0,1), so darkest will be dark blue
    colormap = sns.light_palette(rgb, as_cmap=True)    
    # all seaborn palettes: 
    # https://medium.com/@morganjonesartist/color-guide-to-seaborn-palettes-da849406d44f

    # plot the heatmap;  see: https://seaborn.pydata.org/generated/seaborn.heatmap.html
    # (fmt = "d" indicates to show annotation with integer format)
    sns.heatmap(pixels, annot = showNumbers, fmt = "d", linewidths = 0.5, \
                ax = axes, cmap = colormap)
    plt.show(block = False)

def cleanTheData(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    '''
    This function cleans the digits DataFrame and converts it to a numpy array.
    Parameters:
        df: pandas DataFrame read from digits.csv
    Returns:
        A tuple (A, df_clean)
            A: numpy array, first 64 columns are pixel values, column 64 is the label
            df_clean: cleaned DataFrame
    '''
    col65name = df.columns[65]
    df_clean = df.drop(columns=[col65name]).copy()
    df_clean = df_clean.dropna().copy()
    df_clean = df_clean.astype(int)
    A = df_clean.values

    return A, df_clean

def predictiveModel(trainA: np.ndarray, x_features: np.ndarray) -> int:
    '''
    This function calculates the distance by lines and help make predictions 
    Parameters
        trainA : np.ndarray, training data of shape (n_train, 65).
        x_features : np.ndarray, single test sample features of shape (64,).
    Returns:
        int: Predicted digit label for the test sample.
    '''
    x = np.asarray(x_features).ravel()

    best_idx = -1
    best_dist = float("inf")

    for i in range(trainA.shape[0]):
        xi = trainA[i, 0:64]                
        d = np.linalg.norm(xi - x) 
        if d < best_dist:
            best_dist = d
            best_idx = i

    return int(trainA[best_idx, 64])

def main():

    filename = "digits.csv"
    df = pd.read_csv(filename, header=0)
    print(f"{filename}: file read into a pandas DataFrame.")
    df.head()
    df.info()
    A, df_clean = cleanTheData(df)

    n = A.shape[0]
    cut = int(round(0.8 * n))
    trainA = A[:cut, :]
    testA  = A[cut:, :]

    X_train = trainA[:, 0:64]
    y_train = trainA[:, 64].astype(int)
    X_test  = testA[:, 0:64]
    y_test  = testA[:, 64].astype(int)

    preds_test = np.zeros(len(X_test), dtype=int)
    for i in range(len(X_test)):
        preds_test[i] = predictiveModel(trainA, X_test[i])

        if (i + 1) % 50 == 0 or (i + 1) == len(X_test):
            print(f"[80/20] Progress: {i + 1}/{len(X_test)}")

    correct_8020 = int(np.sum(preds_test == y_test))
    acc_8020 = correct_8020 / len(y_test)
    print(f"[1-NN] 80/20 split, Correct: {correct_8020}/{len(y_test)} | Accuracy = {acc_8020:.4f}")

    trainB = A[cut:, :]
    testB  = A[:cut, :]

    X_train_B = trainB[:, 0:64]
    y_train_B = trainB[:, 64].astype(int)
    X_test_B  = testB[:, 0:64]
    y_test_B  = testB[:, 64].astype(int)

    preds_test_B = np.zeros(len(X_test_B), dtype=int)
    for i in range(len(X_test_B)):
        preds_test_B[i] = predictiveModel(trainB, X_test_B[i])
        if (i + 1) % 50 == 0 or (i + 1) == len(X_test_B):
            print(f"[20/80] Progress: {i + 1}/{len(X_test_B)}")

    correct_2080 = int(np.sum(preds_test_B == y_test_B))
    acc_2080 = correct_2080 / len(y_test_B)
    print(f"[1-NN] 20/80 split, Correct: {correct_2080}/{len(y_test_B)} | Accuracy = {acc_2080:.4f}")

    mis_idx = np.where(preds_test != y_test)[0]
    for j in mis_idx[:5]:
        pixels_8x8 = X_test[j].reshape(8, 8).astype(int)
        print(f"Misclassified idx={j}: true={y_test[j]}, pred={preds_test[j]}")
        drawDigitHeatmap(pixels_8x8)

if __name__ == "__main__":
    main()
