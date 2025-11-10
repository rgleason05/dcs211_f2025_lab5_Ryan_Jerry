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

def fetchDigit(df: pd.core.frame.DataFrame, which_row: int) -> tuple[int, np.ndarray]:
    ''' For digits.csv data represented as a dataframe, this fetches the digit from
        the corresponding row, reshapes, and returns a tuple of the digit and a
        numpy array of its pixel values.
    Parameters:
        df: pandas data frame expected to be obtained via pd.read_csv() on digits.csv
        which_row: an integer in 0 to len(df)
    Returns:
        a tuple containing the reprsented digit and a numpy array of the pixel
        values
    '''
    digit  = int(round(df.iloc[which_row, 64]))
    pixels = df.iloc[which_row, 0:64]   # don't want the rightmost rows
    pixels = pixels.values              # converts to numpy array
    pixels = pixels.astype(int)         # convert to integers for plotting
    pixels = np.reshape(pixels, (8,8))  # makes 8x8
    return (digit, pixels)              # return a tuple

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

from sklearn.model_selection import train_test_split
from typing import List

def splitData(A: np.ndarray) -> List[np.ndarray]:
    ''' 
    Splits the full data array into training and testing sets.

    Parameters:
        A : np.ndarray 
            The full dataset array where the first 64 columns are features 
            and the last column is the label.

    Returns:
       List[np.ndarray] 
            A list containing [X_test, y_test, X_train, y_train], in that order.
    '''
    X = A[:, :-1]   # feature columns
    y = A[:, -1]    # label column

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state =42
    )

    return [X_test, y_test, X_train, y_train]

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def runInitialKNN(X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray, k_guess: int = 3) -> None:
    """
    Trains and tests a k-NN classifier with an initial guessed k value.

    Parameters:

    X_train, y_train : np.ndarray
        Training data and corresponding labels.
    X_test, y_test : np.ndarray
        Testing data and corresponding labels.
    k_guess : int, optional
        The guessed value for k (default is 3).

    Returns: None- Prints out the resulting accuracy.
    """
    knn = KNeighborsClassifier(n_neighbors=k_guess)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Guessed k = {k_guess}")
    print(f"Accuracy = {accuracy:.4f}")

def compareLabels(predicted_labels: np.ndarray, actual_labels: np.ndarray) -> int:
    '''
    Prints a neatly formatted comparison of predicted vs. actual labels,
    returning the number of correct predictions.

    Parameters:
        predicted_labels : np.ndarray
            The predicted digit labels (from the classifier).
        actual_labels : np.ndarray
            The true digit labels from the test data.

    Returns:
        int
            The number of correctly predicted labels.
    '''
    num_labels: int = len(predicted_labels)
    num_correct: int = 0

    for i in range(num_labels):
        predicted: int = int(round(predicted_labels[i]))  # handle float rounding
        actual: int = int(round(actual_labels[i]))
        result: str = "incorrect"
        if predicted == actual:
            result = ""
            num_correct += 1

        # formatting for consistent alignment
        print(f"row {i:>3d} : predicted={predicted:<3d} actual={actual:<3d}   {result}")

    print()
    print(f"Correct: {num_correct} out of {num_labels}")
    return num_correct

from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import cross_val_score

def findBestK(X_train: np.ndarray, y_train: np.ndarray, max_k: int = 64) -> dict:
    """
    Determines the best k value for a k-NN classifier using cross-validation
    for multiple seeds.

    Parameters:
    X_train : np.ndarray
        Training feature data.
    y_train : np.ndarray
        Training labels.
    max_k : int, optional
        The maximum k value to test (default is 20).

    Returns: dict
        Dictionary mapping each seed to the best k value found.
    """

    seeds = [8675309, 5551212, 123456]
    best_k_per_seed = {}

    for seed in seeds:
        np.random.seed(seed)
        best_k = 1
        best_acc = 0.0

        for k in range(1, max_k + 1):
            knn = KNeighborsClassifier(n_neighbors=k)
            cv_scores = cross_val_score(knn, X_train, y_train, cv=5)
            acc = cv_scores.mean()

            if acc > best_acc:
                best_acc = acc
                best_k = k

        best_k_per_seed[seed] = best_k
        print(f"Seed {seed}: Best k = {best_k} with CV accuracy = {best_acc:.4f}")

    return best_k_per_seed

def trainAndTest( X_train: np.ndarray,  y_train: np.ndarray,X_test: np.ndarray,y_test: np.ndarray, best_k: int) -> np.ndarray:
    """
    Trains and tests a k-NN classifier using the best determined k value.

    Parameters:

    X_train : np.ndarray
        Training feature data.
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Test feature data.
    y_test : np.ndarray
        Test labels.
    best_k : int
        Optimal k value determined from tuning.

    Returns:np.ndarray- Predicted labels for the test set.
    """
    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f" Model trained and tested with k = {best_k}")
    print(f"Accuracy = {accuracy:.4f}")

    
    return predictions
 
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


    # scikit-learn-assisted portion

    print("\n[INFO] Starting scikit-learn-assisted tests...")

    # Step 7: Split data
    X_test_sk, y_test_sk, X_train_sk, y_train_sk = splitData(A)

    # Step 8: Run guessed k-NN
    guessed_k = 3
    print(f"\n[STEP 8] Running guessed k-NN with k={guessed_k}")
    knn_guess = KNeighborsClassifier(n_neighbors=guessed_k)
    knn_guess.fit(X_train_sk, y_train_sk)
    preds_guess = knn_guess.predict(X_test_sk)

    print("[COMPARE LABELS] Guessed k-NN (k=3):")
    compareLabels(preds_guess, y_test_sk)

    # Step 9: Find best k
print("\n[STEP 9] Determining best k...")
best_k_per_seed = findBestK(X_train, y_train, max_k=64)

print("\nSummary of best k per seed:")
for seed, k in best_k_per_seed.items():
    print(f"Seed {seed}: Best k = {k}")




    # Step 10: Train and test using best k
    print("\n[STEP 10] Training and testing with best k...")
    final_preds = trainAndTest(X_train_sk, y_train_sk, X_test_sk, y_test_sk, best_k)

    print("[COMPARE LABELS] Final model with best k:")
    compareLabels(final_preds, y_test_sk)


if __name__ == "__main__":
    main()
