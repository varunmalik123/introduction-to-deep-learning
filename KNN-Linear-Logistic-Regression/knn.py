"""
K Nearest Neighbours Model
"""
import numpy as np
from scipy import stats

class KNN(object):
    def __init__(
        self,
        num_class: int
    ):
        self.num_class = num_class

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        k: int
    ):
        """
        Train KNN Classifier

        KNN only need to remember training set during training

        Parameters:
            x_train: Training samples ; np.ndarray with shape (N, D)
            y_train: Training labels  ; snp.ndarray with shape (N,)
        """
        self._x_train = x_train
        self._y_train = y_train
        self.k = k

    def predict(
        self,
        x_test: np.ndarray,
        k: int = None,
        loop_count: int = 1
    ):
        """
        Use the contained training set to predict labels for test samples

        Parameters:
            x_test    : Test samples                                     ; np.ndarray with shape (N, D)
            k         : k to overwrite the one specificed during training; int
            loop_count: parameter to choose different knn implementation ; int

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        k_test = k if k is not None else self.k

        if loop_count == 1:
            distance = self.calc_dis_one_loop(x_test)
        elif loop_count == 2:
            distance = self.calc_dis_two_loop(x_test)

        
        # Your Code Here

        ##distance is a 500 x 5000 matrix 
        from scipy.stats import mode
        
        index_sorted = np.argsort(distance, axis = 1)
       

        y_hat = []
        for i in range(index_sorted.shape[0]): 
            
            index_dists = index_sorted[i][:k_test]
            min_label = self._y_train[index_dists]
        
            # print(min_label)
            y_hat.append(stats.mode(min_label)[0][0])
            
   


        return np.asarray(y_hat)
        
       

    def calc_dis_one_loop(
        self,
        x_test: np.ndarray
    ):
        """
        Calculate distance between training samples and test samples

        This function could one for loop

        Parameters:
            x_test: Test samples; np.ndarray with shape (N, D)
        """
        distance_array = np.ndarray((x_test.shape[0],self._x_train.shape[0]))
        x_train_squared = np.sum(self._x_train * self._x_train)


        for n in range(x_test.shape[0]):
            
            dist_temp = np.sum(x_test[n]**2) + x_train_squared - np.sum(2*x_test[n] * x_train_squared)
        
            distance_array[n]

        return distance_array




    def calc_dis_two_loop(
        self,
        x_test: np.ndarray
    ):
        """
        Calculate distance between training samples and test samples

        This function could contain two loop

        Parameters:
            x_test: Test samples; np.ndarray with shape (N, D) (5000, 3072)
        """

        distance_array = np.ndarray((x_test.shape[0],self._x_train.shape[0]))
        for n in range(x_test.shape[0]): 
            for m in range(self._x_train.shape[0]): 
                distance_array[n,m] = np.linalg.norm(x_test[n] - self._x_train[m])
        
        return distance_array  

        

