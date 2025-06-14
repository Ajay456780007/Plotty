from sklearn.metrics import accuracy_score
from termcolor import colored, cprint
from mealpy import FloatVar
import numpy as np
from mealpy.Proposed import HYBRID


class DataBalancing:
    """
    Optimization is optimizing the trained model weights for better performance.
    We combine the Harris cat, coyote Optimization and leopard Optimization.
    """

    def __init__(self, Features, Labels):
        """
        Initialize the Optimization class.

        Args:
        - Featurest: Test data used for evaluation.
        - Labels: Ground truth labels for the test data.
        """
        self.Features = Features
        self.Labels = Labels

        # Calculate unique labels and their counts in y_train
        unique, counts = np.unique(self.Labels, return_counts=True)
        # Find the minimum and maximum count of labels
        min_count = np.min(counts)
        max_count = np.max(counts)
        # max_count = 500
        self.min_count_features = []
        self.min_count_labels = []
        self.T = []
        for i in range(len(counts)):
            if counts[i] != max_count:
                # Find the index of the minimum count labels
                # min_count_index = counts.tolist().index(unique[i])
                oversample_count = max_count - counts[i]
                feat_indices_for_minimum_count = np.where(self.Labels == unique[i])[0]

                # Extract features and labels for the class with minimum count
                self.min_count_features.append(self.Features[feat_indices_for_minimum_count.tolist(), :])
                self.min_count_labels.append(self.Labels[feat_indices_for_minimum_count])

                self.T.append(oversample_count)
        # # Find the index of the minimum and maximum count labels
        # min_count_index = counts.tolist().index(min_count)
        # max_count_index = counts.tolist().index(max_count)
        # # Calculate the oversample count needed for balancing
        # oversample_count = max_count - min_count
        # # Get indices of features belonging to the class with minimum count
        # feat_indices_for_minimum_count = np.where(self.Labels == min_count_index)[0]
        #
        # # Extract features and labels for the class with minimum count
        # self.min_count_features = self.Features[feat_indices_for_minimum_count.tolist(), :]
        # self.min_count_labels = self.Labels[feat_indices_for_minimum_count]
        # # Set oversampling count as T
        # self.T = oversample_count

    def fitness_function1(self, solution):
        print(colored("Fitness Function >> ", color='blue', on_color='on_grey'))
        # Get rounded array element using numpy.round() and convert float into integer
        solution = np.round(solution).astype("int")
        # Extract oversampled features and labels using solution
        oversampled_feat = self.min_count_features[self.lab][solution]
        oversampled_labels = self.min_count_labels[self.lab][solution]
        # stack features and oversampled features to get new features
        new_feat = np.vstack((self.Features, oversampled_feat))
        # stack labels and oversampled labels to get new labels
        new_label = np.hstack((self.Labels, oversampled_labels))
        # train and predict features using knn models
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier()
        model.fit(new_feat[::100], new_label[::100])
        preds = model.predict(new_feat[::100])
        acc = accuracy_score(new_label[::100], preds)
        return acc

    def Oversampling(self):

        new_feat = self.Features
        new_label = self.Labels

        for self.lab in range(len(self.T)):
            """
            Perform weight optimization using various optimization algorithms.

            Returns:
            - best_position2: The optimized weights.
            """
            # Define problem parameters for optimization
            problem_dict = {
                "bounds": FloatVar(lb=[0 for i in range(self.T[self.lab])],
                                   ub=[len(self.min_count_features[self.lab]) - 1 for i in range(self.T[self.lab])],
                                   name="delta"),
                "obj_func": self.fitness_function1,
                "minmax": "max",
            }

            # Use Coyote Optimization algorithm
            cprint("[🚀🚀] SMOTE based Data balancing using Hybrid Optimization ", 'magenta', on_color='on_grey')
            model = HYBRID(epoch=1, pop_size=5)
            best_position2 = model.solve(problem_dict)

            indexes = np.round(best_position2.solution).astype("int")
            # Extract oversampled features and labels using the generated indexes
            oversampled_feat = self.min_count_features[self.lab][indexes]
            oversampled_labels = self.min_count_labels[self.lab][indexes]
            # Concatenate original and oversampled features and labels

            new_feat = np.vstack((new_feat, oversampled_feat))
            new_label = np.hstack((new_label, oversampled_labels))

        return new_feat, new_label