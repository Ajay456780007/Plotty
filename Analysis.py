import numpy as np
from termcolor import cprint
from SubFunctions.Model import RecommendationNetwork, YieldNetwork
from SubFunctions.Evaluate import Evaluation_Metrics, Evaluation_Metrics1
from sklearn.model_selection import train_test_split



def train_test_split1(data, train_size):

    labels = data['labels']
    # Get the unique classes in the target variable 'y'
    num_classes = np.unique(labels)

    # Initialize empty lists to store training and testing data
    x_train = []
    y_train = []

    x_test = []
    y_test = []

    # Loop through each unique class
    for i in range(len(num_classes)):
        # Find indices of samples belonging to the current class
        indices = np.where(labels == num_classes[i])

        # Split the indices based on the specified 'train_size'
        train_index = indices[0][:int(len(indices[0]) * train_size)]
        test_index = indices[0][int(len(indices[0]) * train_size):]

        # Extract features and labels for training set
        train_feat = data['features'][train_index]
        train_lab = labels[train_index]

        # Extract features and labels for testing set
        test_feat = data['features'][test_index]
        test_lab = labels[test_index]

        # Extend the lists with the current class data
        x_train.extend(train_feat)
        y_train.extend(train_lab)

        x_test.extend(test_feat)
        y_test.extend(test_lab)

    # Convert the lists to numpy arrays
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    train_samples = x_train.shape[0]
    train_indices = np.random.permutation(train_samples)

    x_train = x_train[train_indices]
    y_train = y_train[train_indices]

    test_samples = x_test.shape[0]
    test_indices = np.random.permutation(test_samples)

    x_test = x_test[test_indices]

    y_test = y_test[test_indices]

    # Separate features and labels after shuffling
    return [x_train, x_test, y_train.astype(int), y_test.astype(int)]


class Analysis:

    def __init__(self, data, exec_type):
        """
        Initialize the Analysis class.

        Args:
        - Features: The feature data for analysis.
        - Labels: The labels corresponding to the feature data.
        """
        self.data = data
        self.exec_type = exec_type
        self.epochs = 500
        self.perf_epochs = [100, 200, 300, 400, 500]

    def ComparativeAnalysis(self):
        """
        Perform Comparative Analysis to compare the proposed method with existing methods.

        Vary the training percentage and use different classification methods.

        Save the results in numpy files for each method and training percentage.
        """
        # Initialize lists to store comparative analysis results
        ComparativeResults = []

        TrainingPercentage = 0.4

        for i in range(6):
            cprint(f"[⚠️] Comparative Analysis Count Is {i} Out Of 6", 'cyan', on_color='on_grey')

            # Split the data into training and testing sets based on the training percentage

            if self.exec_type == "Crop Recommendation":

                data = train_test_split1(self.data, train_size=TrainingPercentage)


                params = {'x_train': data[0], 'x_test': data[1],
                          'y_train': data[2], 'y_test': data[3], 'epochs': self.epochs}

                Ne = RecommendationNetwork(**params)

            else:
                x_train, x_test, y_train, y_test = train_test_split(self.data['features'], self.data['labels'], train_size=TrainingPercentage)

                params = {'x_train': x_train, 'x_test': x_test,
                          'y_train': y_train, 'y_test': y_test, 'epochs': self.epochs}

                Ne = YieldNetwork(**params)

            # Perform cl classification using different methods and get predictions
            output = [
                Ne.CYPA(),
                Ne.MMML_CRYP(),
                Ne.BSVFM(),
                Ne.IOF_LSTM(),
                Ne.XAI_BMLSTM(opt=0, epochs=self.epochs),
                Ne.XAI_BMLSTM(opt=1, epochs=self.epochs),
                Ne.XAI_BMLSTM(opt=2, epochs=self.epochs),
                Ne.XAI_BMLSTM(opt=3, epochs=self.epochs)]

            # Calculating the Performance
            if self.exec_type == "Crop Recommendation":
                ComparativeResults.append([Evaluation_Metrics(data[3], y_pred) for y_pred in output])

            else:
                ComparativeResults.append([Evaluation_Metrics1(y_test, y_pred) for y_pred in output])

            # Increase the training percentage for the next iteration
            TrainingPercentage += 0.1

        perf_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        # , 'L', 'M'
        file_names = [f'Analysis1\\{self.exec_type}\\COM_{name}.npy' for name in perf_names]

        for i, file_name in enumerate(file_names):
            np.save(file_name, [perf[i] for perf in ComparativeResults])
        cprint("[✅] Execution of Comparative Analysis Completed ", 'green', on_color='on_grey')


    def PerformanceAnalysis(self):
        """
        Perform Performance Analysis to check the maximum performance of the proposed method.

        Vary the training percentage and epochs.

        Save the results in numpy files for each training percentage and epoch combination.
        """
        # Initialize lists to store performance analysis results
        PerformanceResults = []

        TrainingPercentage = 0.4

        for i in range(6):
            cprint(f"[⚠️] Performance Analysis Count Is {i} Out Of 6", 'cyan', on_color='on_grey')

            # Split the data into training and testing sets based on the training percentage
            if self.exec_type == "Crop Recommendation":

                data = train_test_split1(self.data, train_size=TrainingPercentage)

                params = {'x_train': data[0], 'x_test': data[1],
                          'y_train': data[2], 'y_test': data[3], 'epochs': self.epochs}

                Ne = RecommendationNetwork(**params)

            else:
                x_train, x_test, y_train, y_test = train_test_split(self.data['features'], self.data['labels'],
                                                                    train_size=TrainingPercentage)

                params = {'x_train': x_train, 'x_test': x_test,
                          'y_train': y_train, 'y_test': y_test, 'epochs': self.epochs}

                Ne = YieldNetwork(**params)

            # Perform cl classification using different methods and get predictions
            output = [
                Ne.XAI_BMLSTM(opt=3, epochs=self.perf_epochs[0]),
                Ne.XAI_BMLSTM(opt=3, epochs=self.perf_epochs[1]),
                Ne.XAI_BMLSTM(opt=3, epochs=self.perf_epochs[2]),
                Ne.XAI_BMLSTM(opt=3, epochs=self.perf_epochs[3]),
                Ne.XAI_BMLSTM(opt=3, epochs=self.perf_epochs[4])]

            # Calculating the Performance
            if self.exec_type == "Crop Recommendation":
                PerformanceResults.append([Evaluation_Metrics(data[3], y_pred) for y_pred in output])

            else:
                PerformanceResults.append([Evaluation_Metrics1(y_test, y_pred) for y_pred in output])

            # Increase the training percentage for the next iteration
            TrainingPercentage += 0.1

        perf_names = ['A', 'B', 'C', 'D', 'E'] 
        # , 'L', 'M'
        file_names = [f'Analysis1\\{self.exec_type}\\PERF_{name}.npy' for name in perf_names]

        for i, file_name in enumerate(file_names):
            np.save(file_name, [perf[i] for perf in PerformanceResults])


        # Print a completion message
        cprint("[✅] Execution of Performance Analysis Completed ", 'green', on_color='on_grey')

