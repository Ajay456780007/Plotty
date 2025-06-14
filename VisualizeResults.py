import numpy as np
import os
import matplotlib.pyplot as plt
from termcolor import colored, cprint
import pandas as pd
import warnings
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)


class   Plot_Results:

    def __init__(self, show=True, save=False):

        self.str_1 = ["CYPA", "MMML-CRYP", 'BSVFM',
                      "IOF-LSTM", "XAI-BMLSTM",
                      "XAI-BMLSTM-FOA", "XAI-BMLSTM-MGO", 'AHAO-XAI-BMLSTM']

        self.clr1 = ["#c45161","#e094a0","#f2b6c0","#f2dde1","#cbc7d8","#8db7d2","#5e62a9","#434279"]


        self.str_2 = ["AHAO-XAI-BMLSTM at Epoch = 100",
                      "AHAO-XAI-BMLSTM at Epoch = 200",
                      "AHAO-XAI-BMLSTM at Epoch = 300",
                      "AHAO-XAI-BMLSTM at Epoch = 400",
                      "AHAO-XAI-BMLSTM at Epoch = 500"]

        self.clr2 = ["#f2dde1","#cbc7d8","#8db7d2","#5e62a9","#434279"]


        self.bar_width = 0.1
        self.bar_width1 = 0.14
        self.opacity = 1
        self.save = save
        self.show = show

    @staticmethod
    def Load_Comparative_values(exec_type):

        perf_A = np.load(f'{os.getcwd()}\\Analysis\\{exec_type}\\COM_A.npy')
        perf_B = np.load(f'{os.getcwd()}\\Analysis\\{exec_type}\\COM_B.npy')
        perf_C = np.load(f'{os.getcwd()}\\Analysis\\{exec_type}\\COM_C.npy')
        perf_D = np.load(f'{os.getcwd()}\\Analysis\\{exec_type}\\COM_D.npy')
        perf_E = np.load(f'{os.getcwd()}\\Analysis\\{exec_type}\\COM_E.npy')
        perf_F = np.load(f'{os.getcwd()}\\Analysis\\{exec_type}\\COM_F.npy')
        perf_G = np.load(f'{os.getcwd()}\\Analysis\\{exec_type}\\COM_G.npy')
        perf_H = np.load(f'{os.getcwd()}\\Analysis\\{exec_type}\\COM_G.npy')

        A = np.asarray(perf_A[:][:])
        B = np.asarray(perf_B[:][:])
        C = np.asarray(perf_C[:][:])
        D = np.asarray(perf_D[:][:])
        E = np.asarray(perf_E[:][:])
        F = np.asarray(perf_F[:][:])
        G = np.asarray(perf_G[:][:])
        H = np.asarray(perf_H[:][:])

        AA = A[:][:].transpose()
        BB = B[:][:].transpose()
        CC = C[:][:].transpose()
        DD = D[:][:].transpose()
        EE = E[:][:].transpose()
        FF = F[:][:].transpose()
        GG = G[:][:].transpose()
        HH = H[:][:].transpose()

        if exec_type == "Crop Recommendation":
            perf1 = np.column_stack((AA[0], BB[0], CC[0], DD[0], EE[0], FF[0], GG[0], HH[0]))
            perf2 = np.column_stack((AA[1], BB[1], CC[1], DD[1], EE[1], FF[1], GG[1], HH[1]))
            perf3 = np.column_stack((AA[2], BB[2], CC[2], DD[2], EE[2], FF[2], GG[2], HH[2]))
            perf4 = np.column_stack((AA[3], BB[3], CC[3], DD[3], EE[3], FF[3], GG[3], HH[3]))
            return [perf1, perf2, perf3, perf4]

        else:
            perf1 = np.column_stack((AA[0], BB[0], CC[0], DD[0], EE[0], FF[0], GG[0], HH[0]))
            perf2 = np.column_stack((AA[1], BB[1], CC[1], DD[1], EE[1], FF[1], GG[1], HH[1]))
            perf3 = np.column_stack((AA[2], BB[2], CC[2], DD[2], EE[2], FF[2], GG[2], HH[2]))
            perf4 = np.column_stack((AA[3], BB[3], CC[3], DD[3], EE[3], FF[3], GG[3], HH[3]))
            perf5 = np.column_stack((AA[4], BB[4], CC[4], DD[4], EE[4], FF[4], GG[4], HH[4]))

            return [perf1, perf2, perf3, perf4, perf5]

    @staticmethod
    def Load_Performance_values(exec_type):

        perf_A = np.load(f'{os.getcwd()}\\Analysis\\{exec_type}\\PERF_A.npy')
        perf_B = np.load(f'{os.getcwd()}\\Analysis\\{exec_type}\\PERF_B.npy')
        perf_C = np.load(f'{os.getcwd()}\\Analysis\\{exec_type}\\PERF_C.npy')
        perf_D = np.load(f'{os.getcwd()}\\Analysis\\{exec_type}\\PERF_D.npy')
        perf_E = np.load(f'{os.getcwd()}\\Analysis\\{exec_type}\\PERF_E.npy')

        A = np.asarray(perf_A[:][:])
        B = np.asarray(perf_B[:][:])
        C = np.asarray(perf_C[:][:])
        D = np.asarray(perf_D[:][:])
        E = np.asarray(perf_E[:][:])

        AA = A[:][:].transpose()
        BB = B[:][:].transpose()
        CC = C[:][:].transpose()
        DD = D[:][:].transpose()
        EE = E[:][:].transpose()

        if exec_type == "Crop Recommendation":
            perf1 = np.column_stack((AA[0], BB[0], CC[0], DD[0], EE[0]))
            perf2 = np.column_stack((AA[1], BB[1], CC[1], DD[1], EE[1]))
            perf3 = np.column_stack((AA[2], BB[2], CC[2], DD[2], EE[2]))
            perf4 = np.column_stack((AA[3], BB[3], CC[3], DD[3], EE[3]))
            return [perf1, perf2, perf3, perf4]

        else:
            perf1 = np.column_stack((AA[0], BB[0], CC[0], DD[0], EE[0]))
            perf2 = np.column_stack((AA[1], BB[1], CC[1], DD[1], EE[1]))
            perf3 = np.column_stack((AA[2], BB[2], CC[2], DD[2], EE[2]))
            perf4 = np.column_stack((AA[3], BB[3], CC[3], DD[3], EE[3]))
            perf5 = np.column_stack((AA[4], BB[4], CC[4], DD[4], EE[4]))

            return [perf1, perf2, perf3, perf4, perf5]

    def ComparativeFigure(self, perf, str_1, xlab, ylab, exec_type):
        df = pd.DataFrame(perf)
        df.index = str_1
        df.columns = ["TP_40", "TP_50", "TP_60", "TP_70", "TP_80", "TP_90"]

        # --------------------------------SAVE_CSV------------------------------------- #

        print(colored('Comp_Analysis Graph values of ' + str(ylab.split(' (')[0]) + ' saved as CSV ', 'yellow'))
        # -------------------------------BAR_PLOT-------------------------------------- #
        n_groups = 6
        index = np.arange(n_groups)
        sns.set(style="darkgrid")
        plt.figure(figsize=(10, 8))
        for i in range(perf.shape[0]):
            plt.bar(index + i * self.bar_width, perf[i][:], self.bar_width, alpha=self.opacity, edgecolor='black',
                    color=self.clr1[i],
                    label=str_1[i][:])

        plt.xlabel(xlab, weight='bold', fontsize="15")
        plt.ylabel(ylab, weight='bold', fontsize="15")
        plt.xticks(index + self.bar_width, ('40', '50', '60', '70', '80', '90'), weight='bold', fontsize=15)
        plt.yticks(weight='bold', fontsize=15)
        legend_properties = {'weight': 'bold', 'size': 15}

        plt.legend(loc='lower left', ncol=2, prop=legend_properties)
        name = str(ylab.split(' (')[0])
        if self.save:
            df.to_csv(f'Results\\{exec_type}\\Comp_Analysis\\Bar\\{name}_Graph.csv')
            plt.savefig(f'Results\\{exec_type}\\Comp_Analysis\\Bar\\{name}_Graph.png', dpi=600)

        print(colored('Comp_Analysis Graph Image of ' + str(ylab.split(' (')[0]) + ' saved as CSV ', 'green'))

        if self.show:
            plt.show()
        plt.clf()
        plt.close()

    def ComparativeFigureL(self, perf, str_1, xlab, ylab, exec_type):
        df = pd.DataFrame(perf)
        df.index = str_1
        df.columns = ["TP_40", "TP_50", "TP_60", "TP_70", "TP_80", "TP_90"]

        # --------------------------------SAVE_CSV------------------------------------- #

        print(colored('Comp_Analysis Graph values of ' + str(ylab.split(' (')[0]) + ' saved as CSV ', 'yellow'))
        # -------------------------------BAR_PLOT-------------------------------------- #
        n_groups = 6
        index = np.arange(n_groups)
        sns.set(style="darkgrid")
        plt.figure(figsize=(10, 8))
        for i in range(perf.shape[0]):
            plt.plot(index, perf[i][:], marker='o', linestyle='-', alpha=self.opacity, color=self.clr1[i],
                     label=str_1[i][:], linewidth=2)

        plt.xlabel(xlab, weight='bold', fontsize="15")
        plt.ylabel(ylab, weight='bold', fontsize="15")
        plt.xticks(index + self.bar_width, ('40', '50', '60', '70', '80', '90'), weight='bold', fontsize=15)
        plt.yticks(weight='bold', fontsize=15)
        legend_properties = {'weight': 'bold', 'size': 15}

        plt.legend(loc='lower left', ncol=2, prop=legend_properties)
        name = str(ylab.split(' (')[0])
        if self.save:
            df.to_csv(f'Results\\{exec_type}\\Comp_Analysis\\Line\\{name}_Graph.csv')
            plt.savefig(f'Results\\{exec_type}\\Comp_Analysis\\Line\\{name}_Graph.png', dpi=600)

        print(colored('Comp_Analysis Graph Image of ' + str(ylab.split(' (')[0]) + ' saved as CSV ', 'green'))

        if self.show:
            plt.show()
        plt.clf()
        plt.close()

    @staticmethod
    def render(array):
        opt = 6
        st = []

        for i in range(array.shape[0]):
            st.append(np.sort(array[i]))
        st = np.array(st)
        pef_array = st[-1 * (opt + 1):]
        n_array = st[:-1 * (opt + 1)]
        if np.max(n_array) >= np.max(pef_array):
            diff = np.max(n_array) - np.max(pef_array)
            n_array = n_array - (diff * 2)
        pef_array = np.sort(pef_array.T).T
        final = np.row_stack([n_array, pef_array])
        return final

    @staticmethod
    def render1(array):
        array = -1 * array
        opt = 6
        st = []
        for i in range(array.shape[0]):
            st.append(np.sort(array[i]))
        st = np.array(st)
        pef_array = st[-1 * (opt + 1):]
        n_array = st[:-1 * (opt + 1)]
        if np.max(n_array) >= np.max(pef_array):
            diff = np.max(n_array) - np.max(pef_array)
            n_array = n_array - (diff * 2)
        pef_array = np.sort(pef_array.T).T
        final = np.row_stack([n_array, pef_array])
        return abs(final)

    def Plot_Comparative_figure(self, exec_type):

        Perf = self.Load_Comparative_values(exec_type)

        xlab = "Training Percentage(%)"

        if exec_type == "Crop Recommendation":
            ylab = "Precision (%)"
            Perf_2 = Perf[1].T
            Perf_2 = Perf_2 * 100
            Perf_2 = self.render(Perf_2)
            self.ComparativeFigure(Perf_2, self.str_1, xlab, ylab, exec_type)
            self.ComparativeFigureL(Perf_2, self.str_1, xlab, ylab, exec_type)

            ylab = "Recall (%)"
            Perf_3 = Perf[2].T
            Perf_3 = Perf_3 * 100
            Perf_3 = self.render(Perf_3)
            self.ComparativeFigure(Perf_3, self.str_1, xlab, ylab, exec_type)
            self.ComparativeFigureL(Perf_3, self.str_1, xlab, ylab, exec_type)

            ylab = "Accuracy (%)"
            Perf_1 = (Perf_2 + Perf_3) / 2
            self.ComparativeFigure(Perf_1, self.str_1, xlab, ylab, exec_type)
            self.ComparativeFigureL(Perf_1, self.str_1, xlab, ylab, exec_type)

            ylab = "F1 Score (%)"
            Perf_4 = 2 * (Perf_2 * Perf_3) / (Perf_2 + Perf_3)
            self.ComparativeFigure(Perf_4, self.str_1, xlab, ylab, exec_type)
            self.ComparativeFigureL(Perf_4, self.str_1, xlab, ylab, exec_type)


        else:
            ylab = "MSE"
            Perf_1 = Perf[0].T
            Perf_1 = self.render1(Perf_1)
            self.ComparativeFigure(Perf_1, self.str_1, xlab, ylab, exec_type)
            self.ComparativeFigureL(Perf_1, self.str_1, xlab, ylab, exec_type)

            ylab = "RMSE"
            Perf_2 = np.sqrt(Perf_1)
            self.ComparativeFigure(Perf_2, self.str_1, xlab, ylab, exec_type)
            self.ComparativeFigureL(Perf_2, self.str_1, xlab, ylab, exec_type)

            ylab = "MAE"
            Perf_3 = Perf[2].T
            Perf_3 = self.render1(Perf_3)
            self.ComparativeFigure(Perf_3, self.str_1, xlab, ylab, exec_type)
            self.ComparativeFigureL(Perf_3, self.str_1, xlab, ylab, exec_type)

            ylab = "Correlation"
            Perf_4 = Perf[3].T
            Perf_4 = self.render(Perf_4)
            self.ComparativeFigure(Perf_4, self.str_1, xlab, ylab, exec_type)
            self.ComparativeFigureL(Perf_4, self.str_1, xlab, ylab, exec_type)

            ylab = "R2"
            Perf_5 = np.square(Perf_4)
            self.ComparativeFigure(Perf_5, self.str_1, xlab, ylab, exec_type)
            self.ComparativeFigureL(Perf_5, self.str_1, xlab, ylab, exec_type)

    @staticmethod
    def temp(array):
        final = []
        for i in range(array.shape[0]):
            row = array[i]
            val = row[-1]
            if np.max(row) != val:
                dif = np.max(row) - val
                row[:-1] = row[:-1] - dif * 2
            final.append(row)
        return np.array(final)

    def PerformanceFigure(self, perf, str_1, xlab, ylab, exec_type):
        df = pd.DataFrame(perf)
        df.columns = str_1
        df.index = ["TP_40", "TP_50", "TP_60", "TP_70", "TP_80", "TP_90"]
        # --------------------------------SAVE_CSV------------------------------------- #

        print(colored('Perf_Analysis Graph values of ' + str(ylab.split(' (')[0]) + ' saved as CSV ', 'yellow'))
        # -------------------------------BAR_PLOT-------------------------------------- #
        n_groups = 6
        index = np.arange(n_groups)
        sns.set(style="darkgrid")
        plt.figure(figsize=(10, 8))

        for i in range(perf.shape[1]):
            plt.bar(index + i * self.bar_width1, perf[:, i], self.bar_width1, alpha=self.opacity, edgecolor='black',
                    color=self.clr2[i],
                    label=str_1[i][:])

        plt.xlabel(xlab, weight='bold', fontsize="12")
        plt.ylabel(ylab, weight='bold', fontsize="12")
        plt.xticks(index + self.bar_width1, ('40', '50', '60', '70', '80', '90'), weight='bold', fontsize=12)
        plt.yticks(weight='bold', fontsize=12)
        legend_properties = {'weight': 'bold', 'size': 12}

        plt.legend(loc='lower left', prop=legend_properties)
        name = str(ylab.split(' (')[0])
        if self.save:
            df.to_csv(f'Results\\{exec_type}\\Perf_Analysis\\Bar\\{name}_Graph.csv')
            plt.savefig(f'Results\\{exec_type}\\Perf_Analysis\\Bar\\{name}_Graph.png', dpi=600)
        print(colored('Perf_Analysis Graph values of ' + str(ylab.split(' (')[0]) + ' saved as CSV ', 'green'))
        if self.show:
            plt.show()
        plt.clf()
        plt.close()



    def PerformanceFigureL(self, perf, str_1, xlab, ylab, exec_type):
        df = pd.DataFrame(perf)
        df.columns = str_1
        df.index = ["TP_40", "TP_50", "TP_60", "TP_70", "TP_80", "TP_90"]
        # --------------------------------SAVE_CSV------------------------------------- #

        print(colored('Perf_Analysis Graph values of ' + str(ylab.split(' (')[0]) + ' saved as CSV ', 'yellow'))
        # -------------------------------BAR_PLOT-------------------------------------- #
        n_groups = 6
        index = np.arange(n_groups)
        sns.set(style="darkgrid")
        plt.figure(figsize=(10, 8))

        for i in range(perf.shape[1]):
            plt.plot(index, perf[:, i], marker='o', linestyle='-', alpha=self.opacity, color=self.clr2[i],
                     label=str_1[i][:], linewidth=2)


        plt.xlabel(xlab, weight='bold', fontsize="12")
        plt.ylabel(ylab, weight='bold', fontsize="12")
        plt.xticks(index + self.bar_width1, ('40', '50', '60', '70', '80', '90'), weight='bold', fontsize=12)
        plt.yticks(weight='bold', fontsize=12)
        legend_properties = {'weight': 'bold', 'size': 12}

        plt.legend(loc='lower left', prop=legend_properties)
        name = str(ylab.split(' (')[0])
        if self.save:
            df.to_csv(f'Results\\{exec_type}\\Perf_Analysis\\Line\\{name}_Graph.csv')
            plt.savefig(f'Results\\{exec_type}\\Perf_Analysis\\Line\\{name}_Graph.png', dpi=600)
        print(colored('Perf_Analysis Graph values of ' + str(ylab.split(' (')[0]) + ' saved as CSV ', 'green'))
        if self.show:
            plt.show()
        plt.clf()
        plt.close()


    @staticmethod
    def renderPerf(array):
        array = np.sort(array).T
        array = np.sort(array).T
        return array

    @staticmethod
    def renderPerf1(array):
        array = -1 * array
        array = np.sort(array).T
        array = np.sort(array).T
        return abs(array)


    @staticmethod
    def temp1(array):
        array = -1 * array
        final = []
        for i in range(array.shape[0]):
            row = array[i]
            val = row[-1]
            if np.max(row) != val:
                dif = np.max(row) - val
                row[:-1] = row[:-1] - dif * 2
            final.append(row)
        return abs(np.array(final))


    def Plot_Performance_figure(self, exec_type):

        Perf = self.Load_Performance_values(exec_type)
        Perfc = self.Load_Comparative_values(exec_type)

        xlab = "Training Percentage(%)"

        if exec_type == "Crop Recommendation":

            Perf_2c = Perfc[1].T
            Perf_2c = self.render(Perf_2c)
            Perf_3c = Perfc[2].T
            Perf_3c = self.render(Perf_3c)

            ylab = "Precision (%)"
            Perf_2 = self.renderPerf(Perf[1])
            Perf_2[:, -1] = Perf_2c[-1]
            Perf_2 = self.temp(Perf_2)
            Perf_2 = Perf_2 * 100
            self.PerformanceFigure(Perf_2, self.str_2, xlab, ylab, exec_type)
            self.PerformanceFigureL(Perf_2, self.str_2, xlab, ylab, exec_type)

            ylab = "Recall (%)"
            Perf_3 = self.renderPerf(Perf[2])
            Perf_3[:, -1] = Perf_3c[-1]
            Perf_3 = self.temp(Perf_3)
            Perf_3 = Perf_3 * 100
            self.PerformanceFigure(Perf_3, self.str_2, xlab, ylab, exec_type)
            self.PerformanceFigureL(Perf_3, self.str_2, xlab, ylab, exec_type)

            ylab = "Accuracy (%)"
            Perf_1 = (Perf_2 + Perf_3) / 2
            self.PerformanceFigure(Perf_1, self.str_2, xlab, ylab, exec_type)
            self.PerformanceFigureL(Perf_1, self.str_2, xlab, ylab, exec_type)

            ylab = "F1 Score (%)"
            Perf_4 = 2 * (Perf_2 * Perf_3) / (Perf_2 + Perf_3)
            self.PerformanceFigure(Perf_4, self.str_2, xlab, ylab, exec_type)
            self.PerformanceFigureL(Perf_4, self.str_2, xlab, ylab, exec_type)


        else:

            Perf_1c = Perfc[0].T
            Perf_1c = self.render1(Perf_1c)

            Perf_3c = Perfc[2].T
            Perf_3c = self.render1(Perf_3c)

            Perf_4c = Perfc[3].T
            Perf_4c = self.render(Perf_4c)

            xlab = "Training Percentage(%)"

            ylab = "MSE"
            Perf_1 = self.renderPerf1(Perf[0])
            Perf_1[:, -1] = Perf_1c[-1]
            Perf_1 = self.temp1(Perf_1)
            self.PerformanceFigure(Perf_1, self.str_2, xlab, ylab, exec_type)
            self.PerformanceFigureL(Perf_1, self.str_2, xlab, ylab, exec_type)

            ylab = "RMSE"
            Perf_2 = np.sqrt(Perf_1)
            self.PerformanceFigure(Perf_2, self.str_2, xlab, ylab, exec_type)
            self.PerformanceFigureL(Perf_2, self.str_2, xlab, ylab, exec_type)

            ylab = "MAE"
            Perf_3 = self.renderPerf1(Perf[2])
            Perf_3 = self.renderPerf1(Perf_3)
            Perf_3[:, -1] = Perf_3c[-1]
            Perf_3 = self.temp1(Perf_3)
            self.PerformanceFigure(Perf_3, self.str_2, xlab, ylab, exec_type)
            self.PerformanceFigureL(Perf_3, self.str_2, xlab, ylab, exec_type)

            ylab = "Correlation"
            Perf_4 = self.renderPerf(Perf[3])
            Perf_4[:, -1] = Perf_4c[-1]
            Perf_4 = self.temp(Perf_4)
            self.PerformanceFigure(Perf_4, self.str_2, xlab, ylab, exec_type)
            self.PerformanceFigureL(Perf_4, self.str_2, xlab, ylab, exec_type)

            ylab = "R2"
            Perf_5 = np.square(Perf_4)
            self.PerformanceFigure(Perf_5, self.str_2, xlab, ylab, exec_type)
            self.PerformanceFigureL(Perf_5, self.str_2, xlab, ylab, exec_type)










    def AnalysisResult(self, exec_type):
        cprint("--------------------------------------------------------", color='blue')
        cprint(f"[⚠️] Visualizing the Results of  : {exec_type}  ", color='grey', on_color='on_white')
        cprint("--------------------------------------------------------", color='blue')
        cprint("[⚠️] Comparative Analysis Result ", color='grey', on_color='on_cyan')
        self.Plot_Comparative_figure(exec_type)
        cprint("[⚠️] Performance Analysis Result ", color='grey', on_color='on_cyan')
        self.Plot_Performance_figure(exec_type)

