######################################################################################################
##                              Import required Libraries.                                          ##
######################################################################################################
import math
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=6, suppress=True)

######################################################################################################
##                                   Inputs provided.                                               ##
##                     Training pattern is provided through .txt file                               ##
######################################################################################################
learningRate = float(input("Enter the Learning Rate: "))
numItr = int(input("Enter the number of Iterations: "))
trainingPattern = str(input("Enter the Training File name: "))
printOutput = True
if numItr >= 10:
    plotOutput = False
else:
    plotOutput = True


######################################################################################################
##                                      LVQ Class.                                                  ##
######################################################################################################


class LVQ:
    def __init__(self, learning_Rate, num_Itr, data_set, code_book):
        self.learningRate = learning_Rate
        self.numItr = num_Itr
        self.codebook = code_book
        self.dataset = data_set

    @staticmethod
    def euclideanDistance(copy_Code_book, copy_Dataset):  # To find the euclidean distance.
        distance = []
        for ii in range(len(copy_Dataset)):
            index = []
            for jj in range(len(copy_Code_book)):
                tmp = copy_Dataset[ii] - copy_Code_book[jj]
                tmp = tmp[0] ** 2 + tmp[1] ** 2
                index.append(tmp)
            distance.append(index)
        return distance

    @staticmethod
    def priority(dis):  # to find the winning class
        winningClass = []
        winningPos = []
        for element in range(len(dis)):
            index = dis[element].index(min(dis[element]))
            winningPos.append(index)
            if index == 0:
                winningClass.append('+')
            if index == 1:
                winningClass.append('^')
            if index == 2:
                winningClass.append('*')
            if index == 3:
                winningClass.append('o')
        return winningClass, winningPos

    @staticmethod
    def target(copy_Dataset, data_set):  # to find the Target class
        targetClass = []
        for ii in range(len(copy_Dataset)):
            for jj in range(len(data_set)):
                if copy_Dataset[ii][0] == data_set[jj][0][0] and copy_Dataset[ii][1] == data_set[jj][0][1]:
                    targetClass.append(data_set[jj][1][0])
        return targetClass

    @staticmethod
    def accuracy(targetClass, winningClass):  # to find accuracy
        correct = 0
        wrong = 0
        for ii in range(len(targetClass)):
            if targetClass[ii] == winningClass[ii]:
                correct += 1
            if targetClass[ii] != winningClass[ii]:
                wrong += 1
        return (correct / (correct + wrong)) * 100

    def plotting(self, winningClass, acc, itr):  # for plotting outputs for subsequent iterations during learning
        marker = 0
        for element in range(int(math.sqrt(len(winningClass)))):
            for element_2 in range(int(math.sqrt(len(winningClass)))):
                if winningClass[marker] == '+':
                    plt.plot(element, element_2, 'r+')
                if winningClass[marker] == '^':
                    plt.plot(element, element_2, 'k^')
                if winningClass[marker] == '*':
                    plt.plot(element, element_2, 'b*')
                if winningClass[marker] == 'o':
                    plt.plot(element, element_2, 'go')
                marker = marker + 1

        plt.title(
            f"LVQ datapoints in {x_limit}x{y_limit} square.\nIteration: {itr + 1}\nAccuracy: {acc:.3f}%, LearningRate: {self.learningRate:.3f}",
            fontsize=9)
        plt.xlabel(f"X-axis", fontsize=10)
        plt.ylabel('Y-axis', fontsize=10)
        plt.show()

    def updateWeight(self, copy_Codebook, copy_Dataset):  # updating the codebook as per the winning class
        file_output = open(r"output.txt", "w")
        file_output.write(f"Training:\n")

        for itr in range(self.numItr):
            # self.learningRate = self.learningRate * (1.0 - (itr / float(self.numItr)))

            dis = self.euclideanDistance(copy_Codebook, copy_Dataset)
            winningClass, winningPos = self.priority(dis)
            targetClass = self.target(copy_Dataset, self.dataset)

            for element in range(len(targetClass)):
                if targetClass[element] == winningClass[element]:
                    copy_Codebook[winningPos[element]] = copy_Codebook[winningPos[element]] + self.learningRate * (
                            copy_Dataset[element] - copy_Codebook[winningPos[element]])
                if targetClass[element] != winningClass[element]:
                    copy_Codebook[winningPos[element]] = copy_Codebook[winningPos[element]] - self.learningRate * (
                            copy_Dataset[element] - copy_Codebook[winningPos[element]])
            acc = self.accuracy(targetClass, winningClass)

            if printOutput:
                file_output.write(f"#---------------------------------------#\n")
                file_output.write(f"#---------------------------------------#\n")
                file_output.write(f"> Iteration: {itr + 1}\n")
                file_output.write(f"> Accuracy: {acc:.4f}\n")
                file_output.write(f"> LearningRate: {self.learningRate:.4f}\n")
                file_output.write(f"> Updated Weight Matrix: \n")
                for element in range(len(self.codebook)):
                    file_output.write(f"  {copy_Codebook[element]} : {self.codebook[element][1][0]}\n")
                file_output.write('\n')
                wc = np.array(winningClass).reshape(len(ds), len(ds))
                wc1 = wc.T
                for element in range(len(ds) - 1, -1, -1):
                    for element2 in range(len(ds)):
                        file_output.write(f" {wc1[element][element2]} ")
                    file_output.write('\n')
            if plotOutput:
                self.plotting(winningClass, acc, itr)
            self.learningRate = self.learningRate * (1 - 5 / 100)
            if self.learningRate < 5 / 100 * learningRate:
                print(
                    f"Minimum Learning Rate reached !\nIteration: {itr}\nAccuracy: {acc}\nLearningRate: {self.learningRate} ")
                file_output.write(f"#---------------------------------------#\n")
                file_output.write(
                    f"Minimum Learning Rate reached !\nIteration: {itr}\nAccuracy: {acc}\nLearningRate: {self.learningRate}\n ")
                file_output.write(f"#---------------------------------------#\n")
                break
        file_output.close()
        return copy_Codebook


######################################################################################################
##                        Creating DataSet and required Codebook from the .txt file                 ##
######################################################################################################
classes = ['+', '^', 'o', '*']

ds = []
file = open(trainingPattern, "r")
line = file.readline()

for i in range(len(line)):  # Checking the size of the training pattern.
    if line[i] == 'x':
        temp = i
x_limit = int(line[temp - 1]) + int(line[temp - 2]) * 10
y_limit = int(line[temp + 1]) * 10 + int(line[temp + 2])

for i in range(y_limit):
    line = file.readline()
    ds.append(line)

x, y = range(0, len(ds[0]) - 1), range(0, len(ds))
x, y = np.array(x), np.array(y)

dataset_reverse = ds
dataset_reverse.reverse()
dataset11 = []

for i in range(len(dataset_reverse)):
    for j in range(len(dataset_reverse[i])):
        if dataset_reverse[i][j] != '\n':
            dataset11.append([[j, i], [dataset_reverse[i][j]]])

dataset11.sort(key=lambda var: var[0])
dataset = dataset11

codebook = [dataset[0], dataset[len(ds) - 1], dataset[(len(ds) - 1) * len(ds)], dataset[-1]]
codebook.sort(key=lambda var: var[0])

######################################################################################################
##                               Plotting the Training dataset                                      ##
######################################################################################################
for i in range(len(dataset)):
    if dataset[i][1][0] == '+':
        plt.plot(dataset[i][0][0], dataset[i][0][1], 'r+')
    if dataset[i][1][0] == 'o':
        plt.plot(dataset[i][0][0], dataset[i][0][1], 'go')
    if dataset[i][1][0] == '*':
        plt.plot(dataset[i][0][0], dataset[i][0][1], 'b*')
    if dataset[i][1][0] == '^':
        plt.plot(dataset[i][0][0], dataset[i][0][1], 'k^')
plt.title(f"LVQ datapoints in {x_limit}x{y_limit} square.\nTraining Set", fontsize=10)
plt.xlabel(f"X-axis", fontsize=10)
plt.ylabel('Y-axis', fontsize=10)
plt.show()

## Creating copyDataset and copyCodebook as numpy arrays for carrying out
## vector Arithmetic.

copyDataset = []
for i in range(len(dataset)):
    copyDataset.append(dataset[i][0])

copyCodebook = []
for i in range(len(codebook)):
    copyCodebook.append(codebook[i][0])
copyDataset = np.array(copyDataset, dtype=float)
copyCodebook = np.array(copyCodebook, dtype=float)

######################################################################################################
##                                      Creating LVQ Network                                        ##
######################################################################################################
network = LVQ(learningRate, numItr, dataset, codebook)
updated_weight = network.updateWeight(copyCodebook, copyDataset)

# targetClass = network.target(copyDataset, dataset)
# print(f"\nTargetClass:\n{targetClass}")
######################################################################################################
##                                              END                                                 ##
######################################################################################################
