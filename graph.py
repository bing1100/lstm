import matplotlib.pyplot as plt
import numpy as np


# Avg:  0.8486640383386582
# Avg:  0.8370811999999999
# Avg:  0.846154
# max:  0.9688
# max:  0.8799
# max:  0.8717

# Avg:  0.853979552715655
# Avg:  0.8433912000000001
# Avg:  0.8561159999999999
# max:  0.9844
# max:  0.8803
# max:  0.8761


files = [
    # "lstm_imdb_no_update1.txt",
    # "lstm_imdb_no_update2.txt",
    # "lstm_imdb_no_update3.txt",
    # "lstm_imdb_no_update4.txt",
    # "lstm_imdb_no_update5.txt",
    "lstm_imdb_update1.txt",
    "lstm_imdb_update2.txt",
    "lstm_imdb_update3.txt",
    "lstm_imdb_update4.txt",
    "lstm_imdb_update5.txt",
]

sumTrainAcc = 0
sumValAcc = 0
sumTestAcc = 0

maxTrainAcc = 0
maxValAcc = 0
maxTestAcc = 0

for file in files:
    f = open(file, "r")
    trainAcc = []
    valAcc = []
    testAcc = []
    for x in f:
        if "Training acc" in x:
            trainAcc.append(float(x[-6:]))
        if "Validation acc" in x:
            valAcc.append(float(x[-6:]))
        if "Testing Accuracy" in x:
            testAcc.append(float(x[-7:-2])/100)
    f.close()

    # Create trainAcc Graph
    x2 = np.arange(0, 1565, 1)
    y2 = trainAcc[:1566]

    # plotting the line 2 points 
    plt.plot(x2, y2, label = "Training Acc")

    # Create trainAcc Graph
    x2 = np.arange(0, 1565, 31.3)
    y2 = valAcc[:50]

    # plotting the line 2 points 
    plt.plot(x2, y2, label = "Validation Acc")

    # Create trainAcc Graph
    x2 = np.arange(0, 1565, 160)
    y2 = testAcc[:10]

    # plotting the line 2 points 
    plt.plot(x2, y2, label = "Testing Acc")
    
    sumTrainAcc += np.mean(trainAcc)
    sumValAcc += np.mean(valAcc)
    sumTestAcc += np.mean(testAcc)
    
    maxTrainAcc = max(max(trainAcc), maxTrainAcc)
    maxValAcc = max(max(valAcc), maxValAcc)
    maxTestAcc = max(max(testAcc), maxTestAcc)
    
    print(np.mean(trainAcc))
    print(np.mean(valAcc))
    print(np.mean(testAcc))
    
print("Avg: ",sumTrainAcc/len(files))
print("Avg: ",sumValAcc/len(files))
print("Avg: ",sumTestAcc/len(files))

print("max: ",maxTrainAcc)
print("max: ",maxValAcc)
print("max: ",maxTestAcc)

# naming the x axis
plt.xlabel('Iterations')
# naming the y axis
plt.ylabel('valAcc/IoU')
# giving a title to my graph
plt.title('Model valAcc and testAcc vs Iterations')
  
# show a legend on the plot
plt.legend()
plt.grid()
  
# function to show the plot
plt.show()