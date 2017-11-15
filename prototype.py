import numpy as np
import matplotlib.pyplot as plt
import csv
from matplotlib import style
style.use("ggplot")
from sklearn import svm



#x = [1, 5, 1.5, 8, 1, 9]
#y = [2, 8, 1.8, 8, 0.6, 11]

X = np.array([[11.58,11],
             [4.19,13],
             [9.44,15],
             [9.22,19],
             [7.12,20],
             [11.23,15],
             [10.34,18]])
y = [1,0,1,1,0,1,1]

clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(X,y)


w = clf.coef_[0]
print(w)

a = -w[0] / w[1]

xx = np.linspace(0,12)
yy = a * xx - clf.intercept_[0] / w[1]

h0 = plt.plot(xx, yy, 'k-', label="Linear SVC")

plt.scatter(X[:, 0], X[:, 1], c = y)
plt.legend()
plt.show()