import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y1 = [2, 3, 5, 7, 11]
y2 = [1, 4, 6, 8, 10]

plt.plot(x, y1, label='Series A', color='blue', linestyle='--')
plt.plot(x, y2, label='Series B', color='red', marker='o')

plt.title("Comparison of Series A and B")
plt.legend()
plt.show()
