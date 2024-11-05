
import matplotlib.pyplot as plt

size = [4,10,20,50,100]

CPU = [0,
0.00001001358032,
0.00002002716064,
0.0004298686981,
0.003179788589]

GPU = [0.0001997947693,
0.0001199245453,
0.0001399517059,
0.0001299381256,
0.0001201629639]

tiled = [0.0001399517059,
0.01360988617,
0.0001602172852,
0.01271986961,
0.01487994194]

cublas = [0.002210140228,
0.000519990921,
0.000550031662,
0.000519990921,
0.0005300045013]


# plot lines
plt.plot(size, CPU, label = "CPU", color = "blue")
plt.plot(size, GPU, label = "GPU", color = "red")
plt.plot(size, tiled, label = "tiled", color = "black")
plt.plot(size, cublas, label = "cublas", color = "green")
plt.legend()
plt.title("size of matrix vs time")
#whole number ticks
plt.locator_params(axis="both", integer=True, tight=True)
plt.show()
