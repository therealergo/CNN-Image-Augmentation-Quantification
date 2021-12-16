import numpy as np

vals = []
for param in range(0, 180):
	try:
		arr_v = np.loadtxt(open("./out/" + str(param) + "val_accuracy_last.csv", "rb"), delimiter=",", skiprows=0)
		arr_t = np.loadtxt(open("./out/" + str(param) + "tra_accuracy_last.csv", "rb"), delimiter=",", skiprows=0)
		arr_v_end = arr_v[75-10:75]
		arr_t_end = arr_t[75-10:75]
		val_v = np.average(arr_v_end)
		val_t = np.average(arr_t_end)
		vals.append([param, val_v, val_t])
	except FileNotFoundError as e:
		pass
print(vals)
np.savetxt("./res.csv", vals, delimiter=",")
