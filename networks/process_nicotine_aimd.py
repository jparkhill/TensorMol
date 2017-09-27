log = open("nicotine_aimd_log.txt")

for i in range(100000):
	line = log[i]
	if (line[range(0,4)] == "2017-" and line[range(45,49)] == "step:" and line[73] == "."):
			train_loss[i] = float(line[range(93,104)])
			energy_loss[i] = float(line[range(120,131)])
			grad_loss[i] = float(line[range(145,156)])

print(train_loss)
print(energy_loss)
print(grad_loss)

