import os
import numpy as np
import subprocess
import shutil

file_list = []
class_size = 0
y = []
k=0

training_folder = 'shapenet/train'
testing_folder = 'shapenet/test'

X_train, y_train = read_data(training_folder)
X_val, y_val = read_data(testing_folder)

#Saving data in the numpy files    
np.save('points_train.npy',X_train)
np.save('labels_train.npy',y_train)
np.save('points_test.npy',X_val)
np.save('labels_test.npy',y_val)


#Function to read the data
read_data(data_path):
	for i in range(0,len(dir_list)):
	    path = os.path.join(data_path,dir_list[i])
	    print(path)
	    for file in [doc for doc in os.listdir(path) if doc.endswith(".txt")]:
	        file_list.append(os.path.join(path,file))
	        y.append(k)
	    
	    print('Class '+str(k)+' :',len(file_list)-class_size)
	    class_size = len(file_list)
	    k = k+1
	    #print(file_list)
	#print(y)
	fpoints = np.zeros((class_size, 2000,3))
	flabels = np.zeros((class_size, 2000))

	for idx,file_prefix in enumerate(file_list):
	    
	    points = []
	    labels = []
	    with open(file_prefix, 'r') as txt_f:
	        for line in txt_f:
	            tokens = line.split()
	            points.append(tokens[:-1])
	            labels.append(tokens[-1])

	    lbs = np.asarray(labels).astype(np.float)
	    pts = np.asarray(points).astype(np.float)
	    pts = pts[:,:3]
	    print(pts.shape, pts.max(), pts.min())
	    
	    #Random sampling
	    rand_idx = np.random.randint(0, pts.shape[0]-1, 2000)
	    fpoints[idx] = pts[rand_idx]
	    flabels[idx] = lbs[rand_idx]

	return fpoints, flabels    
