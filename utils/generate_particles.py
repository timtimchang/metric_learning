import os
import sys

import numpy as np
from sphere_space import SphereSpace

if(len(sys.argv) != 5):
    print ("\nusage: python generate_particles.py dimension num_samples start_idx end_idx\n")
    sys.exit()


DIMENSION = int(sys.argv[1])
num_samples = int(sys.argv[2])
start = int(sys.argv[3])
end = int(sys.argv[4])

sphere = SphereSpace(DIMENSION)

for i in range(start, end):
    samples = sphere.sample(num_samples)
    #if (i == 0): print (samples)
    temp_str = str(DIMENSION) + "_" + str(num_samples)
    directory = '../data/sphere_particles_' + temp_str 
    if not os.path.exists(directory):
            os.makedirs(directory)

    file_name = directory + '/' + temp_str + '_p' + str(i) + '.npy'
    np.save(file_name, samples)
    print(file_name)
    print("generate_particles success")




