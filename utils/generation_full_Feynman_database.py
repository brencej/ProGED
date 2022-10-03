# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 16:04:47 2021

@author: jureb
"""
import os

import pandas as pd

if __name__ == "__main__":
    
    name = "progedDimTest3"
    workers = 1
    
    output_folder = name + "/" 
    os.makedirs(os.path.dirname(output_folder), exist_ok=True)
    
    eqfile = "../source/FeynmanEquations.csv"
    reference = pd.read_csv(eqfile)
    
    for eqN in range(0, 5):
        datafile = reference["Filename"][eqN]
              
        job_name = name + "_eq" + str(eqN) 
        job_folder = output_folder + job_name + "/"
        os.makedirs(os.path.dirname(job_folder), exist_ok = True)
            
        txt = ''
        txt += '&\n'
        txt += '(executable = "run_'+name+ '.sh")\n'
        txt += '(inputFiles =\n'
        txt += '  ("run_'+name+'.sh" "run_'+name+'.sh")\n'
        txt += '  ("feynman_env.sif" "/media/sf_shared/cluster_jobs/feynman_env.sif")\n'
        txt += '  ("source.tar" "/media/sf_shared/cluster_jobs/feynman_universal_2/source.tar")\n'
        #txt += '  ("Feynman_with_units_p'+str(i)+'.npz" "gsiftp://dcache.arnes.si/data/arnes.si/gen.vo.sling.si/brence/feynman/Feynman_with_units_p'+str(i)+'.npz")\n'
        txt += '  ("' + datafile + '" "/media/sf_shared/cluster_jobs/Feynman_with_units/' + datafile +'")\n'
        txt += ')\n'
        txt += '(outputFiles =\n'
        txt += '("results.tar" "results.tar")\n'
        txt += '("feynman.log" "feynman.log")\n'
        txt += ')\n'
        txt += '(walltime = "2 days")\n'
        txt += '(memory=2000)\n'
        txt += '(stdout="standard.log")\n'
        txt += '(count='+str(workers)+')\n'
        txt += '(countpernode='+str(workers)+')\n' 
        #txt += '(queue = "grid")\n'
        txt += '(jobname = "'+job_name+'")'
        with open(job_folder + name + ".xrsl", "w", newline='\n') as file:
            file.write(txt)
            
        txt2 = ''
        txt2 += 'tar -xf source.tar\n'
        txt2 += 'mkdir -p results\n'
        txt2 += 'singularity exec feynman_env.sif python3.7 source/utils/estimation_all_Feynman_database.py '+str(eqN)+' ' + str(workers) + ' >> feynman.log\n'
        txt2 += 'wait\n'
        txt2 += 'tar -cf results.tar results'
        with open(job_folder + 'run_'+name+'.sh', "w", newline='\n') as file:
            file.write(txt2)