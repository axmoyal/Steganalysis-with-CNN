import os 
os.chdir('data')
for directory in ['Cover','JMiPOD','JUNIWARD','UERD','Test']:
    os.chdir(directory)
    L_image=os.listdir()
    i=0
    for i in range(len(L_image)):
        os.rename(L_image[i],str(i).rjust(5,"0")+".jpeg")
    os.chdir('..')
    print(os.listdir())