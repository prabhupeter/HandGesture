import os
os.chdir('C:/Users/Admin/Documents/Day-14/images/Harry_potter')
i=1
for file in os.listdir():
    src=file
    dst="abc"+str(i)+".png"
    os.rename(src,dst)
    i+=1

