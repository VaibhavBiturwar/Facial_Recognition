###to install dlib
#$ sudo apt-get install build-essential cmake
#$ sudo apt-get install libgtk-3-dev
#$ sudo apt-get install libboost-all-dev


import face_recognition as fr


image = fr.load_image_file('./img/groups/team2.jpg')
floc = fr.face_locations(image)

#print(floc)
print(f"total no of faces  {len(floc)}")