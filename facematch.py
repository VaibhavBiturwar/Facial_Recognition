import face_recognition as fr


known = fr.load_image_file('./img/known/Steve Jobs.jpg')
# to get face features
known_face_features = fr.face_encodings(known)[0]
#print(known_face_features)

#print("-"*20)

#unknown = fr.load_image_file('./img/unknown/obama.jpeg')
unknown = fr.load_image_file('./img/unknown/stevejobs.png')
unknown_face_features = fr.face_encodings(unknown)[0]
#print(unknown_face_features)


result = fr.compare_faces([known_face_features] , unknown_face_features)

if result[0]:
    print("Match Found")
else:
    print("No match Found")



