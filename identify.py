import face_recognition as fr
from PIL import Image , ImageDraw


bill_image = fr.load_image_file('./img/known/Bill Gates.jpg')
bill_face_features = fr.face_encodings(bill_image)[0]

steve_image = fr.load_image_file('./img/known/Steve Jobs.jpg')
steve_face_features = fr.face_encodings(steve_image)[0]

elon_image = fr.load_image_file('./img/known/Elon Musk.jpg')
elon_face_features = fr.face_encodings(elon_image)[0]


known_face_encodings = [ bill_face_features , steve_face_features , elon_face_features ]
known_face_names = [ "Bill Gates" , "Steve Jobs" , "Elon Musk"]

#loading text file

test_image = fr.load_image_file('./img/groups/bill-steve-elon.jpg')
face_locations = fr.face_locations(test_image)
face_encodings = fr.face_encodings(test_image , face_locations)


pil_image = Image.fromarray(test_image)
draw = ImageDraw.Draw(pil_image)

for (top,right,bottom,left) , face_encoding in zip(face_locations , face_encodings):
    match = fr.compare_faces( known_face_encodings , face_encoding )

    name = "Unknown Person"

    if True in match:
        match_index = match.index(True)
        name = known_face_names[match_index]

    draw.rectangle(((left, top), (right, bottom)), outline='red')

    text_width, text_height = draw.textsize(name)

    draw.rectangle(((left, top  - text_height), (right, top)), fill="black", outline='red')

    draw.text((left + 6, top - text_height), name, fill='white')

del draw

pil_image.show()
