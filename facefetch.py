from PIL import Image
import face_recognition as fr

print("Loading Image...")
image = fr.load_image_file('./img/groups/team1.jpg')
floc = fr.face_locations(image)
print("Loading Completed")

print("Finding Faces ....")
for loc in floc:
    top,right,bottom,left = loc


    face_image = image[top:bottom , left:right]
    pil_image = Image.fromarray(face_image)
    print("Found Face")
    pil_image.show()

print("Finding Faces completed")