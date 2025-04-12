from PIL import Image

image_dir = "/Users/hubbleloo/Documents/ResearchAssistant/Disease Classification/soyabean/DiseaseClassification/src/images/bacterial_blight"

# a background imitating soil
base_image_size = 500
base_image = Image.new("RGBA", (base_image_size, base_image_size), color=(70, 50, 32))

# Loading leaf image
leaf_image_path = "/Users/hubbleloo/Documents/ResearchAssistant/Disease Classification/soyabean/DiseaseClassification/src/images/healthy/output_healthy_118.png"
leaf_image = Image.open(leaf_image_path).convert("RGBA")
leaf_image1 = leaf_image.copy()
leaf_image2 = leaf_image.copy()


# Reduce the size of the leaf image
new_size = (100, 100)  # Target size (width, height)
leaf_image = leaf_image.resize(new_size, Image.Resampling.LANCZOS)
leaf_image1 = leaf_image1.resize(new_size, Image.Resampling.LANCZOS)
leaf_image2 = leaf_image2.resize(new_size, Image.Resampling.LANCZOS)
# Rotate the overlay image
angle1 = 90
angle2 = 180
rotated_image1 = leaf_image1.rotate(angle1)
rotated_image2 = leaf_image2.rotate(angle2)


# coordinate
x_offset = 150
y_offset = 150

x_offset1 = 150
y_offset1 = 50

x_offset2 = 50
y_offset2 = 150

# # Create a transparent canvas the same size as the base image
# canvas = Image.new("RGBA", base_image.size, (255, 255, 255, 0))
#
# # Paste the rotated overlay onto the canvas at the specified coordinates
# canvas.paste(rotated_image1, (x_offset1, y_offset1), rotated_image1)
# canvas.paste(leaf_image, (x_offset, y_offset), leaf_image1)
# canvas.paste(rotated_image2, (x_offset2, y_offset2), rotated_image2)
#
# canvas.paste(rotated_image1, (x_offset1+150, y_offset1), rotated_image1)
# canvas.paste(leaf_image, (x_offset+150, y_offset), leaf_image1)
# canvas.paste(rotated_image2, (x_offset2+150, y_offset2), rotated_image2)
#
# canvas.paste(rotated_image1, (x_offset1+50, y_offset1), rotated_image1)
# canvas.paste(leaf_image, (x_offset+50, y_offset), leaf_image1)
# canvas.paste(rotated_image2, (x_offset2+50, y_offset2), rotated_image2)
#
# canvas.paste(rotated_image1, (x_offset1, y_offset1+125), rotated_image1)
# canvas.paste(leaf_image, (x_offset+150, y_offset+150), leaf_image1)
# canvas.paste(rotated_image2, (x_offset2, y_offset2+150), rotated_image2)
#
# canvas.paste(rotated_image1, (x_offset1+65, y_offset1), rotated_image1)
# canvas.paste(leaf_image, (x_offset+100, y_offset), leaf_image1)
# canvas.paste(rotated_image2, (x_offset2+25, y_offset2), rotated_image2)
#
# canvas.paste(rotated_image1, (x_offset1, y_offset1+100), rotated_image1)
# canvas.paste(leaf_image, (x_offset, y_offset+75), leaf_image1)
# canvas.paste(rotated_image2, (x_offset2+75, y_offset2+100), rotated_image2)
#
# # composite_image = Image.alpha_composite(base_image, canvas)
#
#
# # composite_image.show()
#
# bacterial_blight_trifolite = trifoliate.get_trifoliate("bacterial_blight", (250,250), 25 )
# frogeye_trifolite = trifoliate.get_trifoliate("frogeye", (350,350), 75 )
# healthy_trifolite = trifoliate.get_trifoliate("healthy", (150,450), 155 )
#
# composite_image = Image.alpha_composite(bacterial_blight_trifolite, frogeye_trifolite)
# composite_image = Image.alpha_composite(composite_image, healthy_trifolite)
