from flask import Flask, render_template, request, redirect, url_for
import os
import uuid
from PIL import Image, ImageEnhance, ImageFont, ImageDraw
from rembg import remove
#import backgroundremover
import torch
import numpy as np
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.static_folder = 'uploads'

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":

        # Get the uploaded image file
        image_file = request.files["image"]
        background_option = request.form.get("background") 
        shape_option = request.form.get("shape")
        led_option = request.form.get("led")
        fonts_option = request.form.get("fonts")

        # Generate a unique filename for the uploaded image
        filename = str(uuid.uuid4()) + ".png"
        image_path = os.path.join(
            app.config["UPLOAD_FOLDER"], filename
        )
        image_file.save(image_path)
        def resize_and_convert(image, max_width, max_height):
            width, height = image.size

            # Calculate the ratios of the dimensions
            ratio_w = float(max_width) / width
            ratio_h = float(max_height) / height  

            # Use the smaller ratio to resize the image
            ratio = min(ratio_w, ratio_h)

            new_width = int(max_width)
            new_height = int(max_height)

            # Resize the image
            resized_image = image.resize((new_width, new_height), Image.BICUBIC)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        #reseize image
        img_r = Image.open(image_path)
        img_r = resize_and_convert(img_r, 200,200)

        # Get the user input text
        text = request.form["text"]

        # Open the crystal image
        if shape_option == "Rectangle" and led_option == "without": 
            crystal = Image.open("ImageLoader_without_led.png")
        elif shape_option == "Rectangle" and led_option == "with": 
            crystal = Image.open("Small.jpg")
        elif shape_option == "MRectangle" and led_option == "with": 
            crystal = Image.open("Meduim.jpg")   
        elif shape_option == "LRectangle" and led_option == "with": 
            crystal = Image.open("Large.jpg") 
        elif shape_option == "XLRectangle" and led_option == "with": 
            crystal = Image.open("XL-1.jpg")
        elif shape_option == "XXLRectangle" and led_option == "with": 
            crystal = Image.open("XXL.jpg")
        elif shape_option == "SHeart" and led_option == "with": 
            crystal = Image.open("SHeart.jpg")
        elif shape_option == "LHeart" and led_option == "with": 
            crystal = Image.open("LHeart.jpg")
        elif shape_option == "horizental" and led_option == "with":
            crystal = Image.open("horizental_with_led.jpg")
        elif shape_option == "horizental" and led_option == "without":
            crystal = Image.open("horizental_without_led.jpg")
        elif shape_option == "Diamond" and led_option == "without":
            crystal = Image.open("Diamond_Without_Led.jpg")
        elif shape_option == "Diamond" and led_option == "with":
            crystal = Image.open("Diamond-1.jpg")
        elif shape_option == "Round" and led_option == "with":
            crystal = Image.open("round-1.jpg")
        elif shape_option == "LRound" and led_option == "with":
            crystal = Image.open("Lround.jpg")
        elif shape_option == "Iceberg" and led_option == "without":
            crystal = Image.open("iceberg_without_led.jpg")
        else:
            crystal = Image.open("iceberg_without_led.jpg")


        # Open the user image
        #user_image = Image.open(image_path)
        #user_image = user_image.resize((285, 193), Image.LANCZOS)
        #user_image.save("resized_image.png")

        # ... (continue integrating your existing image processing code)
        
        # Convert user image to grayscale
        user_image_gray = img_r.convert('L')

        # Resize user image to 50% of crystal size
        crystal_w, crystal_h = crystal.size
        if shape_option == "Diamond":
            new_w = int(crystal_w * 0.2)
            new_h = int(crystal_h * 0.2)
        elif shape_option =="Iceberg" and led_option == "with":
            new_w = int(crystal_w * 0.24)
            new_h = int(crystal_h * 0.37)
        elif shape_option =="Iceberg" and led_option == "without":
            new_w = int(crystal_w * 0.26)
            new_h = int(crystal_h * 0.26)
        elif shape_option == "horizental" and led_option == "with":
            new_w = int(crystal_w * 0.50)
            new_h = int(crystal_h * 0.50)
        elif shape_option == "Rectangle":
            new_w = int(crystal_w * 0.31)
            new_h = int(crystal_h * 0.31)
        elif shape_option == "LRectangle":
            new_w = int(crystal_w * 0.34)
            new_h = int(crystal_h * 0.34)
        elif shape_option == "MRectangle":
            new_w = int(crystal_w * 0.33)
            new_h = int(crystal_h * 0.33)
        elif shape_option == "LRound":
            new_w = int(crystal_w * 0.36)
            new_h = int(crystal_h * 0.42)
        elif shape_option == "SHeart":
            new_w = int(crystal_w * 0.20)
            new_h = int(crystal_h * 0.17)
        elif shape_option == "XLRectangle":
            new_w = int(crystal_w * 0.36)
            new_h = int(crystal_h * 0.36)
        elif shape_option == "XXLRectangle":
            new_w = int(crystal_w * 0.46)
            new_h = int(crystal_h * 0.46)
        else:
            new_w = int(crystal_w * 0.4)
            new_h = int(crystal_h * 0.4)
        user_image_resize = user_image_gray.resize((new_w, new_h))

        # Apply blending to darken user image
        blended_image = Image.new('L', user_image_resize.size, 130)  # Grayscale with intensity 125
        user_image_blend = Image.blend(user_image_resize, blended_image, 0.55)

        # Increase contrast of user image
        enhancer = ImageEnhance.Contrast(user_image_blend)
        user_image_contrast = enhancer.enhance(1.5)

        # Convert user image back to RGB
        user_image_rgb = user_image_contrast.convert('RGB')

        # Minimize width of user image by 50%
        new_width = int(new_w * 0.5)
        new_height = new_h
        user_image_minimize = user_image_rgb.resize((new_width, new_height))

        # Load the pre-trained U-Net model
        model = deeplabv3_resnet50(pretrained=True, progress=True)

        # Preprocess the user image
        preprocess = transforms.Compose([
            transforms.ToTensor()  # Convert image to tensor
        ])
        input_tensor = preprocess(user_image_minimize)

        # Set the model to evaluation mode
        model.eval()

        # Make predictions with U-Net
        with torch.no_grad():
            output = model(input_tensor.unsqueeze(0))['out']
            predictions = output.argmax(1)

        # Convert the resulting mask to a binary mask
        mask = predictions.squeeze().byte()

        # Convert mask to numpy array
        mask_np = np.array(mask)

        # Convert mask to alpha channel format (0s and 255s)
        alpha = np.where(mask_np == 0, 0, 255).astype(np.uint8)

        # Create a new image with alpha channel
        removed_background = remove(user_image_contrast)


        # Position the user image inside the crystal image

        if shape_option == "Diamond":
            x = int((crystal_w - new_width) / 2.25) # /3
            y = int((crystal_h - new_height) / 2.2) # /3
        elif shape_option =="Iceberg" and led_option == "with":
            x = int((crystal_w - new_width) / 2.45) # /3
            y = int((crystal_h - new_height) / 2.1) # /3
        elif shape_option == "Rectangle":
            x = int((crystal_w - new_width) / 2.5) # /3
            y = int((crystal_h - new_height) / 1.9) # /3
        elif shape_option == "MRectangle":
            x = int((crystal_w - new_width) / 2.6) # /3
            y = int((crystal_h - new_height) / 2.07) # /3
        elif shape_option == "LRectangle":
            x = 186
            y = int((crystal_h - new_height) / 2.45) # /3
        elif shape_option == "XLRectangle":
            x = 165
            y = int((crystal_h - new_height) / 2.45) # /3
        elif shape_option == "XXLRectangle":
            x = 141
            y = 74
        elif shape_option == "horizental" and led_option == "without":
            x = int((crystal_w - new_width) / 2.8) # /3
            y = int((crystal_h - new_height) / 3) # /3 
        elif shape_option == "Round":
            x = int((crystal_w - new_width) / 2.7) # /3
            y = int((crystal_h - new_height) / 2.35) # /3
        elif shape_option == "LRound":
            x = int((crystal_w - new_width) / 2.65) # /3
            y = int((crystal_h - new_height) / 3.65) # /3
        elif shape_option == "SHeart":
            x = int((crystal_w - new_width) / 2.28) # /3
            y = int((crystal_h - new_height) / 1.75) # /3
        else:
            x = int((crystal_w - new_width) / 2.8) # /3
            y = int((crystal_h - new_height) / 3.2) # /3

        # Paste the removed background image into the crystal image
        if background_option == 'with':
             crystal.paste(user_image_contrast, (x, y), user_image_contrast)
        else:
            crystal.paste(removed_background, (x, y), removed_background)

        # Save the final result
        crystal.save('result1.png')

# Ouvrir l'image finale
        image = Image.open('result1.png')

# Configurer le texte
        text = request.form["text"]
        if fonts_option=="Helvetica":
            fonttext='Helvetica.ttf'
        elif fonts_option=="Arial":
            fonttext='arial.ttf'
        elif fonts_option=="TNR":
            fonttext='times.ttf'
        elif fonts_option=="Calibri":
            fonttext='CALIBRI.ttf'
        elif fonts_option=="Cambria":
            fonttext='Cambria.ttf'
        elif fonts_option=="Oswald":
            fonttext="Oswald-Regular.ttf"
        else:
            fonttext='font.ttf'

        if shape_option == "Rectangle":
            font = ImageFont.truetype(fonttext, 18)
        elif shape_option == "Diamond" and led_option == "without":
            font = ImageFont.truetype(fonttext, 18)
        elif shape_option == "Diamond" and led_option == "with":
            font = ImageFont.truetype(fonttext, 19)
        else:
            font = ImageFont.truetype(fonttext, 21)
        fill = (255, 240, 255)

# Créer un objet Drawing
        draw = ImageDraw.Draw(image)

# Calculer la taille du texte
        textwidth, textheight = draw.textsize(text, font)

# Positionner le texte au centre
        if shape_option == "Diamond":
            x = (image.width - textwidth) / 2
        elif shape_option == "Rectangle":
            x = (image.width - textwidth) / 2.05
        elif shape_option == "MRectangle":
            x = (image.width - textwidth) / 2.02 
        elif shape_option == "LRectangle":
            x = (image.width - textwidth) / 2  
        elif shape_option == "XLRectangle":
            x = (image.width - textwidth) / 2.05  
        elif shape_option == "horizental" and led_option == "with":
            x = (image.width - textwidth) / 1.9
        elif shape_option == "Iceberg" and led_option == "without":
            x = (image.width - textwidth) / 2.1
        elif shape_option == "Iceberg" and led_option == "with":
            x = (image.width - textwidth) / 2.1
        elif shape_option == "Round":
            x = (image.width - textwidth) / 2
        elif shape_option == "LRound":
            x = (image.width - textwidth) / 2
        elif shape_option == "SHeart":
            x = (image.width - textwidth) / 2
        else:
            x = (image.width - textwidth) / 2.1
        if shape_option == "MRectangle": 
            y = image.height - 205
        elif shape_option == "LRectangle": 
            y = image.height - 205
        elif shape_option == "XLRectangle": 
            y = image.height - 192
        elif shape_option == "XXLRectangle": 
            y = image.height - 198
        elif shape_option == "LRound": 
            y = image.height - 262
        elif shape_option == "Rectangle":
            y = image.height - 200
        elif shape_option == "horizental":
            y = image.height - 190
        elif shape_option == "Diamond":
            y = image.height - 265
        elif shape_option =="Iceberg" and led_option == "with":
            y = image.height - 205  
        elif shape_option == "Round":
            y = image.height - 205
        elif shape_option == "SHeart":
            y = image.height - 210
        else:
            y = image.height - 155

# Ajouter le texte sur l'image
        draw.text((x, y), text, font=font, fill=fill)

# Enregistrer l'image finale
        saved_filename = 'saved_result22.png'
        full_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        image.save(full_path)

        # Redirect to the result page with the processed image filename
        return redirect(url_for('show_image', filename=saved_filename))

    return render_template('index.html')


from PIL import Image, ImageDraw, ImageFont
import os
from flask import request, redirect, url_for, render_template
import uuid

@app.route("/LED", methods=["GET", "POST"])
def led():
    if request.method == "POST":
        try:
            # Get form inputs
            text = request.form["text"]
            font_option = request.form.get("fonts", "Helvetica")

            background_image_path = "LED.jpeg"
            font_mapping = {
                "Helvetica": 'Helvetica.ttf',
                "Arial": 'arial.ttf',
                "TNR": 'times.ttf',
                "Calibri": 'CALIBRI.ttf',
                "Cambria": 'Cambria.ttf',
                "Oswald": "Oswald-Regular.ttf"
            }
            font_path = font_mapping.get(font_option, 'font.ttf')

            with Image.open(background_image_path) as background_image:
                font = ImageFont.truetype(font_path, 78)
                draw = ImageDraw.Draw(background_image)

                letter_spacing = 4
                text_width, text_height = draw.textsize(text, font)
                text_width += letter_spacing * (len(text) - 1)
                x_start = (background_image.width - text_width) // 2
                y = background_image.height - 280

                draw.text((x_start, y), text, font=font, fill='#664223', spacing=letter_spacing)

                center = (background_image.width // 2, y + text_height // 2)
                midpoint = y + text_height // 2

                rotated_top = background_image.crop((0, 0, background_image.width, midpoint)).rotate(-3, center=center, resample=Image.BICUBIC)
                rotated_bottom = background_image.crop((0, midpoint, background_image.width, background_image.height)).rotate(3, center=center, resample=Image.BICUBIC)

                final_image = Image.new('RGBA', (background_image.width, background_image.height))
                final_image.paste(rotated_top, (0, 0))
                final_image.paste(rotated_bottom, (0, midpoint))

                saved_filename = f'led_text_{uuid.uuid4()}.png'
                full_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
                final_image.save(full_path, optimize=True)

                return redirect(url_for('show_image', filename=saved_filename))

        except Exception as e:
            app.logger.error(f"Error processing image: {str(e)}")
            return "An error occurred while processing the image. Please check the logs for more details.", 500

    return render_template('index1.html')
@app.route('/show_image/<filename>')  # New route for accessing image path in template
def show_image(filename):
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return render_template('result.html', processed_image=filename)

if __name__ == '__main__':
    app.run(debug=True)


#from PIL import Image, ImageDraw, ImageFont
#im = Image.new("RGB", (1600, 500), "black")
#font_file = "Tests/fonts/FreeMono.ttf"
#text_var = ImageDraw.Draw(im)

#color_var = (255, 255, 255)
#tuple_var = (200, 400)_
#string_var = "Today's date: 11/06/2024"
#saved_filename = 'saved_result33.png'
#full_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
#image.save(full_path)
#redirect to the result page with the processed image filename
#return redirect(url_for('show_image', filename=saved_filename))
#return render_template('index.html')
#font_var = ImageFont.truetype(font_file, 70)
#gap = 5

#for char in string_var:
#	text_var.text(tuple_var, char, color_var, font = font_var, align = 'center')
#	width = text_var.textsize(char, font = font_var)[0] + gap
#	tuple_var = (tuple_var[0]+width, tuple_var[1])

#im.save("out.png")
