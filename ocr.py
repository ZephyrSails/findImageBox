import os
from PIL import Image
import pytesseract

def image2Text(cropped_image):
	# print cropped_image
	# img = Image.open(cropped_image)
	# print img
	text = pytesseract.image_to_string(Image.open(cropped_image))
	# text = pytesseract.image_to_string(cropped_image)
	# print (text)
	return text

def main():
	crop_path = './crop/'
	predicted_image_text_csv = open('predicted_image_text.csv', 'w')
	# print crop_path
	images = os.listdir(crop_path)
	for image in images:
		print image
		if image == '.DS_Store': continue
		cropped_images_path = crop_path+image+'/'
		dirs = os.listdir(cropped_images_path)
		text_list = []
		for cropped_image in dirs:
			# print cropped_image
			# e.g. flowchart_image ='crop_0.jpg'
			text = image2Text(cropped_images_path + cropped_image)
			if (len(text) > 1):
				text_list.append(text.replace('\n', ''))
		text = ';'.join(text for text in text_list)
		print text
		predicted_image_text_csv.write(image + '\t' + text+'\n')
	predicted_image_text_csv.close()

if __name__ == "__main__":
    main()
