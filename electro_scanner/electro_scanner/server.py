from flask import Flask, render_template, request, jsonify
from PIL import Image
import easyocr
from selen import find_element
from compar import compare_to_database
import re
from ocr import process_image

app = Flask(__name__)

reader = easyocr.Reader(['en'])

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        print('Got request')
        extracted_text = None
        if 'text' in request.form:
            extracted_text = request.form.get('text').upper()
            result = find_element(extracted_text)
            print(result)
            return send_result(result, extracted_text)
        elif 'file' in request.files:
            file = request.files["file"]

            if file.filename == "":
                return jsonify(error="No selected file")

            if file:
                is_image = request.form.get('is_image') == 'true'  # Sprawdzenie czy przesłano obraz

                if is_image:
                    # Przetwarzanie obrazu za pomocą OCR
                    extracted_text = ocr_image(file)
                    _extracted_text = None
                    if len(extracted_text) > 3:
                        try:
                            extr_split = extracted_text.split(' ')
                            for text in extr_split:
                                if not text.isdigit():
                                    try:
                                        __extracted_text = compare_to_database(text)
                                        if __extracted_text:
                                            _extracted_text = __extracted_text
                                            print('Compared element: ', text)
                                            break
                                    except:
                                        pass
                        except:
                            if not extracted_text.isnumeric():
                                _extracted_text = compare_to_database(extracted_text)
                        if _extracted_text:
                            extracted_text = _extracted_text
                            print('Using database improvement')

                if not extracted_text:
                    return jsonify(error="Unable to extract text from image or text")
                print('Extracted text: ', extracted_text)
                extracted_text = extracted_text.split(' ')
                result = None
                for ex in extracted_text:
                    if not ex.isdigit():
                        print('Finding: ', ex)
                        result = find_element(ex)
                        if result:
                            break
                    else:
                        continue
                return send_result(result, extracted_text)
        else:
            return jsonify(error="No file part")

    return render_template("index.html")

def ocr_image(file):
    file_name = process_image(file)
    image = Image.open(file_name)
    result = reader.readtext(image=image, rotation_info=[10,20,30,40,50,60,350,340,330,320,310,300])
    text = ' '.join([res[1] for res in result])
    print('OCR text: ', text)
    return clean_string(text)

def clean_string(string):
    # Użyj wyrażeń regularnych, aby znaleźć tylko litery, liczby i spacje
    cleaned_string = re.sub(r'[^a-zA-Z0-9\s]', '', string)
    return cleaned_string

def send_result(result, extracted_text):
    if result:
        if result == 'Error':
            return jsonify(error="Unable to find part from image or text", extracted_text=extracted_text)
        elif 'Resistor' in result:
            return jsonify(extracted_text=extracted_text, resistor=result[1])
        elif len(result[0]) == 1:
            return jsonify(extracted_text=extracted_text, data=result[0][0], link=result[1][0])
        elif len(result[0]) == 2:
            return jsonify(extracted_text=extracted_text, data=result[0][0], link=result[1][0],
                           data1=result[0][1], link1=result[1][1])
        elif len(result[0]) == 3:
            return jsonify(extracted_text=extracted_text, data=result[0][0], link=result[1][0],
                           data1=result[0][1], link1=result[1][1], data2=result[0][2], link2=result[1][2])
        elif result:
            return jsonify(extracted_text=extracted_text, data=result[0], link=result[1])
        else:
            return jsonify(error='Error')
    else:
        return jsonify(error='Error')

if __name__ == "__main__":
    app.run(debug=True)
