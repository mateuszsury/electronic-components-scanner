from flask import Flask, render_template, request, jsonify
from own_ocr.wyswietl_tekst_multinomial import read_text_from_image
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from selen import find_element
from compar import compare_to_database
import re

app = Flask(__name__)

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
                    extracted_text = clean_string(read_text_from_image(file, r'own_ocr\model_and_vectorizer_multinomial.pkl')[0].upper())
                    print('Extracted text by OCR:', extracted_text)
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
                if ' ' in extracted_text:
                    extracted_text = extracted_text.split(' ')
                result = None
                if isinstance(extracted_text, list):
                    print('Element is list')
                    for ex in extracted_text:
                        if not ex.isdigit():
                            print('Finding: ', ex)
                            result = find_element(ex)
                            if result:
                                extracted_text = ex
                                break
                            else:
                                if 'S' in ex:
                                    _ex = ex.replace('S', '5')
                                    result = find_element(_ex)
                                    if result:
                                        extracted_text = ex
                                        break
                        else:
                            continue
                else:
                    print('Element is not list')
                    result = find_element(extracted_text)
                return send_result(result, extracted_text)
        else:
            return jsonify(error="No file part")

    return render_template("index.html")


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
    app.run(debug=True, host='0.0.0.0', port=5000)
