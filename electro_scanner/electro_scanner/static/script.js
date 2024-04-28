
$(document).ready(function() {
    $('#submit-button').hide();
    // Funkcja obsługująca wysyłkę pliku
    function sendFile() {

        var formData = new FormData();
        var fileInput = $('input[type=file]')[0];

        // Sprawdzenie czy został wybrany plik obrazu
        var isImageFile = fileInput.files.length > 0 && fileInput.files[0].type.startsWith('image');

        // Pobranie tekstu z pola tekstowego
        var textInput = $('#text-input').val().trim();

        if (isImageFile) {
            formData.append('is_image', 'true'); // Oznaczenie, że przesłano plik obrazu
            formData.append('file', fileInput.files[0]); // Dodanie pliku obrazu do danych formularza
        } else {
            formData.append('is_image', 'false'); // Oznaczenie, że przesłano tekst
            formData.append('text', textInput); // Dodanie tekstu do danych formularza
        }

        $.ajax({
            url: '/',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(data) {
                if (data.error) {
                    $('#result').html('<div class="error-message"><h2>Error:</h2><p>' + data.error + '</p><br><h2>Extracted Text:</h2><p>' + data.extracted_text + '</p></div>');
                } else {
                    var html = '<div class="extracted-text"><h2>Extracted Text:</h2><p>' + data.extracted_text + '</p></div>';

                    // Dodajemy obsługę innych danych zwracanych przez serwer
                    if (data.resistor) {
                        html += '<div class="result"><h2>Resistor:</h2><p>' + data.resistor + '</p></div>';
                    }
                    if (data.data && data.link) {
                        html += '<div class="result"><h2>First match:</h2><p>' + data.data + '</p><a href="' + data.link + '" target="_blank">Datasheet</a></div>';
                    }
                    if (data.data1 && data.link1) {
                        html += '<div class="result"><h2>Second match:</h2><p>' + data.data1 + '</p><a href="' + data.link1 + '" target="_blank">Datasheet</a></div>';
                    }
                    if (data.data2 && data.link2) {
                        html += '<div class="result"><h2>Third match:</h2><p>' + data.data2 + '</p><a href="' + data.link2 + '" target="_blank">Datasheet</a></div>';
                    }
                    $('#result').html(html);
                }

                // Po otrzymaniu odpowiedzi, pokazanie przycisku "Prześlij" i ukrycie kręcącego się kółka
                $('#loading-spinner').hide();
            },
            error: function() {
                // Obsługa błędu, pokazanie przycisku "Prześlij" i ukrycie kręcącego się kółka
                $('#loading-spinner').hide();
            }
        });
    }

    // Dodanie obsługi zdarzenia kliknięcia przycisku "Prześlij"
    $('#submit-button').on('click', function(event) {
        event.preventDefault(); // Zapobieganie domyślnemu zachowaniu formularza

        // Ukrycie przycisku "Prześlij" i pokazanie kręcącego się kółka
        $('#submit-button').hide();
        $('#loading-spinner').show();

        sendFile(); // Wywołanie funkcji wysyłki pliku
    });

    // Dodanie obsługi zdarzenia zmiany pliku
    $('input[type=file]').on('change', function() {
        var fileName = $(this).val().split('\\').pop(); // Pobranie nazwy pliku
        if (fileName) {
            $('label[for="file-upload"]').text('Wybrano plik: ' + fileName); // Zmiana tekstu na "Wybrano plik: nazwa_pliku"
            // Ukrycie wszystkich znalezionych poprzednio danych
            $('#result').empty();
        } else {
            $('label[for="file-upload"]').text('Wybierz plik'); // Jeśli nie wybrano pliku, zmień tekst z powrotem
        }
    });
});