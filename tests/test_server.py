"""Exercise Flask routes without loading OCR weights or making network requests."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
ocr = types.ModuleType('own_ocr.wyswietl_tekst_multinomial')
ocr.read_text_from_image = Mock()
lookup = types.ModuleType('selen')
lookup.find_element = Mock(return_value='Error')
comparison = types.ModuleType('compar')
comparison.compare_to_database = Mock()
with patch.dict(sys.modules, {ocr.__name__: ocr, 'selen': lookup, 'compar': comparison}):
    spec = importlib.util.spec_from_file_location('scanner_server', ROOT / 'electro_scanner/server.py')
    server = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = server
    spec.loader.exec_module(server)

class ServerTests(unittest.TestCase):
    def test_home_renders(self):
        self.assertEqual(server.app.test_client().get('/').status_code, 200)

    def test_manual_search_normalizes_input_without_calling_ocr(self):
        response = server.app.test_client().post('/', data={'text':'abc123'})
        lookup.find_element.assert_called_with('ABC123')
        self.assertEqual(response.status_code, 200)
        self.assertIn('error', response.get_json())
        ocr.read_text_from_image.assert_not_called()

if __name__ == '__main__': unittest.main()
