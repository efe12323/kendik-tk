# my_library/examples/usage_examples.py

from my_library.algorithms.machine_learning import custom_classifier
from my_library.algorithms.nlp import custom_nlp_pipeline
from my_library.utils.helpers import load_data

# Örnek kullanım
if __name__ == "__main__":
    # Veri yükleme örneği
    data = load_data('data.csv')

    # Özel sınıflandırıcı kullanımı
    predictions = custom_classifier(data)
    print("Sınıflandırma sonuçları:", predictions)

    # Özel NLP işlemleri kullanımı
    text = "Örnek bir metin."
    processed_text = custom_nlp_pipeline(text)
    print("İşlenmiş metin:", processed_text)
