# Import işlemleri
from my_library.algorithms.machine_learning import custom_classifier
from my_library.algorithms.nlp import custom_nlp_pipeline
from my_library.utils.helpers import load_data

def main():
    # Veri yükleme
    data = load_data('veri.csv')

    # Özel sınıflandırıcı kullanımı
    predictions = custom_classifier(data)
    print("Sınıflandırma sonuçları:", predictions)

    # Özel NLP işlemleri kullanımı
    text = "Örnek bir metin."
    processed_text = custom_nlp_pipeline(text)
    print("İşlenmiş metin:", processed_text)

if __name__ == "__main__":
    main()
