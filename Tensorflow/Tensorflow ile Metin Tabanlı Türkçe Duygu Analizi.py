import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
import numpy as np
from tensorflow.python.client import device_lib

# Adım 1: Genelleştirilmiş ve Net Veri Setini Oluşturma (100 İyi, 100 Kötü Yorum)
iyi_yorumlar = [
    "Harika bir video, izlerken çok keyif aldım!",
    "Bu kadar detaylı anlatımı beklemiyordum, süper olmuş.",
    "Enerjin ve anlatım tarzın çok motive edici.",
    "Görseller ve ses mükemmel bir uyum içindeydi.",
    "Videoyu bitirir bitirmez tekrar başlamak istedim.",
    "Kanalını keşfettiğim için çok mutluyum, harikasın!",
    "Kısa ama dolu dolu bir içerik olmuş, teşekkürler.",
    "Her videon bir öncekinden daha iyi, tebrikler!",
    "Anlatımın o kadar akıcı ki zamanın nasıl geçtiğini fark etmedim.",
    "Bu kadar net ve anlaşılır anlatımı nadiren görüyorum.",
    "Videonun başındaki özet kısmı çok faydalıydı.",
    "Kameranın açısı ve renkler çok göze hitap ediyor.",
    "Bu konuya dair kafamda oluşan tüm sorulara cevap aldım.",
    "Çok yaratıcı bir yaklaşım, kesinlikle ilham verici.",
    "Enerjin ekrandan hissediliyor, çok samimisin.",
    "Bu kadar sade ama etkili anlatım beklemiyordum.",
    "Videonun temposu izleyici için tam ideal.",
    "Kanalındaki içerik çeşitliliği gerçekten etkileyici.",
    "Her detayın üzerinde özenle durulmuş, bravo!",
    "Ses kalitesi mükemmel, anlatım net ve anlaşılır.",
    "Videoyu arkadaşlarıma da önerdim, herkes bayıldı.",
    "İzlerken hem öğrendim hem de keyif aldım.",
    "Her videonda farklı bir bakış açısı sunuyorsun.",
    "Bu içerik gerçekten zaman ayırmaya değerdi.",
    "Videonun giriş kısmı çok merak uyandırıcıydı.",
    "Kendi deneyimlerini paylaşman çok değerli bir detay.",
    "Bu tarz içerikleri daha fazla görmek isterim.",
    "Videonun kapanışı çok etkileyiciydi, tebrikler.",
    "Anlatımın hem öğretici hem eğlenceli olmuş.",
    "Kanalını keşfettiğim için şanslı hissediyorum.",
    "Bu videoyu kaydettim, tekrar tekrar izlemeyi planlıyorum.",
    "Her cümlen çok yerinde ve anlaşılır.",
    "Kurgunun temposu çok iyi, sıkılmadan izledim.",
    "Videonun her saniyesi değerli, eline sağlık.",
    "Detaylara gösterdiğin özen çok etkileyici.",
    "Bu içerik beni gerçekten motive etti.",
    "Videonun uzunluğu tam kararında, sıkmadı.",
    "Her videoda yeni bir şey öğreniyorum, teşekkürler.",
    "Kanalın kalitesi her videoda artıyor.",
    "Videoların arka plan müzikleri çok uyumlu seçilmiş.",
    "Bu anlatım tarzı benim için çok öğreticiydi.",
    "Videonun görselliği anlatımı çok destekliyor.",
    "Sade ama etkili bir şekilde tüm konuyu aktarmışsın.",
    "Bu içerik benim için çok faydalı oldu.",
    "Videonun temposu ve anlatımı mükemmel dengede.",
    "Her videonun sonunda yeni bilgiler öğreniyorum.",
    "Kanalındaki içeriklerin özgünlüğü gerçekten fark yaratıyor.",
    "Videoların başlıkları çok ilgi çekici ve açıklayıcı.",
    "Bu tarz videolar motivasyonumu artırıyor.",
    "Detaylı anlatımın sayesinde konuyu kolayca kavradım.",
    "Videoyu izlerken hiç sıkılmadım, çok akıcıydı.",
    "Kanalındaki içeriklerin güncelliği çok iyi.",
    "Videonun başlangıcı merak uyandırıcıydı.",
    "Her videoda yeni bir perspektif sunuyorsun.",
    "İçerik çok kaliteli ve özenle hazırlanmış.",
    "Videonun temposu izleyici için ideal.",
    "Anlatım tarzın çok doğal ve samimi.",
    "Bu kadar detaylı içerik için teşekkürler.",
    "Videonun görsel sunumu çok hoş ve estetik.",
    "Bu videoyu izleyince kafamdaki tüm sorular çözüldü.",
    "Enerjin izleyiciye çok iyi geçiyor.",
    "Bu kanalın içerikleri her zaman güvenilir ve kaliteli.",
    "Videonun anlatımı çok net ve anlaşılır.",
    "Bu videoyu izlemek gerçekten çok keyifliydi.",
    "Her videon bana yeni bir bakış açısı kazandırıyor.",
    "Kanalın içerikleri sürekli gelişiyor, takipteyim.",
    "Videonun açıklaması ve detayları çok faydalıydı.",
    "Bu kadar kısa sürede bu kadar bilgi vermen harika.",
    "Videonun giriş kısmı çok etkileyici ve merak uyandırıcı.",
    "Her bir örnek çok açıklayıcı ve anlaşılır.",
    "Kanalındaki içeriklerin çeşitliliği çok iyi düşünülmüş.",
    "Videonun temposu ve anlatım tarzı çok dengeli.",
    "Bu içerik sayesinde konuya dair kafamdaki her şey netleşti.",
    "Videoların kalitesi izleyiciye değer verildiğini gösteriyor.",
    "Her videon sonunda yeni bilgiler öğreniyorum.",
    "Videonun sunumu çok profesyonel ve akıcı.",
    "Bu anlatım tarzı beni gerçekten etkiledi.",
    "Videonun her bölümü özenle hazırlanmış.",
    "Kanalındaki içeriklerin özgünlüğü çok etkileyici.",
    "Videonun görsel ve işitsel kalitesi mükemmel.",
    "Bu tarz içeriklerle bilgilenmek çok keyifli.",
    "Videonun temposu ve anlatım tarzı çok başarılı.",
    "Her videon bana yeni bir şey öğretiyor.",
    "Kanalındaki içeriklerin kalitesi her zaman yüksek.",
    "Videonun başlığı çok ilgi çekici ve doğru seçilmiş.",
    "Bu videoyu izlemek gerçekten çok öğreticiydi.",
    "Videonun kurgusu ve temposu çok akıcı.",
    "Bu içerik izlemeye değer, çok faydalıydı.",
    "Videonun görselliği anlatımı çok iyi destekliyor.",
    "Videoların her biri ayrı bir öğrenme deneyimi sunuyor.",
    "Bu kanalın içerikleri her zaman güncel ve faydalı.",
    "Videonun anlatımı hem öğretici hem de eğlenceli.",
    "Kanalını keşfettiğim için çok memnunum.",
    "Videonun temposu çok iyi ayarlanmış.",
    "Her videon farklı bir bakış açısı kazandırıyor.",
    "Bu kadar etkileyici bir anlatım beklemiyordum.",
    "Videonun görsel sunumu ve anlatımı çok uyumlu.",
    "Bu içerik benim için çok değerli oldu.",
    "Videonun başlığı ve içeriği tam uyumlu.",
    "Her videoda yeni bir şey öğreniyorum ve keyif alıyorum.",
    "Videonun temposu izleyiciyi sıkmadan ilerliyor.",
    "Anlatım tarzın çok etkileyici ve net.",
    "Bu video sayesinde kafamdaki tüm sorular çözüldü.",
    "Videonun içeriği çok özgün ve kaliteli.",
    "Her videon çok ilgi çekici ve öğretici.",
    "Videonun sunumu çok profesyonel ve akıcı.",
    "Bu kanalın içerikleri gerçekten çok faydalı.",
    "Videonun temposu ve anlatımı çok dengeli ve net.",
    "Her videon bana yeni bir bilgi kazandırıyor.",
    "Videonun görselliği çok güzel, izlerken keyif aldım.",
    "Bu içerik sayesinde konuyu çok iyi anladım.",
    "Videonun anlatımı çok net ve anlaşılır.",
    "Her videon farklı bir bakış açısı sunuyor.",
    "Videonun temposu çok akıcı ve izleyici dostu.",
    "Kanalındaki içeriklerin kalitesi her zaman çok yüksek.",
    "Videonun başlığı çok ilgi çekici ve açıklayıcı.",
    "Bu video izlemeye değer, çok faydalı bilgiler içeriyor.",
    "Videonun kurgusu ve anlatımı çok başarılı ve etkileyici.",
    "Her videon ayrı bir öğrenme deneyimi sunuyor.",
    "Videonun görsel ve işitsel kalitesi çok yüksek.",
    "Bu kanalın içerikleri her zaman güvenilir ve kaliteli.",
]


np.random.shuffle(iyi_yorumlar)
#iyi_yorumlar = iyi_yorumlar[:100]

kotu_yorumlar = [
    "Video çok sıkıcıydı, izlerken zaman nasıl geçti anlamadım.",
    "Beklediğim kadar bilgilendirici değildi.",
    "Ses biraz daha net olabilirdi, zor duydum bazı kısımları.",
    "Anlatım tarzı bana göre ağır kaçtı.",
    "Konu biraz yüzeysel işlenmiş, detay eksikti.",
    "Başlıkla içerik uyumsuz geldi.",
    "Görseller anlatımı pek desteklemedi.",
    "Videonun başı çok uzun sürdü, hemen konuya girilseydi daha iyi olurdu.",
    "Bazı teknik terimler açıklanmamış, anlamak zordu.",
    "Kurguda kopukluk vardı, akıcı değildi.",
    "Motivasyon düşmanı gibiydi bu anlatım.",
    "İçerik özenli hazırlanmış gibi görünmüyordu.",
    "Senaryo biraz klişe ve tahmin edilebilirdi.",
    "Renkler ve görseller gözü yordu.",
    "Videoda bazı bilgiler eksik kalmış gibi hissedildi.",
    "Ses seviyesi dalgalıydı, rahatsız ediciydi.",
    "Kamera açıları bazı sahnelerde kötüydü.",
    "Anlatım monoton ve sıkıcıydı.",
    "Videonun temposu çok yavaştı.",
    "Bazı örnekler çok basit ve sıradandı.",
    "Videonun bazı bölümleri kafa karıştırıcıydı.",
    "Başlangıç kısmı gereğinden uzun sürmüştü.",
    "Detay eksikliği nedeniyle konu tam anlaşılmadı.",
    "Videoyu izlerken motivasyonum düştü.",
    "Eksik bilgi bırakılmış gibi hissettim.",
    "Daha organize bir yapı olabilirdi.",
    "Dil akıcılığı düşüktü, izlemek zor oldu.",
    "Videonun kapanışı hızlı ve aceleye gelmiş gibiydi.",
    "İçerik çok yüzeysel ve sıradan kalmış.",
    "Kurguda ani geçişler vardı, rahatsız ediciydi.",
    "Bazı bölümlerde tekrarlar vardı, sıkıcıydı.",
    "Videonun temposu izleyici için uygun değildi.",
    "Görsel efektler çok basit ve amatör görünüyordu.",
    "Videodaki açıklamalar yeterince detaylı değildi.",
    "Ses ve görsel uyumu bazı kısımlarda bozulmuş.",
    "Videoyu anlamak için tekrar izlemek zorunda kaldım.",
    "Bu video beklentimi karşılamadı.",
    "Bazı örnekler anlaşılması güçtü.",
    "Videonun akışı kopuk ve düzensizdi.",
    "Görseller çok basit ve dikkat dağıtıcıydı.",
    "Anlatım tarzı bana hitap etmedi.",
    "Videonun temposu çok yavaş ilerliyordu.",
    "Detay eksikliği nedeniyle konuyu tam kavrayamadım.",
    "Videoda sık sık duraklamalar vardı.",
    "İzlerken dikkatim dağıldı, çok monotondu.",
    "Başlık ve içerik arasında uyumsuzluk vardı.",
    "Videonun bazı kısımları gereksiz uzundu.",
    "Ses tonu çok monoton ve dikkat dağıtıcıydı.",
    "Bazı bilgiler yanlış veya eksik gibiydi.",
    "Kamera açıları bazı sahnelerde rahatsız ediciydi.",
    "Videonun temposu çok dalgalıydı.",
    "Görsel ve işitsel kalite beklentimin altındaydı.",
    "Videodaki örnekler yeterince açıklayıcı değildi.",
    "Anlatım tarzı sıkıcı ve monotondu.",
    "Videonun giriş kısmı çok uzun sürdü ve dikkat dağıttı.",
    "Eksik bilgiler nedeniyle konu tam anlaşılmadı.",
    "Bazı kısımlarda tekrarlar vardı, sıkıcıydı.",
    "Videonun temposu izleyici için uygun değildi.",
    "Ses ve görseller uyumsuzdu, rahatsız ediciydi.",
    "Videoyu izlemek zorlayıcıydı.",
    "İçerik beklentimi karşılamadı, yüzeysel kaldı.",
    "Bazı örnekler çok basit ve anlaşılması zordu.",
    "Videonun akışı düzensiz ve kopuktu.",
    "Görseller dikkat dağıtıcı ve amatör görünüyordu.",
    "Anlatım tarzı bana uygun değildi.",
    "Videonun temposu çok yavaş ve monotondu.",
    "Detay eksikliği nedeniyle konu net değildi.",
    "Videoda gereksiz duraklamalar vardı.",
    "İzlerken dikkatim dağıldı, sıkıcıydı.",
    "Başlık ile içerik arasında ciddi uyumsuzluk vardı.",
    "Videonun bazı bölümleri gereksiz uzunluktaydı.",
    "Ses tonu çok monoton ve rahatsız ediciydi.",
    "Bazı bilgiler eksik veya yanlış gibiydi.",
    "Kamera açıları bazı sahnelerde kötüydü.",
    "Videonun temposu çok dengesizdi.",
    "Görsel ve işitsel kalite beklentinin altındaydı.",
    "Videodaki örnekler açıklayıcı değildi.",
    "Anlatım tarzı sıkıcıydı.",
    "Videonun giriş kısmı çok uzun sürdü.",
    "Eksik bilgiler nedeniyle konuyu anlamak zor oldu.",
    "Bazı kısımlarda tekrarlar vardı, sıkıcıydı.",
    "Videonun temposu izleyiciye uygun değildi.",
    "Ses ve görseller uyumsuzdu.",
    "Videoyu izlemek zorlayıcı ve sıkıcıydı.",
    "İçerik beklentilerimin altındaydı.",
    "Bazı örnekler çok basit ve sıkıcıydı.",
    "Videonun akışı düzensizdi.",
    "Görseller amatör ve dikkat dağıtıcıydı.",
    "Anlatım tarzı bana uygun değildi.",
    "Videonun temposu çok yavaştı ve monotondu.",
    "Detay eksikliği nedeniyle konuyu anlamadım.",
    "Videoda duraklamalar çok fazlaydı.",
    "İzlerken dikkatim dağıldı, sıkıldım.",
    "Başlık ile içerik uyumsuzdu.",
    "Videonun bazı kısımları gereksiz uzunlukta.",
    "Ses tonu çok monoton ve rahatsız ediciydi.",
    "Bazı bilgiler eksikti veya yanlış gibiydi.",
    "Kamera açıları rahatsız ediciydi.",
    "Videonun temposu dengesizdi.",
    "Görsel ve işitsel kalite yetersizdi.",
    "Videodaki örnekler açıklayıcı değildi.",
    "Anlatım tarzı sıkıcıydı ve monotondu.",
    "Videonun başlangıcı çok uzundu.",
    "Eksik bilgiler yüzünden konuyu anlamak zor oldu.",
    "Bazı bölümler tekrarlayıcıydı, sıkıcıydı.",
    "Videonun temposu izleyici için uygun değildi.",
    "Ses ve görseller birbirine uyumsuzdu.",
    "Videoyu izlemek yorucuydu.",
    "İçerik beklentilerimin altındaydı.",
    "Bazı örnekler basit ve sıkıcıydı.",
    "Videonun akışı kopuktu ve düzensizdi.",
    "Görseller amatör ve rahatsız ediciydi.",
    "Anlatım tarzı bana hitap etmedi.",
    "Videonun temposu çok yavaş ve sıkıcıydı.",
    "Detay eksikliği konuyu anlaşılmaz kıldı.",
    "Videoda gereksiz duraklamalar vardı.",
    "İzlerken dikkatimi toplamak zordu, sıkıldım.",
    "Başlık ile içerik arasında ciddi uyumsuzluk vardı.",
    "Videonun bazı bölümleri gereksiz uzundu.",
    "Ses tonu monoton ve rahatsız ediciydi.",
    "Bazı bilgiler eksik ya da yanlış gibiydi.",
    "Kamera açıları rahatsız ediciydi ve düzensizdi."
]


np.random.shuffle(kotu_yorumlar)
#kotu_yorumlar = kotu_yorumlar[:100]

# Veri setini oluşturma
egitim_cumleler = iyi_yorumlar + kotu_yorumlar
egitim_etiketler = [1] * len(iyi_yorumlar) + [0] * len(kotu_yorumlar)

c = list(zip(egitim_cumleler, egitim_etiketler))
np.random.shuffle(c)
egitim_cumleler, egitim_etiketler = zip(*c)

# Adım 2: Metni Sayılara Dönüştürme (Tokenization)
tokenizer = keras.preprocessing.text.Tokenizer(num_words=None, oov_token="<unk>")
tokenizer.fit_on_texts(egitim_cumleler)
word_index = tokenizer.word_index

egitim_dizileri = tokenizer.texts_to_sequences(egitim_cumleler)
egitim_padded = keras.preprocessing.sequence.pad_sequences(egitim_dizileri, padding='post', maxlen=100)
max_len = 100

# Adım 3: Cihazı Bulma ve Zorunlu Kılma
dml_device_name = None
devices = device_lib.list_local_devices()
for dev in devices:
    if "DML" in dev.physical_device_desc:
        dml_device_name = dev.name
        break

if dml_device_name is None:
    raise ValueError("DirectML cihazı bulunamadı. Program GPU hızlandırması olmadan devam edemez.")
else:
    print(f"\nDirectML cihazı bulundu: {dml_device_name}. Eğitim GPU'da yapılacak.")

# Adım 4: Modeli Oluşturma ve GPU'ya Yerleştirme
with tf.device(dml_device_name):
    model = Sequential([
        layers.Embedding(len(word_index) + 1, 16),
        layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
        layers.GlobalMaxPooling1D(),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    # Modeli derleme
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    print("\nModel eğitimi başlıyor...")
    # Adım 5: Modeli Eğitme
    model.fit(egitim_padded, np.array(egitim_etiketler), epochs=20, verbose=1)
    print("\nModel eğitimi tamamlandı!")

    # Adım 6: Kullanıcı Girişi ile Tahmin Yapma
    print("\nLütfen bir yorum girin ('exit' yazarak çıkabilirsiniz):")
    while True:
        kullanici_yorumu = input("> ")

        if kullanici_yorumu.lower() == 'exit':
            print("Programdan çıkılıyor...")
            break

        test_dizileri = tokenizer.texts_to_sequences([kullanici_yorumu])
        test_padded = keras.preprocessing.sequence.pad_sequences(test_dizileri, padding='post', maxlen=max_len)

        tahmin = model.predict(test_padded)[0]

        tahmin_sinif = "Olumlu" if tahmin > 0.5 else "Olumsuz"
        print(f"'{kullanici_yorumu}' -> Tahmin: {tahmin_sinif} (Skor: {tahmin[0]:.2f})")
