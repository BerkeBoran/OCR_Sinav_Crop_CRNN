# Kutu Etiketleme Rehberi (Label Studio)

Yeni sınav şablonlarında **öğrenci numarası** ve **not** alanlarını kutu içine
alma (bounding box) işi burada anlatılıyor. Bu, YOLO detektörünü yeni
şablonlarda çalışır hale getirmek için gereken adımdır.

> Rakam değerlerini (85, 221450046 gibi) girme işi bu araçta DEĞİL, mevcut
> Streamlit etiketleme sitesinde yapılır. Bu araç yalnızca **alanların yerini**
> işaretlemek içindir.

---

## Neden ön-etiketleme?

Sıfırdan kutu çizmiyoruz. `prepare_tasks.py` mevcut `models/yolo_fields.pt`
modelini çalıştırıp her fotoğrafa **önerilen kutuları** koyar. Ekip yalnızca
yanlış olanları düzeltir — çizmek yerine kaydırmak kat kat hızlıdır.

Yeni şablonlarda model çoğu kutuyu kaçıracak (zaten sorun bu); kaçırdıklarını
elle çizeceksiniz. Yine de tanıdık şablonlarda ciddi zaman kazandırır.

---

## 1. Görselleri hazırla (bir kişi, bir kez)

HEIC dosyaları doğrudan kullanılamaz; script hepsini EXIF yönü **gömülmüş**
JPG'ye çevirir (böylece tarayıcı ile model aynı yönü görür).

```bash
source .venv/bin/activate
python3 scripts/labelstudio/prepare_tasks.py "/Users/berkeboran/Desktop/Şablonlar/Şablon_1" --parti sablon_1
python3 scripts/labelstudio/prepare_tasks.py "/Users/berkeboran/Desktop/Şablonlar/Şablon_2" --parti sablon_2
```

Üretilenler:

- `data/labeling_images/<parti>/*.jpg` — Label Studio'nun sunacağı görseller
- `data/labelstudio_tasks/<parti>.json` — içe aktarılacak görev dosyası (ön-etiketli)

**Not:** Şablon başına 30–50 fotoğraf genelde yeterlidir. 550 fotoğrafın
hepsini etiketlemeye kalkmayın; önce azıyla eğitip `evaluate_pipeline.py` ile
ölçün, yetmezse ekleyin.

---

## 2. Label Studio'yu başlat

### Yerel (tek kişi)

```bash
mkdir -p data/labelstudio && chmod 777 data/labelstudio
docker compose up -d labelstudio
```

Tarayıcı: <http://localhost:8080>

### VPS (ekipçe)

```bash
cd /opt/ocr-sinav-crop-crnn
git pull
mkdir -p data/labelstudio data/labeling_images && chmod 777 data/labelstudio
docker compose -f docker-compose.prod.yml up -d --build
```

Tarayıcı: `https://<DOMAIN>:8443`

> **Güvenlik duvarı:** 8443 portu dışarı açık olmalı.
> `sudo ufw allow 8443/tcp` (ufw kullanılıyorsa).

> **Kayıt kapalı:** Rastgele kişiler hesap açamaz
> (`LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK`). İlk hesabı siz açın, ekibi
> Organization → Members ekranındaki davet bağlantısıyla çağırın.

---

## 2.5. Görselleri VPS'e taşı (ekipçe çalışıyorsanız)

Görseller sizin bilgisayarınızda üretiliyor ama Label Studio VPS'te çalışıyor.
Görselleri oraya kopyalamanız gerekir (git'e dahil değiller):

```bash
rsync -avz --progress data/labeling_images/sablon_1/ \
  root@<VPS_IP>:/opt/ocr-sinav-crop-crnn/data/labeling_images/sablon_1/
```

`prepare_tasks.py` görselleri 1600 piksele küçülttüğü için boyut makuldür
(200 görsel ≈ 100 MB). Görev dosyasını (`<parti>.json`) taşımanıza gerek yok —
onu tarayıcıdan kendi bilgisayarınızdan yükleyeceksiniz.

---

## 3. Projeyi kur (her parti için bir kez)

1. **Create Project** → ad: `Sinav Alan Tespiti`

2. **Labeling Setup** → `Custom template` → `scripts/labelstudio/label_config.xml`
   içeriğini yapıştır → Save

   ⚠️ **Etiket sırasını değiştirmeyin.** `not` üstte, `ogrenci_numara` altta
   olmalı; YOLO sınıf indeksleri (0/1) bu sıradan üretiliyor.

3. **Settings → Cloud Storage → Add Source Storage → Local files**

   - Absolute local path: **`/label-studio/files/<parti>`**
     (ör. `/label-studio/files/sablon_1`)
   - Save

   ⚠️ İki tuzak (ikisi de test edilip doğrulandı):

   - **Yol, parti klasörü olmalı.** `/label-studio/files` (kök) yazarsanız
     Label Studio güvenlik gerekçesiyle reddeder:
     *"cannot be the same as LOCAL_FILES_DOCUMENT_ROOT"*. Her parti için
     ayrı bir storage ekleyin.
   - **Bu bağlantı olmadan görseller görünmez.** Ayarlar doğru olsa bile,
     kayıtlı storage yoksa görsel isteği 404 döner.

4. **Sync'e BASMAYIN.** Sync, klasördeki her dosya için sıfırdan görev
   oluşturur; JSON'u da import edince her fotoğraf **iki kez** listeye girer
   (testte 6 görev 12'ye çıktı) ve ön-etiketler ikinci kopyada olmaz.

5. **Import** → `data/labelstudio_tasks/<parti>.json` dosyasını yükleyin.
   Görevler ön-etiketleriyle birlikte gelir.

---

## 3.5. Ekipçe çalışma

### Hesaplar

1. **İlk kişi** (yönetici) `https://<DOMAIN>:8443` adresinden hesap açar.
   Bu kişi organizasyonun sahibi olur.
2. Sağ üst → **Organization** → **Invite people** → çıkan **davet bağlantısını**
   kopyalayıp ekibe gönderir.
3. Diğerleri o bağlantıyla kendi hesabını açar. Bağlantı olmadan kayıt
   kapalıdır — dışarıdan kimse giremez.

### İş bölümü — iki seçenek

**A) Ortak kuyruk (önerilen, kurulum yok)**

Herkes aynı projeye girer, **Label Next Task** ile sıradaki görevi alır.
Label Studio görevi açan kişi için kilitler ve bir görev tamamlandıktan sonra
başkasına göstermez — yani **aynı fotoğrafı iki kişi etiketlemez**, kimin ne
zaman çalıştığını koordine etmenize gerek kalmaz.

**B) Parti başına ayrı proje (takibi kolay)**

Her şablon partisi için ayrı proje açıp (`Şablon 1`, `Şablon 2`, …) kişi
başına bir parti verirsiniz. Kimin nerede olduğu net görünür, ama her proje
için ayarları (etiket şeması + Local files) tekrar kurmanız gerekir.

> 4 kişi ve birkaç yüz görsel için **A** yeterli. Kişi başı sorumluluk
> istiyorsanız B'ye geçin.

### Kontrol edilmesi gereken ayar

**Settings → Annotation → Overlap / "Annotations per task"** değeri **1**
olmalı (varsayılan). 2 yaparsanız her fotoğrafı iki kişi etiketler — ölçüm
için faydalı ama iş yükü ikiye katlanır.

### Başlamadan önce: 5 dakikalık kalibrasyon

Ekipçe **aynı 5 fotoğrafı** birlikte etiketleyip kutuları karşılaştırın.
Herkes kutuyu aynı yerden başlatıp aynı yerde bitirmeli (bkz. aşağıdaki
kurallar). Bu 5 dakika, sonradan tutarsız veriyle model eğitmekten çok daha
ucuza gelir.

### İlerleme takibi

Proje ana ekranında toplam / tamamlanan görev sayısı görünür. Kişi bazlı
dağılım için **Members** sekmesine bakabilirsiniz.

### Dürüst sınır

Açık kaynak (Community) sürümde **rol ayrımı ve onay/red (review) akışı
yoktur**: organizasyondaki herkes projeyi görebilir ve birbirinin etiketini
değiştirebilir. 4 kişilik güvenilir bir ekip için sorun değil, ama "sadece
şu kişi düzeltsin" gibi bir kısıt kurmak mümkün değil.

---

## 4. Etiketleme kuralları (ekip için)

- Kutu, **basılı etiketi de içerecek** şekilde çizilir: "Öğrenci No: 22145004"
  ifadesinin tamamı kutuya girer, sadece rakamlar değil. Mevcut veri seti böyle
  etiketlendi; farklı yaparsanız model tutarsız öğrenir.
- Sayfada bir alan **yoksa** kutu çizmeyin, `alan_yok` seçeneğini işaretleyin.
- Alan var ama **okunamıyorsa** kutuyu yine çizin, `okunamiyor` işaretleyin.
- Her sayfada her sınıftan **en fazla bir** kutu olmalı.
- Modelin koyduğu kutu doğruysa dokunmayın, sadece **Submit** deyin.

---

## 5. Export ve veri setine katma

Label Studio'da: **Project → Export → YOLO** → inen zip'i açın.

```bash
python3 scripts/labelstudio/merge_export.py ~/Downloads/project-1-at-2026-07-19-yolo --kuru   # önce dene
python3 scripts/labelstudio/merge_export.py ~/Downloads/project-1-at-2026-07-19-yolo          # gerçek
```

Script sınıf indekslerini projenin sırasına göre yeniden eşler (Label Studio
farklı sırada verirse veri sessizce bozulurdu), train/valid/test'e böler ve
`data/roboflow_export` içine kopyalar. Aynı dosyayı iki kez katmaz.

---

## 6. Yeniden eğit ve ölç

```bash
python3 training/train_yolo.py           # yerelde ~5 saat
# veya notebooks/train_yolo_colab.ipynb  # T4'te ~1 saat

python3 training/evaluate_pipeline.py    # uçtan uca karne
```

Ardından yeni kırpımları üretip rakam etiketlemesine gönderin:

```bash
python3 scripts/crop_new_to_pending.py   # -> data/cropped_fields_pending
# Streamlit sitesinde "Yeni Veriler (Kontrol Bekleyen)" ile değerleri girin
python3 scripts/merge_pending_data.py    # -> labels.csv
python3 training/train_crnn.py           # CRNN'i güncelle
```

---

## Notlar

- `data/labelstudio/` (Label Studio veritabanı) ve `data/labeling_images/`
  git'e dahil edilmez. **Etiketler Label Studio veritabanında durur** — VPS'te
  çalışıyorsanız asıl kopya oradadır; export alıp `merge_export.py` ile veri
  setine katana kadar git'te bir izi olmaz. Partiyi bitirir bitirmez export
  alıp katın.
- Docker imajı **1.23.0** sürümüne sabitlendi — bu rehberdeki adımlar o
  sürümde test edildi. Sürüm yükseltirseniz özellikle Local Storage
  davranışını yeniden doğrulayın; Label Studio bu alanda sürümler arasında
  değişiklik yapıyor (ör. 1.23'te eski API token'ları varsayılan olarak
  kapalı geliyor).
