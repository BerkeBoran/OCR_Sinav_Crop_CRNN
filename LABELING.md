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

## 3. Projeyi kur (bir kez)

1. **Create Project** → ad: `Sinav Alan Tespiti`
2. **Labeling Setup** → `Custom template` → `scripts/labelstudio/label_config.xml`
   içeriğini yapıştır → Save

   ⚠️ **Etiket sırasını değiştirmeyin.** `not` üstte, `ogrenci_numara` altta
   olmalı; YOLO sınıf indeksleri (0/1) bu sıradan üretiliyor.

3. **Settings → Cloud Storage → Add Source Storage → Local files**
   - Absolute local path: `/label-studio/files`
   - Treat every bucket object as a source file: **kapalı**
   - Save → **Sync**

4. **Import** → `data/labelstudio_tasks/<parti>.json` dosyasını yükle

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
- Docker imajı `latest` olarak bırakıldı. Kurulum çalıştıktan sonra sürümü
  sabitlemek isterseniz `docker compose images` ile mevcut etiketi görüp
  compose dosyasında sabitleyebilirsiniz.
