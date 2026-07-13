# VPS Deploy Rehberi

Bu doküman, VPS üzerinde sadece `git clone` + `docker compose up` çalıştırarak
projeyi ayağa kaldırmak için gereken adımları içerir.

## Ön koşullar

- VPS'te Docker ve Docker Compose kurulu.
- SSL sertifikaları host'ta `/etc/letsencrypt/live/<DOMAIN>/fullchain.pem` ve
  `privkey.pem` olarak hazır.
- 80 ve 443 portları şu an başka bir docker compose yığını tarafından
  kullanılıyor — bu adımlarda önce o durdurulacak.

Aşağıdaki komutlarda `<DOMAIN>` yerine gerçek alan adınızı,
`<REPO_URL>` yerine bu projenin git repo adresini yazın.

---

## 1. Mevcut yığını durdur (silme, sadece durdur)

Eski compose projesinin bulunduğu dizine gidip servisleri durdurun.
`down` DEĞİL `stop` kullanın; böylece container'lar, network'ler ve
volume'ler kaybolmaz, gerekirse geri dönebilirsiniz.

```bash
cd /path/to/eski-proje
docker compose stop
```

Port çakışmasını doğrulamak için:

```bash
sudo ss -tlnp | grep -E ':80|:443'
```

Çıktı boşsa 80/443 boşta demektir, devam edebilirsiniz.

---

## 2. Bu projeyi klonla

```bash
cd /opt
sudo git clone <REPO_URL> ocr-sinav-crop-crnn
cd ocr-sinav-crop-crnn
sudo chown -R "$USER":"$USER" .
```

---

## 3. .env dosyasını oluştur

```bash
cp .env.example .env
nano .env
```

`.env` içinde `DOMAIN` değerini gerçek alan adınızla değiştirin:

```
DOMAIN=<DOMAIN>
```

---

## 4. Sertifikaların yerinde olduğunu doğrula

```bash
sudo ls -la /etc/letsencrypt/live/<DOMAIN>/
```

`fullchain.pem` ve `privkey.pem` dosyaları görünmelidir.

---

## 5. Prod compose ile başlat

```bash
docker compose -f docker-compose.prod.yml up -d --build
```

Durumu kontrol edin:

```bash
docker compose -f docker-compose.prod.yml ps
docker compose -f docker-compose.prod.yml logs -f
```

Tarayıcıdan `https://<DOMAIN>` adresini açarak doğrulayın, veya:

```bash
curl -Ik https://<DOMAIN>
```

---

## Güncelleme (yeniden deploy)

```bash
cd /opt/ocr-sinav-crop-crnn
git pull
docker compose -f docker-compose.prod.yml up -d --build
```

---

## Geri dönüş (rollback)

Bu projeyi durdurun:

```bash
cd /opt/ocr-sinav-crop-crnn
docker compose -f docker-compose.prod.yml down
```

Eski yığını tekrar başlatın:

```bash
cd /path/to/eski-proje
docker compose start
```

80/443 portlarının eski yığına geri döndüğünü doğrulayın:

```bash
sudo ss -tlnp | grep -E ':80|:443'
```

---

## Notlar

- Uygulama container'ı (`app`) sadece Docker iç ağında `8501` portundan
  erişilebilir; dışarıya açılan tek servis `nginx`'tir (80/443).
- `data/` ve `outputs/` klasörleri host'tan bind-mount edilir; container
  yeniden build edilse veya silinse bile veriler VPS'te kalır.
- Yerel geliştirme için (nginx olmadan, doğrudan 127.0.0.1:8002):

  ```bash
  docker compose up --build
  ```
