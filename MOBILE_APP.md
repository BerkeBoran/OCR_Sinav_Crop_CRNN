# Mobil Uygulama Kimliği ve Çalışma Düzeni

Bu belge, aynı repoda geliştirilen Flutter mobil uygulamasının ekip genelinde
değişmemesi gereken teknik kimliğini kaydeder.

## Proje kimliği

| Alan | Değer |
|---|---|
| Flutter proje adı | `sinav_ocr_mobile` |
| Kuruluş kimliği (`org`) | `com.sinavocr` |
| Android uygulama kimliği | `com.sinavocr.sinav_ocr_mobile` |
| iOS bundle kimliği | `com.sinavocr.sinavOcrMobile` |
| Uygulama klasörü | `mobile_app/` |
| Flutter sürümü | `3.44.0` |
| Dart sürümü | `3.12.0` |

Bu değerler gizli bilgi değildir. `.env` yerine sürüm kontrolündeki bu belgede
tutulmaları, bütün ekip üyelerinin aynı proje kimliğini kullanmasını sağlar.

## Repo düzeni

```text
OCR_Sinav_Crop_CRNN/
├── mevcut Python OCR sistemi
├── docker/flutter/Dockerfile
├── docker-compose.mobile.yml
└── mobile_app/                 # Flutter uygulaması
```

## Dal politikası

Mehmet'in veri, yerel saklama ve Excel dışa aktarma çalışmaları
`Mehmet-Branch` dalında yapılır.

## Kişi C kapsamı

- `mobile_app/lib/models/`
- `mobile_app/lib/services/local_storage_service.dart`
- `mobile_app/lib/services/excel_export_service.dart`
- `mobile_app/lib/viewmodels/records_viewmodel.dart`
- `mobile_app/lib/viewmodels/export_viewmodel.dart`
- `mobile_app/lib/views/records_view.dart`
- `mobile_app/lib/views/export_view.dart`
- `mobile_app/lib/views/widgets/student_record_tile.dart`
- `mobile_app/lib/views/widgets/stat_widget.dart`

Bu dosyaların uygulama kodu Mehmet tarafından öğrenme amaçlı olarak adım adım
yazılacaktır. Ortam kurulumu ve standart Flutter iskeleti bu kapsamın dışındadır.

## Container komut biçimi

Flutter komutları repo kökünden şu kalıpla çalıştırılır:

```powershell
docker compose -f docker-compose.mobile.yml run --rm flutter <flutter-komutu>
```

Örnek sürüm kontrolü:

```powershell
docker compose -f docker-compose.mobile.yml run --rm flutter flutter --version
```
