# Local-RAG-Assistant 
Bu proje, PDF belgeleri üzerinde çalışan, kullanıcıların yüklediği dokümanlar üzerinden doğal dilde soru sormasına ve ilgili içeriklere dayalı yanıtlar almasına olanak sağlayan, tamamen yerel bir **Retrieval-Augmented Generation (RAG)** uygulamasıdır.

Sistem; belge işleme, metin parçalama, embedding oluşturma, vektör arama ve yerel Large Language Model kullanarak uçtan uca bir doküman tabanlı soru-cevap sistemi sunmaktadır.

## Öne Çıkan Özellikler

* **Yerel RAG Mimarisi:** PDF belgelerinden alınan içerikler, Türkçe dili özelinde efektif bir şekilde çalışan embedding ve vektör arama süreçlerinden geçirilerek ilgili bilgiler dil modeline bağlam olarak sunulmaktadır.
* **Yerel LLM Kullanımı:** Yanıt üretimi için Türkçe dili özelinde efektif bir şekilde çalışan **Qwen2.5-3B-Instruct** modeli kullanılmakta ve temel çıkarım süreci Colab T4 yerel/GPU ortamında gerçekleştirilmektedir.
* **FAISS ile Vektör Arama:** Belgelerden oluşturulan embedding'ler FAISS üzerinde indekslenerek kullanıcı sorusuyla en alakalı içeriklerin hızlı bir şekilde bulunması sağlanmaktadır.
* **Çoklu PDF Desteği:** Birden fazla PDF aynı oturum içerisinde işlenebilmekte ve ortak bir vektör indeksinde kullanılabilmektedir.
* **Bağlama Dayalı Yanıt Üretimi:** Sistem, modelin yalnızca sağlanan belge bağlamını kullanmasını sağlayarak dokümanlarda bulunmayan bilgilerin üretilmesini azaltmayı hedeflemektedir. (Alakalı olmayan yanıtlar sınırlandırılmış aynı zamanda bilgisi olmadığı noktada halüsinasyon verisi sunması engellenmiştir.)
* **Gradio Arayüzü:** PDF yükleme, soru-cevap ve kayıtların görüntülenmesi için kullanıcı dostu ve hızlı bir web arayüzü sunulmaktadır.
* **SQLite ile Etkileşim Kaydı:** Kullanıcıların gerçekleştirdiği soru-cevap etkileşimleri SQLite veritabanında saklanmakta ve Gradio arayüzü üzerinden cevap kayıtları incelenbilmektedir.
* **Güvenli Dosya İşleme:** Yüklenen dosyalar `secure_filename` kullanılarak işlenmekte ve yalnızca PDF dosyalarının kabul edilmesi sağlanmaktadır. (Pdf dışı, fotoğraf - video gibi erişimler sonraki güncellemelerde eklenebilir.)
* **GPU Optimizasyonu:** Model çıkarımı için `fp16` hassasiyeti ve `sdpa` attention gibi GPU optimizasyonlarından yararlanılmaktadır.

## RAG Pipeline

Sistem, kullanıcı tarafından yüklenen PDF belgelerini aşağıdaki işlem adımlarından geçirmektedir:

```text
PDF Yükleme
     ↓
PDF İşleme
     ↓
Metin Parçalama
     ↓
Embedding Oluşturma
     ↓
FAISS Vektör İndeksi
     ↓
Retriever
     ↓
Qwen2.5-3B-Instruct (Türkçe odaklı)
     ↓
Yanıt
     ↓
SQLite Kayıt (Aynı zamanda arayüz içi sorgulama aktifliği)
```

Kullanıcı bir soru gönderdiğinde sistem, FAISS vektör indeksinde arama gerçekleştirerek en alakalı iki belge parçasını getirir. Bu içerikler ChatML formatındaki prompt içerisine bağlam olarak eklenir ve Qwen2.5-3B-Instruct modeli tarafından yanıt oluşturulur. Soru-cevap çifti daha sonra SQLite veritabanına kaydedilir.

## Veri İşleme ve Retrieval

PDF belgeleri `PyPDFLoader` kullanılarak işlenmekte ve metinler `RecursiveCharacterTextSplitter` ile parçalara ayrılmaktadır.

Mevcut yapılandırmada:

* **Chunk Size:** 700 karakter
* **Chunk Overlap:** 150 karakter (hızı artırmak için 150 karakter sınırı eklendi)
* **Embedding Model:** `paraphrase-multilingual-mpnet-base-v2`
* **Vector Store:** FAISS
* **Retrieval:** En alakalı `k=2` içerik (hızı artırmak için k = 4 yerine 2 tercih edildi.)

Bu yapı, belge içerisindeki ilgili bölümlerin bulunarak dil modeline bağlam olarak aktarılmasını sağlamaktadır.

## Kullanılan Teknolojiler

* **Programlama Dili:** `Python`
* **Large Language Model:** `Qwen2.5-3B-Instruct`
* **RAG Framework:** `LangChain` : Çoklu API entegrasyonu sağlanabilmesi için kullanıldı
* **Vector Search:** `FAISS`
* **Embedding:** `Sentence Transformers`
* **Model Framework:** `Hugging Face Transformers` (API kullanımı istenmediği için tercih edildi)
* **PDF İşleme:** `PyPDF`
* **Kullanıcı Arayüzü:** `Gradio` (hızlı ve basit kullanım için tercih edildi)
* **Veritabanı:** `SQLite` : python üzerinde çalışabilmesi için tercih edildi
* **GPU / Deep Learning:** `PyTorch`, `CUDA`

## Proje Yapısı

```text
Local RAG Assistant/
├── Local_RAG_Assistant.ipynb
├── app.py
├── guvenli_pdf_deposu/
├── sohbet_gecmisi.db
├── screenshots/
├── LICENSE
└── README.md
```

`guvenli_pdf_deposu/` yüklenen PDF dosyalarının saklandığı, `sohbet_gecmisi.db` ise soru-cevap kayıtlarının tutulduğu çalışma dizinleridir. Bu dosyalar uygulama çalışırken otomatik olarak oluşturulmaktadır.

<img width="1901" height="1025" alt="Ekran görüntüsü 2026-08-15 142810" src="https://github.com/user-attachments/assets/0f541364-b766-401d-ac04-4149bcc0863b" />


## Kurulum ve Çalıştırma

Projeyi kendi ortamınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz.

### 1. Repoyu klonlayın

```bash
git clone https://github.com/Meryem-Erdogdu/Local-RAG-Assistant.git
cd Local-RAG-Assistant
```

### 2. Gerekli kütüphaneleri yükleyin

```bash
pip install -qU transformers accelerate langchain langchain-community langchain-huggingface
pip install -qU pypdf faiss-cpu sentence-transformers gradio werkzeug
```

### 3. Uygulamayı çalıştırın

```bash
python app.py
```

Uygulama başlatıldıktan sonra terminal üzerinde görüntülenen Gradio bağlantısı üzerinden arayüze erişebilirsiniz.

## Google Colab

Proje, GPU desteği bulunan Google Colab ortamında da çalıştırılabilir.

Önerilen GPU:

```text
NVIDIA T4
```

Ana Colab notebook'u:

```text
Local_RAG_Assistant.ipynb
```

Model ve embedding bileşenlerinin ilk çalıştırmada otomatik olarak indirilmesi nedeniyle yaklaşık 6 GB depolama alanı gerekmektedir.

## Projenin Amacı

Bu proje, herhangi bir PDF doküman koleksiyonu üzerinde çalışabilecek yerel bir RAG sisteminin uçtan uca geliştirilmesini ve uygulanmasını amaçlamaktadır.

Proje kapsamında;

* Retrieval-Augmented Generation mimarisi,
* Yerel Large Language Model kullanımı,
* FAISS tabanlı vektör arama,
* PDF belge işleme,
* Embedding tabanlı retrieval,
* GPU üzerinde model çıkarımı,
* Gradio ile yapay zeka arayüzü,
* SQLite ile etkileşim kayıt sistemi

bir araya getirilmiştir.

## Gelecek Geliştirmeler

* Daha gelişmiş retrieval ve reranking yöntemlerinin eklenmesi
* Kaynak/citation gösteriminin geliştirilmesi
* Streaming yanıt desteği
* Kullanıcı ve oturum yönetimi
* Docker desteği
* Farklı yerel LLM modellerinin desteklenmesi
* Production ortamına uygun deployment yapısının oluşturulması
* Türkçe dili ( Veya bazı sondan eklemeli diller) bazında daha efektif bir token algılama sistemi oluşturulması için ekstra güncellemeler planlanması

## Lisans
Bu proje **MIT License** ile lisanslanmıştır.

