# Deteksi Ujaran Kebencian Teks Panjang Berbahasa Indonesia Menggunakan Data Facebook

Nofa Aulia - 1606964742
Magister Ilmu Komputer

Kode ini merupakan program yang digunakan untuk eksperimen dalam tesis 
"Deteksi Ujaran Kebencian Teks Panjang Berbahasa Indonesia Menggunakan Data Facebook".

-   Bahasa pemrograman yang digunakan adalah Python 3.5.2
-   Library-library yang dibutuhkan silakan lihat file requirement.txt.
-   Program untuk scraping data Facebook en merujuk pada link berikut:
    https://github.com/adibPr/facebook_grab
-   Pada pengaturan awal, eksperimen dilakukan tanpa proses balancing data.
    Jika proses tersebut diperlukan, silakan uncomment baris 403 pada file 
    experiment.py. Ubah `# self.data_balancing()` menjadi `self.data_balancing()`.


## Pengaturan Environment
-   Membuat dan Mengaktifkan Virtual Environment
    Buka terminal lalu jalankan perintah berikut

```
$ pip install virtualenv
$ virtualenv <nama environment>
$ source <nama environment>/bin/activate
```
-   Install library yang dibutuhkan
    Untuk menginstall library yang dibutuhkan, jalankan perintah berikut.
```
$ pip install -r requirement.txt
```

## Cara Menjalankan Program

-   Menggunakan terminal
    Buka terminal lalu jalankan perintah berikut
```
$ python
from experiment import Experiment
k = 10  # masukkan jumlah k yang diinginkan
input_data = 'data/dataset.csv'  # masukkan lokasi file input

# Parameter: input_file, classifier, positive_words, negative_words, harsh_words, positive_prop, negative_prop, harsh_prop, char_ngram, word_ngram, k_cross_val
# untuk classifier, pilihannya adalah: rfdt, log_reg, dan svm
# sesuaikan parameter dengan kebutuhkan

exp = Experiment(input_data, False, False, "rfdt", True, False, False, False, False, False, False, False, False, 10)
exp.run()

```

atau 

-   Menggunakan file testing_scenario.py
    Pastikan telah memasukkan skenario testing pada file tersebut. Hasil eksperimen akan ditulis ke dalam sebuah file csv pada folder data/output/. Lokasi nama dan nama file output dapat diubah dengan memodifikasi baris kode
    `with open("path/to/file.csv",'w') as f:`. Berikut adalah perintah untuk menjalankan program.
  
```
$ python testing_scenario.py
```
