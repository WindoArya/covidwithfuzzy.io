import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import requests
import json

url = "https://data.covid19.go.id/public/api/prov.json"

provinsi = input("Masukkan nama Provinsi: ").upper()

#Mengambil data covid
response = requests.get(url)
data = response.text
pick_json = json.loads(data)
i = 0
while(i<=33):
    active_case = pick_json['list_data'][i]['key']
    if(active_case == provinsi):
         case = pick_json['list_data'][i]['jumlah_kasus']
         meninggal = pick_json['list_data'][i]['jumlah_meninggal']
    i += 1

#Mengambil range case terbesar
j = 0
case_max = 0
while(j<=33):
    if(pick_json['list_data'][j]['jumlah_kasus'] > case_max):
        case_max = pick_json['list_data'][j]['jumlah_kasus']
    j += 1

#Mengambil range jumlah meninggal terbesar
k = 0
meninggal_max = 0
while(k<=33):
    if(pick_json['list_data'][k]['jumlah_meninggal'] > meninggal_max):
        meninggal_max = pick_json['list_data'][k]['jumlah_meninggal']
    k += 1

#Membuat Variable Global
x_case = np.arange(0, case_max, 1)
x_meninggal = np.arange(0, meninggal_max, 1)
x_zona = np.arange(0, 11, 1)

case_lo = fuzz.trimf(x_case, [0, 0, case_max/2])
case_md = fuzz.trimf(x_case, [0, case_max/2, case_max])
case_hi = fuzz.trimf(x_case, [case_max/2, case_max, case_max])
meninggal_lo = fuzz.trimf(x_meninggal, [0, 0, meninggal_max/2])
meninggal_md = fuzz.trimf(x_meninggal, [0, meninggal_max/2, meninggal_max])
meninggal_hi = fuzz.trimf(x_meninggal, [meninggal_max/2, meninggal_max, meninggal_max])
zona_hijau = fuzz.trimf(x_zona, [0, 0, 5])
zona_kuning = fuzz.trimf(x_zona, [0, 5, 10])
zona_merah = fuzz.trimf(x_zona, [5, 10, 10])

#Fuzzy Rule
case_level_lo = fuzz.interp_membership(x_case, case_lo, case)
case_level_md = fuzz.interp_membership(x_case, case_md, case)
case_level_hi = fuzz.interp_membership(x_case, case_hi, case)

meninggal_level_lo = fuzz.interp_membership(x_meninggal, meninggal_lo, meninggal)
meninggal_level_md = fuzz.interp_membership(x_meninggal, meninggal_md, meninggal)
meninggal_level_hi = fuzz.interp_membership(x_meninggal, meninggal_hi, meninggal)

#Rule 1
rule1 = np.fmax(case_level_lo, meninggal_level_lo)
zona_activation_hijau = np.fmin(rule1, zona_hijau)

#Rule 2
zona_activation_kuning = np.fmin(meninggal_level_md, zona_kuning)

#Rule 3
rule3 = np.fmax(case_level_hi, meninggal_level_hi)
zona_activation_merah = np.fmin(rule3, zona_merah)
zona0 = np.zeros_like(x_zona)

#Defuzzifikasi
gabungan = np.fmax(zona_activation_hijau, np.fmax(zona_activation_kuning, zona_activation_merah))

zona = fuzz.defuzz(x_zona, gabungan, 'centroid')
zona_activation = fuzz.interp_membership(x_zona, gabungan, zona)


#1. Visualisasid dari membership
def grafMembership():
    fig, (ax0, ax1, ax2) = plt.subplots(nrows = 3, figsize = (8, 9))
    ax0.plot(x_case, case_lo, 'g', linewidth = 1.5, label = 'Rendah')
    ax0.plot(x_case, case_md, 'y', linewidth = 1.5, label = 'Sedang')
    ax0.plot(x_case, case_hi, 'r', linewidth = 1.5, label = 'Tinggi')
    ax0.set_title('Jumlah kasus')
    ax0.legend()

    ax1.plot(x_meninggal, meninggal_lo, 'g', linewidth = 1.5, label = 'Rendah')
    ax1.plot(x_meninggal, meninggal_md, 'y', linewidth = 1.5, label = 'Sedang')
    ax1.plot(x_meninggal, meninggal_hi, 'r', linewidth = 1.5, label = 'Tinggi')
    ax1.set_title('Jumlah Meninggal')
    ax1.legend()

    ax2.plot(x_zona, zona_hijau, 'g', linewidth = 1.5, label = 'Hijau')
    ax2.plot(x_zona, zona_kuning, 'y', linewidth = 1.5, label = 'Kuning')
    ax2.plot(x_zona, zona_merah, 'r', linewidth = 1.5, label = 'Merah')
    ax2.set_title('Zona')
    ax2.legend()

    for ax in (ax0, ax1, ax2):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    plt.tight_layout()
    plt.show()

#2. Visualisasi input
def grafInput():
    fig, ax0 = plt.subplots(figsize=(8, 3))

    ax0.fill_between(x_zona, zona0, zona_activation_hijau, facecolor='g', alpha=0.7)
    ax0.plot(x_zona, zona_hijau, 'g', linewidth=0.5, linestyle='--', )
    ax0.fill_between(x_zona, zona0, zona_activation_kuning, facecolor='y', alpha=0.7)
    ax0.plot(x_zona, zona_kuning, 'y', linewidth=0.5, linestyle='--')
    ax0.fill_between(x_zona, zona0, zona_activation_merah, facecolor='r', alpha=0.7)
    ax0.plot(x_zona, zona_merah, 'r', linewidth=0.5, linestyle='--')
    ax0.set_title('Output membership activity')

    for ax in (ax0,):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    plt.tight_layout()
    plt.show()

#3. Visualisasi tahap akhir
def grafResult():
    fig, ax0 = plt.subplots(figsize=(8, 3))

    ax0.plot(x_zona, zona_hijau, 'g', linewidth=0.5, linestyle = '--', )
    ax0.plot(x_zona, zona_kuning, 'y', linewidth=0.5, linestyle = '--')
    ax0.plot(x_zona, zona_merah, 'r', linewidth=0.5, linestyle = '--')
    ax0.fill_between(x_zona, zona0, gabungan, facecolor='Orange', alpha=0.7)
    ax0.plot([zona, zona], [0, zona_activation], 'k', linewidth=1.5, alpha=0.9)
    ax0.set_title('Result')

    for ax in (ax0,):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    plt.tight_layout()
    plt.show()


grafMembership()
grafInput()
grafResult()



