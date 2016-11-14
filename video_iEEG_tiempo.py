# Calcula los segundos a tomar del archivo TRC


def calc_tiempo(min_ini=None, seg_ini=None, min_fin=None, seg_fin=None, video_nr=None):
    videos_t_ini = [0, 2002, 4004, 6006]
    ini_t_en_segundos = (min_ini * 60) + seg_ini + videos_t_ini[video_nr - 1]
    fin_t_en_segundos = (min_fin * 60) + seg_fin + videos_t_ini[video_nr - 1]

    print('Tomar desde el segundo', ini_t_en_segundos, 'hasta el segundo', fin_t_en_segundos)

calc_tiempo(min_ini=0,
            seg_ini=0,
            min_fin=25,
            seg_fin=0,
            video_nr=4)


