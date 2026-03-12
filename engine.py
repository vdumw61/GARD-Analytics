import numpy as np
import pandas as pd
import requests
import io
import time
import os
import random
import logging
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# =============================================================================
# ГЕОДЕЗИЧЕСКИЕ ПРЕОБРАЗОВАНИЯ
# =============================================================================
def geocentric_to_geodetic(x, y, z):
    a, f = 6378137.0, 1 / 298.257223563
    b    = a * (1 - f)
    e2   = 1 - (b / a) ** 2
    ep2  = (a / b) ** 2 - 1
    lon  = np.degrees(np.arctan2(y, x))
    p    = np.sqrt(x**2 + y**2)
    th   = np.arctan2(a * z, b * p)
    lat  = np.degrees(np.arctan2(
        z  + ep2 * b * np.sin(th)**3,
        p  - e2  * a * np.cos(th)**3
    ))
    return float(lat), float(lon)

def ecef_to_enu_vel(lat, lon, vx, vy, vz):
    phi, lam = np.radians(lat), np.radians(lon)
    R = np.array([
        [-np.sin(lam),               np.cos(lam),               0.0        ],
        [-np.sin(phi)*np.cos(lam),  -np.sin(phi)*np.sin(lam),   np.cos(phi)],
        [ np.cos(phi)*np.cos(lam),   np.cos(phi)*np.sin(lam),   np.sin(phi)]
    ])
    v = R @ np.array([vx, vy, vz])
    return R, v

# =============================================================================
# ПРЕДОБРАБОТКА ДАННЫХ (УДАЛЕНИЕ ВЫБРОСОВ И ПОИСК СКАЧКОВ)
# =============================================================================
def remove_spikes(t, m, s, window=7, k_mad=5.0):
    """Удаляет одиночные аномальные выбросы до запуска фильтра."""
    n = len(t)
    valid = np.ones(n, dtype=bool)
    for i in range(3):
        series = pd.Series(m[:, i])
        roll_med = series.rolling(window, center=True, min_periods=1).median()
        roll_mad = series.rolling(window, center=True, min_periods=1).apply(
            lambda x: np.median(np.abs(x - np.median(x))), raw=True
        )
        # Ограничитель минимального шума (чтобы не делить на 0 на идеальных участках)
        floor = 0.002
        invalid = np.abs(m[:, i] - roll_med) > (k_mad * np.maximum(roll_mad, floor))
        valid &= ~invalid
    return t[valid], m[valid], s[valid], valid

def detect_jumps(times, measurements, threshold=0.030, window=10, min_gap=60):
    """
    Скользящий медианный фильтр + подавление двойной детекции (эха).
    Берется только самый большой скачок из кластера.
    """
    n = len(times)
    if n < window * 3:
        return [], []

    raw_jumps, raw_mags = [], []
    for k in range(window, n - window):
        after  = np.median(measurements[k : k + window], axis=0)
        before = np.median(measurements[k - window : k], axis=0)
        delta  = np.linalg.norm(after - before)
        if delta > threshold:
            raw_jumps.append(k)
            raw_mags.append(delta)

    if not raw_jumps:
        return [], []

    # Кластеризация: всё что ближе min_gap — один кластер
    clusters, cur_cluster = [], [0]
    for i in range(1, len(raw_jumps)):
        if raw_jumps[i] - raw_jumps[cur_cluster[-1]] <= min_gap:
            cur_cluster.append(i)
        else:
            clusters.append(cur_cluster)
            cur_cluster = [i]
    clusters.append(cur_cluster)

    # Из каждого кластера — только максимальный скачок
    clean, mags = [], []
    for cl in clusters:
        best = cl[np.argmax([raw_mags[i] for i in cl])]
        clean.append(raw_jumps[best])
        mags.append(raw_mags[best])

    return clean, mags

# =============================================================================
# ПОЛЮС ЭЙЛЕРА — IRLS И ДЕФОРМАЦИИ
# =============================================================================
def estimate_euler_pole(lats, lons, ve, vn):
    phi, lam = np.radians(lats), np.radians(lons)
    n = len(lats)
    R = 6371000.0
    A = np.zeros((2 * n, 3)); L = np.zeros(2 * n)
    for i in range(n):
        slat, clat = np.sin(phi[i]), np.cos(phi[i])
        slon, clon = np.sin(lam[i]), np.cos(lam[i])
        A[2*i,   :] = [-R*slat*clon, -R*slat*slon,  R*clat]
        L[2*i]      =  ve[i] / 1000.0
        A[2*i+1, :] = [ R*slon,      -R*clon,        0.0  ]
        L[2*i+1]    =  vn[i] / 1000.0

    try: Omega, _, _, _ = np.linalg.lstsq(A, L, rcond=None)
    except: Omega = np.zeros(3)

    weights = np.ones(2 * n)
    for _ in range(6):
        W = np.diag(weights)
        try:
            AtW   = A.T @ W
            Omega = np.linalg.solve(AtW @ A + np.eye(3) * 1e-12, AtW @ L)
        except np.linalg.LinAlgError: break
        res   = L - A @ Omega
        mad   = np.median(np.abs(res - np.median(res)))
        sigma = max(mad / 0.6745, 1e-9)
        u     = res / (4.685 * sigma)
        weights = np.where(np.abs(u) <= 1.0, (1 - u**2)**2, 1e-6)

    magnitude = float(np.linalg.norm(Omega) * 180 / np.pi * 1e6)
    euler_valid = magnitude < 50.0

    if not euler_valid:
        Omega, V_model = np.zeros(3), np.zeros(2 * n)
    else:
        V_model = A @ Omega

    return Omega, V_model[0::2] * 1000.0, V_model[1::2] * 1000.0, euler_valid


def _perfect_initialization_12d(times, measurements, window_years=2.0):
    """
    Умная инициализация: вычисляет тренд и проверяет, 
    нужно ли добавлять сезонную модель (синусоиды).
    """
    t_span = times[-1] - times[0]
    # Берем данные для инициализации (не более 2 лет)
    mask = times <= (times[0] + window_years) if t_span > window_years else np.ones(len(times), dtype=bool)
    
    t_fit, m_fit = times[mask], measurements[mask]
    t_shifted = t_fit - times[0]
    omega = 2 * np.pi * t_fit
    
    x0_12d = np.zeros(12)
    
    # Решаем МНК для каждой координаты X, Y, Z
    for i in range(3):
        try:
            # Шаг А: Считаем только линейный тренд
            A_lin = np.column_stack([np.ones(len(t_fit)), t_shifted])
            c_lin = np.linalg.lstsq(A_lin, m_fit[:, i], rcond=None)[0]
            
            # Шаг Б: Проверяем остатки на наличие годовой волны
            resids = m_fit[:, i] - (c_lin[0] + c_lin[1] * t_shifted)
            A_seas = np.column_stack([np.sin(omega), np.cos(omega)])
            c_seas = np.linalg.lstsq(A_seas, resids, rcond=None)[0]
            
            amplitude = np.sqrt(c_seas[0]**2 + c_seas[1]**2)
            
            # ПРЕДОХРАНИТЕЛЬ: 
            # Включаем сезонность только если ряд > 0.8 года И амплитуда вменяемая (от 1мм до 5см)
            if 0.001 < amplitude < 0.05 and t_span > 0.8:
                s_amp, c_amp = c_seas[0], c_seas[1]
            else:
                s_amp, c_amp = 0.0, 0.0
                
            x0_12d[i] = c_lin[0]      # Позиция
            x0_12d[i+3] = c_lin[1]    # Скорость
            x0_12d[i+6] = s_amp       # Sin
            x0_12d[i+9] = c_amp       # Cos
        except:
            x0_12d[i] = m_fit[0, i]
            
    return x0_12d
# =============================================================================
# ФИЛЬТР КАЛМАНА 12D (ОБНОВЛЕННЫЙ)
# =============================================================================
class SeasonalEKF12D:
    def __init__(self, dt, x0):
        self.dt = dt
        self.x = x0.copy()
        
        # Начальная ковариация P
        self.P = np.eye(12) * 1e-6
        # Если сезонность в x0 нулевая (отключена), ставим P=0, чтобы фильтр её не "выдумывал"
        for i in range(6, 12):
            if abs(self.x[i]) < 1e-10:
                self.P[i, i] = 0.0
        
        self.F = np.eye(12)
        self.F[0, 3] = dt; self.F[1, 4] = dt; self.F[2, 5] = dt
        
        # Матрица шума процесса Q (Ультра-жесткая настройка для стабильности)
        self.Q = np.zeros((12, 12))
        for i in range(3):
            self.Q[i, i] = 1e-10       # Шум позиции
            self.Q[i+3, i+3] = 1e-13   # Тектоническая скорость (почти константа)
            # Шум сезонности (только если она активна)
            self.Q[i+6, i+6] = 1e-13 if self.P[i+6, i+6] > 0 else 0.0
            self.Q[i+9, i+9] = 1e-13 if self.P[i+9, i+9] > 0 else 0.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z, R_obs, t_year):
        omega = 2 * np.pi * t_year
        sin_w, cos_w = np.sin(omega), np.cos(omega)
        H = np.zeros((3, 12))
        for i in range(3):
            H[i, i] = 1.0; H[i, i+6] = sin_w; H[i, i+9] = cos_w

        innov = z - (H @ self.x)
        S = H @ self.P @ H.T + R_obs

        # Mahalanobis gating: раздуваем R для явных выбросов вместо
        # жёсткого отклонения — фильтр продолжает работать, но меньше доверяет
        try:
            S_inv = np.linalg.inv(S)
            d2 = float(innov.T @ S_inv @ innov)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
            d2 = float(innov.T @ S_inv @ innov)

        gating_threshold = 25.0   # ~5σ в 3D
        if d2 > gating_threshold:
            inflation = max(1.0, d2 / gating_threshold)
            R_eff = R_obs * inflation
            S = H @ self.P @ H.T + R_eff
            try:
                K = self.P @ H.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                K = self.P @ H.T @ np.linalg.pinv(S)
        else:
            K = self.P @ H.T @ S_inv

        delta = K @ innov
        # Ограничение шага обновления (защита от численных взрывов)
        for i in range(3):
            delta[i]   = np.clip(delta[i],   -0.5,  0.5)   # позиция, м
            delta[i+3] = np.clip(delta[i+3], -0.02, 0.02)  # скорость, м/год
            delta[i+6] = np.clip(delta[i+6], -0.05, 0.05)  # сезонность, м
            delta[i+9] = np.clip(delta[i+9], -0.05, 0.05)

        if not (np.any(np.isnan(delta)) or np.any(np.isinf(delta))):
            self.x = self.x + delta
            # Joseph form — численно устойчивая, симметричная
            I_KH = np.eye(12) - K @ H
            self.P = I_KH @ self.P @ I_KH.T + K @ R_obs @ K.T

        return innov, (H @ self.x), self.x[0:3].copy()

# =============================================================================
# ОСНОВНОЙ ДВИЖОК
# =============================================================================
class GeodeticEngine:
    def __init__(self):
        self.url_fmt = "https://geodesy.unr.edu/gps_timeseries/IGS20/txyz/{}.txyz2"
        self.master_file = "data/raw_data/DataHoldings.txt"

    def _ensure_master_file(self):
        if os.path.exists(self.master_file): return True
        os.makedirs("data/raw_data", exist_ok=True)
        try:
            r = requests.get("http://geodesy.unr.edu/NGLStationPages/DataHoldings.txt", timeout=30)
            r.raise_for_status()
            with open(self.master_file, 'w') as f: f.write(r.text)
            return True
        except: return False

    def _fetch_and_process(self, sid, target_epoch):
        try:
            r = requests.get(self.url_fmt.format(sid), timeout=10, verify=False, headers={'User-Agent': 'Mozilla/5.0'})
            if r.status_code != 200: return None

            df_raw = pd.read_csv(io.StringIO(r.text), sep=r'\s+', header=None, usecols=range(9)).dropna()
            df_raw.columns = ['site','date','t','x','y','z','sx','sy','sz']

            # Проверка: есть ли данные хотя бы близко к нужной эпохе?
            if df_raw.t.min() > target_epoch + 0.5 or df_raw.t.max() < target_epoch - 2.0:
                return None

            # Вырезаем окно для фильтра Калмана (вокруг эпохи)
            df_w = df_raw[(df_raw.t >= target_epoch - 4.0) & (df_raw.t <= target_epoch + 1.5)]
            if len(df_w) < 80: return None

            t, m, s = df_w.t.values, df_w[['x','y','z']].values, df_w[['sx','sy','sz']].values
            
            # Предварительная очистка от спайков
            t, m, s, _ = remove_spikes(t, m, s, window=7, k_mad=5.0)

            # Скорости для стрелок (МНК)
            vels_lsq, sigma_vels = [], []
            for _ci in range(3):
                try:
                    _c, _cov = np.polyfit(t, m[:, _ci], 1, cov=True)
                    vels_lsq.append(float(_c[0]))
                    sigma_vels.append(float(np.sqrt(max(_cov[0, 0], 0.0))))
                except Exception:
                    vels_lsq.append(float(np.polyfit(t, m[:, _ci], 1)[0]))
                    sigma_vels.append(1e-4)
            vx_r, vy_r, vz_r = vels_lsq
            sigma_vx, sigma_vy, sigma_vz = sigma_vels

            # Детекция скачков
            jump_idxs, _ = detect_jumps(t, m, threshold=0.030, min_gap=60)
            n_jumps = len(jump_idxs)

            segments = []; start_seg = 0
            for j in jump_idxs:
                segments.append((start_seg, j))
                start_seg = j
            segments.append((start_seg, len(t)))

            dt_med = float(np.median(np.diff(t)))
            resids, plot_xyz, trend_xyz = [], [], []

            for (st, end) in segments:
                if end - st < 15: continue
                
                x0 = _perfect_initialization_12d(t[st:end], m[st:end])
                ekf = SeasonalEKF12D(dt_med, x0)
                
                for i in range(st, end):
                    R_obs = np.diag(s[i]**2) + np.eye(3) * 0.002**2 
                    ekf.predict()
                    y_res, full_pt, trend_pt = ekf.update(m[i], R_obs, t[i])
                    
                    resids.append(y_res)
                    plot_xyz.append(full_pt)
                    trend_xyz.append(trend_pt)

            if not resids: return None

            res_arr, plot_arr, trend_arr = np.array(resids), np.array(plot_xyz), np.array(trend_xyz)
            rmse = float(np.sqrt(np.mean(res_arr**2)) * 1000)

            # --- ИСПРАВЛЕННЫЙ БЛОК НАДЕЖНОСТИ И ЭКСТРАПОЛЯЦИИ ---
            
            # Технический сдвиг для Калмана (от конца окна до эпохи)
            dt_back_tech = target_epoch - t[-1]
            
            # Реальная экстраполяция (смотрит на ВСЕ данные станции, а не на окно)
            # Если эпоха внутри ряда, экстраполяция = 0
            dt_extrap_real = max(0, target_epoch - df_raw.t.max(), df_raw.t.min() - target_epoch)
            
            # Дыры вокруг целевой эпохи (+/- 1 год)
            w_mask = np.abs(t - target_epoch) < 1.0
            dt_gap = float(np.max(np.diff(t[w_mask]))) if w_mask.sum() > 2 else 9.9
            
            # Истинная надежность: не экстраполируем больше чем на год, нет огромных дыр
            epoch_reliable = bool((dt_extrap_real < 1.0) and (dt_gap < 0.5))
            
            # Предвестник зажигаем только если с эпохой все ок, но шумит фильтр
            is_precursor = bool(rmse > 15.0) if epoch_reliable else False

            # --- КОНЕЦ ИСПРАВЛЕНИЙ ---

            # Координата на эпоху
            x_ep, y_ep, z_ep = (ekf.x[0:3] + ekf.x[3:6] * dt_back_tech).tolist()

            # СКО координат
            sigma_x = float(np.sqrt(max(ekf.P[0,0] + ekf.P[3,3]*dt_back_tech**2, 0)) * 1000)
            sigma_y = float(np.sqrt(max(ekf.P[1,1] + ekf.P[4,4]*dt_back_tech**2, 0)) * 1000)
            sigma_z = float(np.sqrt(max(ekf.P[2,2] + ekf.P[5,5]*dt_back_tech**2, 0)) * 1000)

            # Геодезия + ENU
            lat, lon = geocentric_to_geodetic(x_ep, y_ep, z_ep)
            R_mat, v_enu = ecef_to_enu_vel(lat, lon, vx_r, vy_r, vz_r)
            ve, vn, vu = float(v_enu[0]), float(v_enu[1]), float(v_enu[2])

            sigma_v_ecef = np.array([sigma_vx, sigma_vy, sigma_vz])
            sigma_v_enu  = np.sqrt(np.maximum((R_mat ** 2) @ (sigma_v_ecef ** 2), 0))

            amp_annual_x = float(np.sqrt(ekf.x[6]**2 + ekf.x[9]**2) * 1000)

            step = max(1, len(t) // 120)
            x0r, y0r, z0r = m[0, 0], m[0, 1], m[0, 2]

            return {
                'id': sid, 'lat': lat, 'lon': lon,
                'x': x_ep, 'y': y_ep, 'z': z_ep,
                'sigma_x': round(sigma_x, 3), 'sigma_y': round(sigma_y, 3), 'sigma_z': round(sigma_z, 3),
                've': round(ve * 1000, 3), 'vn': round(vn * 1000, 3), 'vu': round(vu * 1000, 3),
                'sigma_ve': round(float(sigma_v_enu[0] * 1000), 3), 'sigma_vn': round(float(sigma_v_enu[1] * 1000), 3), 'sigma_vu': round(float(sigma_v_enu[2] * 1000), 3),
                'rmse': round(rmse, 3), 'jumps': n_jumps,
                'is_precursor': is_precursor,
                'epoch_reliable': epoch_reliable,
                't_span_start': round(float(df_raw.t.min()), 3), # Истинное начало
                't_span_end':   round(float(df_raw.t.max()), 3), # Истинный конец
                'n_obs':        int(len(t)),
                'amp_annual_x': round(amp_annual_x, 3),
                'dt_extrap':    round(float(dt_extrap_real), 3), # Истинная экстраполяция
                'dt_gap':       round(float(dt_gap), 3),
                'graph': {
                    't': t[::step].tolist(),
                    'raw_x': (m[::step, 0] - x0r).tolist(), 'filt_x': (plot_arr[::step, 0] - x0r).tolist(), 'trend_x': (trend_arr[::step, 0] - x0r).tolist(),
                    'raw_y': (m[::step, 1] - y0r).tolist(), 'filt_y': (plot_arr[::step, 1] - y0r).tolist(), 'trend_y': (trend_arr[::step, 1] - y0r).tolist(),
                    'raw_z': (m[::step, 2] - z0r).tolist(), 'filt_z': (plot_arr[::step, 2] - z0r).tolist(), 'trend_z': (trend_arr[::step, 2] - z0r).tolist(),
                }
            }
        except Exception as e:
            log.error(f"Error {sid}: {e}"); return None

    def analyze_region(self, bbox, target_epoch=2020.0, target_count=15, max_load=0):
        t_start = time.time()
        if not self._ensure_master_file(): return {"error": "Не удалось загрузить каталог станций"}

        sids = []
        with open(self.master_file, 'r') as f:
            for line in f:
                p = line.split()
                if len(p) < 3: continue
                try:
                    lat, lon_r = float(p[1]), float(p[2])
                    lon = lon_r - 360.0 if lon_r > 180.0 else lon_r
                    if bbox[0] <= lat <= bbox[1] and bbox[2] <= lon <= bbox[3]: sids.append(p[0])
                except ValueError: continue

        if not sids: return {"error": "В выбранном регионе нет станций в каталоге"}

        target_sids = random.sample(sids, max_load) if (max_load > 0 and len(sids) > max_load) else sids
        log.info(f"Найдено {len(sids)} станций, к обработке: {len(target_sids)}")

        results = []
        with ThreadPoolExecutor(max_workers=24) as ex:
            futures = {ex.submit(self._fetch_and_process, sid, target_epoch): sid for sid in target_sids}
            for fut in as_completed(futures):
                res = fut.result()
                if res is not None: results.append(res)

        if len(results) < 4: return {"error": "Недостаточно данных (< 4 станций успешно обработано)"}

        df = pd.DataFrame(results)
        Omega, ve_mod, vn_mod, euler_valid = estimate_euler_pole(df.lat.values, df.lon.values, df.ve.values, df.vn.values)
        df['ve_plate'] = (df['ve'] - ve_mod).clip(-150.0, 150.0)
        df['vn_plate'] = (df['vn'] - vn_mod).clip(-150.0, 150.0)

        df['score'] = (100.0 - df['rmse'] * 2.0 - (~df['epoch_reliable']).astype(int) * 20.0 - df['jumps'] * 2.0).clip(lower=0.0)

        n_cl = min(len(df), target_count)
        km = KMeans(n_clusters=n_cl, n_init=10, random_state=42).fit(df[['lat', 'lon']])
        df['cluster'] = km.labels_
        ref_ids = df.loc[df.groupby('cluster')['score'].idxmax()]['id'].tolist()

        df_tri_prelim = df[df['epoch_reliable']].reset_index(drop=True)
        if len(df_tri_prelim) < 4: df_tri_prelim = df.copy().reset_index(drop=True)


        stations_list = []
        for rec in df.to_dict(orient='records'):
            clean = {}
            for k, v in rec.items():
                if isinstance(v, np.integer): clean[k] = int(v)
                elif isinstance(v, np.floating): clean[k] = float(v)
                elif isinstance(v, np.bool_): clean[k] = bool(v)
                else: clean[k] = v
            stations_list.append(clean)


        elapsed = round(time.time() - t_start, 1)

        # Определяем зону по центру bbox для station_detail (без триангуляции, но поле нужно)
        lat_c = (bbox[0] + bbox[1]) / 2.0
        lon_c = (bbox[2] + bbox[3]) / 2.0
        is_pacific = ((28 < lat_c < 65 and 128 < lon_c < 200) or (48 < lat_c < 72 and -172 < lon_c < -128))
        is_alpine  = ((25 < lat_c < 48 and 22 < lon_c < 92)   or (-18 < lat_c < 22 and 26 < lon_c < 46))
        if is_pacific:
            zone_name, rmse_max = "Тихоокеанское кольцо", 7.0
        elif is_alpine:
            zone_name, rmse_max = "Альпийский пояс", 6.0
        else:
            zone_name, rmse_max = "Стабильная платформа", 6.0

        return {
            'meta': {
                'processed': int(len(df)), 'epoch': target_epoch,
                'euler_pole': (Omega * 180 / np.pi * 1e6).tolist(),
                'euler_valid': euler_valid,
                'time_sec': elapsed,
                'zone_name': zone_name,
                'rmse_max': rmse_max,
            },
            'stations': stations_list,
            'ref_network_ids': ref_ids,
        }
