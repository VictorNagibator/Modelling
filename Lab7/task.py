import math
import csv
import sys

class SMOCalculator:
    def __init__(self):
        self.params = {}
        self.system_type = ""
        self.results = {}
    
    def load_from_csv(self, filename='Lab7\\input.csv'):
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                # Читаем строку, разделяем точкой с запятой
                line = file.readline().strip()
                if not line:
                    print("Файл пустой")
                    return False
                
                # Разделяем значения
                values = line.split(';')
                
                if len(values) >= 4:
                    try:
                        # Преобразуем значения
                        self.params['lambda'] = float(values[0])
                        self.params['mu'] = float(values[1])
                        self.params['n'] = int(float(values[2]))
                        
                        # Обработка m (может быть числом или 'inf')
                        m_str = values[3].strip().lower()
                        if m_str in ['inf', 'бесконечность', '∞', '']:
                            self.params['m'] = None  # Бесконечная очередь
                        else:
                            self.params['m'] = int(float(m_str))
                        
                        print("\nПараметры успешно загружены:")
                        print(f"  λ = {self.params['lambda']}")
                        print(f"  μ = {self.params['mu']}")
                        print(f"  n = {self.params['n']}")
                        print(f"  m = {self.params['m'] if self.params['m'] is not None else 'бесконечность'}")
                        return True
                        
                    except ValueError as e:
                        print(f"Ошибка преобразования данных: {e}")
                        return False
                else:
                    print("Недостаточно параметров в файле")
                    return False
                    
        except FileNotFoundError:
            print(f"Файл {filename} не найден")
            print("Создайте файл input.csv с содержимым вида: λ; μ; n; m")
            print("где m может быть 'inf' для бесконечной очереди")
            return False
        except Exception as e:
            print(f"Ошибка при чтении файла: {e}")
            return False
    
    def select_system_type(self):
        print("\nВыберете тип системы:")
        print("1. M|M|n|0  (многоканальная с отказами)")
        print("2. M|M|1|m  (одноканальная с ограниченной очередью)")
        print("3. M|M|1|∞  (одноканальная с неограниченной очередью)")
        print("4. M|M|n|m  (многоканальная с ограниченной очередью)")
        print("5. M|M|n|∞  (многоканальная с неограниченной очередью)")
        print("6. Замкнутая одноканальная СМО")
        print("7. Замкнутая многоканальная СМО")
        
        while True:
            choice = input("Введите номер типа системы (1-7): ").strip()
            
            if choice == '1':
                self.system_type = "M|M|n|0"
                return True
            elif choice == '2':
                self.system_type = "M|M|1|m"
                return True
            elif choice == '3':
                self.system_type = "M|M|1|∞"
                return True
            elif choice == '4':
                self.system_type = "M|M|n|m"
                return True
            elif choice == '5':
                self.system_type = "M|M|n|∞"
                return True
            elif choice == '6':
                self.system_type = "Замкнутая одноканальная"
                return True
            elif choice == '7':
                self.system_type = "Замкнутая многоканальная"
                return True
            else:
                print("Неверный выбор. Попробуйте снова.")
    
    # ФУНКЦИИ РАСЧЕТА ДЛЯ КАЖДОГО ТИПА СИСТЕМЫ
    
    def calculate_mmn_0(self):
        lambd = self.params['lambda']
        mu = self.params['mu']
        n = self.params['n']
        
        # ρ = λ/μ - приведенная интенсивность потока
        rho = lambd / mu
        
        # Вычисление суммы для p0
        sum_series = 0
        for k in range(n + 1):
            sum_series += rho**k / math.factorial(k)
        
        # p0 - вероятность того, что все каналы свободны
        p0 = 1 / sum_series
        
        # pn - вероятность того, что все n каналов заняты
        pn = (rho**n / math.factorial(n)) * p0
        
        # Вероятность отказа
        p_otk = pn
        
        # Относительная пропускная способность
        q = 1 - p_otk
        
        # Абсолютная пропускная способность
        A = lambd * q
        
        # Среднее число занятых каналов
        k_sr = rho * (1 - pn)
        
        # Сохраняем все результаты
        self.results = {
            'Тип системы': self.system_type,
            'Параметры': f'λ={lambd}, μ={mu}, n={n}',
            'ρ (приведенная интенсивность)': rho,
            'p₀ (вероятность простоя)': p0,
            'pₙ (вероятность занятости всех каналов)': pn,
            'p_отк (вероятность отказа)': p_otk,
            'q (относительная пропускная способность)': q,
            'A (абсолютная пропускная способность)': A,
            'k_ср (среднее число занятых каналов)': k_sr,
            'Формулы': 'Использованы формулы Эрланга (4.3-4.8)'
        }
        
        return True
    
    def calculate_mm1_m(self):
        lambd = self.params['lambda']
        mu = self.params['mu']
        m = self.params['m']
        
        if m is None:
            print("Для этого типа системы требуется параметр m (ограничение очереди)")
            return False
        
        # ρ = λ/μ
        rho = lambd / mu
        
        # Вероятность простоя системы
        if rho == 1:
            p0 = 1 / (m + 2)
        else:
            p0 = (1 - rho) / (1 - rho**(m + 2))
        
        # Вероятности состояний
        p = [0] * (m + 2)  # Индексы 0..m+1
        p[0] = p0
        for k in range(1, m + 2):
            p[k] = rho**k * p0
        
        # Вероятность отказа
        p_otk = p[m + 1]
        
        # Относительная пропускная способность
        q = 1 - p_otk
        
        # Абсолютная пропускная способность
        A = lambd * q
        
        # Средняя длина очереди
        if rho == 1:
            r_sr = m * (m + 1) / (2 * (m + 2))
        else:
            numerator = rho**2 * (1 - rho**m * (m + 1 - m * rho))
            denominator = (1 - rho**(m + 2)) * (1 - rho)
            r_sr = numerator / denominator
        
        # Среднее число заявок под обслуживанием
        omega_sr = (rho - rho**(m + 2)) / (1 - rho**(m + 2))
        
        # Среднее число заявок в системе
        v_sr = r_sr + omega_sr
        
        # Среднее время ожидания
        t_ozh = r_sr / lambd
        
        # Среднее время пребывания в системе
        t_sist = t_ozh + q / mu
        
        # Сохраняем все результаты
        self.results = {
            'Тип системы': self.system_type,
            'Параметры': f'λ={lambd}, μ={mu}, m={m}',
            'ρ (коэффициент загрузки)': rho,
            'p₀ (вероятность простоя)': p0,
            'p_отк (вероятность отказа)': p_otk,
            'q (относительная пропускная способность)': q,
            'A (абсолютная пропускная способность)': A,
            'r_ср (средняя длина очереди)': r_sr,
            'ω_ср (среднее число заявок под обслуживанием)': omega_sr,
            'v_ср (среднее число заявок в системе)': v_sr,
            't_ож (среднее время ожидания)': t_ozh,
            't_сист (среднее время пребывания в системе)': t_sist
        }
        
        return True
    
    def calculate_mm1_inf(self):
        lambd = self.params['lambda']
        mu = self.params['mu']
        
        # ρ = λ/μ
        rho = lambd / mu
        
        # Проверка условия существования стационарного режима
        if rho >= 1:
            self.results = {
                'Тип системы': self.system_type,
                'Параметры': f'λ={lambd}, μ={mu}',
                'ρ (коэффициент загрузки)': rho,
                'Статус': 'Очередь растёт бесконечно',
                'Причина': 'ρ >= 1, стационарного режима не существует',
                'Рекомендация': 'Увеличьте интенсивность обслуживания μ или уменьшите λ'
            }
            return True
        
        # Вероятность простоя системы
        p0 = 1 - rho
        
        # Средняя длина очереди
        r_sr = rho**2 / (1 - rho)
        
        # Среднее число заявок в системе
        v_sr = rho / (1 - rho)
        
        # Среднее время ожидания
        t_ozh = rho / (mu * (1 - rho))
        
        # Среднее время пребывания в системе
        t_sist = 1 / (mu * (1 - rho))
        
        # Сохраняем все результаты
        self.results = {
            'Тип системы': self.system_type,
            'Параметры': f'λ={lambd}, μ={mu}',
            'ρ (коэффициент загрузки)': rho,
            'Условие стационарности': 'ρ < 1 - выполнено',
            'p₀ (вероятность простоя)': p0,
            'r_ср (средняя длина очереди)': r_sr,
            'v_ср (среднее число заявок в системе)': v_sr,
            't_ож (среднее время ожидания)': t_ozh,
            't_сист (среднее время пребывания в системе)': t_sist,
            'Формула Литтла (проверка)': {
                'v_ср = λ·t_сист': v_sr,
                'λ·t_сист': lambd * t_sist,
                'r_ср = λ·t_ож': r_sr,
                'λ·t_ож': lambd * t_ozh
            }
        }
        
        return True
    
    def calculate_mmn_m(self):
        lambd = self.params['lambda']
        mu = self.params['mu']
        n = self.params['n']
        m = self.params['m']
        
        if m is None:
            print("Для этого типа системы требуется параметр m (ограничение очереди)")
            return False
        
        # ρ = λ/μ, γ = ρ/n
        rho = lambd / mu
        gamma = rho / n
        
        # Вычисление суммы для p0
        sum1 = 0
        for k in range(n + 1):
            sum1 += rho**k / math.factorial(k)
        
        # Вычисление второй суммы
        if gamma == 1:
            sum2 = m
        else:
            sum2 = gamma * (1 - gamma**m) / (1 - gamma)
        
        # p0 - вероятность простоя системы
        p0 = 1 / (sum1 + (rho**n / math.factorial(n)) * sum2)
        
        # Вероятность отказа
        p_otk = (rho**(n + m) / (n**m * math.factorial(n))) * p0
        
        # Относительная пропускная способность 
        q = 1 - p_otk
        
        # Абсолютная пропускная способность 
        A = lambd * q
        
        # Среднее число занятых каналов 
        k_sr = rho * (1 - p_otk)
        
        # Средняя длина очереди
        if gamma == 1:
            r_sr = (rho**(n + 1) * p0 / (math.factorial(n) * n)) * (m * (m + 1) / 2)
        else:
            numerator = rho**(n + 1) * p0 * (1 - (m + 1) * gamma**m + m * gamma**(m + 1))
            denominator = math.factorial(n) * n * (1 - gamma)**2
            r_sr = numerator / denominator
        
        # Среднее число заявок в системе
        v_sr = k_sr + r_sr
        
        # Среднее время ожидания 
        t_ozh = r_sr / lambd
        
        # Среднее время пребывания в системе
        t_sist = t_ozh + q / mu
        
        # Сохраняем все результаты
        self.results = {
            'Тип системы': self.system_type,
            'Параметры': f'λ={lambd}, μ={mu}, n={n}, m={m}',
            'ρ (коэффициент загрузки)': rho,
            'γ (ρ/n)': gamma,
            'p₀ (вероятность простоя)': p0,
            'p_отк (вероятность отказа)': p_otk,
            'q (относительная пропускная способность)': q,
            'A (абсолютная пропускная способность)': A,
            'k_ср (среднее число занятых каналов)': k_sr,
            'r_ср (средняя длина очереди)': r_sr,
            'v_ср (среднее число заявок в системе)': v_sr,
            't_ож (среднее время ожидания)': t_ozh,
            't_сист (среднее время пребывания в системе)': t_sist
        }
        
        return True
    
    def calculate_mmn_inf(self):
        lambd = self.params['lambda']
        mu = self.params['mu']
        n = self.params['n']
        
        # ρ = λ/μ, γ = ρ/n
        rho = lambd / mu
        gamma = rho / n
        
        # Проверка условия существования стационарного режима
        if gamma >= 1:
            self.results = {
                'Тип системы': self.system_type,
                'Параметры': f'λ={lambd}, μ={mu}, n={n}',
                'ρ (коэффициент загрузки)': rho,
                'γ (ρ/n)': gamma,
                'Статус': 'Очередь растёт бесконечно',
                'Причина': 'γ >= 1, стационарного режима не существует',
                'Рекомендация': f'Увеличьте число каналов n > {rho} или уменьшите λ'
            }
            return True
        
        # Вычисление суммы для p0
        sum1 = 0
        for k in range(n + 1):
            sum1 += rho**k / math.factorial(k)
        
        sum2 = rho**(n + 1) / (math.factorial(n) * (n - rho))
        
        # p0 - вероятность простоя системы
        p0 = 1 / (sum1 + sum2)
        
        # Средняя длина очереди
        r_sr = (rho**(n + 1) * p0) / (math.factorial(n) * n * (1 - gamma)**2)
        
        # Среднее число занятых каналов
        k_sr = rho
        
        # Среднее число заявок в системе
        v_sr = k_sr + r_sr
        
        # Среднее время ожидания
        t_ozh = r_sr / lambd
        
        # Среднее время пребывания в системе
        t_sist = t_ozh + 1 / mu
        
        # Сохраняем все результаты
        self.results = {
            'Тип системы': self.system_type,
            'Параметры': f'λ={lambd}, μ={mu}, n={n}',
            'ρ (коэффициент загрузки)': rho,
            'γ (ρ/n)': gamma,
            'Условие стационарности': 'γ < 1 - выполнено',
            'p₀ (вероятность простоя)': p0,
            'r_ср (средняя длина очереди)': r_sr,
            'k_ср (среднее число занятых каналов)': k_sr,
            'v_ср (среднее число заявок в системе)': v_sr,
            't_ож (среднее время ожидания)': t_ozh,
            't_сист (среднее время пребывания в системе)': t_sist,
            'Формула Литтла (проверка)': {
                'v_ср = λ·t_сист': v_sr,
                'λ·t_сист': lambd * t_sist,
                'r_ср = λ·t_ож': r_sr,
                'λ·t_ож': lambd * t_ozh
            }
        }
        
        return True
    
    def calculate_closed_mm1(self):
        # Для замкнутой системы параметры: N = n, λ, μ
        N = self.params['n']  # число источников (станков)
        lambd = self.params['lambda']
        mu = self.params['mu']
        
        # ρ = λ/μ
        rho = lambd / mu
        
        # Вычисление суммы для p0
        sum_series = 0
        for i in range(N + 1):
            sum_series += math.factorial(N) / math.factorial(N - i) * rho**i
        
        # p0 - вероятность простоя системы
        p0 = 1 / sum_series
        
        # Абсолютная пропускная способность
        A = (1 - p0) * mu
        
        # Среднее число заявок в системе 
        z_sr = N - (1 - p0) / rho
        
        # Среднее число заявок под обслуживанием 
        omega_sr = 1 - p0
        
        # Средняя длина очереди 
        r_sr = z_sr - omega_sr
        
        # Средняя интенсивность входящего потока
        Lambda_sr = A
        
        # Среднее время ожидания
        t_ozh = r_sr / Lambda_sr if Lambda_sr > 0 else 0
        
        # Среднее время пребывания в системе
        t_sist = z_sr / Lambda_sr if Lambda_sr > 0 else 0
        
        # Сохраняем все результаты
        self.results = {
            'Тип системы': self.system_type,
            'Параметры': f'N={N} станков, λ={lambd}, μ={mu}',
            'ρ (коэффициент загрузки)': rho,
            'p₀ (вероятность простоя рабочего)': p0,
            'A (абсолютная пропускная способность)': A,
            'z_ср (среднее число неисправных станков)': z_sr,
            'ω_ср (среднее число станков под обслуживанием)': omega_sr,
            'r_ср (среднее число станков в очереди)': r_sr,
            'Λ_ср (средняя интенсивность входящего потока)': Lambda_sr,
            't_ож (среднее время ожидания ремонта)': t_ozh,
            't_сист (среднее время в системе)': t_sist,
            'Коэффициент простоя станков': z_sr / N if N > 0 else 0,
            'Коэффициент занятости рабочего': 1 - p0
        }
        
        return True
    
    def calculate_closed_mmn(self):
        # Для замкнутой многоканальной системы:
        # n - число источников (станков)
        # m - число каналов (рабочих)
        N = self.params['n']  # число источников (станков)
        lambd = self.params['lambda']
        mu = self.params['mu']
        m = self.params['m'] if self.params['m'] is not None else 2  # число каналов (рабочих)
        
        if m is None:
            print("Для этого типа системы требуется параметр m (число каналов/рабочих)")
            return False
        
        # ρ = λ/μ
        rho = lambd / mu
        
        # Вычисление p0 
        sum_series = 0
        
        # Первая сумма: для i=0..m
        for i in range(m + 1):
            sum_series += math.factorial(N) / (math.factorial(i) * math.factorial(N - i)) * rho**i
        
        # Вторая сумма: для i=m+1..N
        for i in range(m + 1, N + 1):
            sum_series += math.factorial(N) / (math.factorial(m) * math.factorial(N - i) * m**(i - m)) * rho**i
        
        p0 = 1 / sum_series
        
        # Вычисление вероятностей состояний
        p = [0] * (N + 1)
        p[0] = p0
        
        for i in range(1, m + 1):
            p[i] = math.factorial(N) / (math.factorial(i) * math.factorial(N - i)) * rho**i * p0
        
        for i in range(m + 1, N + 1):
            p[i] = math.factorial(N) / (math.factorial(m) * math.factorial(N - i) * m**(i - m)) * rho**i * p0
        
        # Среднее число занятых каналов
        k_sr = 0
        for i in range(1, m + 1):
            k_sr += i * p[i]
        
        for i in range(m + 1, N + 1):
            k_sr += m * p[i]
        
        # Абсолютная пропускная способность
        A = k_sr * mu
        
        # Среднее число заявок в системе
        z_sr = N - k_sr / rho
        
        # Средняя длина очереди
        r_sr = z_sr - k_sr
        
        # Средняя интенсивность входящего потока
        Lambda_sr = A
        
        # Среднее время ожидания
        t_ozh = r_sr / Lambda_sr if Lambda_sr > 0 else 0
        
        # Среднее время пребывания в системе
        t_sist = z_sr / Lambda_sr if Lambda_sr > 0 else 0
        
        # Сохраняем все результаты
        self.results = {
            'Тип системы': self.system_type,
            'Параметры': f'N={N} станков, m={m} рабочих, λ={lambd}, μ={mu}',
            'ρ (коэффициент загрузки)': rho,
            'p₀ (вероятность простоя всех рабочих)': p0,
            'A (абсолютная пропускная способность)': A,
            'k_ср (среднее число занятых рабочих)': k_sr,
            'z_ср (среднее число неисправных станков)': z_sr,
            'r_ср (среднее число станков в очереди)': r_sr,
            'Λ_ср (средняя интенсивность входящего потока)': Lambda_sr,
            't_ож (среднее время ожидания ремонта)': t_ozh,
            't_сист (среднее время в системе)': t_sist,
            'Коэффициент простоя станков': z_sr / N if N > 0 else 0,
            'Коэффициент занятости рабочих': k_sr / m if m > 0 else 0
        }
        
        return True
    
    def calculate(self):
        if not self.params:
            print("Нет загруженных параметров")
            return False
        
        # Выбираем функцию расчета в зависимости от типа системы
        if self.system_type == "M|M|n|0":
            return self.calculate_mmn_0()
        elif self.system_type == "M|M|1|m":
            return self.calculate_mm1_m()
        elif self.system_type == "M|M|1|∞":
            return self.calculate_mm1_inf()
        elif self.system_type == "M|M|n|m":
            return self.calculate_mmn_m()
        elif self.system_type == "M|M|n|∞":
            return self.calculate_mmn_inf()
        elif self.system_type == "Замкнутая одноканальная":
            return self.calculate_closed_mm1()
        elif self.system_type == "Замкнутая многоканальная":
            return self.calculate_closed_mmn()
        else:
            print("Неизвестный тип системы")
            return False
    
    def print_results(self):
        """Вывод результатов расчета"""
        if not self.results:
            print("Нет результатов для вывода")
            return
        
        print("\nРезультаты расчёта характеристик СМО")
        print(f"Тип системы: {self.results.get('Тип системы', 'Не указан')}")
        print(f"Параметры: {self.results.get('Параметры', 'Не указаны')}")
        
        # Выводим все характеристики
        for key, value in self.results.items():
            if key not in ['Тип системы', 'Параметры']:
                if isinstance(value, dict):
                    print(f"\n{key}:")
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, float):
                            print(f"  {subkey}: {subvalue:.6f}")
                        else:
                            print(f"  {subkey}: {subvalue}")
                elif isinstance(value, float):
                    print(f"{key}: {value:.6f}")
                else:
                    print(f"{key}: {value}")


def main():
    print("Программа для расчёта характеристик СМО")
    
    calculator = SMOCalculator()
    
    # Основной цикл программы
    while True:
        # Загрузка данных из файла
        filename = input("\nВведите имя файла с параметрами (по умолчанию input.csv): ").strip()
        if filename == '':
            filename = 'Lab7\\input.csv'
        
        if not calculator.load_from_csv(filename):
            choice = input("Хотите попробовать другой файл? (да/нет): ").strip().lower()
            if choice in ['да', 'yes', 'y', 'д']:
                continue
            else:
                break
        
        # Выбор типа системы
        if not calculator.select_system_type():
            break
        
        # Расчет характеристик
        if calculator.calculate():
            calculator.print_results()
        else:
            print("Ошибка при расчете характеристик")
        
        # Предложение повторить
        print("\nВыберите действие:")
        print("1. Загрузить новые параметры")
        print("2. Выйти из программы")
        
        choice = input("Ваш выбор (1-2): ").strip()
        
        if choice == '1':
            continue  # Вернуться к выбору типа системы
        elif choice == '2':
            print("Выход из программы...")
            break
        else:
            print("Неверный выбор. Продолжаем с новыми параметрами.")

if __name__ == "__main__":
    main()