import numpy as np
import matplotlib.pyplot as plt

class QueueSystem:
    def __init__(self, lambda_rate, mu_rate, queue_capacity, T_max, loss_per_refusal=50000):
        """
        Одноканальная СМО с ограниченной очередью
        lambda_rate: интенсивность поступления заявок (12 грузовиков/день)
        mu_rate: интенсивность обслуживания (10 грузовиков/день)
        queue_capacity: максимальная длина очереди (12 мест)
        T_max: время моделирования (дни)
        loss_per_refusal: потери за отказ (руб.)
        """
        self.lambda_rate = lambda_rate
        self.mu_rate = mu_rate
        self.queue_capacity = queue_capacity
        self.T_max = T_max
        self.loss_per_refusal = loss_per_refusal
        
        # Состояния системы: 0 - свободен, 1 - занят + очередь
        self.state_times = [0.0] * (queue_capacity + 2)  # время в каждом состоянии
        self.current_state = 0  # начальное состояние - система свободна
        self.last_state_change_time = 0.0  # время последнего изменения состояния
        
        # Статистика
        self.total_arrivals = 0  # общее число поступивших заявок
        self.served = 0  # обслуженные заявки
        self.refusals = 0  # отказы
        self.total_wait_time = 0.0  # суммарное время ожидания
        self.total_service_time = 0.0  # суммарное время обслуживания
        self.queue_length_history = []  # история длины очереди
        self.time_history = []  # временные метки
        
        # Очередь (храним время прихода заявок)
        self.queue = []
        
        # Канал
        self.channel_busy = False
        self.channel_free_time = 0.0
        
        # Планирование событий
        self.next_arrival_time = self.generate_exponential(self.lambda_rate)
        self.next_service_completion_time = float('inf')
    
    def generate_exponential(self, rate):
        """Генерация экспоненциально распределенной случайной величины"""
        return -np.log(np.random.random()) / rate
    
    def update_state_time(self, current_time):
        """Обновление времени пребывания в текущем состоянии"""
        dt = current_time - self.last_state_change_time
        self.state_times[self.current_state] += dt
        self.last_state_change_time = current_time
    
    def process_arrival(self, current_time):
        """Обработка прибытия заявки"""
        self.total_arrivals += 1
        
        if not self.channel_busy:
            # Канал свободен, начинаем обслуживание
            self.channel_busy = True
            service_time = self.generate_exponential(self.mu_rate)
            self.next_service_completion_time = current_time + service_time
            self.total_service_time += service_time
            self.served += 1
            # Изменение состояния
            self.update_state_time(current_time)
            self.current_state = 1  # канал занят, очередь пуста
            self.channel_free_time = self.next_service_completion_time
        else:
            # Канал занят, проверяем очередь
            if len(self.queue) < self.queue_capacity:
                # Есть место в очереди
                self.queue.append(current_time)  # запоминаем время прихода
                # Изменение состояния
                self.update_state_time(current_time)
                self.current_state = len(self.queue) + 1  # канал занят, в очереди N заявок
            else:
                # Очередь полна - отказ
                self.refusals += 1
        
        # Планируем следующее прибытие
        self.next_arrival_time = current_time + self.generate_exponential(self.lambda_rate)
    
    def process_service_completion(self, current_time):
        """Обработка завершения обслуживания"""
        if self.queue:
            # Есть заявки в очереди, берем первую
            arrival_time = self.queue.pop(0)
            wait_time = current_time - arrival_time
            self.total_wait_time += wait_time
            
            # Начинаем обслуживание следующей заявки
            service_time = self.generate_exponential(self.mu_rate)
            self.next_service_completion_time = current_time + service_time
            self.total_service_time += service_time
            self.served += 1
        else:
            # Очередь пуста, освобождаем канал
            self.channel_busy = False
            self.next_service_completion_time = float('inf')
        
        # Изменение состояния
        self.update_state_time(current_time)
        if self.channel_busy:
            self.current_state = len(self.queue) + 1
        else:
            self.current_state = 0
    
    def simulate(self):
        """Основной цикл моделирования"""
        current_time = 0.0
        
        while current_time < self.T_max:
            # Определяем ближайшее событие
            if self.next_arrival_time < self.next_service_completion_time:
                next_event_time = self.next_arrival_time
                event_type = 'arrival'
            else:
                next_event_time = self.next_service_completion_time
                event_type = 'service'
            
            if next_event_time > self.T_max:
                break
            
            # Обновляем время в текущем состоянии
            dt = next_event_time - current_time
            self.state_times[self.current_state] += dt
            current_time = next_event_time
            self.last_state_change_time = current_time
            
            # Записываем историю длины очереди
            self.queue_length_history.append(len(self.queue))
            self.time_history.append(current_time)
            
            # Обрабатываем событие
            if event_type == 'arrival':
                self.process_arrival(current_time)
            else:
                self.process_service_completion(current_time)
        
        # Завершаем сбор статистики
        dt = self.T_max - current_time
        self.state_times[self.current_state] += dt
    
    def calculate_statistics(self):
        """Расчет статистических характеристик"""
        # Вероятности состояний
        total_time = sum(self.state_times)
        probabilities = [t / total_time for t in self.state_times]
        
        # Среднее число заявок в очереди
        avg_queue_length = sum(self.queue_length_history) / len(self.queue_length_history) if self.queue_length_history else 0
        
        # Вероятность простоя системы
        p0 = probabilities[0]
        
        # Вероятность отказа
        p_refusal = self.refusals / self.total_arrivals if self.total_arrivals > 0 else 0
        
        # Среднее время ожидания
        avg_wait_time = self.total_wait_time / self.served if self.served > 0 else 0
        
        # Среднее время обслуживания
        avg_service_time = self.total_service_time / self.served if self.served > 0 else 0
        
        # Среднее время в системе
        avg_system_time = avg_wait_time + avg_service_time
        
        # Вероятность того, что прибывающему грузовику придется ждать
        p_wait = 1 - p0
        
        # Потери порта в день
        daily_losses = (self.refusals / self.T_max) * self.loss_per_refusal
        
        # Относительная пропускная способность
        q = 1 - p_refusal
        
        # Абсолютная пропускная способность
        A = self.lambda_rate * q
        
        return {
            'p0': p0,
            'avg_queue_length': avg_queue_length,
            'avg_wait_time': avg_wait_time,
            'avg_service_time': avg_service_time,
            'avg_system_time': avg_system_time,
            'p_wait': p_wait,
            'p_refusal': p_refusal,
            'daily_losses': daily_losses,
            'q': q,
            'A': A,
            'probabilities': probabilities,
            'total_arrivals': self.total_arrivals,
            'served': self.served,
            'refusals': self.refusals
        }
    
    def plot_results(self, stats):
        """Визуализация результатов"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # График 1: Распределение вероятностей состояний
        states = list(range(len(stats['probabilities'])))
        axes[0, 0].bar(states, stats['probabilities'])
        axes[0, 0].set_xlabel('Состояние системы (число заявок)')
        axes[0, 0].set_ylabel('Вероятность')
        axes[0, 0].set_title('Распределение вероятностей состояний системы')
        axes[0, 0].grid(True, alpha=0.3)
        
        # График 2: Длина очереди во времени
        if self.time_history and self.queue_length_history:
            axes[0, 1].plot(self.time_history[:1000], self.queue_length_history[:1000])
            axes[0, 1].set_xlabel('Время (дни)')
            axes[0, 1].set_ylabel('Длина очереди')
            axes[0, 1].set_title('Длина очереди во времени (первые 1000 событий)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # График 3: Основные характеристики
        labels = ['p0', 'Lq', 'Wq', 'W', 'Pотк', 'Потери']
        values = [
            stats['p0'],
            stats['avg_queue_length'],
            stats['avg_wait_time'],
            stats['avg_system_time'],
            stats['p_refusal'],
            stats['daily_losses'] / 1000  # в тысячах рублей
        ]
        
        bars = axes[1, 0].bar(labels, values)
        axes[1, 0].set_ylabel('Значение')
        axes[1, 0].set_title('Основные характеристики СМО')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Подписи значений на столбцах
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}',
                           ha='center', va='bottom')
        
        # График 4: Сравнение с аналитическим решением
        analytical = self.analytical_solution()
        categories = ['p0', 'Lq', 'Wq', 'Pотк']
        simulated = [stats['p0'], stats['avg_queue_length'], 
                    stats['avg_wait_time'], stats['p_refusal']]
        analytical_vals = [analytical['p0'], analytical['Lq'], 
                          analytical['Wq'], analytical['p_refusal']]
        
        x = np.arange(len(categories))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, simulated, width, label='Моделирование')
        axes[1, 1].bar(x + width/2, analytical_vals, width, label='Аналитика')
        axes[1, 1].set_xlabel('Характеристика')
        axes[1, 1].set_ylabel('Значение')
        axes[1, 1].set_title('Сравнение с аналитическим решением')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(categories)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analytical_solution(self):
        """Аналитическое решение для M/M/1/m системы"""
        n = 1  # один канал
        m = self.queue_capacity
        ρ = self.lambda_rate / self.mu_rate
        
        # Вероятность простоя системы
        if ρ != 1:
            p0 = (1 - ρ) / (1 - ρ**(m + 2))
        else:
            p0 = 1 / (m + 2)
        
        # Вероятность отказа
        p_refusal = ρ**(m + 1) * p0
        
        # Среднее число заявок в очереди
        if ρ != 1:
            Lq = (ρ**2 * (1 - ρ**m * (m + 1 - m * ρ))) / ((1 - ρ**(m + 2)) * (1 - ρ))
        else:
            Lq = m * (m + 1) / (2 * (m + 2))
        
        # Среднее время ожидания
        λ_eff = self.lambda_rate * (1 - p_refusal)
        Wq = Lq / λ_eff
        
        # Среднее время обслуживания
        t_ob = 1 / self.mu_rate
        
        # Среднее время в системе
        W = Wq + t_ob
        
        return {
            'p0': p0,
            'p_refusal': p_refusal,
            'Lq': Lq,
            'Wq': Wq,
            'W': W,
            'ρ': ρ
        }

# Параметры из задания
lambda_rate = 12  # 12 грузовиков в день
mu_rate = 10      # 10 грузовиков в день
queue_capacity = 12  # мест в очереди
T_max = 1000  # дней моделирования
loss_per_refusal = 50000  # руб. за отказ

# Создание и запуск модели
print("Моделирование одноканальной СМО методом Монте-Карло")
print(f"Параметры системы:")
print(f"  λ = {lambda_rate} груз./день")
print(f"  μ = {mu_rate} груз./день")
print(f"  m = {queue_capacity} мест в очереди")
print(f"  T = {T_max} дней моделирования")
print(f"  Потери за отказ: {loss_per_refusal} руб.")

# Запуск моделирования
system = QueueSystem(lambda_rate, mu_rate, queue_capacity, T_max, loss_per_refusal)
system.simulate()
stats = system.calculate_statistics()
analytical = system.analytical_solution()

# Вывод результатов
print("\nРезультаты моделирования:")
print(f"Общее число поступивших заявок: {stats['total_arrivals']}")
print(f"Обслужено заявок: {stats['served']}")
print(f"Отказов: {stats['refusals']}")
print(f"Вероятность простоя системы (p0): {stats['p0']:.4f}")
print(f"Среднее число грузовиков в очереди (r_ср): {stats['avg_queue_length']:.4f}")
print(f"Среднее время ожидания (t_ож): {stats['avg_wait_time']:.4f} дня")
print(f"Среднее время обслуживания: {stats['avg_service_time']:.4f} дня")
print(f"Среднее время в системе (t_сист): {stats['avg_system_time']:.4f} дня")
print(f"Вероятность ожидания (p_ожид): {stats['p_wait']:.4f}")
print(f"Вероятность отказа (p_отк): {stats['p_refusal']:.4f}")
print(f"Относительная пропускная способность (q): {stats['q']:.4f}")
print(f"Абсолютная пропускная способность (A): {stats['A']:.4f} груз./день")
print(f"Потери порта в день: {stats['daily_losses']:.2f} руб.")

print("\nАналитическое решение (M/M/1/m):")
print(f"Коэффициент загрузки ρ = λ/μ = {analytical['ρ']:.4f}")
print(f"Вероятность простоя системы (p0): {analytical['p0']:.4f}")
print(f"Вероятность отказа (p_отк): {analytical['p_refusal']:.4f}")
print(f"Среднее число в очереди (r_ср): {analytical['Lq']:.4f}")
print(f"Среднее время ожидания (t_ож): {analytical['Wq']:.4f} дня")
print(f"Среднее время в системе (t_сист): {analytical['W']:.4f} дня")

# Расчет погрешности
print("\nСравнение результатов:")
print(f"Погрешность p0: {abs(stats['p0'] - analytical['p0']):.6f}")
print(f"Погрешность Pотк: {abs(stats['p_refusal'] - analytical['p_refusal']):.6f}")
print(f"Погрешность Lq: {abs(stats['avg_queue_length'] - analytical['Lq']):.6f}")
print(f"Погрешность Wq: {abs(stats['avg_wait_time'] - analytical['Wq']):.6f}")

# Визуализация результатов
system.plot_results(stats)