import numpy as np
from typing import List, Tuple
import csv

class TransportProblem:
    def __init__(self, supplies, demands, costs):
        self.supplies = list(supplies)
        self.demands = list(demands)
        self.costs = [list(row) for row in costs]
        self.n = len(self.supplies)
        self.m = len(self.demands)
        
        # Проверка сбалансированности
        total_supply = sum(self.supplies)
        total_demand = sum(self.demands)
        
        if total_supply != total_demand:
            print(f"Задача несбалансированна! Сумма поставок: {total_supply}, Сумма потребностей: {total_demand}")
            if total_supply > total_demand:
                self.demands.append(total_supply - total_demand)
                self.m += 1
                for i in range(self.n):
                    self.costs[i].append(0)
            else:
                self.supplies.append(total_demand - total_supply)
                self.n += 1
                self.costs.append([0] * self.m)
        
        self.x = np.zeros((self.n, self.m), dtype=float)
    
    # Метод северо-западного угла
    def north_west_corner(self):
        x = np.zeros((self.n, self.m), dtype=float)
        basis = np.zeros((self.n, self.m), dtype=bool)
        
        a = self.supplies.copy()
        b = self.demands.copy()
        i, j = 0, 0
        
        while i < self.n and j < self.m:
            if abs(a[i]) < 1e-12:
                i += 1
                continue
            if abs(b[j]) < 1e-12:
                j += 1
                continue
                
            amount = min(a[i], b[j])
            x[i, j] = amount
            basis[i, j] = True
            
            a[i] -= amount
            b[j] -= amount
            
            if abs(a[i]) < 1e-12 and abs(b[j]) < 1e-12:
                if i < self.n - 1:
                    basis[i + 1, j] = True
                    x[i + 1, j] = 0
                elif j < self.m - 1:
                    basis[i, j + 1] = True
                    x[i, j + 1] = 0
                i += 1
                j += 1
            elif abs(a[i]) < 1e-12:
                i += 1
            elif abs(b[j]) < 1e-12:
                j += 1
        
        # Гарантируем, что базис содержит ровно n + m - 1 клеток
        self._ensure_proper_basis(x, basis)
        return x, basis
    
    # Гарантирует, что базис содержит ровно n + m - 1 клеток
    def _ensure_proper_basis(self, x, basis):
        needed = self.n + self.m - 1
        current = np.sum(basis)
        
        if current < needed:
            # Добавляем искусственные нулевые базисные клетки
            for i in range(self.n):
                for j in range(self.m):
                    if not basis[i, j] and abs(x[i, j]) < 1e-12:
                        basis[i, j] = True
                        current += 1
                        if current == needed:
                            return
        
        elif current > needed:
            # Удаляем лишние нулевые базисные клетки
            zero_cells = []
            for i in range(self.n):
                for j in range(self.m):
                    if basis[i, j] and abs(x[i, j]) < 1e-12:
                        zero_cells.append((i, j))
            
            for cell in zero_cells:
                if current > needed:
                    basis[cell] = False
                    current -= 1
                else:
                    break
    
    # Вычисление потенциалов
    def calculate_potentials(self, basis):
        u = [None] * self.n
        v = [None] * self.m
        
        u[0] = 0  # произвольное значение
        
        changed = True
        while changed:
            changed = False
            for i in range(self.n):
                for j in range(self.m):
                    if basis[i, j]:
                        if u[i] is not None and v[j] is None:
                            v[j] = self.costs[i][j] - u[i]
                            changed = True
                        elif v[j] is not None and u[i] is None:
                            u[i] = self.costs[i][j] - v[j]
                            changed = True
        return u, v
    
    # Вычисление оценок для свободных клеток
    def calculate_deltas(self, u, v, basis):
        deltas = np.zeros((self.n, self.m))
        for i in range(self.n):
            for j in range(self.m):
                if not basis[i, j]:
                    if u[i] is not None and v[j] is not None:
                        deltas[i, j] = self.costs[i][j] - (u[i] + v[j])
                    else:
                        deltas[i, j] = np.inf
        return deltas
    
    # Нахождение цикла пересчета
    def find_cycle(self, start_i, start_j, basis):
        def get_possible_moves(bool_table, path):
            # Получим возможные ходы с чередованием направления
            if len(path) < 2:
                # Первый ход: можно двигаться в любом направлении
                current_i, current_j = path[-1]
                moves = []
                # Клетки в той же строке
                for j in range(self.m):
                    if j != current_j and bool_table[current_i, j]:
                        moves.append((current_i, j))
                # Клетки в том же столбце
                for i in range(self.n):
                    if i != current_i and bool_table[i, current_j]:
                        moves.append((i, current_j))
                return moves
            
            # Определяем направление предыдущего хода
            prev_i, prev_j = path[-2]
            current_i, current_j = path[-1]
            
            if prev_i == current_i:  # предыдущий ход был горизонтальным
                # Текущий ход должен быть вертикальным
                return [(i, current_j) for i in range(self.n) 
                        if i != current_i and bool_table[i, current_j]]
            else:  # предыдущий ход был вертикальным
                # Текущий ход должен быть горизонтальным
                return [(current_i, j) for j in range(self.m) 
                        if j != current_j and bool_table[current_i, j]]
        
        # Временный базис с добавленной стартовой клеткой
        temp_basis = basis.copy()
        temp_basis[start_i, start_j] = True
        
        # Поиск в глубину
        stack = [[(start_i, start_j)]]
        
        while stack:
            path = stack.pop()
            current_i, current_j = path[-1]
            
            # Если вернулись в начало и путь достаточно длинный
            if len(path) > 3 and path[0] == (current_i, current_j):
                return path
            
            possible_moves = get_possible_moves(temp_basis, path)
            
            for move in possible_moves:
                if len(path) == 1 or move != path[-2]:  # избегаем возврата назад
                    new_path = path + [move]
                    stack.append(new_path)
        
        return []
    
    # Улучшение решения методом потенциалов
    def improve_solution(self, x, basis):
        iteration = 0
        max_iterations = 50
        
        while iteration < max_iterations:
            iteration += 1
            print(f"Итерация {iteration}")
            
            # Вычисляем потенциалы
            u, v = self.calculate_potentials(basis)
            print(f"Потенциалы u: {[f'{val:.1f}' if val is not None else 'None' for val in u]}")
            print(f"Потенциалы v: {[f'{val:.1f}' if val is not None else 'None' for val in v]}")
            
            # Вычисляем оценки
            deltas = self.calculate_deltas(u, v, basis)
            print("Матрица поставок:")
            for row in deltas:
                print([f'{val:6.1f}' if not np.isinf(val) else '   inf' for val in row])
            
            # Проверяем оптимальность
            min_delta = np.min(deltas)
            if min_delta >= -1e-10:
                print("Решение оптимально!")
                return x, basis, True
            
            # Находим клетку для улучшения
            improving_cells = []
            for i in range(self.n):
                for j in range(self.m):
                    if deltas[i, j] < -1e-10:
                        improving_cells.append((i, j, deltas[i, j]))
            
            if not improving_cells:
                print("Нет клеток для улучшения!")
                return x, basis, True
            
            # Выбираем клетку с наименьшей оценкой
            improving_cells.sort(key=lambda cell: cell[2])
            i0, j0, min_delta_val = improving_cells[0]
            print(f"Выбрана клетка ({i0},{j0}) с оценкой delta = {min_delta_val:.3f}")
            
            # Ищем цикл
            cycle = self.find_cycle(i0, j0, basis)
            if not cycle:
                print("Цикл не найден! Пропускаем эту клетку")
                # Помечаем эту клетку как неподходящую
                deltas[i0, j0] = np.inf
                continue
            
            print(f"Найден цикл: {cycle}")
            
            # Определяем θ (минимальное значение в "минусовых" клетках)
            theta = float('inf')
            minus_cells = []
            for idx in range(1, len(cycle), 2):  # клетки с нечетными индексами
                i, j = cycle[idx]
                if x[i, j] < theta:
                    theta = x[i, j]
                minus_cells.append((i, j))
            
            if theta < 1e-10:
                print("delta = 0, невозможно улучшить")
                continue
            
            print(f"delta = {theta}")
            
            # Перераспределяем поставки
            for idx, (i, j) in enumerate(cycle[:-1]):  # исключаем повтор старта в конце
                if idx % 2 == 0:  # четные позиции - плюс
                    x[i, j] += theta
                else:  # нечетные позиции - минус
                    x[i, j] -= theta
            
            # Обновляем базис
            basis[i0, j0] = True  # добавляем новую клетку
            
            # Удаляем клетку, которая обнулилась
            removed = False
            for i, j in minus_cells:
                if abs(x[i, j]) < 1e-10:
                    basis[i, j] = False
                    print(f"Удалена клетка ({i},{j})")
                    removed = True
                    break
            
            if not removed:
                # Если не нашли нулевую клетку для удаления, удаляем первую из минусовых
                if minus_cells:
                    i, j = minus_cells[0]
                    basis[i, j] = False
                    print(f"Удалена клетка ({i},{j})")
            
            # Гарантируем правильный размер базиса
            self._ensure_proper_basis(x, basis)
            
            print("Новый план перевозок:")
            for row in x:
                print([f'{val:6.1f}' for val in row])
            
            cost = np.sum(x * np.array(self.costs))
            print(f"Текущая стоимость: {cost:.1f}\n")
        
        print("Достигнуто максимальное количество итераций!")
        return x, basis, False
    
    # Решение транспортной задачи
    def solve(self):
        print("\n\tРешение транспортной задачи")
        print(f"Запасы: {self.supplies}")
        print(f"Потребности: {self.demands}")
        print("Матрица стоимостей:")
        for row in self.costs:
            print([f'{val:4.1f}' for val in row])
        
        # Построение начального плана
        print("\n\tПостроение начального плана методом северо-западного угла")
        x, basis = self.north_west_corner()
        
        print("Начальный план перевозок:")
        for row in x:
            print([f'{val:6.1f}' for val in row])
        
        initial_cost = np.sum(x * np.array(self.costs))
        print(f"Начальная стоимость: {initial_cost:.1f}")
        
        # Улучшение решения
        print("\n\tИспользование метода потенциалов")
        x, basis, optimal = self.improve_solution(x, basis)
        
        # Результаты
        print("\n\tОптимальный план перевозок:")
        for i, row in enumerate(x):
            print(f"Поставщик {i+1}: {[f'{val:6.1f}' for val in row]}")
        
        final_cost = np.sum(x * np.array(self.costs))
        print(f"Минимальная стоимость перевозок: {final_cost:.1f}")
        
        return x, final_cost

# Чтение входных данных из CSV файла
def read_input_from_csv(filename: str):
    supplies = []
    demands = []
    costs = []
    
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        rows = []
        for row in reader:
            # Пропускаем пустые строки
            if row and any(cell.strip() for cell in row):
                # Фильтруем пустые ячейки
                filtered_row = [cell for cell in row if cell.strip()]
                rows.append([float(x) for x in filtered_row])
        
        # Последняя строка - потребности
        if rows:
            demands = rows[-1]
            # Все остальные строки - это поставщики (последний элемент - запас, остальные - стоимости)
            for row in rows[:-1]:
                if len(row) > 1:
                    supplies.append(row[-1]) # последний элемент - запас
                    costs.append(row[:-1]) # все кроме последнего - стоимости
        
    return supplies, demands, costs


# Чтение данных из файла
try:
    supplies, demands, costs = read_input_from_csv('input.csv')
        
    print(f"Запасы: {supplies}")
    print(f"Потребности: {demands}")
    print("Матрица стоимостей:")
    for row in costs:
        print(row)
        
    # Создание и решение задачи
    problem = TransportProblem(supplies, demands, costs)
    solution, total_cost = problem.solve()
except FileNotFoundError:
    print("Файл input.csv не найден!")