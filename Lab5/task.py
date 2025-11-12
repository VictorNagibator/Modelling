import numpy as np
import copy
import heapq
from typing import List, Tuple, Dict, Any
import os
import csv

class TSPSolver:
    def __init__(self, cost_matrix):
        self.n = len(cost_matrix)
        self.original_matrix = np.array(cost_matrix, dtype=float)
        # Заменяем диагональные элементы на бесконечность
        for i in range(self.n):
            self.original_matrix[i][i] = float('inf')
        self.log_file = open('Lab5/tsp_solution_log.txt', 'w', encoding='utf-8')
        self.step_counter = 0
        self.tree_nodes = {}
        self.current_node_id = 0
        
    def log_step(self, matrix, reduced_constants, theta, edge, penalty, node_info=""):
        self.step_counter += 1
        self.log_file.write(f"Шаг {self.step_counter}\n")
        
        if node_info:
            self.log_file.write(f"{node_info}\n")
            
        self.log_file.write("Текущая матрица стоимостей:\n")
        for row in matrix:
            self.log_file.write("[" + " ".join(f"{x:6.1f}" if x != float('inf') else "   inf " for x in row) + "]\n")
            
        self.log_file.write(f"Сумма приводящих констант: {reduced_constants}\n")
        self.log_file.write(f"Оценка затрат: {theta}\n")
        
        if edge:
            self.log_file.write(f"Ребро маршрута: ({edge[0]+1},{edge[1]+1})\n")
            
        if penalty is not None:
            self.log_file.write(f"Штраф за неиспользование: {penalty}\n")

    def reduce_matrix(self, matrix):
        reduced_matrix = copy.deepcopy(matrix)
        reduction_cost = 0
        n = len(reduced_matrix)
        
        # Приводим по строкам
        for i in range(n):
            min_val = min(reduced_matrix[i])
            if min_val != float('inf') and min_val > 0:
                reduction_cost += min_val
                for j in range(n):
                    if reduced_matrix[i][j] != float('inf'):
                        reduced_matrix[i][j] -= min_val
        
        # Приводим по столбцам
        for j in range(n):
            col = [reduced_matrix[i][j] for i in range(n)]
            min_val = min(col)
            if min_val != float('inf') and min_val > 0:
                reduction_cost += min_val
                for i in range(n):
                    if reduced_matrix[i][j] != float('inf'):
                        reduced_matrix[i][j] -= min_val
        
        return reduced_matrix, reduction_cost

    def calculate_penalties(self, matrix):
        n = len(matrix)
        penalties = np.full((n, n), -1.0)
        
        for i in range(n):
            for j in range(n):
                if matrix[i][j] == 0:
                    # Находим минимальный элемент в строке (исключая текущий нуль)
                    row_vals = [matrix[i][k] for k in range(n) if k != j and matrix[i][k] != float('inf')]
                    min_row = min(row_vals) if row_vals else 0
                    
                    # Находим минимальный элемент в столбце (исключая текущий нуль)
                    col_vals = [matrix[k][j] for k in range(n) if k != i and matrix[k][j] != float('inf')]
                    min_col = min(col_vals) if col_vals else 0
                    
                    penalties[i][j] = min_row + min_col
        
        return penalties

    def find_max_penalty_edge(self, penalties):
        max_penalty = -1
        best_edge = None
        n = len(penalties)
        
        for i in range(n):
            for j in range(n):
                if penalties[i][j] > max_penalty:
                    max_penalty = penalties[i][j]
                    best_edge = (i, j)
        
        return best_edge, max_penalty

    def include_edge(self, matrix, edge, current_path):
        i, j = edge
        n = len(matrix)
        new_matrix = copy.deepcopy(matrix)
        
        # Запрещаем другие переходы из i и в j
        for k in range(n):
            new_matrix[i][k] = float('inf')
            new_matrix[k][j] = float('inf')
        
        # Запрещаем образование подциклов
        new_matrix[j][i] = float('inf')
        
        # Если у нас уже есть путь, запрещаем ребра, которые могут создать подциклы
        if current_path:
            # Создаем граф из текущего пути
            graph = {}
            for from_node, to_node in current_path:
                graph[from_node] = to_node
            
            # Находим все связанные компоненты
            visited = set()
            components = []
            
            for node in range(n):
                if node not in visited:
                    component = []
                    stack = [node]
                    while stack:
                        current = stack.pop()
                        if current not in visited:
                            visited.add(current)
                            component.append(current)
                            if current in graph:
                                stack.append(graph[current])
                    components.append(component)
            
            # Если добавление ребра (i,j) создает цикл, который не включает все города,
            # запрещаем ребра, которые могут создать подциклы
            if len(current_path) < n - 1:
                # Находим компоненту, содержащую i и j
                i_component = None
                j_component = None
                for comp in components:
                    if i in comp:
                        i_component = comp
                    if j in comp:
                        j_component = comp
                
                # Если i и j в разных компонентах, объединяем их
                if i_component and j_component and i_component != j_component:
                    # Объединяем компоненты
                    merged_component = i_component + j_component
                    # Если объединенная компонента содержит не все города,
                    # запрещаем ребра из этой компоненты в саму себя
                    if len(merged_component) < n:
                        for node_from in merged_component:
                            for node_to in merged_component:
                                if node_from != node_to:
                                    new_matrix[node_from][node_to] = float('inf')
        
        return new_matrix

    def solve(self):
        # Начальное приведение матрицы
        current_matrix, r0 = self.reduce_matrix(self.original_matrix)
        theta_min = r0
        
        self.log_step(current_matrix, r0, theta_min, None, None, "Начальное приведение матрицы")
        
        # Очередь с приоритетом для узлов дерева
        priority_queue = []
        heapq.heappush(priority_queue, (theta_min, self.current_node_id, current_matrix, [], r0))
        self.tree_nodes[self.current_node_id] = {
            'theta': theta_min,
            'matrix': current_matrix,
            'path': [],
            'parent': None,
            'edge': None,
            'included': None
        }
        
        best_solution = None
        best_cost = float('inf')
        max_iterations = 1000  # Ограничение на количество итераций
        iteration = 0
        
        while priority_queue and iteration < max_iterations:
            iteration += 1
            theta, node_id, matrix, path, reduction_cost = heapq.heappop(priority_queue)
            
            if theta >= best_cost:
                continue
                
            # Если путь содержит n-1 ребер, находим последнее ребро
            if len(path) == self.n - 1:
                # Находим последнее ребро
                used_from = set(edge[0] for edge in path)
                used_to = set(edge[1] for edge in path)
                
                from_node = list(set(range(self.n)) - used_from)[0]
                to_node = list(set(range(self.n)) - used_to)[0]
                
                if matrix[from_node][to_node] != float('inf'):
                    complete_path = path + [(from_node, to_node)]
                    complete_cost = self.calculate_path_cost(complete_path)
                    if complete_cost < best_cost:
                        best_solution = complete_path
                        best_cost = complete_cost
                continue
            
            # Если матрица слишком мала, пропускаем
            if len(matrix) < 2:
                continue
            
            # Вычисляем штрафы
            penalties = self.calculate_penalties(matrix)
            edge, max_penalty = self.find_max_penalty_edge(penalties)
            
            if edge is None:
                continue
                
            self.log_step(matrix, reduction_cost, theta, edge, max_penalty, 
                         f"Узел {node_id}, текущий путь: {[(a+1, b+1) for a, b in path]}")
            
            # Ветвление: включаем ребро
            matrix_with_edge = self.include_edge(matrix, edge, path)
            
            # Проверяем, не стала ли матрица пустой
            if len([x for row in matrix_with_edge for x in row if x != float('inf')]) > 0:
                reduced_matrix_with_edge, reduction_with_edge = self.reduce_matrix(matrix_with_edge)
                theta_with_edge = theta + reduction_with_edge
                
                new_path_with_edge = path + [edge]
                self.current_node_id += 1
                heapq.heappush(priority_queue, (theta_with_edge, self.current_node_id, 
                                              reduced_matrix_with_edge, new_path_with_edge, reduction_with_edge))
                
                self.tree_nodes[self.current_node_id] = {
                    'theta': theta_with_edge,
                    'matrix': reduced_matrix_with_edge,
                    'path': new_path_with_edge,
                    'parent': node_id,
                    'edge': edge,
                    'included': True
                }
            
            # Ветвление: исключаем ребро
            matrix_without_edge = copy.deepcopy(matrix)
            matrix_without_edge[edge[0]][edge[1]] = float('inf')
            reduced_matrix_without_edge, reduction_without_edge = self.reduce_matrix(matrix_without_edge)
            theta_without_edge = theta + max_penalty
            
            self.current_node_id += 1
            heapq.heappush(priority_queue, (theta_without_edge, self.current_node_id,
                                          reduced_matrix_without_edge, path, reduction_without_edge))
            
            self.tree_nodes[self.current_node_id] = {
                'theta': theta_without_edge,
                'matrix': reduced_matrix_without_edge,
                'path': path,
                'parent': node_id,
                'edge': edge,
                'included': False
            }
        
        # Записываем оптимальное решение
        self.log_file.write("Оптимальное решение\n")
        
        if best_solution:
            cycle = self.get_cycle_from_edges(best_solution)
            path_str = "->".join(str(x+1) for x in cycle)
            self.log_file.write(f"Оптимальная последовательность: {path_str}\n")
            self.log_file.write(f"Минимальная стоимость: {best_cost}\n")
            
            # Сохраняем дерево
            self.save_tree_structure()
        else:
            self.log_file.write("Решение не найдено\n")
            # Пытаемся найти любое допустимое решение
            if priority_queue:
                # Берем решение с наименьшей оценкой
                theta, node_id, matrix, path, reduction_cost = priority_queue[0]
                if len(path) == self.n - 1:
                    used_from = set(edge[0] for edge in path)
                    used_to = set(edge[1] for edge in path)
                    
                    from_node = list(set(range(self.n)) - used_from)[0]
                    to_node = list(set(range(self.n)) - used_to)[0]
                    
                    if self.original_matrix[from_node][to_node] != float('inf'):
                        complete_path = path + [(from_node, to_node)]
                        complete_cost = self.calculate_path_cost(complete_path)
                        cycle = self.get_cycle_from_edges(complete_path)
                        path_str = "->".join(str(x+1) for x in cycle)
                        self.log_file.write(f"Найдено допустимое решение: {path_str}\n")
                        self.log_file.write(f"Стоимость: {complete_cost}\n")
        
        self.log_file.close()
        return best_solution, best_cost

    def calculate_path_cost(self, edges):
        total_cost = 0
        for i, j in edges:
            total_cost += self.original_matrix[i][j]
        return total_cost

    def get_cycle_from_edges(self, edges):
        # Преобразуем ребра в цикл
        graph = {}
        for i, j in edges:
            graph[i] = j
        
        # Начинаем с первой вершины
        start = 0
        cycle = [start]
        current = start
        visited = set([start])
        
        while len(cycle) < self.n:
            next_node = graph[current]
            if next_node in visited and next_node != start:
                # Нашли подцикл, начинаем с другой вершины
                for i in range(self.n):
                    if i not in visited:
                        start = i
                        cycle = [start]
                        current = start
                        visited = set([start])
                        break
                continue
                
            cycle.append(next_node)
            visited.add(next_node)
            current = next_node
        
        return cycle

    def save_tree_structure(self):
        tree_file = open('Lab5/tsp_tree.txt', 'w', encoding='utf-8')
        tree_file.write("Дерево ветвления\n")
        
        def print_node(node_id, level=0):
            node = self.tree_nodes[node_id]
            indent = "  " * level
            
            if node['edge']:
                i, j = node['edge']
                edge_str = f"{i+1}->{j+1}"
            else:
                edge_str = "корень"
                
            include_str = "вкл" if node['included'] else "искл" if node['included'] is not None else ""
            
            tree_file.write(f"{indent}+ Узел {node_id}: θ={node['theta']:.1f} {edge_str} {include_str}\n")
            
            # Находим дочерние узлы
            children = [nid for nid, n in self.tree_nodes.items() if n['parent'] == node_id]
            for child_id in children:
                print_node(child_id, level + 1)
        
        # Находим корневой узел
        root_nodes = [nid for nid, node in self.tree_nodes.items() if node['parent'] is None]
        if root_nodes:
            print_node(root_nodes[0])
        
        tree_file.close()

# Читает матрицу стоимостей из CSV файла ('inf' для бесконечности )
def read_matrix_from_csv(filename):
    matrix = []
    with open(filename, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            matrix_row = []
            for cell in row:
                cell = cell.strip()
                if cell.lower() == 'inf' or cell == '':
                    matrix_row.append(float('inf'))
                else:
                    try:
                        matrix_row.append(float(cell))
                    except ValueError:
                        matrix_row.append(float('inf'))
            matrix.append(matrix_row)
    return matrix


def main():
    # Проверяем существование файла input.csv
    if not os.path.exists('Lab5/input.csv'):
        print("Файл input.csv не найден!")
        return
    
    # Читаем матрицу из CSV файла
    cost_matrix = read_matrix_from_csv('Lab5/input.csv')
    
    # Проверяем, что матрица квадратная
    n = len(cost_matrix)
    for row in cost_matrix:
        if len(row) != n:
            print("Ошибка: матрица должна быть квадратной!")
            return
    
    # Создаем и запускаем решатель
    solver = TSPSolver(cost_matrix)
    solution, cost = solver.solve()
    
    print("Решение завершено!")
    
    if solution:
        cycle = solver.get_cycle_from_edges(solution)
        path_str = "->".join(str(x+1) for x in cycle)
        print(f"\nОптимальная последовательность: {path_str}")
        print(f"Минимальная стоимость: {cost}")
    else:
        print("\nОптимальное решение не найдено, но проверьте файл лога для деталей")


if __name__ == "__main__":
    main()