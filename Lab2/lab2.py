import random
import itertools
import tkinter as tk
from tkinter import ttk


# Горизонтальный масштаб (пикселей на единицу времени)
DEFAULT_SCALE = 12

# шаг меток времени: 10 = 0,10,20,... 
DEFAULT_TIME_TICK = 10

# Цвета для разных деталей (циклично используется список)
COLORS = ["#ff9999", "#99ff99", "#9999ff", "#ffcc99", "#cc99ff", "#99ffff", "#ffb3d9"]

# Чтение данных
def load_jobs(filename):
    # Загружает данные из файла (формат: a;b;c;...).
    # Возвращает список словарей:
    #  {"id": номер_детали, "times": [t1, t2, ..., tN]}
    jobs = []
    with open(filename, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            times = [int(x) for x in line.split(";")]
            jobs.append({"id": i+1, "times": times})
    return jobs

# Расчёт расписания для N станков
def schedule(jobs, sequence):
    num_machines = len(jobs[0]["times"])
    machine_times = [0] * num_machines  # время освобождения каждого станка
    sched = []

    for jid in sequence:
        times = next(j["times"] for j in jobs if j["id"] == jid)
        start_finish = {}
        for m in range(num_machines):
            if m == 0:
                s = machine_times[m]
            else:
                s = max(machine_times[m], start_finish[m-1][1])
            f = s + times[m]
            start_finish[m] = (s, f)
            machine_times[m] = f
        sched.append({"id": jid, "ops": start_finish})

    makespan = machine_times[-1]  # время завершения на последнем станке
    return sched, makespan

# Полный перебор всех возможных вариантов
def brute_force(jobs):
    best_seq = None # оптимальная последовательность работ
    best_ms = float("inf") # лучшее (минимальное) время завершения всех работ, изначально бесконечность для корректного сравнения

    # itertools.permutations генерирует все возможные перестановки работ
    for perm in itertools.permutations([j["id"] for j in jobs]):
        # для каждой перестановки считаем расписание и общее время выполнения
        _, ms = schedule(jobs, perm)

        # если текущее время выполнения меньше найденного ранее — обновляем лучший результат
        if ms < best_ms:
            best_ms = ms # запоминаем минимальное время
            best_seq = perm # и соответствующую перестановку

    # Преобразуем лучшую последовательность в список
    return list(best_seq), best_ms

# Вычисляет для каждой детали параметры P1, P2 и lambda
def compute_petrov_params(jobs):
    m = len(jobs[0]["times"])
    params = []
    # чётные
    if m % 2 == 0:
        k = m // 2 # граница суммы
        for j in jobs:
            times = j["times"]
            P1 = sum(times[0:k]) # от 0 до k
            P2 = sum(times[k:m]) # от k до m
            lam = P2 - P1
            params.append({"id": j["id"], "P1": P1, "P2": P2, "lambda": lam, "times": times})
    # нечётные
    else:
        k = (m + 1) // 2 # граница суммы
        for j in jobs:
            times = j["times"]
            P1 = sum(times[0:k]) # от 0 до k
            P2 = sum(times[k-1:m]) # от k - 1 до m
            lam = P2 - P1
            params.append({"id": j["id"], "P1": P1, "P2": P2, "lambda": lam, "times": times})
    return params

# Сортировка по P1 (возрастание), при равенстве P1 — по lambda (убывание)
def sort_by_P1(lst):
    return sorted(lst, key=lambda x: (x["P1"], -x["lambda"], x["id"]))

# Сортировка по P2 (убывание), при равенстве P2 — по lambda (убывание)
def sort_by_P2(lst):
    return sorted(lst, key=lambda x: (-x["P2"], -x["lambda"], x["id"]))

# Реализация правила 2
def rule2(params):
    # группируем по lambda с учетом допусков
    used = [False] * len(params)
    groups = []
    for i, p in enumerate(params):
        if used[i]:
            continue
        same = [p]
        used[i] = True
        for j in range(i + 1, len(params)):
            if not used[j] and params[j]["lambda"] == p["lambda"]:
                same.append(params[j])
                used[j] = True
        groups.append((p["lambda"], same))

    # сортируем группы по lambda убыванию
    groups.sort(key=lambda x: -x[0])

    ordered = []
    for lam_val, group in groups:
        if lam_val >= -1e-9:
            # lambda >= 0: сортируем по P1 возрастанию; при равенстве P1 и P2 — случайный порядок
            group_sorted = sorted(group, key=lambda x: (x["P1"], x["id"]))
            i = 0
            while i < len(group_sorted):
                j = i + 1
                while j < len(group_sorted) and group_sorted[j]["P1"] == group_sorted[i]["P1"]:
                    j += 1
                bucket = group_sorted[i:j]
                # если в наборе у всех одинаковый P2 — порядок случайный
                if all(b["P2"] == bucket[0]["P2"] for b in bucket):
                    random.shuffle(bucket)
                ordered.extend(bucket)
                i = j
        else:
            # lambda < 0: сортируем по P2 убыванию; при равенстве P1 и P2 — случайный порядок
            group_sorted = sorted(group, key=lambda x: (-x["P2"], x["id"]))
            i = 0
            while i < len(group_sorted):
                j = i + 1
                while j < len(group_sorted) and group_sorted[j]["P2"] == group_sorted[i]["P2"]:
                    j += 1
                bucket = group_sorted[i:j]
                if all(b["P1"] == bucket[0]["P1"] for b in bucket):
                    random.shuffle(bucket)
                ordered.extend(bucket)
                i = j
    return ordered

# Возвращает 4 варианта последовательностей по правилам Петрова и параметры
def petrov_sequences(jobs):
    params = compute_petrov_params(jobs)

    # делим на множества D1, D0, D2
    D1 = [p for p in params if p["lambda"] > 0]
    D0 = [p for p in params if p["lambda"] == 0]
    D2 = [p for p in params if p["lambda"] < 0]

    # Правило 1: сначала D1 + D0 по возрастанию P1 (с учётом lambda в случае равенства P1),
    # затем D2 по убыванию P2 (опять с учётом lambda)
    D10 = sort_by_P1(D1 + D0)
    D2_sorted = sort_by_P2(D2)
    seq1 = [p["id"] for p in D10] + [p["id"] for p in D2_sorted]

    # Правило 2: весь набор по убыванию lambda; при равенстве lambda — применяем правило 3
    seq2_params = rule2(params)
    seq2 = [p["id"] for p in seq2_params]

    # Правило 3: D1 по возрастанию P1, затем D0 по возрастанию P1, затем D2 по убыванию P2,
    seq3 = [p["id"] for p in sort_by_P1(D1)] + \
           [p["id"] for p in sort_by_P1(D0)] + \
           [p["id"] for p in sort_by_P2(D2)]

    # Правило 4: попарное упорядочение в D1, затем D0, затем D2.
    # Попарное: первая в паре = max по P2 (при равенстве учитываем lambda),
    # вторая = min по P1 (при равенстве учитываем lambda), остатки объединяем
    def pairwise_order(lst):
        a = lst.copy()
        pairs = []
        while len(a) >= 2:
            # первая = max по P2, потом по lambda
            first = max(a, key=lambda x: (x["P2"], x["lambda"], -x["id"]))
            a.remove(first)
            # вторая = min по P1, при равенстве по lambda (правило 2 — по lambda убыванию)
            second = min(a, key=lambda x: (x["P1"], -x["lambda"], x["id"]))
            a.remove(second)
            pairs.append((first, second))
        leftover = a[0] if a else None
        return pairs, leftover

    seq4_list = []

    # D1 попарно
    pairs1, leftover1 = pairwise_order(D1)

    # если остался одиночный из D1 — пытаемся подобрать партнёра из D0 (min P1),
    # если D0 нет — из D2 (min P1)
    if leftover1:
        if D0:
            candidate = min(D0, key=lambda x: (x["P1"], -x["lambda"], x["id"]))
            D0 = [x for x in D0 if x["id"] != candidate["id"]]
            pairs1.append((leftover1, candidate))
            leftover1 = None
        elif D2:
            candidate = min(D2, key=lambda x: (x["P1"], -x["lambda"], x["id"]))
            D2 = [x for x in D2 if x["id"] != candidate["id"]]
            pairs1.append((leftover1, candidate))
            leftover1 = None

    for a,b in pairs1:
        seq4_list.append(a)
        seq4_list.append(b)

    # D0 попарно
    pairs0, leftover0 = pairwise_order(D0)
    for a,b in pairs0:
        seq4_list.append(a)
        seq4_list.append(b)
    if leftover0:
        if D2:
            candidate = min(D2, key=lambda x: (x["P1"], -x["lambda"], x["id"]))
            D2 = [x for x in D2 if x["id"] != candidate["id"]]
            seq4_list.append(leftover0)
            seq4_list.append(candidate)
        else:
            seq4_list.append(leftover0)

    # D2 попарно
    pairs2, leftover2 = pairwise_order(D2)
    for a,b in pairs2:
        seq4_list.append(a)
        seq4_list.append(b)
    if leftover2:
        seq4_list.append(leftover2)

    seq4 = [p["id"] for p in seq4_list]

    return {"Петров-1": seq1, "Петров-2": seq2, "Петров-3": seq3, "Петров-4": seq4, "params": params}

# Простая генерация случайной последовательности "мешок с бумажками"
# Имитируем мешок с бумажками 1..n — тасуем (перестановки) и достаём по одной
def random_sequence_from_bag(jobs):
    ids = [j["id"] for j in jobs]
    seq = ids[:] # делаем копию списка id

    random.shuffle(seq) # случайная перестановка shuffle — это и есть достать бумажки из мешка :)

    _, ms = schedule(jobs, tuple(seq))
    return seq, ms

# График Ганта (аналогично предыдущей работе)
def draw_gantt(canvas, sched, title, y_offset, scale=DEFAULT_SCALE, time_tick=DEFAULT_TIME_TICK):
    if sched is None:
        canvas.create_text(10, y_offset, text=title + " (нет данных)", anchor="w", font=("Arial", 12, "bold"), fill="red")
        return
    
    num_machines = len(sched[0]["ops"])
    canvas.create_text(10, y_offset, text=title, anchor="w", font=("Arial", 12, "bold"))
    for m in range(num_machines):
        y_m = y_offset + 40 + m * 50
        canvas.create_text(10, y_m, text=f"Станок {m+1}", anchor="w")
        for i, r in enumerate(sched):
            s, f = r["ops"][m]
            dur = f - s
            # если длительность = 0 — не рисуем блок
            if dur == 0:
                continue
            c = COLORS[i % len(COLORS)]
            # прямоугольник операции (обработки детали на стнке)
            canvas.create_rectangle(100 + s * scale, y_m - 10, 100 + f * scale, y_m + 10, fill=c, outline="black")
            # метка с id детали по центру блока
            canvas.create_text(100 + (s + f) / 2 * scale, y_m, text=str(r["id"]))
    # рисуем шкалу времени по всей ширине
    makespan = max(r["ops"][num_machines - 1][1] for r in sched)

    # Рисуем метки времени с шагом time_tick.
    # Если time_tick <= 0 — метки времени не рисуем вообще
    if time_tick and time_tick > 0:
        # выбираем ближайший кратный шаг, начинаем с 0
        # рисуем только t = 0, time_tick, 2*time_tick, ...
        t = 0
        while t <= makespan:
            x = 100 + t * scale
            canvas.create_text(x, y_offset + 30 + num_machines * 50, text=str(t), font=("Arial", 8))
            t += time_tick


# загружаем данные
jobs = load_jobs("jobs.csv")

# исходная последовательность по id
orig_seq = [j["id"] for j in jobs]
sched_orig, ms_orig = schedule(jobs, orig_seq)

# Петров — 4 варианта
petrov_res = petrov_sequences(jobs)
petrov_seqs = {k: v for k, v in petrov_res.items() if k.startswith("Петров")}
petrov_schedules = {}
petrov_makespans = {}
for name, seq in petrov_seqs.items():
    sched, ms = schedule(jobs, seq)
    petrov_schedules[name] = sched
    petrov_makespans[name] = ms

# Случайная последовательность
rand_seq, rand_ms = random_sequence_from_bag(jobs)

# Перебор всех вариантов
brute_seq, brute_ms = brute_force(jobs)
sched_brute, _ = schedule(jobs, brute_seq)

# Интерфейс
root = tk.Tk()
root.title(f"Метод Петрова + Генетика + Перебор (станков: {len(jobs[0]['times'])})")

# Верхняя панель
frame = ttk.Frame(root)
frame.pack(fill="x", padx=8, pady=6)
lbl = ttk.Label(frame, text="Исходные данные (по строкам — детали):", font=("Arial", 12, "bold"))
lbl.pack(anchor="w", pady=4)
cols = [f"m{i+1}" for i in range(len(jobs[0]["times"]))]
tree = ttk.Treeview(frame, columns=cols, show="headings", height=len(jobs))
for i, col in enumerate(cols):
    tree.heading(col, text=f"Станок {i+1}")
for j in jobs:
    tree.insert("", "end", values=j["times"])
tree.pack(fill="x", padx=4, pady=4)

canvas = tk.Canvas(root, width=1200, height=700, bg="white")
canvas.pack(fill="both", expand=True, side="left", padx=8, pady=8)

# Полосы прокрутки
vbar = tk.Scrollbar(root, orient=tk.VERTICAL, command=canvas.yview)
vbar.pack(side=tk.RIGHT, fill=tk.Y)
hbar = tk.Scrollbar(root, orient=tk.HORIZONTAL, command=canvas.xview)
hbar.pack(side=tk.BOTTOM, fill=tk.X)
canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)

# Теперь рисуем диаграммы по очереди, одна под другой
y = 20
draw_gantt(canvas, sched_orig, f"Исходная последовательность: {tuple(orig_seq)}  T = {ms_orig}", y)
y += 40 + len(jobs[0]['times']) * 50 + 40

for name, sched in petrov_schedules.items():
    draw_gantt(canvas, sched, f"{name}: {tuple(petrov_seqs[name])}  T = {petrov_makespans[name]}", y)
    y += 40 + len(jobs[0]['times']) * 50 + 40

sched_ga, _ = schedule(jobs, rand_seq)
draw_gantt(canvas, sched_ga, f"Случайная последовательность: {tuple(rand_seq)}  T = {rand_ms}", y)
y += 40 + len(jobs[0]['times']) * 50 + 40

draw_gantt(canvas, sched_brute, f"Перебор (лучшее): {tuple(brute_seq)}  T = {brute_ms}", y)

# Обновляем область прокрутки под всё нарисованное
root.update_idletasks()
canvas.config(scrollregion=canvas.bbox("all"))

root.mainloop()