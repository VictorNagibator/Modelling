import tkinter as tk
from tkinter import ttk
import itertools

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

# Алгоритм Джонсона (для 3 станков)
def johnson_sequence(jobs):
    # Применим только для 3 станков, если выполняется одно из условий:
    #  a >= b для всех работ   или   c >= b для всех работ
    # Тогда можно свести к 2-м станкам:
    # d = a + b
    # e = b + c

    if len(jobs[0]["times"]) != 3:
        return None  # Алгоритм только для 3 станков

    # Всё остальное аналогично второму заданию
    min_a = min(j["a"] for j in jobs)
    max_b = max(j["b"] for j in jobs)
    min_c = min(j["c"] for j in jobs)
    cond1 = min_a >= max_b
    cond2 = min_c >= max_b

    new_jobs = []
    for j in jobs:
        a, b, c = j["times"]
        new_jobs.append({"id": j["id"], "d": a+b, "e": b+c})

    remaining = new_jobs.copy()
    front, back = [], []
    while remaining:
        min_d = min(r["d"] for r in remaining)
        min_e = min(r["e"] for r in remaining)
        if min_d <= min_e:
            for r in remaining:
                if r["d"] == min_d:
                    front.append(r["id"])
                    remaining.remove(r)
                    break
        else:
            for r in remaining:
                if r["e"] == min_e:
                    back.insert(0, r["id"])
                    remaining.remove(r)
                    break
    return front + back

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

# Полный перебор всех возможных вариантов (n!)
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
    return list(best_seq)

# Диаграмма Ганта
def draw_gantt(canvas, sched, title, y_offset, scale=20):
    num_machines = len(sched[0]["ops"])
    canvas.create_text(10, y_offset, text=title, anchor="w", font=("Arial", 12, "bold"))

    colors = ["#ff9999", "#99ff99", "#9999ff", "#ffcc99", "#cc99ff", "#99ffff"]

    for m in range(num_machines):
        y_m = y_offset + 40 + m*50
        canvas.create_text(10, y_m, text=f"Станок {m+1}", anchor="w")

        for i, r in enumerate(sched):
            s, f = r["ops"][m]
            c = colors[i % len(colors)]
            canvas.create_rectangle(100+s*scale, y_m-10,
                                    100+f*scale, y_m+10,
                                    fill=c, outline="black")
            canvas.create_text((100+(s+f)*scale//2), y_m, text=str(r["id"]))

    makespan = max(r["ops"][num_machines-1][1] for r in sched)
    for t in range(0, makespan+1):
        x = 100 + t*scale
        canvas.create_text(x, y_offset + 30 + num_machines*50, text=str(t), font=("Arial",8))



jobs = load_jobs("Lab1\jobs.csv")

orig_seq = [j["id"] for j in jobs]
sched_orig, ms_orig = schedule(jobs, orig_seq)

opt_seq = johnson_sequence(jobs)
sched_opt, ms_opt = (schedule(jobs, opt_seq) if opt_seq else (None, None))

best_seq = brute_force(jobs)
sched_best, ms_best = schedule(jobs, best_seq)

root = tk.Tk()
root.title(f"Перебор + Джонсон (станков: {len(jobs[0]['times'])})")

frame = ttk.Frame(root)
frame.pack(fill="x", padx=10, pady=5)
lbl = ttk.Label(frame, text="Исходные данные (время на станках):", font=("Arial", 12, "bold"))
lbl.pack(anchor="w", pady=5)

# таблица с исходными данными
cols = [f"m{i+1}" for i in range(len(jobs[0]["times"]))]
tree = ttk.Treeview(frame, columns=cols, show="headings", height=len(jobs))
for i, col in enumerate(cols):
    tree.heading(col, text=f"Станок {i+1}")
for j in jobs:
    tree.insert("", "end", values=j["times"])
tree.pack(fill="x")

canvas = tk.Canvas(root, width=1200, height=800, bg="white")
canvas.pack(fill="both", expand=True, padx=10, pady=10)

y = 20

seq_str_orig = "(" + ",".join(map(str, orig_seq)) + ")"
draw_gantt(canvas, sched_orig, "Исходная последовательность: " + seq_str_orig, y_offset=y)

y += 40 + len(jobs[0]['times'])*50 + 40

if sched_opt:
    seq_str_opt = "(" + ",".join(map(str, opt_seq)) + ")"
    draw_gantt(canvas, sched_opt, "Алгоритм Джонсона: " + seq_str_opt, y_offset=y)
    y += 40 + len(jobs[0]['times'])*50 + 40
else:
    canvas.create_text(10, y, text="Алгоритм Джонсона неприменим для этих данных",
                       anchor="w", font=("Arial", 12, "bold"), fill="red")
    y += 75 # произвольно отступим

seq_str_best = "(" + ",".join(map(str, best_seq)) + ")"
draw_gantt(canvas, sched_best, "Перебор: " + seq_str_best, y_offset=y)

root.mainloop()