import tkinter as tk
from tkinter import ttk

# Чтение данных
def load_jobs(filename):
    # Загружает данные из файла (разделитель ;)
    # Возвращает список словарей: {'id': номер_детали, 'a': время_станок_1, 'b': время_станок_2}
    jobs = []
    with open(filename, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            a, b = line.split(";")
            jobs.append({"id": i+1, "a": int(a), "b": int(b)})
    return jobs

# Алгоритм Джонсона для 2-х станков
def johnson_sequence(jobs):
    """
    Алгоритм Джонсона для двух станков:
    1. Выбираем минимальное время среди всех a и b
    2. Если минимальное время принадлежит a (станок 1),
       то ставим эту деталь в начало последовательности
    3. Если минимальное время принадлежит b (станок 2),
       то ставим эту деталь в конец последовательности
    4. Убираем выбранную деталь из списка и повторяем, пока не останутся все детали
    """

    remaining = jobs.copy()
    front, back = [], []
    while remaining:
        # минимальные времена для станка a и b
        min_a = min(r["a"] for r in remaining)
        min_b = min(r["b"] for r in remaining)

        if min_a <= min_b:
            # Если минимальное значение среди a:
            # деталь должна выполняться раньше -> добавляем в начало
            for r in remaining:
                if r["a"] == min_a:
                    front.append(r["id"])
                    remaining.remove(r)
                    break
        else:
            # Если минимальное значение среди b:
            # деталь должна выполняться позже -> добавляем в конец
            for r in remaining:
                if r["b"] == min_b:
                    back.insert(0, r["id"])
                    remaining.remove(r)
                    break
    return front + back

# Расчёт расписания
def schedule(jobs, sequence):
    # Превратим список jobs в словарь для быстрого доступа по id:
    # info[id] = (время на станке 1, время на станке 2)
    info = {j["id"]: (j["a"], j["b"]) for j in jobs}

    sched = [] # сюда складываем результат (по каждой детали)
    t1 = t2 = 0 # текущее время окончания работы на станке 1 и станке 2

    # Проходим детали в том порядке, который задан sequence
    for jid in sequence:
        a, b = info[jid]  # время на станке 1 и станке 2 для этой детали

        # Станок 1:
        # начинает сразу после предыдущей детали (t1),
        # заканчивает через a единиц времени
        s1 = t1
        f1 = s1 + a

        # Станок 2:
        # может начать только когда:
        #   1) закончилась эта деталь на станке 1 (f1)
        #   2) и станок 2 освободился после предыдущей детали (t2)
        # поэтому старт = max(t2, f1)
        s2 = max(t2, f1)
        f2 = s2 + b

        # Записываем результат для этой детали
        sched.append({
            "id": jid,
            "s1": s1, "f1": f1, # начало/конец на станке 1
            "s2": s2, "f2": f2  # начало/конец на станке 2
        })

        # Обновляем "текущее время" для станков
        t1 = f1 # станок 1 освободится в момент f1
        t2 = f2 # станок 2 освободится в момент f2

    # Время завершения всей работы = момент окончания последней детали на станке 2
    makespan = t2
    return sched, makespan

# Построение диаграммы Ганта
def draw_gantt(canvas, sched, title, y_offset, scale=20):
    # Заголовок
    canvas.create_text(10, y_offset, text=title, anchor="w", font=("Arial", 12, "bold"))

    # Позиции строк (станков)
    y_m1, y_m2 = y_offset+30, y_offset+80

    # Подписи станков
    canvas.create_text(10, y_m1, text="Станок 1", anchor="w")
    canvas.create_text(10, y_m2, text="Станок 2", anchor="w")

    # Цвета для задач
    colors = ["#ff9999", "#99ff99", "#9999ff", "#ffcc99", "#cc99ff", "#99ffff"]

    for i, r in enumerate(sched):
        c = colors[i % len(colors)]
        # блок на станке 1
        canvas.create_rectangle(100+r["s1"]*scale, y_m1-10,
                                100+r["f1"]*scale, y_m1+10,
                                fill=c, outline="black")
        canvas.create_text((100+(r["s1"]+r["f1"])*scale//2), y_m1, text=str(r["id"]))

        # блок на станке 2
        canvas.create_rectangle(100+r["s2"]*scale, y_m2-10,
                                100+r["f2"]*scale, y_m2+10,
                                fill=c, outline="black")
        canvas.create_text((100+(r["s2"]+r["f2"])*scale//2), y_m2, text=str(r["id"]))

    # Шкала времени (по оси X)
    makespan = max(r["f2"] for r in sched)
    for t in range(0, makespan+1):
        x = 100 + t*scale
        canvas.create_text(x, y_m2+30, text=str(t), font=("Arial",8))


# Загружаем задачи
jobs = load_jobs("Lab1\jobs.csv")

# Исходная и оптимальная последовательности
orig_seq = [j["id"] for j in jobs]
opt_seq = johnson_sequence(jobs)

# Расписания
sched_orig, ms_orig = schedule(jobs, orig_seq)
sched_opt, ms_opt = schedule(jobs, opt_seq)

# Создаём окно
root = tk.Tk()
root.title("Алгоритм Джонсона (2 станка)")

# Таблица с исходными данными
frame = ttk.Frame(root)
frame.pack(fill="x", padx=10, pady=5)
lbl = ttk.Label(frame, text="Исходные данные (время на станках):", font=("Arial", 12, "bold"))
lbl.pack(anchor="w", pady=5)

tree = ttk.Treeview(frame, columns=("a","b"), show="headings", height=len(jobs))
tree.heading("a", text="Станок 1 (a)")
tree.heading("b", text="Станок 2 (b)")
for j in jobs:
    tree.insert("", "end", values=(j["a"], j["b"]))
tree.pack(fill="x")

# Canvas для графиков
canvas = tk.Canvas(root, width=900, height=400, bg="white")
canvas.pack(fill="both", expand=True, padx=10, pady=10)

# Диаграммы Ганта
seq_str_orig = "(" + ",".join(map(str, orig_seq)) + ")"
seq_str_opt = "(" + ",".join(map(str, opt_seq)) + ")"
draw_gantt(canvas, sched_orig, "Исходная последовательность: " + seq_str_orig, y_offset=20)
draw_gantt(canvas, sched_opt, "Оптимальная последовательность: " + seq_str_opt, y_offset=180)

root.mainloop()