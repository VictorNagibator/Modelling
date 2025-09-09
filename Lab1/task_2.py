import tkinter as tk
from tkinter import ttk

# Чтение данных
def load_jobs(filename):
    # Загружает данные из файла (формат: a;b;c).
    # Возвращает список словарей: {'id': номер, 'a': станок1, 'b': станок2, 'c': станок3}
    jobs = []
    with open(filename, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            a, b, c = line.split(";")
            jobs.append({"id": i+1, "a": int(a), "b": int(b), "c": int(c)})
    return jobs

# Алгоритм Джонсона для 3-х станков
def johnson_sequence(jobs):
    # Проверяем возможность применения алгоритма Джонсона для 3-х станков
    # Если min(a) >= max(b) или min(c) >= max(b) для всех работ, то можно преобразовать к 2-м станкам:
    #    d = a + b
    #    e = b + c
    # После этого применяем обычный алгоритм Джонсона для 2-х станков

    # Проверка условия
    min_a = min(j["a"] for j in jobs)
    max_b = max(j["b"] for j in jobs)
    min_c = min(j["c"] for j in jobs)
    cond1 = min_a >= max_b
    cond2 = min_c >= max_b
    if not (cond1 or cond2):
        return None  # Алгоритм Джонсона неприменим

    # Формируем d и e
    new_jobs = []
    for j in jobs:
        new_jobs.append({"id": j["id"], "d": j["a"]+j["b"], "e": j["b"]+j["c"]})

    # Обычный алгоритм Джонсона
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

# Расчёт расписания
def schedule(jobs, sequence):
    # Строим расписание на 3-х станках
    # s1,f1 – станок 1
    # s2,f2 – станок 2
    # s3,f3 – станок 3
    
    info = {j["id"]: (j["a"], j["b"], j["c"]) for j in jobs}
    sched = []
    t1 = t2 = t3 = 0
    for jid in sequence:
        a, b, c = info[jid]

        # Станок 1
        s1, f1 = t1, t1 + a
        # Станок 2
        s2, f2 = max(t2, f1), max(t2, f1) + b
        # Станок 3
        s3, f3 = max(t3, f2), max(t3, f2) + c

        sched.append({"id": jid, "s1": s1, "f1": f1,
                                "s2": s2, "f2": f2,
                                "s3": s3, "f3": f3})
        t1, t2, t3 = f1, f2, f3

    makespan = t3
    return sched, makespan

# Диаграмма Ганта
def draw_gantt(canvas, sched, title, y_offset, scale=20):
    canvas.create_text(10, y_offset, text=title, anchor="w", font=("Arial", 12, "bold"))

    y_m1, y_m2, y_m3 = y_offset+30, y_offset+80, y_offset+130
    canvas.create_text(10, y_m1, text="Станок 1", anchor="w")
    canvas.create_text(10, y_m2, text="Станок 2", anchor="w")
    canvas.create_text(10, y_m3, text="Станок 3", anchor="w")

    colors = ["#ff9999", "#99ff99", "#9999ff", "#ffcc99", "#cc99ff", "#99ffff"]

    for i, r in enumerate(sched):
        c = colors[i % len(colors)]
        # M1
        canvas.create_rectangle(100+r["s1"]*scale, y_m1-10,
                                100+r["f1"]*scale, y_m1+10,
                                fill=c, outline="black")
        canvas.create_text((100+(r["s1"]+r["f1"])*scale//2), y_m1, text=str(r["id"]))
        # M2
        canvas.create_rectangle(100+r["s2"]*scale, y_m2-10,
                                100+r["f2"]*scale, y_m2+10,
                                fill=c, outline="black")
        canvas.create_text((100+(r["s2"]+r["f2"])*scale//2), y_m2, text=str(r["id"]))
        # M3
        canvas.create_rectangle(100+r["s3"]*scale, y_m3-10,
                                100+r["f3"]*scale, y_m3+10,
                                fill=c, outline="black")
        canvas.create_text((100+(r["s3"]+r["f3"])*scale//2), y_m3, text=str(r["id"]))

    makespan = max(r["f3"] for r in sched)
    for t in range(0, makespan+1):
        x = 100 + t*scale
        canvas.create_text(x, y_m3+30, text=str(t), font=("Arial",8))


jobs = load_jobs("Lab1\jobs.csv")

orig_seq = [j["id"] for j in jobs]
opt_seq = johnson_sequence(jobs)

sched_orig, ms_orig = schedule(jobs, orig_seq)
sched_opt = None
ms_opt = None
if opt_seq:
    sched_opt, ms_opt = schedule(jobs, opt_seq)

root = tk.Tk()
root.title("Алгоритм Джонсона (3 станка)")

frame = ttk.Frame(root)
frame.pack(fill="x", padx=10, pady=5)
lbl = ttk.Label(frame, text="Исходные данные (время на станках):", font=("Arial", 12, "bold"))
lbl.pack(anchor="w", pady=5)

tree = ttk.Treeview(frame, columns=("a","b","c"), show="headings", height=len(jobs))
tree.heading("a", text="Станок 1 (a)")
tree.heading("b", text="Станок 2 (b)")
tree.heading("c", text="Станок 3 (c)")
for j in jobs:
    tree.insert("", "end", values=(j["a"], j["b"], j["c"]))
tree.pack(fill="x")

canvas = tk.Canvas(root, width=1000, height=600, bg="white")
canvas.pack(fill="both", expand=True, padx=10, pady=10)

seq_str_orig = "(" + ",".join(map(str, orig_seq)) + ")"
draw_gantt(canvas, sched_orig, "Исходная последовательность: " + seq_str_orig, y_offset=20)

if sched_opt:
    seq_str_opt = "(" + ",".join(map(str, opt_seq)) + ")"
    draw_gantt(canvas, sched_opt, "Оптимальная последовательность: " + seq_str_opt, y_offset=220)
else:
    canvas.create_text(10, 220, text="Алгоритм Джонсона неприменим для этих данных",
                       anchor="w", font=("Arial", 12, "bold"), fill="red")

root.mainloop()