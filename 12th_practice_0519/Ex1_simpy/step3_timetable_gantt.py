# -*- coding: utf-8 -*-
"""
=============================================================
  Step 3: 타임테이블 + 간트 차트 통합
=============================================================
시뮬레이션 결과를 실시간으로 기록하고:
  1. 타임 테이블 (Treeview) — 부품별 투입/시작/완료/대기/리드타임
  2. 간트 차트 (Matplotlib) — 설비별 작업 스케줄 시각화

실행 방법:
  conda activate mfg_data
  python step3_timetable_gantt.py
=============================================================
"""

import simpy
import tkinter as tk
from tkinter import ttk, filedialog
import random
import csv
from pathlib import Path

RESULT_DIR = Path(__file__).resolve().parent / "0_result"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# 한글 폰트
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 시뮬레이션 파라미터 ── JYC 변수
NUM_MACHINES = 5
PROCESS_TIME = 3.0
ARRIVAL_INTERVAL = 2.0
SIM_TIME = 50
ANIMATION_SPEED = 50

# ── 상태 변수 ──
buffer_queue = []
machine_dict = {i: None for i in range(NUM_MACHINES)}
completed_count = 0
gantt_data = []
log_data = []   # CSV 내보내기용

# 애니메이션 캔버스 높이 (설비 수에 따라 동적 확장)
CANVAS_HEIGHT = max(210, 50 + NUM_MACHINES * 85)
# ── 설비별 색상 팔레트 ──
MACHINE_COLORS = ['#42A5F5', '#66BB6A', '#FFA726', '#EF5350', '#AB47BC']


# ══════════════════════════════════════════════════════════════
#  SimPy 모델
# ══════════════════════════════════════════════════════════════

def part_process(env, name, machines):
    global completed_count

    arrival_time = env.now
    buffer_queue.append(name)

    with machines.request() as req:
        yield req

        start_time = env.now
        buffer_queue.remove(name)

        # 빈 설비 찾기
        my_machine = None
        for m_id, p_name in machine_dict.items():
            if p_name is None:
                my_machine = m_id
                machine_dict[m_id] = name
                break

        # 가공
        process_duration = random.expovariate(1.0 / PROCESS_TIME)
        yield env.timeout(process_duration)

        finish_time = env.now
        if my_machine is not None:
            machine_dict[my_machine] = None
        completed_count += 1

        # 시간 계산
        wait_time = start_time - arrival_time
        lead_time = finish_time - arrival_time

        # 타임 테이블에 행 추가
        tree.insert("", "end", values=(
            name, f"설비 {my_machine + 1}",
            f"{arrival_time:.1f}", f"{start_time:.1f}", f"{finish_time:.1f}",
            f"{wait_time:.1f}", f"{lead_time:.1f}"
        ))
        tree.yview_moveto(1)

        # 간트 차트 데이터 추가
        gantt_data.append({
            'Machine': my_machine,
            'Start': start_time,
            'Duration': process_duration,
            'Part': name
        })
        update_gantt()

        # CSV용 로그
        log_data.append({
            'Part': name, 'Machine': f"설비 {my_machine + 1}",
            'Arrival': round(arrival_time, 2), 'Start': round(start_time, 2),
            'Finish': round(finish_time, 2), 'Wait': round(wait_time, 2),
            'LeadTime': round(lead_time, 2)
        })


def material_arrival(env, machines):
    i = 0
    while True:
        yield env.timeout(random.expovariate(1.0 / ARRIVAL_INTERVAL))
        i += 1
        env.process(part_process(env, f"부품 {i}", machines))


# ══════════════════════════════════════════════════════════════
#  GUI 업데이트
# ══════════════════════════════════════════════════════════════

def update_gui():
    canvas.delete("all")
    time_label.config(text=f"현재 공정 시간: {env.now:.1f} / {SIM_TIME}")

    # 대기 버퍼
    canvas.create_text(80, 18, text="대기 버퍼 (WIP)",
                       font=("Malgun Gothic", 10, "bold"))
    max_wip = max(5, (CANVAS_HEIGHT - 50) // 32)
    for idx, p_name in enumerate(buffer_queue[:max_wip]):
        x, y = 80, 50 + idx * 32
        canvas.create_oval(x - 16, y - 13, x + 16, y + 13, fill="#B0BEC5", outline="#607D8B")
        canvas.create_text(x, y, text=p_name, font=("Malgun Gothic", 8))
    if len(buffer_queue) > max_wip:
        canvas.create_text(80, 50 + max_wip * 32, text=f"+{len(buffer_queue) - max_wip}개",
                           font=("Malgun Gothic", 8), fill="gray")

    # 가공 설비
    canvas.create_text(300, 18, text="가공 설비",
                       font=("Malgun Gothic", 10, "bold"))
    for m_id in range(NUM_MACHINES):
        x0, y0 = 220, 38 + m_id * 85
        x1, y1 = 380, 105 + m_id * 85
        p_name = machine_dict[m_id]
        color = MACHINE_COLORS[m_id % len(MACHINE_COLORS)]

        if p_name:
            canvas.create_rectangle(x0, y0, x1, y1, fill="#FFF9C4", outline=color, width=2)
            canvas.create_text((x0 + x1) // 2, y0 + 20, text=f"설비 {m_id + 1}",
                               font=("Malgun Gothic", 9, "bold"))
            canvas.create_text((x0 + x1) // 2, y0 + 45, text=f"가공: {p_name}",
                               font=("Malgun Gothic", 9), fill="#E65100")
        else:
            canvas.create_rectangle(x0, y0, x1, y1, fill="#E8E8E8", outline="#999", width=1)
            canvas.create_text((x0 + x1) // 2, y0 + 32, text=f"설비 {m_id + 1} (대기)",
                               font=("Malgun Gothic", 9), fill="#757575")

    # 생산 완료
    canvas.create_text(530, 55, text="완료", font=("Malgun Gothic", 10, "bold"))
    canvas.create_text(530, 90, text=f"{completed_count}개",
                       font=("Malgun Gothic", 20, "bold"), fill="#2E7D32")


def update_gantt():
    """간트 차트 갱신"""
    ax.clear()
    ax.set_title("설비별 작업 스케줄 (Gantt Chart)", fontsize=10, fontweight='bold')
    ax.set_xlabel("시간 (초)", fontsize=9)

    ax.set_yticks([i * 10 for i in range(NUM_MACHINES)])
    ax.set_yticklabels([f"설비 {i + 1}" for i in range(NUM_MACHINES)])
    ax.set_ylim(-5, NUM_MACHINES * 10 + 2)
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    for data in gantt_data:
        m = data['Machine']
        y_pos = m * 10 - 3
        color = MACHINE_COLORS[m % len(MACHINE_COLORS)]
        ax.broken_barh([(data['Start'], data['Duration'])], (y_pos, 6),
                        facecolors=color, edgecolor='white', linewidth=0.5)

        # 부품 번호 텍스트 (막대가 충분히 넓을 때만)
        if data['Duration'] > 1.5:
            part_num = data['Part'].replace("부품 ", "")
            ax.text(data['Start'] + data['Duration'] / 2, y_pos + 3, part_num,
                    ha='center', va='center', color='white', fontsize=7, fontweight='bold')

    canvas_gantt.draw()


def export_csv():
    """타임테이블 데이터를 CSV로 내보내기"""
    if not log_data:
        return

    filepath = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV 파일", "*.csv")],
        initialdir=RESULT_DIR,
        initialfile="simulation_log.csv"
    )
    if filepath:
        with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=log_data[0].keys())
            writer.writeheader()
            writer.writerows(log_data)
        status_label.config(text=f"✅ CSV 저장 완료: {filepath}")


def save_gantt_chart():
    """간트 차트를 이미지로 저장"""
    filename = RESULT_DIR / "gantt_chart_result.png"
    try:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ 간트 차트가 저장되었습니다: {filename}")
    except Exception as e:
        print(f"❌ 간트 차트 저장 실패: {e}")


def show_summary():
    """시뮬레이션 종료 후 요약 통계 팝업"""
    if not log_data:
        return

    waits = [d['Wait'] for d in log_data]
    leads = [d['LeadTime'] for d in log_data]

    msg = (
        f"=== 시뮬레이션 요약 ===\n\n"
        f"총 생산 수량: {len(log_data)}개\n"
        f"시뮬레이션 시간: {SIM_TIME}초\n"
        f"처리량: {len(log_data) / SIM_TIME:.2f}개/초\n\n"
        f"평균 대기 시간: {sum(waits) / len(waits):.2f}초\n"
        f"최대 대기 시간: {max(waits):.2f}초\n"
        f"평균 리드 타임: {sum(leads) / len(leads):.2f}초\n"
        f"최대 리드 타임: {max(leads):.2f}초\n"
    )

    popup = tk.Toplevel(root)
    popup.title("시뮬레이션 요약 통계")
    popup.geometry("300x280")
    tk.Label(popup, text=msg, font=("Malgun Gothic", 10), justify="left", padx=15, pady=15).pack()
    tk.Button(popup, text="닫기", command=popup.destroy,
              font=("Malgun Gothic", 10)).pack(pady=5)


# ══════════════════════════════════════════════════════════════
#  시뮬레이션 루프
# ══════════════════════════════════════════════════════════════

def run_simulation_step():
    if env.peek() < SIM_TIME:
        try:
            env.step()
            update_gui()
            root.after(ANIMATION_SPEED, run_simulation_step)
        except simpy.core.EmptySchedule:
            save_gantt_chart()
            show_summary()
    else:
        status_label.config(text=f"✅ 시뮬레이션 완료! 총 {completed_count}개 생산")
        save_gantt_chart()
        show_summary()


# ══════════════════════════════════════════════════════════════
#  GUI 레이아웃 구성
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    env = simpy.Environment()
    machines = simpy.Resource(env, capacity=NUM_MACHINES)
    env.process(material_arrival(env, machines))

    root = tk.Tk()
    root.title("제조 공정 시뮬레이션 — 타임테이블 & 간트 차트")
    
    # 설비 수에 따라 창 높이 동적 조절
    window_height = max(800, 450 + NUM_MACHINES * 90)
    root.geometry(f"720x{window_height}")

    # ── 시간 표시 ──
    time_label = tk.Label(root, text="현재 공정 시간: 0.0",
                           font=("Malgun Gothic", 13, "bold"))
    time_label.pack(pady=3)

    # ── 1. 애니메이션 캔버스 ──
    canvas = tk.Canvas(root, width=680, height=CANVAS_HEIGHT, bg="white")
    canvas.pack(padx=10, pady=3)

    # ── 2. 타임 테이블 ──
    table_frame = tk.LabelFrame(root, text=" 📋 타임 테이블 ",
                                 font=("Malgun Gothic", 10, "bold"))
    table_frame.pack(fill="x", padx=10, pady=3)

    columns = ("Part", "Machine", "Arrival", "Start", "Finish", "Wait", "LeadTime")
    tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=5)
    headers = ["부품명", "설비", "투입 시간", "가공 시작", "가공 완료", "대기 시간", "리드 타임"]
    for col, header in zip(columns, headers):
        tree.heading(col, text=header)
        tree.column(col, width=92, anchor="center")
    tree.pack(fill="x", padx=5, pady=3)

    # 버튼 행
    btn_frame = tk.Frame(table_frame)
    btn_frame.pack(pady=3)
    tk.Button(btn_frame, text="📥 CSV 내보내기", command=export_csv,
              font=("Malgun Gothic", 9)).pack(side="left", padx=5)
    tk.Button(btn_frame, text="📊 요약 통계", command=show_summary,
              font=("Malgun Gothic", 9)).pack(side="left", padx=5)

    # ── 3. 간트 차트 (Matplotlib) ──
    gantt_frame = tk.LabelFrame(root, text=" 📊 간트 차트 ",
                                 font=("Malgun Gothic", 10, "bold"))
    gantt_frame.pack(fill="both", expand=True, padx=10, pady=3)

    fig_height = max(2.2, NUM_MACHINES * 0.5)
    fig = Figure(figsize=(6.5, fig_height), dpi=100)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(bottom=0.22, left=0.12, right=0.95)

    canvas_gantt = FigureCanvasTkAgg(fig, master=gantt_frame)
    canvas_gantt.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=3)
    update_gantt()

    # ── 상태 표시줄 ──
    status_label = tk.Label(root, text="▶ 시뮬레이션 실행 중...",
                             font=("Malgun Gothic", 9), bg='#e0e0e0', anchor="w", padx=10)
    status_label.pack(fill="x", side="bottom")

    # 시뮬레이션 시작
    run_simulation_step()
    root.mainloop()
