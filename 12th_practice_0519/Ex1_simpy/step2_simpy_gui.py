# -*- coding: utf-8 -*-
"""
=============================================================
  Step 2: 실시간 GUI 시뮬레이션 + KPI 대시보드
=============================================================
제조 공정을 실시간 애니메이션으로 관찰하며,
파라미터(설비 수, 투입 간격, 가공 시간)를 슬라이더로 조절합니다.

실행 방법:
  conda activate mfg_data
  python step2_simpy_gui.py
=============================================================
"""

import random
import sys

import simpy

try:
    import tkinter as tk
    from tkinter import ttk
except ModuleNotFoundError as exc:
    if exc.name != "_tkinter":
        raise
    print("[실행 오류] 현재 Python에는 Tkinter GUI 모듈(_tkinter)이 없습니다.")
    print("이 실습은 GUI 창을 사용하므로 Tk가 포함된 Python으로 실행해야 합니다.")
    print("권장 실행 명령:")
    print("  /opt/anaconda3/bin/python3 12th_practice_0519/simpy/step2_simpy_gui.py")
    sys.exit(1)


class ManufacturingSimulator:
    """
    제조 공정 시뮬레이터 (SimPy + Tkinter 통합)

    구성요소:
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ 부품 투입  │ →  │ 대기 버퍼  │ →  │ 가공 설비  │ →  완료
    │(Material) │    │  (WIP)   │    │(Machine) │
    └──────────┘    └──────────┘    └──────────┘
    """

    def __init__(self):
        # ── 시뮬레이션 파라미터 (슬라이더로 조절 가능) ──
        self.num_machines = 2
        self.process_time = 3.0
        self.arrival_interval = 2.0
        self.sim_time = 100
        self.animation_speed = 150  # ms

        # ── 상태 변수 ──
        self.buffer_queue = []
        self.machine_dict = {}
        self.completed_count = 0
        self.total_wait_time = 0
        self.total_busy_time = {}
        self.is_running = False
        self.env = None

        # ── GUI 구축 ──
        self._build_gui()

    # ══════════════════════════════════════════════════════════
    #  GUI 구축
    # ══════════════════════════════════════════════════════════

    def _build_gui(self):
        self.root = tk.Tk()
        self.root.title("제조 공정 시뮬레이션 (SimPy)")
        self.root.configure(bg='#f0f0f0')
        self.root.geometry("750x650")

        # ── 상단: 제어 패널 ──
        control_frame = tk.LabelFrame(self.root, text=" ⚙️ 파라미터 조절 ",
                                       font=("Malgun Gothic", 11, "bold"),
                                       bg='#f0f0f0', padx=10, pady=5)
        control_frame.pack(fill="x", padx=10, pady=5)

        # 슬라이더: 설비 수
        tk.Label(control_frame, text="설비 수:", bg='#f0f0f0',
                 font=("Malgun Gothic", 10)).grid(row=0, column=0, sticky="e")
        self.machine_slider = tk.Scale(control_frame, from_=1, to=5,
                                        orient="horizontal", length=150,
                                        font=("Malgun Gothic", 9))
        self.machine_slider.set(2)
        self.machine_slider.grid(row=0, column=1, padx=5)

        # 슬라이더: 투입 간격
        tk.Label(control_frame, text="투입 간격(초):", bg='#f0f0f0',
                 font=("Malgun Gothic", 10)).grid(row=0, column=2, sticky="e")
        self.arrival_slider = tk.Scale(control_frame, from_=0.5, to=5.0,
                                        orient="horizontal", length=150,
                                        resolution=0.5,
                                        font=("Malgun Gothic", 9))
        self.arrival_slider.set(2.0)
        self.arrival_slider.grid(row=0, column=3, padx=5)

        # 슬라이더: 가공 시간
        tk.Label(control_frame, text="가공 시간(초):", bg='#f0f0f0',
                 font=("Malgun Gothic", 10)).grid(row=0, column=4, sticky="e")
        self.process_slider = tk.Scale(control_frame, from_=1.0, to=8.0,
                                        orient="horizontal", length=150,
                                        resolution=0.5,
                                        font=("Malgun Gothic", 9))
        self.process_slider.set(3.0)
        self.process_slider.grid(row=0, column=5, padx=5)

        # 버튼: 시작 / 리셋
        btn_frame = tk.Frame(control_frame, bg='#f0f0f0')
        btn_frame.grid(row=1, column=0, columnspan=6, pady=5)

        self.start_btn = tk.Button(btn_frame, text="▶ 시작", width=10,
                                    font=("Malgun Gothic", 10, "bold"),
                                    bg='#4CAF50', fg='white',
                                    command=self._start_simulation)
        self.start_btn.pack(side="left", padx=5)

        self.reset_btn = tk.Button(btn_frame, text="🔄 리셋", width=10,
                                    font=("Malgun Gothic", 10, "bold"),
                                    bg='#FF9800', fg='white',
                                    command=self._reset_simulation)
        self.reset_btn.pack(side="left", padx=5)

        # 속도 조절
        tk.Label(btn_frame, text="  애니메이션 속도:", bg='#f0f0f0',
                 font=("Malgun Gothic", 9)).pack(side="left")
        self.speed_slider = tk.Scale(btn_frame, from_=50, to=500,
                                      orient="horizontal", length=120,
                                      font=("Malgun Gothic", 8))
        self.speed_slider.set(150)
        self.speed_slider.pack(side="left", padx=5)

        # ── 중단: 시뮬레이션 시간 표시 ──
        self.time_label = tk.Label(self.root, text="현재 공정 시간: 0.0",
                                    font=("Malgun Gothic", 13, "bold"),
                                    bg='#f0f0f0')
        self.time_label.pack(pady=3)

        # ── 애니메이션 캔버스 ──
        self.canvas = tk.Canvas(self.root, width=720, height=280, bg="white",
                                 highlightthickness=1, highlightbackground="#ccc")
        self.canvas.pack(padx=10, pady=5)

        # ── 하단: KPI 대시보드 ──
        kpi_frame = tk.LabelFrame(self.root, text=" 📊 실시간 KPI 대시보드 ",
                                   font=("Malgun Gothic", 11, "bold"),
                                   bg='#f0f0f0', padx=10, pady=8)
        kpi_frame.pack(fill="x", padx=10, pady=5)

        # KPI 라벨들
        kpi_style = {"font": ("Malgun Gothic", 12), "bg": "#f0f0f0"}
        self.kpi_labels = {}

        kpi_items = [
            ("throughput", "처리량(Throughput)"),
            ("wip", "재공(WIP)"),
            ("avg_wait", "평균 대기시간"),
            ("utilization", "설비 가동률"),
        ]

        for i, (key, label_text) in enumerate(kpi_items):
            tk.Label(kpi_frame, text=f"{label_text}:", **kpi_style).grid(
                row=0, column=i * 2, sticky="e", padx=(10, 2))
            self.kpi_labels[key] = tk.Label(kpi_frame, text="—",
                                             font=("Malgun Gothic", 13, "bold"),
                                             fg="#1565C0", bg="#f0f0f0")
            self.kpi_labels[key].grid(row=0, column=i * 2 + 1, sticky="w", padx=(0, 15))

        # ── 상태 표시줄 ──
        self.status_label = tk.Label(self.root, text="⏸ 시작 버튼을 눌러주세요",
                                      font=("Malgun Gothic", 10),
                                      bg='#e0e0e0', anchor="w", padx=10)
        self.status_label.pack(fill="x", side="bottom")

    # ══════════════════════════════════════════════════════════
    #  SimPy 모델
    # ══════════════════════════════════════════════════════════

    def _part_process(self, env, name, machines):
        """부품 하나가 대기 → 가공 → 완료되는 과정"""
        arrival_time = env.now
        self.buffer_queue.append(name)

        with machines.request() as req:
            yield req

            # 대기 완료 → 설비 배정
            start_time = env.now
            wait_time = start_time - arrival_time
            self.total_wait_time += wait_time
            self.buffer_queue.remove(name)

            # 빈 설비 찾기
            my_machine = None
            for m_id in self.machine_dict:
                if self.machine_dict[m_id] is None:
                    my_machine = m_id
                    self.machine_dict[m_id] = name
                    break

            # 가공 진행
            proc_time = random.expovariate(1.0 / self.process_time)
            yield env.timeout(proc_time)

            # 가공 완료
            if my_machine is not None:
                self.total_busy_time[my_machine] = self.total_busy_time.get(my_machine, 0) + proc_time
                self.machine_dict[my_machine] = None
            self.completed_count += 1

    def _material_arrival(self, env, machines):
        """부품이 랜덤 간격으로 연속 투입"""
        i = 0
        while True:
            yield env.timeout(random.expovariate(1.0 / self.arrival_interval))
            i += 1
            env.process(self._part_process(env, f"부품{i}", machines))

    # ══════════════════════════════════════════════════════════
    #  시뮬레이션 제어
    # ══════════════════════════════════════════════════════════

    def _start_simulation(self):
        """시작 버튼 클릭 시"""
        if self.is_running:
            return

        # 슬라이더에서 파라미터 읽기
        self.num_machines = int(self.machine_slider.get())
        self.arrival_interval = float(self.arrival_slider.get())
        self.process_time = float(self.process_slider.get())

        # 상태 초기화
        self.buffer_queue.clear()
        self.machine_dict = {i: None for i in range(self.num_machines)}
        self.completed_count = 0
        self.total_wait_time = 0
        self.total_busy_time = {}

        # SimPy 환경 생성
        self.env = simpy.Environment()
        machines = simpy.Resource(self.env, capacity=self.num_machines)
        self.env.process(self._material_arrival(self.env, machines))

        self.is_running = True
        self.start_btn.config(state="disabled", bg='#9E9E9E')
        self.status_label.config(text=f"▶ 실행 중... (설비 {self.num_machines}대, "
                                       f"투입간격 {self.arrival_interval}초, "
                                       f"가공시간 {self.process_time}초)")

        self._run_step()

    def _reset_simulation(self):
        """리셋 버튼 클릭 시"""
        self.is_running = False
        self.buffer_queue.clear()
        self.machine_dict = {}
        self.completed_count = 0
        self.total_wait_time = 0
        self.total_busy_time = {}
        self.env = None
        self.canvas.delete("all")
        self.time_label.config(text="현재 공정 시간: 0.0")
        for lbl in self.kpi_labels.values():
            lbl.config(text="—")
        self.start_btn.config(state="normal", bg='#4CAF50')
        self.status_label.config(text="⏸ 리셋 완료. 파라미터를 조절한 후 시작하세요.")

    def _run_step(self):
        """시뮬레이션 한 스텝 실행 + GUI 갱신"""
        if not self.is_running or self.env is None:
            return

        if self.env.peek() < self.sim_time:
            try:
                self.env.step()
                self._update_canvas()
                self._update_kpi()
                speed = int(self.speed_slider.get())
                self.root.after(speed, self._run_step)
            except simpy.core.EmptySchedule:
                self._on_simulation_end()
        else:
            self._on_simulation_end()

    def _on_simulation_end(self):
        """시뮬레이션 종료 처리"""
        self.is_running = False
        self.start_btn.config(state="normal", bg='#4CAF50')
        self.status_label.config(text=f"✅ 시뮬레이션 완료! (총 {self.completed_count}개 생산)")

    # ══════════════════════════════════════════════════════════
    #  GUI 그리기
    # ══════════════════════════════════════════════════════════

    def _update_canvas(self):
        """애니메이션 캔버스 갱신"""
        c = self.canvas
        c.delete("all")

        self.time_label.config(text=f"현재 공정 시간: {self.env.now:.1f}")

        # ── 대기 버퍼 (WIP) ──
        c.create_text(80, 18, text="대기 버퍼 (WIP)", font=("Malgun Gothic", 11, "bold"))
        for idx, p_name in enumerate(self.buffer_queue[:6]):  # 최대 6개 표시
            x, y = 80, 55 + idx * 35
            c.create_oval(x - 18, y - 15, x + 18, y + 15, fill="#B0BEC5", outline="#607D8B")
            c.create_text(x, y, text=p_name, font=("Malgun Gothic", 8))
        if len(self.buffer_queue) > 6:
            c.create_text(80, 55 + 6 * 35, text=f"+{len(self.buffer_queue) - 6}개 더...",
                          font=("Malgun Gothic", 8), fill="gray")

        # ── 가공 설비 (Machines) ──
        num = self.num_machines
        machine_start_y = 45
        machine_height = min(70, 250 // max(num, 1))
        machine_gap = min(10, 30 // max(num, 1))

        c.create_text(350, 18, text="가공 설비 (Machines)", font=("Malgun Gothic", 11, "bold"))

        for m_id in range(num):
            x0 = 250
            y0 = machine_start_y + m_id * (machine_height + machine_gap)
            x1 = x0 + 200
            y1 = y0 + machine_height

            p_name = self.machine_dict.get(m_id)

            if p_name:
                # 가공 중
                c.create_rectangle(x0, y0, x1, y1, fill="#FFF9C4", outline="#F9A825", width=2)
                c.create_text(x0 + 100, y0 + machine_height // 3,
                              text=f"설비 {m_id + 1} — 가공 중",
                              font=("Malgun Gothic", 9, "bold"), fill="#E65100")
                c.create_text(x0 + 100, y0 + machine_height * 2 // 3,
                              text=p_name, font=("Malgun Gothic", 9))
            else:
                # 대기 중
                c.create_rectangle(x0, y0, x1, y1, fill="#E0E0E0", outline="#9E9E9E", width=1)
                c.create_text(x0 + 100, y0 + machine_height // 2,
                              text=f"설비 {m_id + 1} — 대기중",
                              font=("Malgun Gothic", 9), fill="#757575")

        # ── 생산 완료 ──
        c.create_text(600, 60, text="생산 완료", font=("Malgun Gothic", 11, "bold"))
        c.create_text(600, 100, text=f"{self.completed_count}개",
                      font=("Malgun Gothic", 22, "bold"), fill="#2E7D32")

        # ── 화살표 ──
        c.create_text(170, 140, text="→", font=("Arial", 20), fill="#999")
        c.create_text(475, 140, text="→", font=("Arial", 20), fill="#999")

    def _update_kpi(self):
        """KPI 대시보드 갱신"""
        now = self.env.now if self.env and self.env.now > 0 else 1

        # 처리량 (개/시간단위)
        throughput = self.completed_count / now
        self.kpi_labels["throughput"].config(text=f"{throughput:.2f} 개/초")

        # WIP (재공 수)
        wip = len(self.buffer_queue)
        color = "#2E7D32" if wip < 3 else ("#FF8F00" if wip < 6 else "#C62828")
        self.kpi_labels["wip"].config(text=f"{wip}개", fg=color)

        # 평균 대기시간
        avg_wait = self.total_wait_time / self.completed_count if self.completed_count > 0 else 0
        self.kpi_labels["avg_wait"].config(text=f"{avg_wait:.2f}초")

        # 가동률
        if self.num_machines > 0 and now > 0:
            total_busy = sum(self.total_busy_time.values())
            utilization = (total_busy / (self.num_machines * now)) * 100
            util_color = "#C62828" if utilization > 90 else ("#2E7D32" if utilization > 50 else "#1565C0")
            self.kpi_labels["utilization"].config(text=f"{utilization:.1f}%", fg=util_color)

    def run(self):
        self.root.mainloop()


# ── 실행 ──
if __name__ == '__main__':
    app = ManufacturingSimulator()
    app.run()
