# streamlit_app.py
# Math Arcade 🎯 — a fun Streamlit math game for all school levels (not too complicated)
# Run with: streamlit run streamlit_app.py

import math
import random
import time
from fractions import Fraction

import streamlit as st

# ---------- Page Setup ----------
st.set_page_config(page_title="Math Arcade 🎯", page_icon="🎯", layout="centered")

PRIMARY_TOPICS = [
    "Add/Subtract",
    "Multiply/Divide",
    "Mixed Ops",
    "Fractions & Percent",
    "Squares & Roots",
    "Number Properties",
    "Times Tables",
]

LEVELS = {
    "Easy": {"min": 0, "max": 12, "ops": ["+", "-"], "bonus": 1.0},
    "Medium": {"min": 0, "max": 50, "ops": ["+", "-", "×", "÷"], "bonus": 1.3},
    "Hard": {"min": -50, "max": 100, "ops": ["+", "-", "×", "÷"], "bonus": 1.6},
    "Challenge": {"min": -200, "max": 200, "ops": ["+", "-", "×", "÷"], "bonus": 2.0},
}

GAME_MODES = {
    "Practice (no limit)": {"type": "practice", "desc": "Endless practice. No timer, no lives."},
    "Classic (3 lives)": {"type": "lives", "lives": 3, "desc": "You get 3 hearts. Lose one per wrong answer."},
    "Speedrun (60s)": {"type": "timed", "seconds": 60, "desc": "Answer as many as you can in 60 seconds."},
}

# ---------- Helpers ----------
def init_state():
    defaults = {
        "started": False,
        "score": 0,
        "streak": 0,
        "best_streak": 0,
        "lives": None,
        "time_left": None,
        "question": None,
        "answer": None,
        "explanation": "",
        "t_start": None,
        "total_answered": 0,
        "total_correct": 0,
        "used_hint": False,
        "history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def emoji_hearts(n):
    if n is None:
        return ""
    return "❤️" * n + "🤍" * max(0, 3 - n)

def format_fraction(frac: Fraction):
    if frac.denominator == 1:
        return str(frac.numerator)
    return f"{frac.numerator}/{frac.denominator}"

def parse_numeric(user_text: str):
    """Accept integers, decimals, simple fractions like '3/4' or '-7/2', and percentage with %."""
    s = user_text.strip().replace(" ", "")
    if not s:
        raise ValueError("empty")
    # percentage like 25% -> 0.25
    if s.endswith("%"):
        val = float(s[:-1]) / 100.0
        return val
    # fraction
    if "/" in s:
        num, den = s.split("/", 1)
        return Fraction(int(num), int(den))
    # integer or float
    if "." in s:
        return float(s)
    return int(s)

def approx_equal(a, b):
    # compare numbers that may be Fraction / int / float
    def to_float(x):
        return float(x) if isinstance(x, (Fraction, float)) else float(int(x))
    # exact for fractions/ints
    if isinstance(a, Fraction) or isinstance(b, Fraction):
        return Fraction(a) == Fraction(b)
    # floats: tolerance
    return abs(to_float(a) - to_float(b)) <= 1e-6

def pick_int(a, b, excl_zero_div=False):
    while True:
        n = random.randint(a, b)
        if excl_zero_div and n == 0:
            continue
        return n

def times_table_question(level):
    table = random.choice([2,3,4,5,6,7,8,9,10,11,12]) if LEVELS[level]["max"] >= 12 else random.randint(2, 10)
    b = random.randint(2, 12)
    ans = table * b
    return {
        "q": f"{table} × {b} = ?",
        "a": ans,
        "exp": f"Multiply: {table} × {b} = {ans}."
    }

# ---------- Question Generators ----------
def gen_add_sub(level):
    lo, hi = LEVELS[level]["min"], LEVELS[level]["max"]
    a, b = pick_int(lo, hi), pick_int(lo, hi)
    op = random.choice(["+", "-"]) if level == "Easy" else random.choice(["+", "-"])
    ans = a + b if op == "+" else a - b
    return {"q": f"{a} {op} {b} = ?", "a": ans, "exp": f"{a} {op} {b} = {ans}."}

def gen_mul_div(level):
    lo, hi = LEVELS[level]["min"], LEVELS[level]["max"]
    op = random.choice(["×", "÷"])
    if op == "×":
        a, b = pick_int(lo, hi), pick_int(lo, hi)
        ans = a * b
        return {"q": f"{a} × {b} = ?", "a": ans, "exp": f"Multiply: {a} × {b} = {ans}."}
    else:
        b = pick_int(1 if lo <= 0 <= hi else max(1, lo), hi, excl_zero_div=True)
        ans = pick_int(lo, hi)
        a = ans * b
        return {"q": f"{a} ÷ {b} = ?", "a": ans, "exp": f"Divide: {a} ÷ {b} = {ans} because {ans}×{b}={a}."}

def gen_mixed(level):
    lo, hi = LEVELS[level]["min"], LEVELS[level]["max"]
    ops = ["+", "-", "×", "÷"]
    # (a op1 b) op2 c with precedence
    a, b, c = pick_int(lo, hi), pick_int(lo, hi), pick_int(lo, hi if hi != 0 else 1)
    op1, op2 = random.choice(ops), random.choice(ops)
    # build safe expression using Python operators
    def apply(x, op, y):
        if op == "+": return x + y
        if op == "-": return x - y
        if op == "×": return x * y
        if op == "÷": return x / y if y != 0 else x / 1
    # ensure no div by zero
    if op1 == "÷" and b == 0: b = 1
    if op2 == "÷" and c == 0: c = 1
    # compute step by step respecting × ÷ before + -
    exp_str = f"Compute {a} {op1} {b} {op2} {c} with ×/÷ first."
    # Convert to tuple list for precedence
    nums = [a, b, c]
    ops_list = [op1, op2]
    # first pass for × ÷
    work_nums = [nums[0]]
    work_ops = []
    i = 0
    while i < len(ops_list):
        o = ops_list[i]
        if o in ("×", "÷"):
            left = work_nums.pop()
            right = nums[i+1] if o in ("×","÷") else nums[i+1]
            tmp = left * right if o == "×" else left / (right if right != 0 else 1)
            work_nums.append(tmp)
            exp_str += f"\n• {left} {o} {right} = {tmp}"
        else:
            work_nums.append(nums[i+1])
            work_ops.append(o)
        i += 1
    # second pass for + -
    result = work_nums[0]
    for j, o in enumerate(work_ops):
        before = result
        result = result + work_nums[j+1] if o == "+" else result - work_nums[j+1]
        exp_str += f"\n• {before} {o} {work_nums[j+1]} = {result}"
    q = f"{a} {op1} {b} {op2} {c} = ?"
    # round if close to int
    if abs(result - round(result)) < 1e-9:
        result = int(round(result))
    return {"q": q, "a": result, "exp": exp_str}

def gen_frac_percent(level):
    # 50/50 either fraction simplification / addition or a percent question
    if random.random() < 0.5:
        # Percent of a number
        base = random.randint(10, 500)
        pct = random.choice([5, 10, 12.5, 20, 25, 33.33, 50])
        ans = round(base * (pct / 100.0), 2)
        exp = f"{pct}% of {base} = {pct/100} × {base} = {ans}."
        return {"q": f"{pct}% of {base} = ?", "a": ans, "exp": exp}
    else:
        # Simple fraction addition or simplification
        a, b = random.randint(1, 9), random.randint(2, 9)
        c, d = random.randint(1, 9), random.randint(2, 9)
        if random.random() < 0.5:
            # simplify
            frac = Fraction(a * c, b * c)
            simp = frac  # already reduced by Fraction
            return {
                "q": f"Simplify: {(a*c)}/{(b*c)}",
                "a": simp,
                "exp": f"Divide numerator and denominator by {c}: {(a*c)}/{(b*c)} → {format_fraction(simp)}",
            }
        else:
            # add
            f1, f2 = Fraction(a, b), Fraction(c, d)
            s = f1 + f2
            return {
                "q": f"{a}/{b} + {c}/{d} = ?",
                "a": s,
                "exp": f"Find common denominator {b*d}, add: {a*d}+{c*b}={a*d + c*b}; result {format_fraction(s)}.",
            }

def gen_squares_roots(level):
    if random.random() < 0.5:
        n = random.choice([2,3,4,5,6,7,8,9,10,11,12,13,15])
        ans = n * n
        return {"q": f"{n}² = ?", "a": ans, "exp": f"{n}² = {n}×{n} = {ans}."}
    else:
        n = random.choice([1,4,9,16,25,36,49,64,81,100,121,144,169,196,225])
        ans = int(math.sqrt(n))
        return {"q": f"√{n} = ?", "a": ans, "exp": f"Square root of {n} is {ans} because {ans}² = {n}."}

def gen_number_properties(level):
    # prime/composite or even/odd or LCM/GCD small
    choice = random.choice(["prime", "parity", "gcd"])
    if choice == "prime":
        n = random.randint(2, 199)
        # compute primality
        def is_prime(x):
            if x < 2: return False
            if x % 2 == 0: return x == 2
            r = int(math.sqrt(x))
            for i in range(3, r+1, 2):
                if x % i == 0:
                    return False
            return True
        ans = "Prime" if is_prime(n) else "Composite"
        return {"q": f"Is {n} prime or composite?", "a": ans, "exp": f"{n} is {ans}."}
    elif choice == "parity":
        n = random.randint(-200, 200)
        ans = "Even" if n % 2 == 0 else "Odd"
        return {"q": f"Is {n} even or odd?", "a": ans, "exp": f"{n} ÷ 2 leaves remainder {n%2}. So it is {ans}."}
    else:
        a, b = random.randint(2, 30), random.randint(2, 30)
        # ask for GCD
        from math import gcd
        g = gcd(a, b)
        return {"q": f"GCD({a}, {b}) = ?", "a": g, "exp": f"Prime factors → gcd = {g}."}

def gen_question(topic, level):
    if topic == "Add/Subtract":
        return gen_add_sub(level)
    if topic == "Multiply/Divide":
        return gen_mul_div(level)
    if topic == "Mixed Ops":
        return gen_mixed(level)
    if topic == "Fractions & Percent":
        return gen_frac_percent(level)
    if topic == "Squares & Roots":
        return gen_squares_roots(level)
    if topic == "Number Properties":
        return gen_number_properties(level)
    if topic == "Times Tables":
        return times_table_question(level)
    # fallback
    return gen_add_sub(level)

# ---------- Scoring ----------
def base_points(level):
    return {"Easy": 10, "Medium": 15, "Hard": 20, "Challenge": 25}[level]

def time_bonus(seconds_taken, level):
    # reward faster answers, capped
    if seconds_taken is None:
        return 0
    cap = 8 if level in ("Hard", "Challenge") else 6
    bonus = max(0, (cap - seconds_taken)) * 2
    return int(bonus)

def streak_bonus(streak):
    if streak <= 1: return 0
    return min(10 + (streak - 2) * 2, 30)

# ---------- Game Flow ----------
def start_game(mode_key, level, topic):
    st.session_state.started = True
    st.session_state.score = 0
    st.session_state.streak = 0
    st.session_state.best_streak = 0
    st.session_state.total_answered = 0
    st.session_state.total_correct = 0
    st.session_state.used_hint = False
    st.session_state.history = []

    mode = GAME_MODES[mode_key]
    if mode["type"] == "lives":
        st.session_state.lives = mode["lives"]
        st.session_state.time_left = None
    elif mode["type"] == "timed":
        st.session_state.lives = None
        st.session_state.time_left = mode["seconds"]
        st.session_state.timer_started_at = time.time()
    else:
        st.session_state.lives = None
        st.session_state.time_left = None

    q = gen_question(topic, level)
    st.session_state.question = q["q"]
    st.session_state.answer = q["a"]
    st.session_state.explanation = q["exp"]
    st.session_state.t_start = time.time()

def next_question(level, topic):
    q = gen_question(topic, level)
    st.session_state.question = q["q"]
    st.session_state.answer = q["a"]
    st.session_state.explanation = q["exp"]
    st.session_state.t_start = time.time()
    st.session_state.used_hint = False

def end_game():
    st.session_state.started = False

# ---------- UI ----------
init_state()

st.title("🎯 Math Arcade")
st.caption("Fun, fast, and friendly math practice for school levels — pick a topic, set the level, and play!")

with st.sidebar:
    st.header("⚙️ Game Setup")
    topic = st.selectbox("Topic", PRIMARY_TOPICS, index=0)
    level = st.selectbox("Level", list(LEVELS.keys()), index=0, help="Harder levels use bigger numbers and more ops.")
    mode_key = st.selectbox("Mode", list(GAME_MODES.keys()), index=1,
                            help="Practice = endless | Classic = 3 lives | Speedrun = 60 seconds")
    st.info(GAME_MODES[mode_key]["desc"])
    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶️ Start / Restart", use_container_width=True):
            start_game(mode_key, level, topic)
    with c2:
        if st.button("⏹️ Stop", use_container_width=True):
            end_game()

# HUD
hud1, hud2, hud3 = st.columns([1,1,1])
with hud1:
    st.metric("Score", st.session_state.score)
with hud2:
    st.metric("Streak", f"{st.session_state.streak} 🔥" if st.session_state.streak >= 3 else st.session_state.streak)
with hud3:
    if st.session_state.lives is not None:
        st.metric("Lives", emoji_hearts(st.session_state.lives))
    elif st.session_state.time_left is not None and st.session_state.started:
        # update timer
        elapsed = int(time.time() - st.session_state.get("timer_started_at", time.time()))
        st.session_state.time_left = max(0, GAME_MODES[mode_key]["seconds"] - elapsed)
        st.metric("Time Left", f"{st.session_state.time_left}s ⏱️")
        if st.session_state.time_left <= 0 and st.session_state.started:
            st.warning("⏰ Time's up!")
            end_game()

st.divider()

# Main game panel
if not st.session_state.started:
    st.subheader("How to play")
    st.markdown(
        """
- Pick a **Topic**, **Level**, and **Mode** on the left.
- Press **Start / Restart** to begin.
- Type your answer and hit **Submit**.  
- Use **Hint** if stuck (small score penalty).
- In **Classic**, you have 3 hearts. In **Speedrun**, you have 60 seconds.  
- Answers accept integers, decimals, simple fractions like `3/4`, and percentages like `25%` where appropriate.
        """
    )
else:
    st.subheader(f"📚 {topic} — {level} — {mode_key}")
    st.write("**Question:**")
    st.markdown(f"### {st.session_state.question}")

    hint_area = st.empty()
    answer_feedback = st.empty()

    # Buttons: Hint / Skip (Skip only in Practice)
    bcol1, bcol2, bcol3 = st.columns([1,1,1])
    with bcol1:
        if st.button("💡 Hint (-3 pts)"):
            st.session_state.used_hint = True
            hint_area.info("Hint: " + st.session_state.explanation.split("\n")[0])
    with bcol2:
        can_skip = GAME_MODES[mode_key]["type"] == "practice"
        if st.button("⏭️ Skip" + ("" if can_skip else " (Practice only)"), disabled=not can_skip):
            # record skip
            st.session_state.history.append(
                {"q": st.session_state.question, "your": "—", "correct": st.session_state.answer, "result": "skipped"}
            )
            next_question(level, topic)
            st.experimental_rerun()
    with bcol3:
        st.write("")  # spacer

    with st.form("answer_form", clear_on_submit=True):
        user_input = st.text_input("Your answer", value="", placeholder="e.g., 42 or 3/4 or 25%")
        submitted = st.form_submit_button("Submit ✅")

    if submitted:
        seconds_taken = time.time() - (st.session_state.t_start or time.time())
        st.session_state.total_answered += 1
        try:
            parsed = parse_numeric(user_input)
        except Exception:
            parsed = user_input.strip()

        correct = st.session_state.answer
        is_correct = False

        if isinstance(correct, str):
            is_correct = str(parsed).strip().lower() == correct.lower()
        else:
            try:
                is_correct = approx_equal(parsed, correct)
            except Exception:
                is_correct = False

        if is_correct:
            st.session_state.total_correct += 1
            st.session_state.streak += 1
            st.session_state.best_streak = max(st.session_state.best_streak, st.session_state.streak)
            pts = base_points(level) + time_bonus(seconds_taken, level) + streak_bonus(st.session_state.streak)
            if st.session_state.used_hint:
                pts = max(1, pts - 3)
            st.session_state.score += pts
            answer_feedback.success(
                f"✅ Correct! +{pts} points  •  ⏱ {seconds_taken:.1f}s  •  Streak {st.session_state.streak} 🔥"
            )
            st.session_state.history.append(
                {"q": st.session_state.question, "your": user_input, "correct": correct, "result": "✅"}
            )
        else:
            st.session_state.streak = 0
            answer_feedback.error(f"❌ Not quite. Correct answer: **{correct}**")
            st.session_state.history.append(
                {"q": st.session_state.question, "your": user_input or "—", "correct": correct, "result": "❌"}
            )
            if st.session_state.lives is not None:
                st.session_state.lives -= 1
                if st.session_state.lives <= 0:
                    st.warning("💥 Out of lives!")
                    end_game()
                    st.rerun()  # ← ensure HUD updates to game over immediately

        # 👉 ALWAYS advance if the game is still running (prevents the “décalage”)
        if st.session_state.started and (
            st.session_state.lives is None or st.session_state.lives > 0
        ):
            next_question(level, topic)
            st.rerun()  # ← immediate UI refresh to show the new question

    # Show explanation toggle
    with st.expander("See full explanation / steps"):
        st.markdown(f"**How to solve:**  \n{st.session_state.explanation}")

    # Stats
    st.divider()
    s1, s2, s3, s4 = st.columns(4)
    with s1: st.metric("Answered", st.session_state.total_answered)
    with s2: st.metric("Correct", st.session_state.total_correct)
    with s3:
        acc = 0 if st.session_state.total_answered == 0 else int(100*st.session_state.total_correct/st.session_state.total_answered)
        st.metric("Accuracy", f"{acc}%")
    with s4: st.metric("Best Streak", st.session_state.best_streak)

    # Recent history (last 8)
    if st.session_state.history:
        st.write("### Recent")
        for row in st.session_state.history[-8:][::-1]:
            correct_fmt = row["correct"] if isinstance(row["correct"], str) else (format_fraction(row["correct"]) if isinstance(row["correct"], Fraction) else row["correct"])
            st.markdown(
                f"- {row['result']} **{row['q']}** — Your answer: `{row['your']}` • Correct: `{correct_fmt}`"
            )

st.divider()
st.caption("Tip: you can accept answers like `3/4` (fractions) and `25%` (percentages) when it makes sense.")
