"""
Operation CodeQuest — Episode 1: The Lost Core
Streamlit single-file app for a playful CS/Math escape-game.

INSTRUCTIONS
- Place this file as `app.py` (or any name) and run with: streamlit run app.py
- Provide the following assets in your repo:
  data/images/control_room.jpg
  data/images/data_vault.jpg
  data/images/air_chamber.jpg
  data/images/bit_bot.jpg
  data/images/debug_chamber.jpg
  data/images/final_core.jpg

  (You can replace names, but keep the same relative structure or edit paths below.)

- (Optional sounds) put in:
  data/sounds/tick.mp3
  data/sounds/alert.mp3
  data/sounds/success.mp3

This app keeps state in st.session_state and is designed for 9–10 year olds.
It includes: logic patterns, opposite operations (square vs square root, etc.),
fractions/multiplications, basic volume (cube/rectangular prism), and very simple Python debugging.

Storyline: Alex Byte (the player) helps TechNova recover the locked AI Core (Project Nova)
by collecting 5 digital keys. A coherent narrative runs from start to final win.
"""

from __future__ import annotations
import time
import math
import re
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, List

from PIL import Image, ImageDraw, ImageFont
import streamlit as st

# --------------------------
# --- CONFIG & UTILITIES ---
# --------------------------

st.set_page_config(page_title="Operation CodeQuest — Episode 1", page_icon="🧠", layout="wide")
# Debugging helper: show current working directory and whether assets exist (hidden under expander)
with st.expander("🔧 Debug info (optional)"):
    st.write("CWD:", str(Path.cwd()))
    for k, v in ASSETS.items():
        rp = _resolve_path(v)
        st.write(k, v, "→", str(rp) if rp else "NOT FOUND")

ASSETS = {
    "control_room": "data/images/control_room.jpg",
    "data_vault": "data/images/data_vault.jpg",
    "air_chamber": "data/images/air_chamber.jpg",
    "bit_bot": "data/images/bit_bot.jpg",
    "debug_chamber": "data/images/debug_chamber.jpg",
    "final_core": "data/images/final_core.jpg",
}

SOUNDS = {
    "tick": "data/sounds/tick.mp3",
    "alert": "data/sounds/alert.mp3",
    "success": "data/sounds/success.mp3",
}

# --- Robust asset loading ---

SEARCH_ROOTS = [Path.cwd(), Path.cwd() / "app", Path.cwd() / "src", Path.cwd() / "streamlit_app"]

def _resolve_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    p = Path(path_str)
    if p.exists():
        return p
    for root in SEARCH_ROOTS:
        cand = root / path_str
        if cand.exists():
            return cand
    return None

def _load_image(image_key: str) -> Image.Image:
    path = _resolve_path(ASSETS.get(image_key))
    if path and path.exists():
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            pass
    # Fallback: generate a simple placeholder so app never crashes
    img = Image.new("RGB", (1200, 700), (10, 25, 45))
    draw = ImageDraw.Draw(img)
    text = f"Missing image: {image_key}"
    draw.text((30, 30), text, fill=(160, 230, 255))
    return img

# Helper to display a scene background and title nicely

def scene_header(title: str, image_key: str) -> None:
    col_img, col_title = st.columns([3, 2])
    with col_img:
        st.image(_load_image(image_key), use_container_width=True)
    with col_title:
        st.markdown(f"# {title}")


# Safe int conversion

def to_int(x: str) -> int | None:
    try:
        return int(x.strip())
    except Exception:
        return None


# Safe float conversion

def to_float(x: str) -> float | None:
    try:
        return float(x.strip().replace(",", "."))
    except Exception:
        return None


# Normalize text for checking answers

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


# Inventory badge renderer

def render_inventory() -> None:
    inv = st.session_state.get("inventory", [])
    if not inv:
        st.info("**Inventory:** (empty)")
        return
    chips = " ".join([f"`{item}`" for item in inv])
    st.success(f"**Inventory:** {chips}")


# Progress renderer

def render_progress() -> None:
    total = 5
    solved = len(st.session_state.get("keys", set()))
    st.progress(solved/total)
    st.caption(f"Keys collected: {solved}/{total}")


# Play audio if the file exists. Streamlit does not autoplay; user can click play.

def audio_player(kind: str, label: str) -> None:
    path = SOUNDS.get(kind)
    if path:
        with st.expander(f"🔊 {label}"):
            st.audio(path)


# Simple timer widget (manual start). Returns seconds_remaining (can be None if not started)

def timer_widget(timer_key: str, seconds: int = 120) -> int | None:
    if f"{timer_key}_start" not in st.session_state:
        st.session_state[f"{timer_key}_start"] = None

    cols = st.columns([1, 2])
    with cols[0]:
        if st.button("⏱️ Start timer", key=f"start_{timer_key}"):
            st.session_state[f"{timer_key}_start"] = time.time()
    with cols[1]:
        if st.button("🔁 Reset timer", key=f"reset_{timer_key}"):
            st.session_state[f"{timer_key}_start"] = None

    start = st.session_state.get(f"{timer_key}_start")
    if start is None:
        st.info(f"Timer: {seconds} seconds (not started)")
        return None

    elapsed = int(time.time() - start)
    remaining = max(0, seconds - elapsed)
    st.write(f"⏳ Time left: **{remaining} s**")
    if remaining == 0:
        st.warning("⛔ Time's up! You can still try to answer.")
    return remaining


# Mark a key as collected and add an inventory item

def collect_key(key_name: str, item_label: str) -> None:
    st.session_state.setdefault("keys", set()).add(key_name)
    inv = st.session_state.setdefault("inventory", [])
    if item_label not in inv:
        inv.append(item_label)


# Reset story state

def reset_game() -> None:
    st.session_state.clear()


# Init session state
if "keys" not in st.session_state:
    st.session_state["keys"] = set()
if "inventory" not in st.session_state:
    st.session_state["inventory"] = []
if "scene" not in st.session_state:
    st.session_state["scene"] = 0  # 0..5


# --------------------------
# --- APP HEADER / NAV   ---
# --------------------------

left, mid, right = st.columns([2, 3, 1])
with left:
    st.markdown("## 🧠 Operation CodeQuest — Episode 1: The Lost Core")
with right:
    if st.button("🔄 Reset Game"):
        reset_game()
        st.rerun()

st.write("")
render_inventory()
render_progress()

st.divider()

# --------------------------
# --- STORY & SCENES     ---
# --------------------------

# Scene 0: Intro
if st.session_state["scene"] == 0:
    scene_header("Arrival at TechNova (Intro)", "control_room")

    st.markdown(
        """
        You are **Alex Byte**, the youngest computer scientist at **TechNova Labs**.
        As you step inside the futuristic glass building, alarms flash **red**.

        > **ALERT:** System breach detected. The main AI — *Project Nova* — is **locked**.
        > Only the smartest coder can recover it.

        **Mission:** Collect **5 digital keys** by solving puzzles in each room to reboot *Project Nova*.
        """
    )

    with st.expander("Optional: Play pressure sound (ticking)"):
        audio_player("tick", "Ticking clock")
        audio_player("alert", "Alert siren")

    if st.button("🚪 Enter the Control Room"):
        st.session_state["scene"] = 1
        st.rerun()

# Scene 1: Control Room — Pattern Enigma
elif st.session_state["scene"] == 1:
    scene_header("The Control Room — Pattern Enigma", "control_room")
    st.markdown(
        """
        The main terminal displays a blinking number pattern. Solve it to unlock the **first firewall**.
        """
    )

    timer_widget("ctrl_room", seconds=90)

    st.markdown("**Pattern:** 2, 4, 8, 16, ?")
    ans = st.text_input("What number comes next?", key="p1")

    if st.button("Submit answer", key="p1_submit"):
        if to_int(ans) == 32:
            st.success("Correct! Key #1 collected.")
            collect_key("key1", "Firewall Key")
            with st.expander("Mini‑Story Card #1"):
                st.write("✅ The first firewall falls. The AI whispers: *Good job, Alex. Hurry — someone is altering the code!*")
            audio_player("success", "Success chime")
            st.session_state["scene"] = 2
            st.rerun()
        else:
            st.error("Not quite. Hint: each number is **double** the previous.")

# Scene 2: Data Vault — Opposite Operations + fractions/multiplications
elif st.session_state["scene"] == 2:
    scene_header("The Data Vault — Opposite Operations", "data_vault")
    st.markdown("The vault requires you to pair operations with their opposites and evaluate some expressions.")

    timer_widget("vault", seconds=120)

    c1, c2, c3 = st.columns(3)
    with c1:
        add_op = st.selectbox("Opposite of Add", ["(choose)", "Multiply", "Square", "Subtract", "Divide", "Square Root"], index=0, key="op_add")
    with c2:
        mul_op = st.selectbox("Opposite of Multiply", ["(choose)", "Divide", "Subtract", "Square", "Square Root"], index=0, key="op_mul")
    with c3:
        sq_op = st.selectbox("Opposite of Square (x²)", ["(choose)", "Divide", "Square Root", "Subtract"], index=0, key="op_sq")

    st.markdown("#### Compute:")
    a = st.text_input("5² = ?", key="sq5")
    b = st.text_input("√49 = ?", key="rt49")
    c = st.text_input("9 × 8 = ?", key="mul98")
    d = st.text_input("Now divide that result by 8 = ?", key="div8")

    st.markdown("#### Bonus (fractions):")
    fr1 = st.text_input("What is 1/2 + 1/4 ? (give a fraction or decimal)", key="frac1")
    fr2 = st.text_input("Compute 3/5 of 20 = ?", key="frac2")

    if st.button("Unlock the vault"):
        ok_ops = (add_op == "Subtract" and mul_op == "Divide" and sq_op == "Square Root")
        ok_arith = (to_int(a) == 25 and to_int(b) == 7 and to_int(c) == 72 and to_int(d) == 9)

        # Fraction checks (accept decimals)
        def is_half_plus_quarter(s: str) -> bool:
            if not s: return False
            s2 = norm(s)
            return s2 in {"0.75", "0,75", "3/4", "3 / 4", "0.75 ", "0,75 "}
        def is_three_fifths_of_20(s: str) -> bool:
            v = to_float(s)
            return v == 12 if v is not None else False

        ok_frac = is_half_plus_quarter(fr1) and is_three_fifths_of_20(fr2)

        if ok_ops and ok_arith and ok_frac:
            st.success("Vault unlocked! Key #2 collected.")
            collect_key("key2", "Data Key")
            with st.expander("Mini‑Story Card #2"):
                st.write("⚙️ The vault slides open. Inside, a glowing cube reads: **AIR CHAMBER SYSTEM LOCKED**.")
            audio_player("success", "Success chime")
            st.session_state["scene"] = 3
            st.rerun()
        else:
            st.error("Some answers are off. Check operation opposites, arithmetic, and fractions.")

# Scene 3: Air Chamber — Volumes (cube and rectangular prism)
elif st.session_state["scene"] == 3:
    scene_header("The Air Chamber — Balancing Volumes", "air_chamber")
    st.markdown("Pressure is rising! Balance the air by calculating the correct **volumes**.")

    timer_widget("air", seconds=120)

    st.markdown("#### Cube")
    side = st.text_input("Cube side = 3 cm. Volume = ? (in cm³)", key="cube_vol")

    st.markdown("#### Rectangular Box")
    l = st.text_input("Length = 5 cm", key="box_l")
    w = st.text_input("Width = 2 cm", key="box_w")
    h = st.text_input("Height = 4 cm", key="box_h")

    st.markdown("#### Brain Teaser")
    bt = st.selectbox("If you double each dimension of a cube, the volume becomes...", ["(choose)", "2 times", "4 times", "8 times"], index=0, key="cube_bt")

    if st.button("Stabilize Air System"):
        ok_cube = to_int(side) == 27
        ok_box = (to_int(l) == 5 and to_int(w) == 2 and to_int(h) == 4)
        vol_box = 5 * 2 * 4 if ok_box else None
        ok_box_val = (vol_box == 40)
        ok_bt = (bt == "8 times")

        if ok_cube and ok_box_val and ok_bt:
            st.success("Air stabilized! Key #3 collected.")
            collect_key("key3", "Air Chip")
            with st.expander("Mini‑Story Card #3"):
                st.write("🌬️ The air flows smoothly. A metal card drops out: **ACCESS — Bit‑Bot Room**.")
            audio_player("success", "Success chime")
            st.session_state["scene"] = 4
            st.rerun()
        else:
            st.error("Recheck the cube volume (3×3×3), the box (L×W×H), and the brain teaser.")

# Scene 4: Bit‑Bot Room — Algorithmic Thinking
elif st.session_state["scene"] == 4:
    scene_header("Bit‑Bot Room — Guide the Robot", "bit_bot")
    st.markdown(
        """
        A tiny robot **Bit‑Bot** awaits your instructions to reach the **Core**.
        Write a simple plan as steps. Obstacles block some paths.
        """
    )

    timer_widget("bot", seconds=120)

    st.markdown("""
    Grid (conceptual 5×5): Start at bottom‑left → move to top‑right.
    Obstacles at cells (3,2) and (4,4) (you must *avoid* them).

    **Write steps like:** `right, right, up, up, right, up, up` (exact words, comma‑separated).
    """)
    path = st.text_input("Your instruction list:", placeholder="right, right, up, up, right, up, up", key="bot_path")

    # Accept a couple of valid alternative paths that avoid (3,2) and (4,4).
    VALID_PATHS = {
        norm("right, right, right, up, up, right, up, up"),
        norm("right, right, up, right, up, right, up, up"),
        norm("right, right, up, up, right, right, up, up"),
    }

    if st.button("Send to Bit‑Bot"):
        if norm(path) in VALID_PATHS:
            st.success("Bit‑Bot beeps happily and reaches the Core door. Key #4 collected.")
            collect_key("key4", "Robot Pass")
            with st.expander("Mini‑Story Card #4"):
                st.write("🤖 'Thanks, Alex!' says Bit‑Bot. 'But the mainframe code is broken. We must debug it before midnight!' ")
            audio_player("success", "Success chime")
            st.session_state["scene"] = 5
            st.rerun()
        else:
            st.error("Bit‑Bot hits an obstacle or doesn't reach the goal. Try a different sequence of rights/ups.")

# Scene 5: Debugging Chamber — Simple Python
elif st.session_state["scene"] == 5:
    scene_header("Debugging Chamber — Fix the Code", "debug_chamber")
    st.markdown("A small program prevents the final door from opening. Fix it and evaluate an expression.")

    timer_widget("debug", seconds=120)

    st.code("""
for i in range(3)
    print("CodeQuest")
""", language="python")

    bug_fix = st.text_input("What single character is missing after `range(3)`? (tip: punctuation)", key="bug_char")

    st.markdown("**Predict the output:** What does `print(2 + 3 * 4)` show?")
    out = st.text_input("Your answer:", key="order_ops")

    if st.button("Run Fix"):
        ok_bug = norm(bug_fix) in {":", "colon", " :"}
        ok_out = to_int(out) == 14
        if ok_bug and ok_out:
            st.success("Program fixed! Key #5 collected.")
            collect_key("key5", "Core Decoder")
            with st.expander("Mini‑Story Card #5"):
                st.write("💾 The console unlocks and reveals a final **pass‑phrase puzzle**.")
            audio_player("success", "Success chime")
            st.session_state["scene"] = 6
            st.rerun()
        else:
            st.error("Check the missing character (a colon) and order of operations (× before +).")

# Scene 6: Final — Reboot the Core (coherent ending)
elif st.session_state["scene"] == 6:
    scene_header("Final Core Room — Reboot Project Nova", "final_core")

    st.markdown(
        """
        You've collected all **5 keys**:
        - Firewall Key (pattern)
        - Data Key (opposites, arithmetic, fractions)
        - Air Chip (volumes)
        - Robot Pass (algorithm)
        - Core Decoder (debugging)

        The final console asks you to **enter the pass‑phrase** built from earlier clues:
        1) The pattern's next number (Control Room)
        2) The result of 3/5 of 20 (Data Vault)
        3) The cube volume (3 cm each side) (Air Chamber)
        4) The multiplication before division result (9×8) (Data Vault)

        **Format:** type the four numbers in order, separated by dashes (e.g., `A-B-C-D`).
        """
    )

    pp = st.text_input("Enter pass‑phrase:", placeholder="number-number-number-number", key="passphrase")

    if st.button("Reboot Project Nova"):
        # Correct sequence: 32 - 12 - 27 - 72
        if norm(pp) in {"32-12-27-72", "32 - 12 - 27 - 72"}:
            st.balloons()
            st.success("🎉 Project Nova reboots! You solved the entire mission.")
            with st.expander("Epilogue — Victory"):
                st.markdown(
                    """
                    **MISSION COMPLETE!**

                    Project Nova speaks: "**Alex Byte,** you've restored my core and protected TechNova.
                    Your logic, math, and coding skills saved the day. You are now promoted to
                    **Junior Chief of Cyber Adventures**. Prepare for your next mission: **The Mystery of the Binary Planet**."
                    """
                )
            audio_player("success", "Victory sound")
            st.markdown("---")
            st.markdown("Want to play again or let someone else try?")
            if st.button("Play Again from the Start"):
                reset_game()
                st.rerun()
        else:
            st.error("That's not the correct pass‑phrase. Revisit previous results and try again.")

# Fallback
else:
    st.error("Unknown scene. Click Reset.")
