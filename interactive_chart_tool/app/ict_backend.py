from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Body, Header, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from pathlib import Path
from typing import Dict, Set, List
from collections import deque
import uuid, time, asyncio, re, math

app = FastAPI()
app.mount("/public", StaticFiles(directory=Path(__file__).parent / "static"), name="public")

# ----------------------- Defaults / Tuning -----------------------
MAX_HISTORY: int      = 5000
SNAPSHOT_LIMIT: int   = 600

# 250ms Broadcast (UI fluent), Price movement only after trdes
BROADCAST_HZ: float   = 4.0

# Volatilität / Impact
ALPHA: float          = 0.12     # Threshhold for impact for steps (±12%)
IMPACT_K: float       = 4.0      # scale for net-orderflow (smaller = more sensitive)
POWER: float          = 1.35     # Super-linearer amplifierer (>=1.0); large waves -> greater effect
LAMBDA: float         = 0.60     # Easing towards target price direction (0..1)
MAX_MOVE_BP: float    = 400.0    # Cap for updates in basepoints (e.g. 400bp = 4.00%)

# Spiel-Setup
START_PRICE: float    = 100.00
START_CASH: float     = 1000.0
TRADE_SIZE: int       = 1
ACTIVE_WINDOW_MS: int = 1000

# Auth
ADMIN_TOKEN: str      = ""   # <- anpassen
PLAYER_PASSWORD: str  = ""   # <- anpassen

USERNAME_MAXLEN = 12
USERNAME_REGEX  = re.compile(r"^.{1,12}$", re.DOTALL)  # entire German keyboard

# --------------------- In-Memory State -----------------------
STATE: Dict = {
    "session": {
        "id": "demo",
        "price": START_PRICE,
        "alpha": ALPHA,
        "started_at": int(time.time() * 1000),
    },
    "players": {},        # pid -> {"username": str, "cash": float, "shares": int}
    "history": [],        # only points after trades
    "clients": set(),
    "tasks": [],
    "totals": {"buys": 0, "sells": 0},
    "recent": {"buys": deque(), "sells": deque()},
    "paused": False,
    "bucket": {"buys": 0, "sells": 0},   # Orderflow since last update
}

def _now_ms() -> int:
    return int(time.time() * 1000)

def _append_history(price: float) -> Dict:
    pt = {"ts": _now_ms(), "price": round(price, 2)}  # ts stays, client uses index
    STATE["history"].append(pt)
    if len(STATE["history"]) > MAX_HISTORY:
        del STATE["history"][: len(STATE["history"]) - MAX_HISTORY]
    return pt

if not STATE["history"]:
    _append_history(STATE["session"]["price"])

# ------------------------- Models --------------------------
class Action(BaseModel):
    direction: str
    player_id: str

class JoinRequest(BaseModel):
    username: str
    password: str

# ------------------------- Helpers --------------------------
def ensure_player(pid: str) -> None:
    if pid not in STATE["players"]:
        STATE["players"][pid] = {"username": "unknown", "cash": START_CASH, "shares": 0}

def _prune_recent(now_ms: int):
    cutoff = now_ms - ACTIVE_WINDOW_MS
    dq_b, dq_s = STATE["recent"]["buys"], STATE["recent"]["sells"]
    while dq_b and dq_b[0] < cutoff: dq_b.popleft()
    while dq_s and dq_s[0] < cutoff: dq_s.popleft()

def _calc_total(p: Dict, price: float) -> float:
    return round(p["cash"] + p["shares"] * price, 2)

def _stats(now_ms: int) -> Dict:
    _prune_recent(now_ms)
    players = STATE["players"]

    if players:
        totals = {pid: _calc_total(p, STATE["session"]["price"]) for pid, p in players.items()}
        cashes = {pid: p["cash"] for pid, p in players.items()}

        max_cash_pid = max(cashes, key=cashes.get)
        min_cash_pid = min(cashes, key=cashes.get)
        max_total_pid = max(totals, key=totals.get)
        min_total_pid = min(totals, key=totals.get)

        return {
            "clients": len(STATE["clients"]),
            "active": { "buys": len(STATE["recent"]["buys"]), "sells": len(STATE["recent"]["sells"]) },
            "totals": STATE["totals"],
            "max_cash":  {"username": players[max_cash_pid]["username"], "value": round(cashes[max_cash_pid], 2)},
            "min_cash":  {"username": players[min_cash_pid]["username"], "value": round(cashes[min_cash_pid], 2)},
            "max_total": {"username": players[max_total_pid]["username"], "value": round(totals[max_total_pid], 2)},
            "min_total": {"username": players[min_total_pid]["username"], "value": round(totals[min_total_pid], 2)},
        }
    else:
        return {
            "clients": len(STATE["clients"]),
            "active": { "buys": 0, "sells": 0 },
            "totals": STATE["totals"],
            "max_cash": None, "min_cash": None,
            "max_total": None, "min_total": None,
        }

def latest_snapshot() -> Dict:
    hist: List[Dict] = STATE["history"][-SNAPSHOT_LIMIT:]
    now = _now_ms()
    return {
        "type": "snapshot",
        "price": STATE["session"]["price"],
        "history": hist,   # only points after trades
        "count": len(hist),
        "ts": now,
        "stats": _stats(now),
        "paused": STATE["paused"],
    }

async def broadcast_snapshot() -> None:
    msg = latest_snapshot()
    dead: Set[WebSocket] = set()
    for ws in list(STATE["clients"]):
        try:
            await ws.send_json(msg)
        except Exception:
            dead.add(ws)
    for ws in dead:
        STATE["clients"].discard(ws)

def _price_from_bucket() -> bool:
    """
    Bewegt den Preis nur, wenn seit der letzten Aktualisierung Orderflow vorhanden ist.
    Returns True, wenn der Preis (und History) aktualisiert wurden.
    """
    if STATE["paused"]:
        return False

    buys  = STATE["bucket"]["buys"]
    sells = STATE["bucket"]["sells"]

    if buys == 0 and sells == 0:
        return False  # no new history, chart stays static

    # Reset buckets
    STATE["bucket"]["buys"] = 0
    STATE["bucket"]["sells"] = 0

    net_events = buys - sells
    net_volume = net_events * max(1, TRADE_SIZE)

    price = STATE["session"]["price"]

    # Impact (super-linear, restricted by ALPHA)
    raw = net_volume / max(1e-6, IMPACT_K)
    amp = math.tanh(abs(raw) ** max(1.0, POWER))        # 0..1
    sign = 1.0 if raw >= 0 else -1.0
    impact = ALPHA * amp * sign                         # [-ALPHA, +ALPHA]

    target = price * (1.0 + impact)
    new_price = price * (1.0 - LAMBDA) + target * LAMBDA

    # Cap in basepoints
    max_delta = price * (MAX_MOVE_BP / 10000.0)
    new_price = max(price - max_delta, min(price + max_delta, new_price))

    STATE["session"]["price"] = round(new_price, 2)
    _append_history(STATE["session"]["price"])
    return True

async def broadcaster_loop():
    # 250 ms intervall – always snapshot, but price only with orderflow
    try:
        while True:
            await asyncio.sleep(1.0 / max(0.1, BROADCAST_HZ))
            _ = _price_from_bucket()
            await broadcast_snapshot()
    except asyncio.CancelledError:
        return

def _restart_broadcaster():
    for t in STATE["tasks"]:
        t.cancel()
    STATE["tasks"].clear()
    if not STATE["paused"]:
        task = asyncio.create_task(broadcaster_loop())
        STATE["tasks"].append(task)

# -------------------------- Redirect ------------------------
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/public/index.html", status_code=307)

# -------------------------- REST ----------------------------
@app.post("/api/v1/join")
def join(req: JoinRequest):
    if req.password != PLAYER_PASSWORD:
        raise HTTPException(status_code=403, detail="Invalid password")
    uname = req.username.strip()
    if not USERNAME_REGEX.match(uname):
        raise HTTPException(status_code=400, detail="Username invalid or too long (max 12)")
    pid = str(uuid.uuid4())
    STATE["players"][pid] = {"username": uname, "cash": START_CASH, "shares": 0}
    return {"player_id": pid, "session_id": STATE["session"]["id"], "username": uname}

@app.get("/api/v1/state")
def get_state(player_id: str):
    ensure_player(player_id)
    s = STATE["session"]
    p = STATE["players"][player_id]
    total = _calc_total(p, s["price"])
    can_buy  = (not STATE["paused"]) and (p["cash"] >= s["price"] * TRADE_SIZE)
    can_sell = (not STATE["paused"]) and (p["shares"] >= TRADE_SIZE)
    return {"price": s["price"], "portfolio": p, "total": total, "can_buy": can_buy, "can_sell": can_sell, "paused": STATE["paused"]}

@app.get("/api/v1/history")
def get_history(limit: int = Query(600, ge=1, le=MAX_HISTORY)):
    hist = STATE["history"][-limit:]
    return {"history": hist, "count": len(hist)}

@app.post("/api/v1/action")
async def do_action(a: Action):
    direction = a.direction.upper()
    ensure_player(a.player_id)

    if STATE["paused"]:
        return {"ok": False, "error": "paused"}

    s = STATE["session"]
    p = STATE["players"][a.player_id]
    price_now = s["price"]

    if direction == "BUY":
        cost = price_now * TRADE_SIZE
        if p["cash"] < cost:
            return {"ok": False, "error": "insufficient_cash"}
        p["cash"]  -= cost
        p["shares"] += TRADE_SIZE
        STATE["totals"]["buys"] += 1
        STATE["recent"]["buys"].append(_now_ms())
        STATE["bucket"]["buys"] += 1

    elif direction == "SELL":
        if p["shares"] < TRADE_SIZE:
            return {"ok": False, "error": "insufficient_shares"}
        proceeds   = price_now * TRADE_SIZE
        p["shares"] -= TRADE_SIZE
        p["cash"]   += proceeds
        STATE["totals"]["sells"] += 1
        STATE["recent"]["sells"].append(_now_ms())
        STATE["bucket"]["sells"] += 1

    else:
        return {"ok": False, "error": "bad_direction"}

    # Instant price/history update & broadcast (no time delays)
    if _price_from_bucket():
        await broadcast_snapshot()

    return {"ok": True, "price": s["price"], "portfolio": p, "total": _calc_total(p, s["price"])}

# ------------------------ WebSocket -------------------------
@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    STATE["clients"].add(websocket)
    try:
        await websocket.send_json(latest_snapshot())
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=120.0)
            except asyncio.TimeoutError:
                pass
    except WebSocketDisconnect:
        pass
    finally:
        STATE["clients"].discard(websocket)

# ------------------------ Lifecycle -------------------------
@app.on_event("startup")
async def on_startup():
    task = asyncio.create_task(broadcaster_loop())
    STATE["tasks"].append(task)

@app.on_event("shutdown")
async def on_shutdown():
    for t in STATE["tasks"]:
        t.cancel()
    await asyncio.gather(*STATE["tasks"], return_exceptions=True)

# ------------------------ Healthcheck -----------------------
@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "price": STATE["session"]["price"],
        "history_len": len(STATE["history"]),
        "clients": len(STATE["clients"]),
        "hz": BROADCAST_HZ,
        "totals": STATE["totals"],
        "paused": STATE["paused"],
        "bucket": STATE["bucket"],
        "alpha": ALPHA, "impact_k": IMPACT_K, "power": POWER, "lambda": LAMBDA, "max_move_bp": MAX_MOVE_BP
    }

# ------------------------ Admin Endpoints -------------------
def check_admin(x_token: str = Header(...)):
    if x_token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Not authorized")

@app.post("/admin/reset")
def admin_reset(dep=Depends(check_admin)):
    STATE["session"]["price"] = START_PRICE
    STATE["session"]["alpha"] = ALPHA
    STATE["session"]["started_at"] = int(time.time() * 1000)

    STATE["history"].clear()
    _append_history(START_PRICE)

    STATE["players"].clear()
    STATE["totals"] = {"buys": 0, "sells": 0}
    STATE["recent"] = {"buys": deque(), "sells": deque()}
    STATE["bucket"] = {"buys": 0, "sells": 0}

    asyncio.create_task(broadcast_snapshot())
    return {"ok": True, "msg": "Session reset"}

@app.post("/admin/update")
def admin_update(
    alpha: float = Body(...),
    start_price: float = Body(...),
    start_cash: float = Body(...),
    trade_size: int = Body(...),
    broadcast_hz: float = Body(...),
    impact_k: float = Body(...),
    power: float = Body(...),
    lambda_: float = Body(..., embed=True),
    max_move_bp: float = Body(...),
    dep=Depends(check_admin)
):
    global ALPHA, START_PRICE, START_CASH, TRADE_SIZE, BROADCAST_HZ, IMPACT_K, POWER, LAMBDA, MAX_MOVE_BP
    ALPHA = float(alpha)
    START_PRICE = float(start_price)
    START_CASH = float(start_cash)
    TRADE_SIZE = int(trade_size)
    BROADCAST_HZ = float(broadcast_hz)
    IMPACT_K = float(impact_k)
    POWER = float(power)
    LAMBDA = float(lambda_)
    MAX_MOVE_BP = float(max_move_bp)
    STATE["session"]["alpha"] = ALPHA
    asyncio.create_task(broadcast_snapshot())
    return {"ok": True, "msg": "Parameters updated"}

@app.post("/admin/pause")
def admin_pause(dep=Depends(check_admin)):
    STATE["paused"] = True
    _restart_broadcaster()
    asyncio.create_task(broadcast_snapshot())
    return {"ok": True, "msg": "Broadcast paused"}

@app.post("/admin/resume")
async def admin_resume(dep=Depends(check_admin)):
    STATE["paused"] = False
    _restart_broadcaster()
    asyncio.create_task(broadcast_snapshot())
    return {"ok": True, "msg": "Broadcast resumed"}
