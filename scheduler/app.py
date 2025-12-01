from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Optional, List
from datetime import datetime
import sqlite3, json, os

DB_PATH = os.getenv("DB_PATH", "/data/scheduler.sqlite")

app = FastAPI(title="Scheduler API", version="0.1.0")

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""
    CREATE TABLE IF NOT EXISTS schedules (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      cron TEXT NOT NULL,
      timezone TEXT NOT NULL,
      config TEXT NOT NULL,
      enabled INTEGER NOT NULL DEFAULT 1,
      last_run_at TEXT
    )
    """)
    con.commit()
    con.close()

init_db()

class ScheduleIn(BaseModel):
    cron: str = Field(..., description="m H dom mon dow")
    timezone: str = "America/Chicago"
    config: dict[str, Any]
    enabled: bool = True

class ScheduleOut(BaseModel):
    id: int
    cron: str
    timezone: str
    config: dict[str, Any]
    enabled: bool
    last_run_at: Optional[str] = None

@app.get("/api/schedules", response_model=List[ScheduleOut])
def list_schedules():
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("SELECT id, cron, timezone, config, enabled, last_run_at FROM schedules ORDER BY id DESC").fetchall()
    con.close()
    out = []
    for r in rows:
        out.append(ScheduleOut(
            id=r[0], cron=r[1], timezone=r[2],
            config=json.loads(r[3]),
            enabled=bool(r[4]),
            last_run_at=r[5]
        ))
    return out

@app.post("/api/schedules", response_model=ScheduleOut)
def create_schedule(s: ScheduleIn):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO schedules (cron, timezone, config, enabled, last_run_at) VALUES (?,?,?,?,?)",
        (s.cron, s.timezone, json.dumps(s.config), 1 if s.enabled else 0, None)
    )
    sid = cur.lastrowid
    con.commit()
    con.close()
    return ScheduleOut(id=sid, cron=s.cron, timezone=s.timezone, config=s.config, enabled=s.enabled, last_run_at=None)

@app.delete("/api/schedules/{sid}")
def delete_schedule(sid: int):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("DELETE FROM schedules WHERE id=?", (sid,))
    con.commit()
    con.close()
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Not found")
    return {"status": "deleted", "id": sid}
