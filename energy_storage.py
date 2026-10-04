"""
energy_storage.py — SQLite audit trail for PS-13 FedGuard energy system.

Tables:
  energy_local_rounds   — per-node training results
  energy_federation_rounds — FedAvg aggregation runs
  energy_readings       — per-node sensor readings (for dashboard replay)
  anomaly_events        — logged anomaly detections
"""

import os
import sqlite3
import pandas as pd
from datetime import datetime

DB_NAME = "energy_guard.db"


def _conn():
    return sqlite3.connect(DB_NAME)


def init_db():
    con = _conn()
    c = con.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS energy_local_rounds (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            node      TEXT,
            accuracy  REAL,
            n_samples INTEGER,
            baseline_power_w        REAL,
            baseline_vibration_rms  REAL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS energy_federation_rounds (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT,
            nodes         TEXT,
            total_samples INTEGER
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS energy_readings (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp             TEXT,
            node                  TEXT,
            power_w               REAL,
            vibration_rms         REAL,
            energy_kwh_step       REAL,
            cost_inr_step         REAL,
            co2_kg_step           REAL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS anomaly_events (
            id                              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp                       TEXT,
            node                            TEXT,
            severity                        TEXT,
            likely_cause                    TEXT,
            suggested_action                TEXT,
            estimated_monthly_savings_inr   REAL,
            estimated_monthly_co2_reduction_kg REAL,
            power_w                         REAL,
            vibration_rms                   REAL
        )
    """)

    con.commit()
    con.close()


# ─── Writers ──────────────────────────────────────────────────────────────────

def log_energy_round(node, accuracy, n_samples, baseline_power_w, baseline_vib):
    init_db()
    con = _conn()
    con.execute(
        "INSERT INTO energy_local_rounds "
        "(timestamp,node,accuracy,n_samples,baseline_power_w,baseline_vibration_rms) "
        "VALUES (?,?,?,?,?,?)",
        (_now(), node, accuracy, n_samples, baseline_power_w, baseline_vib)
    )
    con.commit(); con.close()


def log_federation_round(nodes: list, total_samples: int):
    init_db()
    con = _conn()
    con.execute(
        "INSERT INTO energy_federation_rounds (timestamp,nodes,total_samples) VALUES (?,?,?)",
        (_now(), ",".join(nodes), total_samples)
    )
    con.commit(); con.close()


def log_energy_reading(node, power_w, vibration_rms,
                       energy_kwh_step, cost_inr_step, co2_kg_step):
    init_db()
    con = _conn()
    con.execute(
        "INSERT INTO energy_readings "
        "(timestamp,node,power_w,vibration_rms,energy_kwh_step,cost_inr_step,co2_kg_step) "
        "VALUES (?,?,?,?,?,?,?)",
        (_now(), node, power_w, vibration_rms,
         energy_kwh_step, cost_inr_step, co2_kg_step)
    )
    con.commit(); con.close()


def log_anomaly_event(node, severity, likely_cause, suggested_action,
                      savings_inr, co2_reduction, power_w, vibration_rms):
    init_db()
    con = _conn()
    con.execute(
        "INSERT INTO anomaly_events "
        "(timestamp,node,severity,likely_cause,suggested_action,"
        "estimated_monthly_savings_inr,estimated_monthly_co2_reduction_kg,"
        "power_w,vibration_rms) VALUES (?,?,?,?,?,?,?,?,?)",
        (_now(), node, severity, likely_cause, suggested_action,
         savings_inr, co2_reduction, power_w, vibration_rms)
    )
    con.commit(); con.close()


# ─── Readers ──────────────────────────────────────────────────────────────────

def get_local_rounds() -> pd.DataFrame:
    init_db()
    con = _conn()
    df = pd.read_sql_query(
        "SELECT * FROM energy_local_rounds ORDER BY id DESC", con)
    con.close(); return df


def get_federation_rounds() -> pd.DataFrame:
    init_db()
    con = _conn()
    df = pd.read_sql_query(
        "SELECT * FROM energy_federation_rounds ORDER BY id DESC", con)
    con.close(); return df


def get_energy_readings(node=None, limit=2000) -> pd.DataFrame:
    init_db()
    con = _conn()
    if node:
        df = pd.read_sql_query(
            "SELECT * FROM energy_readings WHERE node=? ORDER BY id ASC LIMIT ?",
            con, params=(node, limit))
    else:
        df = pd.read_sql_query(
            "SELECT * FROM energy_readings ORDER BY id ASC LIMIT ?",
            con, params=(limit,))
    con.close(); return df


def get_anomaly_events(node=None) -> pd.DataFrame:
    init_db()
    con = _conn()
    if node:
        df = pd.read_sql_query(
            "SELECT * FROM anomaly_events WHERE node=? ORDER BY id DESC", con, params=(node,))
    else:
        df = pd.read_sql_query(
            "SELECT * FROM anomaly_events ORDER BY id DESC", con)
    con.close(); return df


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
