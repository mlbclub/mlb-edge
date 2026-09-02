from __future__ import annotations

import os
from typing import Any

import requests


class SupabaseAuth:
    """Tiny Supabase Auth/PostgREST client for Streamlit.

    This intentionally uses requests directly so the public app does not need the
    heavier supabase Python SDK. The anon key is safe for browser-facing apps only
    when Row Level Security (RLS) policies are enabled in Supabase.
    """

    def __init__(self, url: str | None = None, anon_key: str | None = None, timeout: int = 20):
        self.url = (url or os.getenv("SUPABASE_URL") or "").rstrip("/")
        self.anon_key = anon_key or os.getenv("SUPABASE_ANON_KEY") or ""
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.url and self.anon_key)

    def _headers(self, token: str | None = None, prefer: str | None = None) -> dict[str, str]:
        h = {
            "apikey": self.anon_key,
            "Content-Type": "application/json",
        }
        if token:
            h["Authorization"] = f"Bearer {token}"
        if prefer:
            h["Prefer"] = prefer
        return h

    @staticmethod
    def _error(r: requests.Response) -> str:
        try:
            body = r.json()
            if isinstance(body, dict):
                return str(body.get("msg") or body.get("message") or body.get("error_description") or body.get("error") or body)
            return str(body)
        except Exception:
            return r.text or f"HTTP {r.status_code}"

    def sign_up(self, email: str, password: str, initial_seed: float = 3_000_000) -> dict[str, Any]:
        r = requests.post(
            f"{self.url}/auth/v1/signup",
            headers=self._headers(),
            json={"email": email, "password": password, "data": {"initial_seed": float(initial_seed)}},
            timeout=self.timeout,
        )
        if not r.ok:
            raise RuntimeError(self._error(r))
        return r.json()

    def sign_in(self, email: str, password: str) -> dict[str, Any]:
        r = requests.post(
            f"{self.url}/auth/v1/token?grant_type=password",
            headers=self._headers(),
            json={"email": email, "password": password},
            timeout=self.timeout,
        )
        if not r.ok:
            raise RuntimeError(self._error(r))
        return r.json()

    def get_profile(self, access_token: str, user_id: str) -> dict[str, Any] | None:
        r = requests.get(
            f"{self.url}/rest/v1/profiles",
            headers=self._headers(access_token),
            params={"user_id": f"eq.{user_id}", "select": "user_id,seed,unit_stake,created_at,updated_at", "limit": 1},
            timeout=self.timeout,
        )
        if not r.ok:
            raise RuntimeError(self._error(r))
        rows = r.json()
        return rows[0] if rows else None

    def upsert_profile(self, access_token: str, user_id: str, seed: float, unit_stake: float) -> dict[str, Any]:
        payload = {"user_id": user_id, "seed": float(seed), "unit_stake": float(unit_stake)}
        r = requests.post(
            f"{self.url}/rest/v1/profiles?on_conflict=user_id",
            headers=self._headers(access_token, "resolution=merge-duplicates,return=representation"),
            json=payload,
            timeout=self.timeout,
        )
        if not r.ok:
            raise RuntimeError(self._error(r))
        rows = r.json()
        return rows[0] if rows else payload

    def get_public_site_picks(self, limit: int = 300) -> list[dict[str, Any]]:
        r = requests.get(
            f"{self.url}/rest/v1/site_picks",
            headers=self._headers(),
            params={
                "select": "*",
                "order": "pick_date.desc,game_time.desc",
                "limit": int(limit),
            },
            timeout=self.timeout,
        )
        if not r.ok:
            raise RuntimeError(self._error(r))
        return r.json()

    def get_user_results(self, access_token: str, user_id: str, limit: int = 200) -> list[dict[str, Any]]:
        r = requests.get(
            f"{self.url}/rest/v1/participations",
            headers=self._headers(access_token),
            params={
                "user_id": f"eq.{user_id}",
                "select": "id,pick_date,created_at,participation_picks(id,locked_odds,stake,result,pnl,site_picks(*))",
                "order": "pick_date.desc",
                "limit": int(limit),
            },
            timeout=self.timeout,
        )
        if not r.ok:
            raise RuntimeError(self._error(r))
        return r.json()
