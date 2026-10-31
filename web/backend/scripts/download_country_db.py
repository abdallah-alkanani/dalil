"""Download DB-IP's free IP-to-country database (CC BY 4.0, https://db-ip.com) for
``GEOIP_DB``. It is published monthly; this takes the current month's, or last
month's when the current one is not out yet.

    python scripts/download_country_db.py /opt/geo/country.mmdb
"""
import datetime as dt
import gzip
import sys
import urllib.error
import urllib.request
from pathlib import Path

URL = "https://download.db-ip.com/free/dbip-country-lite-{month}.mmdb.gz"


def main() -> None:
    out = Path(sys.argv[1])
    today = dt.date.today()
    months = [today.strftime("%Y-%m"), (today.replace(day=1) - dt.timedelta(days=1)).strftime("%Y-%m")]
    for month in months:
        try:
            # DB-IP refuses Python's default User-Agent.
            request = urllib.request.Request(URL.format(month=month), headers={"User-Agent": "dalil"})
            with urllib.request.urlopen(request, timeout=120) as resp:
                data = gzip.decompress(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                continue
            raise
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        print(f"wrote {out} ({month}, {len(data) / 1e6:.1f} MB)")
        return
    raise SystemExit(f"no DB-IP country database for {' or '.join(months)}")


if __name__ == "__main__":
    main()
