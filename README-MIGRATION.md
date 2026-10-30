# RevenueGuard-AI PostgreSQL migration patch

This patch changes `web_app/app.py` to use PostgreSQL when `DATABASE_URL` is set and SQLite otherwise. It does not connect to Neon automatically unless you set `DATABASE_URL` in the environment.

## Apply locally
1. Keep your existing `app.py`, `app_sqlite_backup.py`, and `revenueguard.db` unchanged as backups.
2. Copy the migrated `web_app/app.py` and `web_app/requirements.txt` into your project (or compare them first).
3. Activate your existing virtual environment and run `python -m pip install -r requirements.txt`.
4. For SQLite local testing, leave `DATABASE_URL` unset and run `python -m py_compile app.py`, then `python app.py`.
5. For Neon, set `DATABASE_URL` in the process environment using the connection string copied from Neon. Never commit it to Git or paste it into chat. Use the Neon pooled URL for a web service unless your provider guidance says otherwise.

## Important data note
The migration creates compatible tables in whichever database is selected. It does **not** automatically copy existing SQLite customer rows into PostgreSQL. Import the CSV again into Neon after deploying, or perform a separate verified data migration if the SQLite data must be preserved. Do not run a data migration against production until you have confirmed the target project and taken a backup.

## Render
In Render → service → Environment, add `DATABASE_URL` with the Neon connection string and `SECRET_KEY` with a random secret. Save only after reviewing the values. This patch has not been deployed or pushed.
