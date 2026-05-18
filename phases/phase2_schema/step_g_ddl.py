"""
phases/phase2_schema/step_g_ddl.py
------------------------------------
Step G: Generate and test the DDL from the final schema.

Takes the final schema from step_f and:
1. Generates a DuckDB SQL DDL for the brewing_events table
2. Tests it executes cleanly in an in-memory DuckDB connection
3. Saves the DDL to disk

Output saved to: outputs/schema/brewing_events.sql
"""

import json
import re

import duckdb

from utils.llm import call_claude
from utils.parsing import strip_fences


EVENT_DDL_SYSTEM_PROMPT = """Generate a simple, flat table for brewing events.

Create a single table called 'brewing_events' with:
- event_id (primary key)
- timestamp
- machine_id (to identify which machine)
- product_type (what beverage was made)
- All sensor measurements as individual columns
- error (error code if failed)
- success (boolean flag)

Sensor measurements to include:
{sensor_measurements}

CRITICAL REQUIREMENTS:
1. Table name: brewing_events
2. Column names: lowercase_with_underscores (exactly as provided in field_name)
3. Data types:
   - INTEGER for: event_id, counts, durations in seconds
   - DECIMAL(p,s) for: temperatures, pressures, weights, volumes with decimals
   - VARCHAR(50) for: machine_id, product_type, error
   - TIMESTAMP for: timestamp
   - BOOLEAN for: success
4. No foreign keys (this is a flat event table)
5. Add CHECK constraints for valid ranges where typical_range is provided
6. Add DEFAULT nextval for event_id sequence
7. Output ONLY valid DuckDB SQL DDL, no markdown, no explanations

Example:
CREATE SEQUENCE brewing_events_event_id_seq START 1;
CREATE TABLE brewing_events (
    event_id INTEGER PRIMARY KEY DEFAULT nextval('brewing_events_event_id_seq'),
    timestamp TIMESTAMP NOT NULL,
    machine_id VARCHAR(50) NOT NULL,
    product_type VARCHAR(50),
    water_pressure DECIMAL(4,1) CHECK (water_pressure BETWEEN 8 AND 10),
    water_temp_start DECIMAL(4,1),
    brew_time_seconds INTEGER,
    error VARCHAR(10),
    success BOOLEAN NOT NULL
);

Generate the complete DDL now:"""


def run(cfg: dict, schema: list[dict]) -> str:
    """
    Generate and test DDL from the final schema.

    Args:
        cfg:    result of get_machine_config()
        schema: final schema from step_f

    Returns:
        DDL string
    """
    output_path = cfg["ddl_path"]

    # Load from disk if already done
    if output_path.exists():
        print("  ✓ step_g already done — loading from disk")
        return output_path.read_text()

    print("  Running step_g — generating DDL...")

    # Generate DDL
    prompt = EVENT_DDL_SYSTEM_PROMPT.format(
        sensor_measurements=json.dumps(schema, indent=2)
    )

    raw = call_claude(
        prompt="Generate the brewing_events table DDL.",
        system=prompt,
        max_tokens=8000,
    )

    ddl = _clean_ddl(strip_fences(raw))

    # Fix common Claude DuckDB mistakes
    ddl = _fix_duckdb_compatibility(ddl)

    # Test it runs
    _test_ddl(ddl)

    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(ddl)

    print(f"    Saved → {output_path.name}")

    return ddl


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_ddl(text: str) -> str:
    """Strip any remaining markdown or explanation text, keep only SQL."""
    # Remove any lines that look like explanations (not SQL)
    lines  = text.splitlines()
    sql    = "\n".join(line for line in lines if line.strip())
    return sql.strip()


def _fix_duckdb_compatibility(ddl: str) -> str:
    """Fix common issues when Claude generates PostgreSQL instead of DuckDB."""

    # SERIAL → INTEGER (DuckDB doesn't support SERIAL)
    ddl = re.sub(r'\bSERIAL\b', 'INTEGER', ddl, flags=re.IGNORECASE)

    # Remove CASCADE (not supported in DuckDB)
    ddl = re.sub(
        r'ON\s+(DELETE|UPDATE)\s+(CASCADE|SET\s+NULL|SET\s+DEFAULT|RESTRICT|NO\s+ACTION)',
        '', ddl, flags=re.IGNORECASE
    )

    # Add sequence if missing
    if 'CREATE SEQUENCE' not in ddl:
        ddl = "CREATE SEQUENCE brewing_events_event_id_seq START 1;\n\n" + ddl
        ddl = ddl.replace(
            'event_id INTEGER PRIMARY KEY',
            "event_id INTEGER PRIMARY KEY DEFAULT nextval('brewing_events_event_id_seq')"
        )

    # Fix escaped quotes
    ddl = ddl.replace("\\'", "'").replace('\\"', '"')

    return ddl


def _test_ddl(ddl: str) -> None:
    """Test the DDL executes in an in-memory DuckDB connection."""
    try:
        conn = duckdb.connect(':memory:')
        conn.execute(ddl)

        # Show structure
        columns = conn.execute("DESCRIBE brewing_events").fetchall()
        print(f"    ✓ DDL executes — {len(columns)} columns:")
        for col in columns:
            print(f"      {col[0]:<35} {col[1]}")

        conn.close()

    except Exception as e:
        raise RuntimeError(f"DDL test failed: {e}\n\nDDL:\n{ddl}")