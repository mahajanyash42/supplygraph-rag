"""
generate_data.py

Creates a small, realistic-looking supply chain dataset as plain CSV files.
No database involved yet -- just data on disk we can inspect by eye.

Run with:
    python generate_data.py
"""

import csv
import os
import random

random.seed(42)  # same "random" data every time we run this -- makes debugging sane

OUT_DIR = os.path.join(os.path.dirname(__file__), "csv")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Countries
# ---------------------------------------------------------------------------
COUNTRIES = [
    ("C01", "Vietnam"),
    ("C02", "Taiwan"),
    ("C03", "Mexico"),
    ("C04", "Germany"),
    ("C05", "South Korea"),
    ("C06", "India"),
]

# ---------------------------------------------------------------------------
# 2. Facilities -- each one sits in exactly one country
# ---------------------------------------------------------------------------
FACILITIES = [
    ("F01", "Vietnam Battery Plant", "C01"),
    ("F02", "Vietnam Assembly Plant", "C01"),
    ("F03", "Taiwan Chip Fab", "C02"),
    ("F04", "Mexico Wiring Plant", "C03"),
    ("F05", "Germany Precision Parts", "C04"),
    ("F06", "South Korea Display Plant", "C05"),
    ("F07", "Vietnam Lithium Mine", "C01"),
    ("F08", "India Casing Plant", "C06"),
    ("F09", "Mexico Chip Sub-fab", "C03"),
]

# ---------------------------------------------------------------------------
# 3. Suppliers -- each one operates at exactly one facility, and has a tier
#    tier 1 = supplies a component directly used in a product
#    tier 2/3 = supplies materials to another (higher-tier) supplier
# ---------------------------------------------------------------------------
SUPPLIERS = [
    ("S01", "PowerCell Inc.", 1, "F01"),      # tier-1: makes the battery
    ("S02", "Northline Chips", 1, "F03"),     # tier-1: makes the chip
    ("S03", "Delta Wiring Co.", 1, "F04"),    # tier-1: makes the wiring harness
    ("S04", "Vantage Displays", 1, "F06"),    # tier-1: makes the screen
    ("S05", "Lithium Mine Corp.", 3, "F07"),  # tier-3: feeds PowerCell
    ("S06", "Del Rio Semiconductors", 2, "F09"),  # tier-2: feeds Northline Chips
    ("S07", "Ganges Casings Ltd.", 1, "F08"),     # tier-1: makes the phone casing
]

# ---------------------------------------------------------------------------
# 4. Sub-supplier relationships -- who feeds who (child -> parent)
# ---------------------------------------------------------------------------
SUB_SUPPLIER_EDGES = [
    ("S05", "S01"),  # Lithium Mine Corp. feeds into PowerCell Inc.
    ("S06", "S02"),  # Del Rio Semiconductors feeds into Northline Chips
]

# ---------------------------------------------------------------------------
# 5. Components -- each one is made by exactly one tier-1 supplier
# ---------------------------------------------------------------------------
COMPONENTS = [
    ("CM1", "Battery Cell", "S01"),
    ("CM2", "Processor Chip", "S02"),
    ("CM3", "Wiring Harness", "S03"),
    ("CM4", "Display Panel", "S04"),
    ("CM5", "Phone Casing", "S07"),
]

# ---------------------------------------------------------------------------
# 6. Products -- each one requires a few components
# ---------------------------------------------------------------------------
PRODUCTS = [
    ("P01", "Aurora Smartphone"),
    ("P02", "NimbusBook Laptop"),
    ("P03", "Orion Tablet"),
]

REQUIRES_EDGES = [
    ("P01", "CM1"),  # Smartphone needs a battery
    ("P01", "CM2"),  # Smartphone needs a chip
    ("P01", "CM4"),  # Smartphone needs a display
    ("P02", "CM1"),  # Laptop needs a battery
    ("P02", "CM2"),  # Laptop needs a chip
    ("P02", "CM3"),  # Laptop needs wiring
    ("P03", "CM2"),  # Tablet needs a chip
    ("P03", "CM4"),  # Tablet needs a display
    ("P03", "CM5"),  # Tablet needs a casing
]

# ---------------------------------------------------------------------------
# 7. Events -- disruptions, each impacting one facility
# ---------------------------------------------------------------------------
EVENTS = [
    (
        "E01",
        "Flood at Vietnam Lithium Mine",
        "Flood",
        "Heavy flooding has shut down the Vietnam Lithium Mine. Officials estimate "
        "3-4 weeks before operations resume. A backup supply route is being "
        "explored but is not yet confirmed.",
        "F07",
    ),
    (
        "E02",
        "Strike at Taiwan Chip Fab",
        "Strike",
        "Workers at the Taiwan Chip Fab have gone on strike over wage disputes. "
        "Production is at roughly 10% of normal capacity. Similar strikes in the "
        "region have historically resolved within 2-3 weeks.",
        "F03",
    ),
    (
        "E03",
        "Port Closure Near India Casing Plant",
        "Logistics",
        "A nearby port closure has delayed outbound shipments from the India "
        "Casing Plant by an estimated 10-12 days. Air freight is being considered "
        "as a costly but faster alternative for urgent orders.",
        "F08",
    ),
    (
        "E04",
        "Fire at Mexico Chip Sub-fab",
        "Fire",
        "A fire broke out at the Mexico Chip Sub-fab, halting all output. "
        "Investigators estimate a 5-6 week closure while the facility is "
        "rebuilt. This sub-fab supplies raw semiconductor material further "
        "up the chain, so downstream chip production may also be affected.",
        "F09",
    ),
]

# ---------------------------------------------------------------------------
# Write everything out as CSV files
# ---------------------------------------------------------------------------


def write_csv(filename, header, rows):
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {path}")


write_csv("countries.csv", ["id", "name"], COUNTRIES)
write_csv("facilities.csv", ["id", "name", "country_id"], FACILITIES)
write_csv("suppliers.csv", ["id", "name", "tier", "facility_id"], SUPPLIERS)
write_csv("sub_supplier_edges.csv", ["child_id", "parent_id"], SUB_SUPPLIER_EDGES)
write_csv("components.csv", ["id", "name", "supplier_id"], COMPONENTS)
write_csv("products.csv", ["id", "name"], PRODUCTS)
write_csv("requires_edges.csv", ["product_id", "component_id"], REQUIRES_EDGES)
write_csv("events.csv", ["id", "title", "type", "description", "facility_id"], EVENTS)

print("\nDone.")