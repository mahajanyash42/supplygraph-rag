"""
load_graph.py

Reads the CSVs from data/csv/ and loads them into Neo4j as nodes and
relationships. Run this AFTER generate_data.py, and after the constraints
from Step 4a have been created in Neo4j Browser.

Run with:
    python load_graph.py
"""

import csv
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

CSV_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "csv")


def read_csv(filename):
    path = os.path.join(CSV_DIR, filename)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_nodes(tx, label, rows):
    """MERGE means: create it if it doesn't exist, otherwise leave it alone.
    This is what makes it safe to re-run this script without duplicating data."""
    query = f"""
    UNWIND $rows AS row
    MERGE (n:{label} {{id: row.id}})
    SET n += row
    """
    tx.run(query, rows=rows)


def main():
    countries = read_csv("countries.csv")
    facilities = read_csv("facilities.csv")
    suppliers = read_csv("suppliers.csv")
    sub_supplier_edges = read_csv("sub_supplier_edges.csv")
    components = read_csv("components.csv")
    products = read_csv("products.csv")
    requires_edges = read_csv("requires_edges.csv")
    events = read_csv("events.csv")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    with driver.session() as session:
        print("Loading nodes...")
        session.execute_write(load_nodes, "Country", countries)
        session.execute_write(load_nodes, "Facility", facilities)
        session.execute_write(load_nodes, "Supplier", suppliers)
        session.execute_write(load_nodes, "Component", components)
        session.execute_write(load_nodes, "Product", products)
        session.execute_write(load_nodes, "Event", events)

        print("Linking relationships...")

        session.run("""
            UNWIND $rows AS row
            MATCH (f:Facility {id: row.id})
            MATCH (c:Country {id: row.country_id})
            MERGE (f)-[:LOCATED_IN]->(c)
        """, rows=facilities)

        session.run("""
            UNWIND $rows AS row
            MATCH (s:Supplier {id: row.id})
            MATCH (f:Facility {id: row.facility_id})
            MERGE (s)-[:OPERATES_AT]->(f)
        """, rows=suppliers)

        session.run("""
            UNWIND $rows AS row
            MATCH (child:Supplier {id: row.child_id})
            MATCH (parent:Supplier {id: row.parent_id})
            MERGE (child)-[:SUB_SUPPLIER_OF]->(parent)
        """, rows=sub_supplier_edges)

        session.run("""
            UNWIND $rows AS row
            MATCH (comp:Component {id: row.id})
            MATCH (s:Supplier {id: row.supplier_id})
            MERGE (comp)-[:SUPPLIED_BY]->(s)
        """, rows=components)

        session.run("""
            UNWIND $rows AS row
            MATCH (p:Product {id: row.product_id})
            MATCH (comp:Component {id: row.component_id})
            MERGE (p)-[:REQUIRES]->(comp)
        """, rows=requires_edges)

        session.run("""
            UNWIND $rows AS row
            MATCH (e:Event {id: row.id})
            MATCH (f:Facility {id: row.facility_id})
            MERGE (e)-[:IMPACTS]->(f)
        """, rows=events)

    driver.close()
    print("\nDone! Go check Neo4j Browser.")


if __name__ == "__main__":
    main()