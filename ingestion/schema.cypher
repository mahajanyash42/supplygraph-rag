// =============================================================================
// schema.cypher
// Constraints, indexes, and the native vector index used by SupplyGraph.
// Run this once against a fresh Neo4j database before load_graph.py.
// =============================================================================

// --- Uniqueness constraints (also create backing indexes) -------------------
CREATE CONSTRAINT country_id   IF NOT EXISTS FOR (n:Country)   REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT facility_id  IF NOT EXISTS FOR (n:Facility)  REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT supplier_id  IF NOT EXISTS FOR (n:Supplier)  REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT component_id IF NOT EXISTS FOR (n:Component) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT product_id   IF NOT EXISTS FOR (n:Product)   REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT event_id     IF NOT EXISTS FOR (n:Event)     REQUIRE n.id IS UNIQUE;

// --- Vector index over Event embeddings --------------------------------------
// 384 dims matches sentence-transformers/all-MiniLM-L6-v2 (see rag/vector_retriever.py).
// If you swap in a different embedding model, update `vector.dimensions` to match.
CREATE VECTOR INDEX event_embeddings IF NOT EXISTS
FOR (e:Event) ON (e.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine'
  }
};

// =============================================================================
// Example teaching queries — run these directly in Neo4j Browser BEFORE
// introducing any LLM. This builds graph intuition first.
// =============================================================================

// Q1. Which products depend, directly or indirectly, on a given country?
// (4-hop traversal: Product -> Component -> Supplier -> Facility -> Country)
//
// MATCH (p:Product)-[:REQUIRES]->(:Component)-[:SUPPLIED_BY]->(s:Supplier)
//       -[:OPERATES_AT]->(:Facility)-[:LOCATED_IN]->(c:Country {name: 'Vietnam'})
// RETURN DISTINCT p.name;

// Q2. Full upstream chain for a disruption event: which products are affected?
//
// MATCH (e:Event {id: 'E001'})-[:IMPACTS]->(f:Facility)
// MATCH (f)<-[:OPERATES_AT]-(s:Supplier)
// OPTIONAL MATCH (s)<-[:SUB_SUPPLIER_OF*0..2]-(upstream:Supplier)
// MATCH (comp:Component)-[:SUPPLIED_BY]->(s)
// MATCH (p:Product)-[:REQUIRES]->(comp)
// RETURN e.title, f.name, collect(DISTINCT s.name) AS suppliers, collect(DISTINCT p.name) AS affected_products;

// Q3. Which tier-2/3 sub-suppliers ultimately feed a given tier-1 supplier?
//
// MATCH (sub:Supplier)-[:SUB_SUPPLIER_OF*1..3]->(top:Supplier {name: 'Northline Industries'})
// RETURN sub.name, sub.tier;

// Q4. Countries with the highest concentration of active disruption events.
//
// MATCH (e:Event)-[:IMPACTS]->(f:Facility)-[:LOCATED_IN]->(c:Country)
// RETURN c.name, count(DISTINCT e) AS event_count ORDER BY event_count DESC;
