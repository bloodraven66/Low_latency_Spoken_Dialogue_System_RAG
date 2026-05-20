#!/bin/bash

set -euo pipefail

RAG_ROOT="/mnt/matylda4/udupa/exps/RAG"
CYPHER_SHELL="${CYPHER_SHELL:-$RAG_ROOT/neo4j/app/bin/cypher-shell}"
NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-neo4jlocal123}"

if [ ! -x "$CYPHER_SHELL" ]; then
  echo "ERROR: cypher-shell not found or not executable at: $CYPHER_SHELL"
  exit 1
fi

run_query() {
  local title="$1"
  local question="$2"
  local query="$3"

  echo
  echo "============================================================"
  echo "Template: $title"
  echo "Q: $question"
  echo "------------------------------------------------------------"
  "$CYPHER_SHELL" -a "$NEO4J_URI" -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" "$query"
}

echo "Running live Neo4j template query checks..."
echo "URI=$NEO4J_URI, USER=$NEO4J_USER"

run_query \
  "person_contact" \
  "What is the email and office of Barina David?" \
  "MATCH (p:Person {profile_id:'barina david'}) RETURN p.profile_id AS person_id, p.name AS name, p.email AS email, p.office_room AS office_room, p.fit_profile_url AS fit_profile_url, p.fit_person_uid AS fit_uid;"

run_query \
  "course_semester" \
  "In which semester is BAYa taught?" \
  "MATCH (c:Course {code:'BAYa'}) RETURN c.code AS course_code, c.course_name AS course_name, c.semester_label AS semester_label, c.semester_norm AS semester_norm;"

run_query \
  "course_staff" \
  "Who guarantees and lectures BAYa?" \
  "MATCH (p:Person)-[r:GUARANTEES|LECTURES]->(c:Course {code:'BAYa'}) RETURN type(r) AS relation, p.profile_id AS person_id, p.name AS name, p.email AS email, p.office_room AS office_room, r.source AS source, r.confidence AS confidence ORDER BY relation, name;"

run_query \
  "programme_courses_winter" \
  "List winter courses in MIT-EN." \
  "MATCH (pr:Programme {programme_id:'mit-en'})-[:HAS_COURSE]->(c:Course) WHERE c.semester_norm = 'winter' RETURN c.code AS code, c.course_name AS course_name ORDER BY code LIMIT 15;"

run_query \
  "person_projects" \
  "Which projects does Barina David work on?" \
  "MATCH (p:Person {profile_id:'barina david'})-[r:WORKS_ON]->(pr:Project) RETURN pr.title AS project_title, r.source AS source, r.confidence AS confidence ORDER BY r.confidence DESC, project_title LIMIT 10;"

run_query \
  "person_publications" \
  "Which publications is Barina David linked to?" \
  "MATCH (p:Person {profile_id:'barina david'})-[r:AUTHORED]->(pub:Publication) RETURN pub.title AS publication_title, pub.year AS year, r.source AS source, r.confidence AS confidence ORDER BY year DESC, publication_title LIMIT 10;"

run_query \
  "publication_cross_links" \
  "What is publication 'Focus-aware compression and image quality metric for 3D displays' linked to?" \
  "MATCH (pub:Publication {publication_id:'focus-aware compression and image quality metric for 3d displays'}) OPTIONAL MATCH (pub)-[:RELATED_TO_PROJECT]->(pr:Project) OPTIONAL MATCH (pub)-[:RELATED_TO_GROUP]->(g:ResearchGroup) OPTIONAL MATCH (pub)-[:RELATED_TO_DEPARTMENT]->(d:Department) RETURN pub.title AS publication_title, pr.title AS project_title, g.group AS group_name, d.name AS department_name;"

run_query \
  "evidence_distribution" \
  "How many edges do we have per source tier for person-linked relations?" \
  "MATCH ()-[r]->() WHERE type(r) IN ['GUARANTEES','LECTURES','MEMBER_OF','WORKS_ON','AUTHORED'] RETURN type(r) AS relation, coalesce(r.source, 'none') AS source, count(*) AS edge_count, round(avg(coalesce(r.confidence,0.0))*1000)/1000.0 AS avg_confidence ORDER BY relation, edge_count DESC;"

echo
echo "Done."
