import json
import re
import hashlib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ── Case-Insensitive Component Normalizer ───────────────────────────────────
class ComponentNormalizer:
    def __init__(self):
        logging.info("Using case-insensitive matching with deterministic hashing...")
        self.canonical_map = {}  # { "lowercase_key": "canonical_id" }
        self.entities = {}       # { "canonical_id": {"name": str, "aliases": set} }

    def _generate_id(self, text: str) -> str:
        """Generate a stable, deterministic ID using MD5 hashing"""
        clean = re.sub(r'[^a-z0-9\s]', '', text.lower()).strip()
        words = clean.split()
        
        base_id = "_".join(words[:2]) if words else "unknown"
        
        # Use MD5 for stable, order-independent IDs
        stable = hashlib.md5(clean.encode()).hexdigest()[:6]
        
        return f"{base_id}_{stable}"

    def normalize(self, raw_name: str) -> str:
        """Normalize component name using case-insensitive exact matching"""
        if not raw_name or not isinstance(raw_name, str): 
            return "unknown"
        
        raw_name = raw_name.strip()
        if not raw_name: 
            return "unknown"
        
        # Use lowercase for case-insensitive matching
        lookup_key = raw_name.lower()
        
        # Check if we've already seen this component (case-insensitive)
        if lookup_key in self.canonical_map: 
            return self.canonical_map[lookup_key]
        
        # Generate a new deterministic ID for this component
        new_id = self._generate_id(raw_name)
        
        self.entities[new_id] = {"name": raw_name, "aliases": {raw_name}}
        self.canonical_map[lookup_key] = new_id
        
        logging.debug(f"New component: '{raw_name}' -> '{new_id}'")
        return new_id


# ── Stable Fault ID Generator ─────────────────────────────────
def generate_fault_id(symptom: str) -> str:
    """Generate stable, reproducible fault IDs using MD5 hashing"""
    clean = re.sub(r"[^a-z0-9\s]", "", symptom.lower())
    slug = "_".join(clean.split()[:4])
    
    # Use MD5 for stable, reproducible IDs
    stable = hashlib.md5(clean.encode()).hexdigest()[:8]
    
    return f"fault_{slug}_{stable}"


def normalize_component_graph(graph_data, normalizer):
    """Normalize all component references in the component graph"""
    if not graph_data:
        return graph_data
    
    # Handle list format
    if isinstance(graph_data, list):
        normalized_components = []
        for component in graph_data:
            if isinstance(component, dict):
                normalized_comp = {}
                
                # Normalize the main component name/id
                comp_name = None
                if "component" in component:
                    comp_name = component["component"]
                    normalized_comp["component"] = comp_name
                elif "name" in component:
                    comp_name = component["name"]
                    normalized_comp["name"] = comp_name
                elif "id" in component:
                    comp_name = component["id"]
                    normalized_comp["id"] = component["id"]
                
                if comp_name:
                    normalized_comp["component_id"] = normalizer.normalize(comp_name)
                
                # Normalize connected components
                normalized_connections = []
                if "connected_components" in component:
                    for conn in component["connected_components"]:
                        if isinstance(conn, str):
                            normalized_connections.append({
                                "name": conn,
                                "component_id": normalizer.normalize(conn)
                            })
                        elif isinstance(conn, dict):
                            conn_copy = conn.copy()
                            if "name" in conn_copy:
                                conn_copy["component_id"] = normalizer.normalize(conn_copy["name"])
                            elif "component" in conn_copy:
                                conn_copy["component_id"] = normalizer.normalize(conn_copy["component"])
                            normalized_connections.append(conn_copy)
                
                normalized_comp["connected_components"] = normalized_connections
                
                # Build embedding text (always, even without connections)
                if comp_name:
                    if normalized_connections:
                        connection_names = []
                        for c in normalized_connections:
                            if isinstance(c, dict):
                                connection_names.append(c.get("name", str(c)))
                            else:
                                connection_names.append(str(c))
                        connections_text = "\n- ".join(connection_names)
                    else:
                        connections_text = "None"
                    
                    normalized_comp["embedding_text"] = f"""COMPONENT:
{comp_name}

CONNECTED TO:
- {connections_text}
"""
                
                # Normalize parent/child relationships
                if "parent" in component and isinstance(component["parent"], str):
                    normalized_comp["parent_id"] = normalizer.normalize(component["parent"])
                
                if "children" in component:
                    normalized_children = []
                    for child in component["children"]:
                        if isinstance(child, str):
                            normalized_children.append({
                                "name": child,
                                "component_id": normalizer.normalize(child)
                            })
                        elif isinstance(child, dict):
                            child_copy = child.copy()
                            if "name" in child_copy:
                                child_copy["component_id"] = normalizer.normalize(child_copy["name"])
                            normalized_children.append(child_copy)
                    normalized_comp["children"] = normalized_children
                
                # Normalize dependencies if present
                if "depends_on" in component:
                    normalized_deps = []
                    for dep in component["depends_on"]:
                        if isinstance(dep, str):
                            normalized_deps.append({
                                "name": dep,
                                "component_id": normalizer.normalize(dep)
                            })
                        elif isinstance(dep, dict):
                            dep_copy = dep.copy()
                            if "name" in dep_copy:
                                dep_copy["component_id"] = normalizer.normalize(dep_copy["name"])
                            normalized_deps.append(dep_copy)
                    normalized_comp["depends_on"] = normalized_deps
                
                # Preserve all other fields
                for key, value in component.items():
                    if key not in normalized_comp:
                        normalized_comp[key] = value
                
                normalized_components.append(normalized_comp)
        
        return normalized_components
    
    # Handle dict format (single component graph)
    elif isinstance(graph_data, dict):
        normalized_graph = {}
        
        for comp_name, comp_data in graph_data.items():
            normalized_comp_id = normalizer.normalize(comp_name)
            
            if isinstance(comp_data, dict):
                normalized_comp_data = comp_data.copy()
                normalized_comp_data["name"] = comp_name
                normalized_comp_data["component_id"] = normalized_comp_id
                
                # Normalize connections
                normalized_connections = []
                if "connected_components" in comp_data:
                    for conn in comp_data["connected_components"]:
                        if isinstance(conn, str):
                            normalized_connections.append({
                                "name": conn,
                                "component_id": normalizer.normalize(conn)
                            })
                
                normalized_comp_data["connected_components"] = normalized_connections
                
                # Build embedding text (always, even without connections)
                if normalized_connections:
                    connection_names = [c["name"] for c in normalized_connections]
                    connections_text = "\n- ".join(connection_names)
                else:
                    connections_text = "None"
                
                normalized_comp_data["embedding_text"] = f"""COMPONENT:
{comp_name}

CONNECTED TO:
- {connections_text}
"""
                
                normalized_graph[normalized_comp_id] = normalized_comp_data
            
            elif isinstance(comp_data, list):
                # Component name maps to a list of connected components
                normalized_connections = []
                for conn in comp_data:
                    if isinstance(conn, str):
                        normalized_connections.append({
                            "name": conn,
                            "component_id": normalizer.normalize(conn)
                        })
                
                # Build embedding text (always, even without connections) - FIX 2
                if normalized_connections:
                    connection_names = [c["name"] for c in normalized_connections]
                    connections_text = "\n- ".join(connection_names)
                else:
                    connections_text = "None"
                
                normalized_graph[normalized_comp_id] = {
                    "name": comp_name,
                    "component_id": normalized_comp_id,
                    "connected_components": normalized_connections,
                    "embedding_text": f"""COMPONENT:
{comp_name}

CONNECTED TO:
- {connections_text}
"""
                }
        
        return normalized_graph
    
    return graph_data


def enrich_database(target_folder: str, base_filename: str):
    folder = Path(target_folder)
    
    # Load all DBs
    db_paths = {
        "faults": folder / f"{base_filename}_fault_library.json",
        "procedures": folder / f"{base_filename}.json",
        "repairs": folder / f"{base_filename}_repair_procedures.json",
        "images": folder / f"{base_filename}_images.json",
        "index": folder / f"{base_filename}_knowledge_index.json",
        "graph": folder / f"{base_filename}_component_graph.json",
        "trees": folder / f"{base_filename}_diagnostic_trees.json",
        "specs": folder / f"{base_filename}_spec_database.json",
    }
    
    dbs = {}
    for name, path in db_paths.items():
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f: 
                dbs[name] = json.load(f)
            logging.info(f"Loaded: {path.name}")
        else:
            logging.warning(f"File not found: {path}")
            dbs[name] = None if name in ["graph", "trees", "specs"] else []

    # Use case-insensitive matching with deterministic hashing
    normalizer = ComponentNormalizer()

    # 1. Generate Fault IDs and structured embedding text
    logging.info("Enriching Fault Library...")
    fault_map = {}  # map old symptom string to new fault_id
    
    for f in dbs.get("faults", []):
        f_id = generate_fault_id(f.get("symptom", "unknown"))
        f["fault_id"] = f_id
        fault_map[f.get("symptom")] = f_id
        
        # Normalize component using case-insensitive matching
        raw_comp = f.get("component", "")
        f["component_id"] = normalizer.normalize(raw_comp)

        # Build Structured Embedding Text
        causes = "\n- ".join(f.get("likely_causes", []))
        verify = "\n- ".join(f.get("verify", []))
        repair = "\n- ".join(f.get("repair", []))
        
        f["embedding_text"] = f"""FAULT: {f.get('symptom')}
SYMPTOM: {f.get('problem_description', '')}
COMPONENT: {raw_comp}
LIKELY CAUSES:
- {causes}
VERIFY:
- {verify}
REPAIR:
- {repair}
"""

    # 2. Enrich Images (preserve original field names)
    logging.info("Enriching Image Library...")
    for img in dbs.get("images", []):
        # Use original field names: components_shown, search_keywords, fault_relevance
        visible = "\n- ".join(img.get("components_shown", []))
        
        # Combine repair_relevance and fault_relevance for embeddings
        repair_rel = "\n- ".join(img.get("repair_relevance", []))
        fault_rel = "\n- ".join(img.get("fault_relevance", []))
        keywords = "\n- ".join(img.get("search_keywords", []))
        
        useful_for_parts = []
        if img.get("repair_relevance"):
            useful_for_parts.append(f"REPAIR:\n- {repair_rel}")
        if img.get("fault_relevance"):
            useful_for_parts.append(f"FAULTS:\n- {fault_rel}")
        
        useful_for = "\n".join(useful_for_parts) if useful_for_parts else ""
        
        img["embedding_text"] = f"""IMAGE:
{img.get('caption', '')}
VISIBLE COMPONENTS:
- {visible}
KEYWORDS:
- {keywords}
USEFUL FOR:
{useful_for}
"""

    # 3. Update Knowledge Index with embedding rebuild
    logging.info("Updating Knowledge Index...")
    for entry in dbs.get("index", []):
        # Swap string fault_ids for the new stable snake_case IDs
        new_fault_ids = []
        for old_sym in entry.get("fault_ids", []):
            if old_sym in fault_map:
                new_fault_ids.append(fault_map[old_sym])
        entry["fault_ids"] = new_fault_ids
        
        # Normalize the main component using case-insensitive matching
        comp_name = entry.get("component", "")
        entry["component_id"] = normalizer.normalize(comp_name)
        
        # Add embedding rebuild for knowledge index
        faults_text = "\n- ".join(entry.get("fault_ids", []))
        connected_text = "\n- ".join(entry.get("connected_components", []))
        
        entry["embedding_text"] = f"""INDEX
COMPONENT: {comp_name}

FAULT IDS:
- {faults_text}

CONNECTED:
- {connected_text}
"""

    # 4. Normalize Component Graph
    if dbs.get("graph"):
        logging.info("Normalizing Component Graph for AR & dependency tracing...")
        dbs["graph"] = normalize_component_graph(dbs["graph"], normalizer)
        
        # Log summary of normalized graph
        if isinstance(dbs["graph"], list):
            component_count = len(dbs["graph"])
            total_connections = sum(
                len(comp.get("connected_components", [])) 
                for comp in dbs["graph"] 
                if isinstance(comp, dict)
            )
        elif isinstance(dbs["graph"], dict):
            component_count = len(dbs["graph"])
            total_connections = sum(
                len(data.get("connected_components", [])) 
                if isinstance(data, dict) 
                else len(data) 
                for data in dbs["graph"].values()
            )
        else:
            component_count = 0
            total_connections = 0
        
        logging.info(f"✓ Normalized graph with {component_count} components and {total_connections} connections")

    # 5. FIX 1: Enhanced Diagnostic Trees with nested steps support
    if dbs.get("trees"):
        logging.info("Normalizing Diagnostic Trees...")
        
        if isinstance(dbs["trees"], list):
            for tree in dbs["trees"]:
                comp = tree.get("component", "")
                symptom = tree.get("symptom", "")
                
                tree["component_id"] = normalizer.normalize(comp)
                
                if symptom:
                    tree["fault_id"] = generate_fault_id(symptom)
                
                # Extract richer nested data safely
                questions_list = []
                checks_list = []
                actions_list = []
                
                # Get top-level arrays
                questions_list.extend(tree.get("questions", []))
                checks_list.extend(tree.get("checks", []))
                actions_list.extend(tree.get("actions", []))
                
                # Extract from nested steps structure
                for step in tree.get("steps", []):
                    if isinstance(step, dict):
                        if step.get("question"):
                            questions_list.append(step["question"])
                        
                        if step.get("check"):
                            checks_list.append(step["check"])
                        
                        if step.get("action"):
                            actions_list.append(step["action"])
                        
                        if step.get("yes"):
                            actions_list.append(f"YES → {step['yes']}")
                        
                        if step.get("no"):
                            actions_list.append(f"NO → {step['no']}")
                
                questions = "\n- ".join(questions_list) if questions_list else ""
                checks = "\n- ".join(checks_list) if checks_list else ""
                actions = "\n- ".join(actions_list) if actions_list else ""
                
                tree["embedding_text"] = f"""DIAGNOSTIC TREE
SYMPTOM: {symptom}
COMPONENT: {comp}

QUESTIONS:
- {questions}

CHECKS:
- {checks}

ACTIONS:
- {actions}
"""
        logging.info(f"✓ Added embeddings to {len(dbs['trees'])} diagnostic trees")

    # 6. Normalize Spec Database with fallback field names
    if dbs.get("specs"):
        logging.info("Normalizing Spec Database...")
        
        if isinstance(dbs["specs"], list):
            for spec in dbs["specs"]:
                comp = spec.get("component", "")
                
                spec["component_id"] = normalizer.normalize(comp)
                
                # Handle both field name variants
                failures = "\n- ".join(
                    spec.get("failure_if_out_of_range")
                    or spec.get("if_out_of_range")
                    or []
                )
                
                repairs = "\n- ".join(
                    spec.get("repair_actions", [])
                )
                
                spec["embedding_text"] = f"""SPECIFICATION
COMPONENT: {comp}
TYPE: {spec.get("spec_type")}
VALUE: {spec.get("value")} {spec.get("unit")}

FAILS IF:
- {failures}

REPAIR:
- {repairs}
"""
        logging.info(f"✓ Added embeddings to {len(dbs['specs'])} specifications")

    enriched_dir = folder / "enriched_json"
    enriched_dir.mkdir(exist_ok=True)
    logging.info(f"Saving enriched databases to {enriched_dir}...")
    # Save enriched copies (don't overwrite originals)
    logging.info("Saving enriched databases...")
    
    enriched_paths = {
        "faults": enriched_dir / f"{base_filename}_fault_library_enriched.json",
        "procedures": enriched_dir / f"{base_filename}_enriched.json",
        "repairs": enriched_dir / f"{base_filename}_repair_procedures_enriched.json",
        "images": enriched_dir / f"{base_filename}_images_enriched.json",
        "index": enriched_dir / f"{base_filename}_knowledge_index_enriched.json",
        "graph": enriched_dir / f"{base_filename}_component_graph_enriched.json",
        "trees": enriched_dir / f"{base_filename}_diagnostic_trees_enriched.json",
        "specs": enriched_dir / f"{base_filename}_spec_database_enriched.json",
    }
    
    for name, path in enriched_paths.items():
        if name in dbs and dbs[name] is not None:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(dbs[name], f, indent=2, ensure_ascii=False)
            logging.info(f"✓ Saved: {path.name}")

    logging.info("✅ Database enrichment complete!")
    logging.info(f"Generated {len(normalizer.entities)} canonical components with deterministic IDs.")
    logging.info(f"Original files preserved. Enriched copies saved with '_enriched' suffix.")
    
    # Print summary of component graph if available
    if dbs.get("graph"):
        logging.info("Component Graph ready for:")
        logging.info("  • AR component relationships")
        logging.info("  • 'Show me what connects to impeller' queries")
        logging.info("  • Dependency tracing")
        logging.info("  • Smarter troubleshooting")


if __name__ == "__main__":
    TARGET_FOLDER = r"D:\AgriFix_Workspace\AgriFixAR_Python_Client\database_creation\Elect_Water_Pump_Pdfs"
    BASE_FILENAME = "Master_Water_Pump_DB"
    enrich_database(TARGET_FOLDER, BASE_FILENAME)