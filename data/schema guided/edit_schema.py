import json

with open(r"C:\ALL\OJT\SERVER\gradient_server_test\data\schema guided\old_schema.json") as f:
    ontology = json.load(f)
new_ontology = {}
for domain, slot_des in ontology.items():
    new_ontology.setdefault(domain,{})
    for slot, des in slot_des.items():
        new_ontology[domain].setdefault(des, [slot])
with open(r"C:\ALL\OJT\SERVER\gradient_server_test\data\schema guided\schema.json", 'w') as f:
    json.dump(new_ontology, f, indent=4)